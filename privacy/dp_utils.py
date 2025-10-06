"""
Utilities for DP-SGD training (vectorized per-sample gradients with torch.func.vmap)
- Strict per-sample clipping to C
- Add Gaussian noise on averaged gradients: std = sigma * C / B
- Epsilon accounting via Opacus if available (fallback provided)
"""

from typing import Dict, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- privacy accounting ----
try:
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
except Exception:
    OpacusRDPAccountant = None


def compute_epsilon_opacus(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """Prefer Opacus accountant; fallback to a crude RDP if not available."""
    if OpacusRDPAccountant is None:
        # very rough fallback ignoring sampling: Gaussian mech RDP
        if noise_multiplier <= 0:
            return float("inf")
        alphas = np.arange(2, 64, 1.0)
        rdps = [steps * (a / (2 * noise_multiplier ** 2)) for a in alphas]
        eps = [r + math.log(1.0 / delta) / (a - 1.0) for a, r in zip(alphas, rdps)]
        return float(min(eps))
    acc = OpacusRDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(acc.get_epsilon(delta))


def solve_noise_from_epsilon_opacus(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """Binary-search sigma so that epsilon ~= target_epsilon (Opacus accountant)."""
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        eps = compute_epsilon_opacus(mid, sample_rate, steps, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ---- vectorized per-sample grads with torch.func ----
try:
    from torch.func import functional_call, vmap, grad as func_grad
    _FUNC_AVAILABLE = True
except Exception:
    _FUNC_AVAILABLE = False


def _named_param_buffer_dicts(model: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: b for k, b in model.named_buffers()}
    return params, buffers


def _flatten_grads(grads: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    return [g for g in grads.values() if g is not None]


def _per_sample_grad_images(model: nn.Module,
                            params: Dict[str, torch.Tensor],
                            buffers: Dict[str, torch.Tensor],
                            x: torch.Tensor,
                            y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Vectorized per-sample grads for standard image classifier: logits = model(x)
    """
    def loss_of_single(params_, buffers_, x_i, y_i):
        logits = functional_call(model, params_, buffers_, (x_i.unsqueeze(0),))
        if isinstance(logits, tuple):  # (logits, emb)
            logits = logits[0]
        loss = F.cross_entropy(logits, y_i.unsqueeze(0), reduction='mean')
        return loss

    grad_fn = func_grad(loss_of_single)
    # vmap over batch dimension of x, y; params/buffers are shared (None)
    grads_pytree = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)
    return grads_pytree  # dict(name-> per-sample grad tensor with same shape as param); stacked over batch


def _per_sample_grad_fusion(model: nn.Module,
                            params: Dict[str, torch.Tensor],
                            buffers: Dict[str, torch.Tensor],
                            e4: torch.Tensor,
                            e6: torch.Tensor,
                            y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Vectorized per-sample grads for fusion model: logits = model(e4, e6)
    """
    def loss_of_single(params_, buffers_, e4_i, e6_i, y_i):
        logits = functional_call(model, params_, buffers_, (e4_i.unsqueeze(0), e6_i.unsqueeze(0)))
        loss = F.cross_entropy(logits, y_i.unsqueeze(0), reduction='mean')
        return loss

    grad_fn = func_grad(loss_of_single)
    grads_pytree = vmap(grad_fn, in_dims=(None, None, 0, 0, 0))(params, buffers, e4, e6, y)
    return grads_pytree


def _clip_and_aggregate(per_sample_grads: Dict[str, torch.Tensor], C: float) -> Dict[str, torch.Tensor]:
    """
    per_sample_grads[name] has shape [B, ...]; do per-sample L2 norm across all params,
    clip to C, then average over batch.
    """
    # compute per-sample squared norms over all params
    sq_sums = None
    for g in per_sample_grads.values():
        if g is None:
            continue
        # sum over param axes
        g_sq = g.flatten(start_dim=1).pow(2).sum(dim=1)  # [B]
        sq_sums = g_sq if sq_sums is None else (sq_sums + g_sq)
    norms = sq_sums.sqrt()  # [B]

    # scaling factors
    coef = (C / (norms + 1e-12)).clamp(max=1.0)  # [B]

    # apply scaling and average
    agg: Dict[str, torch.Tensor] = {}
    B = norms.numel()
    for name, g in per_sample_grads.items():
        if g is None:
            agg[name] = None
        else:
            # scale each sample, then mean over batch
            # reshape coef for broadcasting: [B, 1, 1, ...]
            while coef.dim() < g.dim():
                coef_ = coef.view(-1, *([1] * (g.dim() - 1)))
                break
            scaled = g * coef_
            agg[name] = scaled.mean(dim=0)
    return agg


class DPOptimizer:
    """
    Vectorized per-sample DP-SGD:
      1) compute per-sample grads with torch.func.vmap
      2) clip each sample to C and average
      3) add Gaussian noise on averaged grads (std = sigma*C/B)
      4) write to .grad and step()
    """
    def __init__(self,
                 model: nn.Module,
                 base_optimizer: torch.optim.Optimizer,
                 noise_multiplier: float,
                 max_grad_norm: float = 1.0):
        if not _FUNC_AVAILABLE:
            raise RuntimeError("torch.func is not available. Please use PyTorch >= 2.0")
        self.model = model
        self.opt = base_optimizer
        self.sigma = float(noise_multiplier)
        self.C = float(max_grad_norm)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _add_noise_on_average(self, batch_size: int):
        if self.sigma <= 0:
            return
        std = (self.sigma * self.C) / max(1, batch_size)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            p.grad.add_(torch.normal(0.0, std, size=p.grad.shape, device=p.grad.device, dtype=p.grad.dtype))

    def dp_step_images(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """DP-SGD step for standard (image) classifier."""
        params, buffers = _named_param_buffer_dicts(self.model)
        # per-sample grads; dict[name] shape [B, ...]
        grads_ps = _per_sample_grad_images(self.model, params, buffers, x, y)
        # average clipped grads
        agg = _clip_and_aggregate(grads_ps, self.C)

        # write averaged grads
        self.opt.zero_grad(set_to_none=True)
        for (name, p) in self.model.named_parameters():
            if p.requires_grad:
                p.grad = agg[name].to(p.device)

        # add noise on averaged gradient
        self._add_noise_on_average(batch_size=y.size(0))
        # update
        self.opt.step()

        # for logging: avg pre-clip norm
        # compute norms from grads_ps (before clipping)
        with torch.no_grad():
            sq_sums = None
            for g in grads_ps.values():
                if g is None: continue
                g_sq = g.flatten(start_dim=1).pow(2).sum(dim=1)
                sq_sums = g_sq if sq_sums is None else (sq_sums + g_sq)
            preclip = float(sq_sums.sqrt().mean().item()) if sq_sums is not None else 0.0
        return preclip

    def dp_step_fusion(self, e4: torch.Tensor, e6: torch.Tensor, y: torch.Tensor) -> float:
        """DP-SGD step for fusion model which takes two embeddings."""
        params, buffers = _named_param_buffer_dicts(self.model)
        grads_ps = _per_sample_grad_fusion(self.model, params, buffers, e4, e6, y)
        agg = _clip_and_aggregate(grads_ps, self.C)

        self.opt.zero_grad(set_to_none=True)
        for (name, p) in self.model.named_parameters():
            if p.requires_grad:
                p.grad = agg[name].to(p.device)

        self._add_noise_on_average(batch_size=y.size(0))
        self.opt.step()

        with torch.no_grad():
            sq_sums = None
            for g in grads_ps.values():
                if g is None: continue
                g_sq = g.flatten(start_dim=1).pow(2).sum(dim=1)
                sq_sums = g_sq if sq_sums is None else (sq_sums + g_sq)
            preclip = float(sq_sums.sqrt().mean().item()) if sq_sums is not None else 0.0
        return preclip


# ---- eval helper ----
def compute_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)
