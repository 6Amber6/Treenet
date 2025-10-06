"""
Utilities for DP-SGD training (per-sample clipping + correct noise scaling)
"""

from typing import Tuple, List, Dict
import math
import numpy as np
import torch
import torch.nn as nn

# -------- Privacy accounting (Opacus if available) --------
try:
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
except Exception:
    OpacusRDPAccountant = None


class PrivacyAccountant:
    """
    Simple fallback RDP accountant (Gaussian mech) when Opacus is unavailable.
    Used only if compute_epsilon_opacus can't import Opacus.
    """
    def __init__(self, noise_multiplier: float):
        self.sigma = float(noise_multiplier)

    def _rdp(self, alpha: float) -> float:
        if self.sigma <= 0:
            return float("inf")
        return alpha / (2.0 * (self.sigma ** 2))

    def epsilon(self, steps: int, delta: float) -> float:
        if self.sigma <= 0:
            return float("inf")
        alphas = np.arange(2, 64, 1.0)
        rdps = [steps * self._rdp(a) for a in alphas]
        eps = [r + math.log(1.0 / delta) / (a - 1.0) for a, r in zip(alphas, rdps)]
        return float(min(eps))


def compute_epsilon_opacus(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """
    Prefer Opacus RDP accountant with sampling rate q and steps.
    Fallback to simple accountant if Opacus isn't available.
    """
    if OpacusRDPAccountant is None:
        # crude fallback: ignore q, only use sigma & steps
        return PrivacyAccountant(noise_multiplier).epsilon(steps, delta)

    acc = OpacusRDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(acc.get_epsilon(delta))


def solve_noise_from_epsilon_opacus(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """
    Binary search sigma such that epsilon ~= target_epsilon under Opacus accountant.
    """
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        eps = compute_epsilon_opacus(mid, sample_rate, steps, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ------------------- Per-sample DP-SGD core -------------------
def _zeros_like_params(model: nn.Module):
    bufs = []
    for p in model.parameters():
        bufs.append(None if (not p.requires_grad) else torch.zeros_like(p, memory_format=torch.preserve_format))
    return bufs


def _add_inplace(dst, src):
    for d, s in zip(dst, src):
        if d is not None and s is not None:
            d.add_(s)


def _scale_inplace(bufs, scale: float):
    for b in bufs:
        if b is not None:
            b.mul_(scale)


def _grab_grads(model: nn.Module):
    out = []
    for p in model.parameters():
        out.append(None if (p.grad is None) else p.grad.detach().clone())
    return out


class DPOptimizer:
    """
    Strict per-sample DP-SGD (framework-free):
      For each batch of size B:
        1) Compute per-sample losses (reduction='none')
        2) For i in [0..B-1]:
             zero_grad(); loss_i.backward(retain_graph)
             collect grads_i; clip to C; sum into accumulator
        3) Average accumulator by B
        4) Add Gaussian noise with std = sigma*C/B to each averaged grad tensor
        5) Write to .grad and optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.opt = base_optimizer
        self.sigma = float(noise_multiplier)
        self.C = float(max_grad_norm)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _add_noise_on_average(self, batch_size: int):
        """Noise is added on averaged gradient: std = sigma * C / B"""
        if self.sigma <= 0:
            return
        std = (self.sigma * self.C) / max(1, batch_size)
        for p in self.model.parameters():
            if p.grad is None:
                continue
            noise = torch.normal(
                mean=0.0, std=std, size=p.grad.shape, device=p.grad.device, dtype=p.grad.dtype
            )
            p.grad.add_(noise)

    def dp_step(self, loss_per_sample: torch.Tensor, batch_size: int) -> float:
        """
        :param loss_per_sample: shape [B]
        :param batch_size:      int B
        :return: average pre-clip per-sample grad L2 norm (for logging only)
        """
        B = int(batch_size)
        preclip_norms = []
        agg = _zeros_like_params(self.model)

        # per-sample loop
        for i in range(B):
            self.opt.zero_grad(set_to_none=True)
            loss_per_sample[i].backward(retain_graph=(i < B - 1))

            grads_i = _grab_grads(self.model)

            # L2 norm
            tot = 0.0
            for g in grads_i:
                if g is not None:
                    tot += g.pow(2).sum()
            l2 = float(tot.sqrt().item())
            preclip_norms.append(l2)

            coef = min(1.0, self.C / (l2 + 1e-12))
            for g, a in zip(grads_i, agg):
                if g is not None and a is not None:
                    a.add_(g * coef)

        # average
        _scale_inplace(agg, 1.0 / max(1, B))

        # write averaged grads back
        self.opt.zero_grad(set_to_none=True)
        for p, a in zip(self.model.parameters(), agg):
            if p.requires_grad and a is not None:
                p.grad = a.to(p.device)

        # add noise on averaged grads
        self._add_noise_on_average(B)

        # update
        self.opt.step()

        return float(np.mean(preclip_norms)) if len(preclip_norms) else 0.0


# ------------------- Eval helper -------------------
def compute_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
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
