"""
Utilities for DP-SGD training (per-sample clipping + correct noise scaling)
"""

import math
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

try:
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
except Exception:
    OpacusRDPAccountant = None


# --------------------------
# Privacy accounting helpers
# --------------------------
class PrivacyAccountant:
    """Fallback RDP accountant (simple closed form for Gaussian mech)."""

    def __init__(self, noise_multiplier: float, batch_size: int, dataset_size: int):
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.sampling_rate = batch_size / max(1, dataset_size)

    def compute_rdp(self, alpha: float) -> float:
        if self.noise_multiplier == 0:
            return float("inf")
        # Gaussian mech RDP at order alpha
        return alpha / (2 * (self.noise_multiplier ** 2))

    def compute_epsilon(self, delta: float, steps: int) -> float:
        if self.noise_multiplier == 0:
            return float("inf")
        alphas = np.arange(2, 128, 1.0)
        rdps = [steps * self.compute_rdp(a) for a in alphas]
        eps = [r + math.log(1 / delta) / (a - 1) for a, r in zip(alphas, rdps)]
        return float(min(eps))

    def get_privacy_spent(self, steps: int, delta: float = 1e-5) -> Tuple[float, float]:
        return self.compute_epsilon(delta, steps), delta


def compute_epsilon_opacus(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """Prefer Opacus accountant when available; else fallback."""
    if OpacusRDPAccountant is None:
        acc = PrivacyAccountant(noise_multiplier, 1, 1)
        return acc.compute_epsilon(delta, steps)
    acc = OpacusRDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(acc.get_epsilon(delta))


def solve_noise_from_epsilon_opacus(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """Binary-search noise so that epsilon ~= target."""
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        eps = compute_epsilon_opacus(mid, sample_rate, steps, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return float(hi)


# --------------------------
# Per-sample DP-SGD optimizer
# --------------------------
def _clone_like_params(model: nn.Module):
    """Create zero buffers shaped like .grad for accumulation."""
    bufs = []
    for p in model.parameters():
        if p.requires_grad:
            bufs.append(torch.zeros_like(p, memory_format=torch.preserve_format))
        else:
            bufs.append(None)
    return bufs


def _accumulate(bufs, scale: float):
    """Scale all buffers by a factor (in-place)."""
    for b in bufs:
        if b is not None:
            b.mul_(scale)


def _add_inplace(dst, src):
    for d, s in zip(dst, src):
        if d is not None and s is not None:
            d.add_(s)


def _grad_list_from_model(model: nn.Module):
    grads = []
    for p in model.parameters():
        grads.append(None if (p.grad is None) else p.grad.detach().clone())
    return grads


class DPOptimizer:
    """
    Pure-PyTorch per-sample DP-SGD (no functorch / no Opacus dependency).
    Pipeline per batch:
      1) per-sample loss (reduction='none')
      2) loop each sample: backward -> get grads_i
      3) clip each grads_i to L2 norm C
      4) sum and average over batch
      5) add Gaussian noise N(0,(sigma*C)^2) to each param
      6) write averaged+noised grads to .grad and step()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float = 1.0,
        momentum_beta: float = 0.9,
        clip_constant: float = 1.0,  # kept for compatibility; not used in scale anymore
    ):
        self.model = model
        self.optimizer = optimizer
        self.sigma = float(noise_multiplier)
        self.C = float(max_grad_norm)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _add_noise(self):
        if self.sigma <= 0:
            return
        for p in self.model.parameters():
            if p.grad is None:
                continue
            noise = torch.normal(
                mean=0.0,
                std=self.sigma * self.C,  # NOTE: std = sigma * C
                size=p.grad.shape,
                device=p.grad.device,
                dtype=p.grad.dtype,
            )
            p.grad.add_(noise)

    def dp_step(self, loss_per_sample: torch.Tensor, batch_size: int):
        """
        Args:
          loss_per_sample: shape [B], unreduced CE losses for current batch
          batch_size:      actual batch size B
        Returns:
          pre_clip_norm_avg: average of per-sample grad norms before clipping（仅监控）
        """
        device = next(self.model.parameters()).device
        B = int(batch_size)
        preclip_norms = []

        # Accumulator for clipped per-sample gradients
        agg = _clone_like_params(self.model)

        # per-sample loop (plain but robust)
        for i in range(B):
            self.optimizer.zero_grad(set_to_none=True)
            loss_per_sample[i].backward(retain_graph=(i < B - 1))

            # Grab grads_i and compute its L2 norm
            grads_i = _grad_list_from_model(self.model)
            total = 0.0
            for g in grads_i:
                if g is not None:
                    total += g.pow(2).sum()
            l2 = total.sqrt()
            preclip_norms.append(float(l2.item()))

            # Clip & accumulate
            coeff = min(1.0, self.C / (l2 + 1e-12))
            for g, buf in zip(grads_i, agg):
                if g is not None:
                    if buf is None:
                        continue
                    buf.add_(g * coeff)

        # Average over batch
        _accumulate(agg, 1.0 / max(1, B))

        # Write averaged grads to .grad
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.model.parameters(), agg):
            if p.requires_grad and g is not None:
                p.grad = g.to(p.device)

        # Add Gaussian noise with std = sigma*C (per parameter)
        self._add_noise()

        # Step
        self.optimizer.step()

        # Return average pre-clip norm for logging
        return float(np.mean(preclip_norms)) if preclip_norms else 0.0


# --------------------------
# Accuracy / IO helpers
# --------------------------
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
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)
