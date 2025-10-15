# """
# Utilities for DP-SGD training including privacy accounting,
# gradient clipping, dataset processing, and per-sample DP-SGD step.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from typing import Tuple, List, Dict
# import math

# # ============================================================
# # Privacy Accountant
# # ============================================================
# class PrivacyAccountant:
#     """Privacy accountant for DP-SGD using RDP (Renyi Differential Privacy)."""

#     def __init__(self, noise_multiplier: float, batch_size: int, dataset_size: int):
#         self.noise_multiplier = noise_multiplier
#         self.batch_size = batch_size
#         self.dataset_size = dataset_size
#         self.sampling_rate = batch_size / dataset_size

#     def compute_rdp(self, alpha: float) -> float:
#         """Compute RDP for Gaussian mechanism at order alpha."""
#         if self.noise_multiplier == 0:
#             return float("inf")
#         return alpha / (2 * self.noise_multiplier ** 2)

#     def compute_epsilon(self, delta: float, steps: int) -> float:
#         """Compute epsilon from RDP given delta and steps."""
#         if self.noise_multiplier == 0:
#             return float("inf")
#         alphas = np.arange(2, 100, 0.5)
#         rdps = [steps * self.compute_rdp(alpha) for alpha in alphas]
#         epsilons = []
#         for alpha, rdp in zip(alphas, rdps):
#             eps = rdp + math.log(1 / delta) / (alpha - 1)
#             epsilons.append(eps)
#         return min(epsilons)

#     def get_privacy_spent(self, steps: int, delta: float = 1e-5) -> Tuple[float, float]:
#         """Return (epsilon, delta) spent after given number of steps."""
#         epsilon = self.compute_epsilon(delta, steps)
#         return epsilon, delta


# def compute_epsilon_opacus(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
#     """
#     Compute epsilon using simplified formula if Opacus not used.
#     """
#     if noise_multiplier == 0:
#         return float("inf")
#     return (steps * (sample_rate ** 2)) / (2 * noise_multiplier ** 2) + math.log(1/delta)


# # ============================================================
# # CIFAR-10 Data Utilities
# # ============================================================
# class FilteredDataset(torch.utils.data.Dataset):
#     """Custom dataset for filtered CIFAR-10 with remapped labels."""
#     def __init__(self, dataset, indices, new_targets):
#         self.dataset = dataset
#         self.indices = indices
#         self.targets = torch.tensor(new_targets)
    
#     def __len__(self):
#         return len(self.indices)
    
#     def __getitem__(self, idx):
#         data, _ = self.dataset[self.indices[idx]]
#         return data, self.targets[idx]


# class DataProcessor:
#     """CIFAR-10 dataset processing utilities."""

#     @staticmethod
#     def get_cifar10_classes() -> Dict[str, List[int]]:
#         return {
#             "animal_classes": [2, 3, 4, 5, 6, 7],   # bird, cat, deer, dog, frog, horse
#             "vehicle_classes": [0, 1, 8, 9],       # airplane, automobile, ship, truck
#             "all_classes": list(range(10)),
#         }

#     @staticmethod
#     def filter_dataset(dataset, target_classes: List[int]) -> torch.utils.data.Dataset:
#         """
#         Filter dataset to keep only given classes.
#         Also remap labels to 0..N-1 to match classifier output size.
#         """
#         class_to_idx = {cls: i for i, cls in enumerate(target_classes)}
#         indices, new_targets = [], []

#         for idx, (_, label) in enumerate(dataset):
#             if label in target_classes:
#                 indices.append(idx)
#                 new_targets.append(class_to_idx[label])

#         # Create a custom dataset that properly handles label remapping
#         return FilteredDataset(dataset, indices, new_targets)

#     @staticmethod
#     def create_data_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4) -> Dict:
#         import torchvision
#         import torchvision.transforms as transforms

#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.2023, 0.1994, 0.2010)),
#         ])
#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.2023, 0.1994, 0.2010)),
#         ])

#         train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
#         test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

#         mappings = DataProcessor.get_cifar10_classes()
#         animal_train = DataProcessor.filter_dataset(train_dataset, mappings["animal_classes"])
#         animal_test = DataProcessor.filter_dataset(test_dataset, mappings["animal_classes"])
#         vehicle_train = DataProcessor.filter_dataset(train_dataset, mappings["vehicle_classes"])
#         vehicle_test = DataProcessor.filter_dataset(test_dataset, mappings["vehicle_classes"])

#         loaders = {
#             "animal_train": torch.utils.data.DataLoader(animal_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
#             "animal_test": torch.utils.data.DataLoader(animal_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
#             "vehicle_train": torch.utils.data.DataLoader(vehicle_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
#             "vehicle_test": torch.utils.data.DataLoader(vehicle_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
#             "full_train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
#             "full_test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
#         }
#         return loaders


# # ============================================================
# # Accuracy Computation
# # ============================================================
# def compute_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
#     """Compute accuracy of a model on a dataset."""
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             if isinstance(output, tuple):
#                 output = output[0]
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             total += target.size(0)
#     return correct / total


# # ============================================================
# # Per-sample DP-SGD Step (torch.func version)
# # ============================================================
# from torch.func import vmap, grad, functional_call

# def dp_step_images(model, optimizer, x, y, noise_multiplier, max_grad_norm):
#     """
#     纯 PyTorch 实现的 DP-SGD 步骤：
#     - 每样本梯度计算 (torch.func.vmap)
#     - 逐样本 L2 裁剪
#     - 聚合 + 加噪声
#     """
#     from torch.func import vmap, grad, functional_call
#     model.train()
#     optimizer.zero_grad()

#     # 定义 loss 函数
#     def compute_loss(params, buffers, sample_x, sample_y):
#         logits = functional_call(model, (params, buffers), (sample_x.unsqueeze(0),))
#         logits = logits[0] if isinstance(logits, tuple) else logits
#         return F.cross_entropy(logits, sample_y.unsqueeze(0))

#     params = {k: v for k, v in model.named_parameters() if v.requires_grad}
#     buffers = {k: v for k, v in model.named_buffers()}

#     # 计算逐样本梯度 [B, ...]
#     grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x, y)

#     # 计算每个样本的梯度范数
#     per_sample_norms = torch.zeros(x.size(0), device=x.device)
#     for p in grads.values():
#         per_sample_norms += p.view(p.shape[0], -1).pow(2).sum(dim=1)
#     per_sample_norms = per_sample_norms.sqrt()

#     # 逐样本裁剪
#     scales = (max_grad_norm / (per_sample_norms + 1e-12)).clamp(max=1.0)
#     for k, p in grads.items():
#         grads[k] = p * scales.view(-1, *([1] * (p.ndim - 1)))

#     # 聚合 + 加噪
#     for (name, param) in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         grad_stack = grads[name]
#         grad_sum = grad_stack.sum(dim=0)
#         if noise_multiplier > 0:
#             noise = torch.randn_like(grad_sum) * noise_multiplier * max_grad_norm
#             grad_sum = grad_sum + noise
#         param.grad = grad_sum / x.size(0)

#     optimizer.step()
#     return per_sample_norms.mean().item()

# # ============================================================
# # Auto-compute noise multiplier σ (from Xiang et al., ICLR 2023)
# # ============================================================

# def get_std(q, total_iterations, epsilon, delta=1e-5, verbose=False):
#     """
#     Compute Gaussian noise std (σ, i.e., noise_multiplier) given:
#     - q:               sampling ratio (e.g., 0.05)
#     - total_iterations: total number of DP-SGD iterations (T_1 + T_3)
#     - epsilon:         target privacy budget
#     - delta:           target δ (default = 1e-5)

#     Implementation based on:
#     "A Theory to Instruct Differentially Private Learning via Clipping Bias Reduction"
#     (Zihang Xiang et al., ICLR 2023)
#     """
#     import math

#     def compute_eps(sigma):
#         # Renyi DP-based epsilon computation
#         alpha = 10.0
#         rdp = total_iterations * (q ** 2) * alpha / (2 * sigma ** 2)
#         eps = rdp + math.log(1 / delta) / (alpha - 1)
#         return eps

#     # binary search over σ
#     low, high = 0.01, 50.0
#     for _ in range(50):
#         mid = (low + high) / 2
#         eps = compute_eps(mid)
#         if eps > epsilon:
#             low = mid
#         else:
#             high = mid

#     sigma = high
#     if verbose:
#         print(f"[get_std] ε={epsilon}, δ={delta}, q={q:.6f}, total_iterations={total_iterations} → σ={sigma:.4f}")
#     return sigma


# dp_utils.py
# -*- coding: utf-8 -*-
"""
Utilities for DP-SGD training:
- Per-sample gradients using functorch (torch.func), clipping, Gaussian noise
- Expected-batch-size normalization to match privacy accountant
- Accuracy helpers
- Dataset filters & label remapping for 4/6/10 classification
- A simple sigma estimator to match (epsilon, delta, q, total_steps)
"""

from typing import Dict, Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.func import vmap, grad, functional_call

# --------------------
# Privacy / DP core
# --------------------

@torch.no_grad()
def _stack_param_buffers(model: nn.Module):
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}
    return params, buffers

def dp_step_images(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   sigma: float,
                   max_grad_norm: float,
                   expected_batchsize: int) -> None:
    """
    Perform exactly ONE DP-SGD step on a minibatch:
    - Compute per-sample grads
    - Clip at max_grad_norm (C)
    - Add Gaussian noise with std C * sigma
    - Normalize by EXPECTED batch size (q*n), *not* the realized len(x)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    params, buffers = _stack_param_buffers(model)

    def compute_loss(p, b, xi, yi):
        logits = functional_call(model, (p, b), (xi.unsqueeze(0),))
        if isinstance(logits, tuple):  # (logits, feat)
            logits = logits[0]
        return F.cross_entropy(logits, yi.unsqueeze(0))

    # per-sample grads
    per_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x, y)

    # clip
    with torch.no_grad():
        per_norms = None
        for g in per_grads.values():
            g2 = g.view(g.shape[0], -1).pow(2).sum(1)
            per_norms = g2 if per_norms is None else (per_norms + g2)
        per_norms = per_norms.sqrt().clamp_min(1e-12)
        scales = (max_grad_norm / per_norms).clamp(max=1.0)

    for name, g in per_grads.items():
        per_grads[name] = g * scales.view(-1, *([1] * (g.ndim - 1)))

    # aggregate, add noise, normalize by expected batch size
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g_sum = per_grads[name].sum(0)
        if sigma > 0:
            g_sum = g_sum + max_grad_norm * sigma * torch.randn_like(g_sum)
        p.grad = g_sum / float(expected_batchsize)

    optimizer.step()


# --------------------
# Accuracy
# --------------------

@torch.no_grad()
def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


# --------------------
# CIFAR-10 splits
# --------------------

# CIFAR-10 classes:
# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
VEHICLE_4 = [0, 1, 8, 9]    # airplane, automobile, ship, truck
ANIMAL_6  = [2, 3, 4, 5, 6, 7]

def _remap_targets(targets, kept_classes: List[int]) -> torch.Tensor:
    mapping = {c: i for i, c in enumerate(kept_classes)}
    return torch.tensor([mapping[int(t)] for t in targets], dtype=torch.long)

def _subset_indices_by_classes(targets: List[int], kept: List[int]) -> List[int]:
    kept_set = set(kept)
    return [i for i, t in enumerate(targets) if int(t) in kept_set]


def get_cifar10_datasets(data_dir: str = "./data"):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf_train)
    test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)
    return train, test


def build_split_loaders(q: float,
                        data_dir: str,
                        batchsize_full: Optional[int],
                        num_workers: int = 2,
                        seed: int = 1) -> Dict[str, DataLoader]:
    """
    Build train/test loaders for:
    - vehicle 4-class (with remapped labels 0..3)
    - animal 6-class (remapped 0..5)
    - full 10-class (0..9)

    Batch sizes are computed as round(q * n_subset).
    """
    g = torch.Generator()
    g.manual_seed(seed)

    train, test = get_cifar10_datasets(data_dir)

    # full
    y_train = torch.tensor(train.targets)
    y_test  = torch.tensor(test.targets)

    # 4-class vehicles
    idx4_tr = _subset_indices_by_classes(y_train, VEHICLE_4)
    idx4_te = _subset_indices_by_classes(y_test,  VEHICLE_4)
    train4 = Subset(train, idx4_tr)
    test4  = Subset(test,  idx4_te)

    # remap labels
    train4_targets = _remap_targets(y_train[idx4_tr], VEHICLE_4)
    test4_targets  = _remap_targets(y_test[idx4_te],  VEHICLE_4)
    # monkey patch targets so default collate works
    train4.dataset.targets = list(y_train.numpy())  # keep original for other splits
    test4.dataset.targets  = list(y_test.numpy())

    # 6-class animals
    idx6_tr = _subset_indices_by_classes(y_train, ANIMAL_6)
    idx6_te = _subset_indices_by_classes(y_test,  ANIMAL_6)
    train6 = Subset(train, idx6_tr)
    test6  = Subset(test,  idx6_te)
    train6_targets = _remap_targets(y_train[idx6_tr], ANIMAL_6)
    test6_targets  = _remap_targets(y_test[idx6_te],  ANIMAL_6)

    n4   = len(idx4_tr)
    n6   = len(idx6_tr)
    n10  = len(train)

    b4  = max(1, round(q * n4))
    b6  = max(1, round(q * n6))
    b10 = max(1, round(q * n10))

    # If user forces a batchsize_full (for reproducibility), respect it for 10-class
    if batchsize_full is not None:
        b10 = batchsize_full

    loader = dict(
        vehicle_train = DataLoader(train4, batch_size=b4, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        vehicle_test  = DataLoader(test4,  batch_size=256, shuffle=False, num_workers=num_workers),
        animal_train  = DataLoader(train6, batch_size=b6, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        animal_test   = DataLoader(test6,  batch_size=256, shuffle=False, num_workers=num_workers),
        full_train    = DataLoader(train,  batch_size=b10, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        full_test     = DataLoader(test,   batch_size=256, shuffle=False, num_workers=num_workers),
        b4 = b4, b6 = b6, b10 = b10, n4 = n4, n6 = n6, n10 = n10
    )
    return loader


# --------------------
# Sigma estimator (simple closed-form upper bound)
# --------------------

def estimate_sigma(epsilon: float, delta: float, q: float, total_steps: int) -> float:
    """
    Very simple, conservative estimate:
    epsilon ≈ q * sqrt(2 * total_steps * log(1/delta)) / sigma
    => sigma ≈ q * sqrt(2 * total_steps * log(1/delta)) / epsilon

    This matches order-of-magnitude used in many DP-SGD tutorials and is
    reasonably close to typical accountants for small q.

    If you have the teacher's accounting function, you can replace this with it.
    """
    total_steps = max(1, int(total_steps))
    l = math.log(1.0 / max(1e-12, delta))
    return (q * math.sqrt(2.0 * total_steps * l)) / max(1e-12, epsilon)
