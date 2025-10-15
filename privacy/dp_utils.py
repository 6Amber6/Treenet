# # dp_utils.py
# # -*- coding: utf-8 -*-
# """
# Utilities for DP-SGD training:
# - Per-sample gradients using functorch (torch.func), clipping, Gaussian noise
# - Expected-batch-size normalization to match privacy accountant
# - Accuracy helpers
# - Dataset filters & label remapping for 4/6/10 classification
# - A simple sigma estimator to match (epsilon, delta, q, total_steps)
# """

# from typing import Dict, Tuple, List, Optional
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset, Subset
# from torchvision import datasets, transforms
# from torch.func import vmap, grad, functional_call

# # --------------------
# # Privacy / DP core
# # --------------------

# @torch.no_grad()
# def _stack_param_buffers(model: nn.Module):
#     params = {k: v for k, v in model.named_parameters() if v.requires_grad}
#     buffers = {k: v for k, v in model.named_buffers()}
#     return params, buffers


# def dp_step_images(model: nn.Module,
#                    optimizer: torch.optim.Optimizer,
#                    x: torch.Tensor,
#                    y: torch.Tensor,
#                    sigma: float,
#                    max_grad_norm: float,
#                    expected_batchsize: int) -> None:
#     """
#     Perform exactly ONE DP-SGD step on a minibatch:
#     - Compute per-sample grads
#     - Clip at max_grad_norm (C)
#     - Add Gaussian noise with std C * sigma
#     - Normalize by EXPECTED batch size (q*n), *not* the realized len(x)
#     """
#     model.train()
#     optimizer.zero_grad(set_to_none=True)

#     params, buffers = _stack_param_buffers(model)

#     def compute_loss(p, b, xi, yi):
#         logits = functional_call(model, (p, b), (xi.unsqueeze(0),))
#         if isinstance(logits, tuple):  # (logits, feat)
#             logits = logits[0]
#         return F.cross_entropy(logits, yi.unsqueeze(0))

#     # per-sample grads
#     per_grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x, y)

#     # clip
#     with torch.no_grad():
#         per_norms = None
#         for g in per_grads.values():
#             g2 = g.view(g.shape[0], -1).pow(2).sum(1)
#             per_norms = g2 if per_norms is None else (per_norms + g2)
#         per_norms = per_norms.sqrt().clamp_min(1e-12)
#         scales = (max_grad_norm / per_norms).clamp(max=1.0)

#     for name, g in per_grads.items():
#         per_grads[name] = g * scales.view(-1, *([1] * (g.ndim - 1)))

#     # aggregate, add noise, normalize by expected batch size
#     for name, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         g_sum = per_grads[name].sum(0)
#         if sigma > 0:
#             g_sum = g_sum + max_grad_norm * sigma * torch.randn_like(g_sum)
#         p.grad = g_sum / float(expected_batchsize)

#     optimizer.step()


# # --------------------
# # Accuracy
# # --------------------

# @torch.no_grad()
# def compute_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
#     model.eval()
#     correct = 0
#     total = 0
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         out = model(x)
#         logits = out[0] if isinstance(out, tuple) else out
#         pred = logits.argmax(dim=1)
#         correct += (pred == y).sum().item()
#         total += y.numel()
#     return correct / max(1, total)


# # --------------------
# # CIFAR-10 splits
# # --------------------

# # CIFAR-10 classes:
# # 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
# VEHICLE_4 = [0, 1, 8, 9]    # airplane, automobile, ship, truck
# ANIMAL_6  = [2, 3, 4, 5, 6, 7]

# def _remap_targets(targets, kept_classes: List[int]) -> torch.Tensor:
#     mapping = {c: i for i, c in enumerate(kept_classes)}
#     return torch.tensor([mapping[int(t)] for t in targets], dtype=torch.long)

# def _subset_indices_by_classes(targets: List[int], kept: List[int]) -> List[int]:
#     kept_set = set(kept)
#     return [i for i, t in enumerate(targets) if int(t) in kept_set]


# class SubsetWithTargets(Dataset):
#     """A dataset wrapper that replaces labels with remapped ones."""
#     def __init__(self, dataset, indices, new_targets):
#         self.dataset = dataset
#         self.indices = indices
#         self.new_targets = new_targets
#     def __len__(self):
#         return len(self.indices)
#     def __getitem__(self, idx):
#         x, _ = self.dataset[self.indices[idx]]
#         y = self.new_targets[idx]
#         return x, y


# def get_cifar10_datasets(data_dir: str = "./data"):
#     tf_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
#     tf_test = transforms.Compose([
#         transforms.ToTensor(),
#     ])

#     train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf_train)
#     test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)
#     return train, test


# def build_split_loaders(q: float,
#                         data_dir: str,
#                         batchsize_full: Optional[int],
#                         num_workers: int = 2,
#                         seed: int = 1) -> Dict[str, DataLoader]:
#     """
#     Build train/test loaders for:
#     - vehicle 4-class (with remapped labels 0..3)
#     - animal 6-class (remapped 0..5)
#     - full 10-class (0..9)
#     """
#     g = torch.Generator()
#     g.manual_seed(seed)

#     train, test = get_cifar10_datasets(data_dir)

#     y_train = torch.tensor(train.targets)
#     y_test  = torch.tensor(test.targets)

#     # 4-class vehicles
#     idx4_tr = _subset_indices_by_classes(y_train, VEHICLE_4)
#     idx4_te = _subset_indices_by_classes(y_test,  VEHICLE_4)
#     train4_targets = _remap_targets(y_train[idx4_tr], VEHICLE_4)
#     test4_targets  = _remap_targets(y_test[idx4_te],  VEHICLE_4)
#     train4 = SubsetWithTargets(train, idx4_tr, train4_targets)
#     test4  = SubsetWithTargets(test,  idx4_te,  test4_targets)

#     # 6-class animals
#     idx6_tr = _subset_indices_by_classes(y_train, ANIMAL_6)
#     idx6_te = _subset_indices_by_classes(y_test,  ANIMAL_6)
#     train6_targets = _remap_targets(y_train[idx6_tr], ANIMAL_6)
#     test6_targets  = _remap_targets(y_test[idx6_te],  ANIMAL_6)
#     train6 = SubsetWithTargets(train, idx6_tr, train6_targets)
#     test6  = SubsetWithTargets(test,  idx6_te,  test6_targets)

#     # full 10-class
#     train10 = train
#     test10  = test

#     n4   = len(idx4_tr)
#     n6   = len(idx6_tr)
#     n10  = len(train10)

#     b4  = max(1, round(q * n4))
#     b6  = max(1, round(q * n6))
#     b10 = max(1, round(q * n10))
#     if batchsize_full is not None:
#         b10 = batchsize_full

#     # Print sanity check for label mapping
#     print("4-class unique labels:", torch.unique(train4_targets))
#     print("6-class unique labels:", torch.unique(train6_targets))

#     loader = dict(
#         vehicle_train = DataLoader(train4, batch_size=b4, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
#         vehicle_test  = DataLoader(test4,  batch_size=256, shuffle=False, num_workers=num_workers),
#         animal_train  = DataLoader(train6, batch_size=b6, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
#         animal_test   = DataLoader(test6,  batch_size=256, shuffle=False, num_workers=num_workers),
#         full_train    = DataLoader(train10,  batch_size=b10, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
#         full_test     = DataLoader(test10,   batch_size=256, shuffle=False, num_workers=num_workers),
#         b4 = b4, b6 = b6, b10 = b10, n4 = n4, n6 = n6, n10 = n10
#     )
#     return loader


# # --------------------
# # Sigma estimator
# # --------------------

# def estimate_sigma(epsilon: float, delta: float, q: float, total_steps: int) -> float:
#     """
#     Very simple, conservative estimate:
#     epsilon ≈ q * sqrt(2 * total_steps * log(1/delta)) / sigma
#     => sigma ≈ q * sqrt(2 * total_steps * log(1/delta)) / epsilon
#     """
#     total_steps = max(1, int(total_steps))
#     l = math.log(1.0 / max(1e-12, delta))
#     return (q * math.sqrt(2.0 * total_steps * l)) / max(1e-12, epsilon)




# dp_utils.py
# -*- coding: utf-8 -*-
"""
Utilities for DP-SGD training:
- Per-sample gradients using torch.func.vmap + functional_call
- Clip per-sample gradients at C, add Gaussian noise C*sigma
- Normalize by EXPECTED batch size (q*n)
- CIFAR-10 splits with correct label remapping for 4/6 classes
- A simple sigma estimator (可替换成老师的 get_std)
"""

from typing import Dict, Tuple, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.func import vmap, grad, functional_call

# --------------------
# DP Core
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
    ONE DP-SGD step:
    - per-sample grads via vmap(grad)
    - clip at C=max_grad_norm
    - add Gaussian noise with std = C * sigma
    - divide by EXPECTED batch size (q*n), not actual len(x)
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    params, buffers = _stack_param_buffers(model)

    def compute_loss(p, b, xi, yi):
        logits = functional_call(model, (p, b), (xi.unsqueeze(0),))
        if isinstance(logits, tuple):
            logits = logits[0]
        return F.cross_entropy(logits, yi.unsqueeze(0))

    # per-sample grads: dict[name] -> [B, ...]
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

    # sum + noise + normalize
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
# CIFAR-10 splits & loaders
# --------------------

# CIFAR-10 classes:
# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
VEHICLE_4 = [0, 1, 8, 9]   # airplane, automobile, ship, truck
ANIMAL_6  = [2, 3, 4, 5, 6, 7]

def _remap_targets(targets, kept_classes: List[int]) -> torch.Tensor:
    mapping = {c: i for i, c in enumerate(kept_classes)}
    return torch.tensor([mapping[int(t)] for t in targets], dtype=torch.long)

def _subset_indices_by_classes(targets: List[int], kept: List[int]) -> List[int]:
    kept_set = set(kept)
    return [i for i, t in enumerate(targets) if int(t) in kept_set]

class SubsetWithTargets(Dataset):
    """Dataset wrapper that replaces labels with remapped ones."""
    def __init__(self, dataset, indices, new_targets):
        self.dataset = dataset
        self.indices = indices
        self.new_targets = new_targets
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x, _ = self.dataset[self.indices[idx]]
        y = self.new_targets[idx]
        return x, y

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
                        num_workers: int = 4,
                        seed: int = 1) -> Dict[str, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)

    train, test = get_cifar10_datasets(data_dir)
    y_train = torch.tensor(train.targets)
    y_test  = torch.tensor(test.targets)

    # 4-class
    idx4_tr = _subset_indices_by_classes(y_train, VEHICLE_4)
    idx4_te = _subset_indices_by_classes(y_test,  VEHICLE_4)
    train4_targets = _remap_targets(y_train[idx4_tr], VEHICLE_4)
    test4_targets  = _remap_targets(y_test[idx4_te],  VEHICLE_4)
    train4 = SubsetWithTargets(train, idx4_tr, train4_targets)
    test4  = SubsetWithTargets(test,  idx4_te,  test4_targets)

    # 6-class
    idx6_tr = _subset_indices_by_classes(y_train, ANIMAL_6)
    idx6_te = _subset_indices_by_classes(y_test,  ANIMAL_6)
    train6_targets = _remap_targets(y_train[idx6_tr], ANIMAL_6)
    test6_targets  = _remap_targets(y_test[idx6_te],  ANIMAL_6)
    train6 = SubsetWithTargets(train, idx6_tr, train6_targets)
    test6  = SubsetWithTargets(test,  idx6_te,  test6_targets)

    # 10-class (full)
    train10 = train
    test10  = test

    n4, n6, n10 = len(idx4_tr), len(idx6_tr), len(train10)
    b4  = max(1, round(q * n4))
    b6  = max(1, round(q * n6))
    b10 = max(1, round(q * n10))
    if batchsize_full is not None:
        b10 = batchsize_full

    # sanity
    print("4-class unique labels:", torch.unique(train4_targets))
    print("6-class unique labels:", torch.unique(train6_targets))

    from torch.utils.data import DataLoader
    loader = dict(
        vehicle_train = DataLoader(train4, batch_size=b4, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        vehicle_test  = DataLoader(test4,  batch_size=256, shuffle=False, num_workers=num_workers),
        animal_train  = DataLoader(train6, batch_size=b6, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        animal_test   = DataLoader(test6,  batch_size=256, shuffle=False, num_workers=num_workers),
        full_train    = DataLoader(train10,  batch_size=b10, shuffle=True, num_workers=num_workers, generator=g, drop_last=True),
        full_test     = DataLoader(test10,   batch_size=256, shuffle=False, num_workers=num_workers),
        b4 = b4, b6 = b6, b10 = b10, n4 = n4, n6 = n6, n10 = n10
    )
    return loader

# --------------------
# Sigma estimator
# --------------------

def estimate_sigma(epsilon: float, delta: float, q: float, total_steps: int) -> float:
    """
    Conservative closed-form:
    epsilon ≈ q * sqrt(2 * T * log(1/delta)) / sigma
    => sigma ≈ q * sqrt(2 * T * log(1/delta)) / epsilon
    """
    total_steps = max(1, int(total_steps))
    l = math.log(1.0 / max(1e-12, delta))
    return (q * math.sqrt(2.0 * total_steps * l)) / max(1e-12, epsilon)
