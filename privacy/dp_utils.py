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

"""
DP-SGD utilities (sampling-rate only version)
Implements per-sample gradient DP-SGD, CIFAR data handling, and noise calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict


# ============================================================
# Noise multiplier calculator (from teacher’s paper)
# ============================================================
def get_std(q, EPOCH, epsilon, delta=1e-5, verbose=False):
    """
    Compute Gaussian noise std (σ) given:
    q: sampling ratio
    EPOCH: total iterations (T1+T3)
    epsilon: privacy budget
    delta: fixed 1e-5
    Based on ICLR 2023 paper.
    """
    def compute_eps(sigma):
        steps = int(EPOCH / q)
        alpha = 10.0
        rdp = steps * (q ** 2) * alpha / (2 * sigma ** 2)
        eps = rdp + math.log(1 / delta) / (alpha - 1)
        return eps

    low, high = 0.01, 50.0
    for _ in range(50):
        mid = (low + high) / 2
        eps = compute_eps(mid)
        if eps > epsilon:
            low = mid
        else:
            high = mid
    sigma = high
    if verbose:
        print(f"[get_std] ε={epsilon}, δ={delta}, q={q:.6f}, EPOCH={EPOCH} → σ={sigma:.4f}")
    return sigma


# ============================================================
# CIFAR-10 Data Utilities
# ============================================================
class FilteredDataset(torch.utils.data.Dataset):
    """Subset of CIFAR-10 for filtered class groups."""
    def __init__(self, dataset, indices, new_targets):
        self.dataset = dataset
        self.indices = indices
        self.targets = torch.tensor(new_targets)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        x, _ = self.dataset[self.indices[idx]]
        return x, self.targets[idx]


class DataProcessor:
    """Handle CIFAR10 splitting by class groups."""
    @staticmethod
    def get_cifar10_classes() -> Dict[str, List[int]]:
        return {
            "animal_classes": [2, 3, 4, 5, 6, 7],
            "vehicle_classes": [0, 1, 8, 9],
            "all_classes": list(range(10)),
        }

    @staticmethod
    def filter_dataset(dataset, target_classes: List[int]):
        mapping = {cls: i for i, cls in enumerate(target_classes)}
        idxs, new_targets = [], []
        for idx, (_, label) in enumerate(dataset):
            if label in target_classes:
                idxs.append(idx)
                new_targets.append(mapping[label])
        return FilteredDataset(dataset, idxs, new_targets)

    @staticmethod
    def create_data_loaders(data_dir: str, sampling_rate: float = 0.05, num_workers: int = 4):
        import torchvision
        import torchvision.transforms as transforms

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        total_size = len(train_dataset)
        batch_size = max(1, int(total_size * sampling_rate))  # derived from q
        print(f"[create_data_loaders] total={total_size}, sampling_rate={sampling_rate}, batch_size={batch_size}")

        mappings = DataProcessor.get_cifar10_classes()
        animal_train = DataProcessor.filter_dataset(train_dataset, mappings["animal_classes"])
        animal_test = DataProcessor.filter_dataset(test_dataset, mappings["animal_classes"])
        vehicle_train = DataProcessor.filter_dataset(train_dataset, mappings["vehicle_classes"])
        vehicle_test = DataProcessor.filter_dataset(test_dataset, mappings["vehicle_classes"])

        loaders = {
            "animal_train": torch.utils.data.DataLoader(animal_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "animal_test": torch.utils.data.DataLoader(animal_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "vehicle_train": torch.utils.data.DataLoader(vehicle_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "vehicle_test": torch.utils.data.DataLoader(vehicle_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "full_train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "full_test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        }
        return loaders


# ============================================================
# Accuracy Computation
# ============================================================
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple): out = out[0]
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# DP-SGD Step
# ============================================================
from torch.func import vmap, grad, functional_call

def dp_step_images(model, optimizer, x, y, noise_multiplier, max_grad_norm):
    """Pure PyTorch DP-SGD step with per-sample gradients."""
    model.train()
    optimizer.zero_grad()

    def compute_loss(params, buffers, xi, yi):
        logits = functional_call(model, (params, buffers), (xi.unsqueeze(0),))
        logits = logits[0] if isinstance(logits, tuple) else logits
        return F.cross_entropy(logits, yi.unsqueeze(0))

    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}

    grads = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(params, buffers, x, y)

    per_norms = torch.zeros(x.size(0), device=x.device)
    for p in grads.values():
        per_norms += p.view(p.shape[0], -1).pow(2).sum(1)
    per_norms = per_norms.sqrt()

    scales = (max_grad_norm / (per_norms + 1e-12)).clamp(max=1.0)
    for k, p in grads.items():
        grads[k] = p * scales.view(-1, *([1] * (p.ndim - 1)))

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        grad_sum = grads[name].sum(0)
        if noise_multiplier > 0:
            noise = torch.randn_like(grad_sum) * noise_multiplier * max_grad_norm
            grad_sum += noise
        param.grad = grad_sum / x.size(0)

    optimizer.step()
    return per_norms.mean().item()
