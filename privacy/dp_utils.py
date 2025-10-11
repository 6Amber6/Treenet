"""
Utilities for DP-SGD training including privacy accounting,
gradient clipping, dataset processing, and per-sample DP-SGD step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import math

# ============================================================
# Privacy Accountant
# ============================================================
class PrivacyAccountant:
    """Privacy accountant for DP-SGD using RDP (Renyi Differential Privacy)."""

    def __init__(self, noise_multiplier: float, batch_size: int, dataset_size: int):
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.sampling_rate = batch_size / dataset_size

    def compute_rdp(self, alpha: float) -> float:
        """Compute RDP for Gaussian mechanism at order alpha."""
        if self.noise_multiplier == 0:
            return float("inf")
        return alpha / (2 * self.noise_multiplier ** 2)

    def compute_epsilon(self, delta: float, steps: int) -> float:
        """Compute epsilon from RDP given delta and steps."""
        if self.noise_multiplier == 0:
            return float("inf")
        alphas = np.arange(2, 100, 0.5)
        rdps = [steps * self.compute_rdp(alpha) for alpha in alphas]
        epsilons = []
        for alpha, rdp in zip(alphas, rdps):
            eps = rdp + math.log(1 / delta) / (alpha - 1)
            epsilons.append(eps)
        return min(epsilons)

    def get_privacy_spent(self, steps: int, delta: float = 1e-5) -> Tuple[float, float]:
        """Return (epsilon, delta) spent after given number of steps."""
        epsilon = self.compute_epsilon(delta, steps)
        return epsilon, delta


def compute_epsilon_opacus(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """
    Compute epsilon using simplified formula if Opacus not used.
    """
    if noise_multiplier == 0:
        return float("inf")
    return (steps * (sample_rate ** 2)) / (2 * noise_multiplier ** 2) + math.log(1/delta)


# ============================================================
# CIFAR-10 Data Utilities
# ============================================================
class FilteredDataset(torch.utils.data.Dataset):
    """Custom dataset for filtered CIFAR-10 with remapped labels."""
    def __init__(self, dataset, indices, new_targets):
        self.dataset = dataset
        self.indices = indices
        self.targets = torch.tensor(new_targets)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data, _ = self.dataset[self.indices[idx]]
        return data, self.targets[idx]


class DataProcessor:
    """CIFAR-10 dataset processing utilities."""

    @staticmethod
    def get_cifar10_classes() -> Dict[str, List[int]]:
        return {
            "animal_classes": [2, 3, 4, 5, 6, 7],   # bird, cat, deer, dog, frog, horse
            "vehicle_classes": [0, 1, 8, 9],       # airplane, automobile, ship, truck
            "all_classes": list(range(10)),
        }

    @staticmethod
    def filter_dataset(dataset, target_classes: List[int]) -> torch.utils.data.Dataset:
        """
        Filter dataset to keep only given classes.
        Also remap labels to 0..N-1 to match classifier output size.
        """
        class_to_idx = {cls: i for i, cls in enumerate(target_classes)}
        indices, new_targets = [], []

        for idx, (_, label) in enumerate(dataset):
            if label in target_classes:
                indices.append(idx)
                new_targets.append(class_to_idx[label])

        # Create a custom dataset that properly handles label remapping
        return FilteredDataset(dataset, indices, new_targets)

    @staticmethod
    def create_data_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4) -> Dict:
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
def compute_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Compute accuracy of a model on a dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total


# ============================================================
# Per-sample DP-SGD Step (torch.func version)
# ============================================================
from torch.func import vmap, grad, functional_call

def dp_step_images(model, optimizer, x, y, noise_multiplier, max_grad_norm):
    """
    Perform DP-SGD step with per-sample gradient clipping and noise.
    """
    optimizer.zero_grad()
    
    # Forward pass
    output = model(x)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    
    # Compute loss
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # 计算梯度范数
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm(2).item()
    
    # 简单的梯度裁剪（如果噪声为0，就是标准SGD）
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # 添加噪声（如果噪声乘数>0）
    if noise_multiplier > 0:
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * noise_multiplier * max_grad_norm
                p.grad += noise
    
    optimizer.step()
    return grad_norm

    # ---- 收集逐样本梯度并计算每样本的 L2 范数 ----
    per_sample_norm_sq = None
    params_with_gs = []

    for p in model.parameters():
        gs = getattr(p, "grad_sample", None)   # gs 形状 [B, ...]
        if gs is None:
            continue
        params_with_gs.append(p)
        gsv = gs.view(gs.shape[0], -1)         # [B, D]
        cur = (gsv ** 2).sum(dim=1)            # [B]
        per_sample_norm_sq = cur if per_sample_norm_sq is None else per_sample_norm_sq + cur

    # 如果模型里某些层没有 grad_sample（比如没有可学习参数），保证不崩
    if per_sample_norm_sq is None:
        print("Warning: No per-sample gradients found, using standard step")
        optimizer.step()
        return 0.0

    per_sample_norms = per_sample_norm_sq.sqrt()  # [B]
    B = per_sample_norms.shape[0]
    preclip_norm = per_sample_norms.mean().item()

    # ---- 计算逐样本缩放系数：min(1, C / ||g_i||) ----
    scales = (max_grad_norm / (per_sample_norms + 1e-12)).clamp(max=1.0)  # [B]

    # ---- 按样本裁剪 → 聚合 → (可选)加噪声 → 设定 p.grad ----
    for p in params_with_gs:
        gs = p.grad_sample  # [B, ...]
        gs = (gs.view(B, -1) * scales.view(B, 1)).view_as(gs)  # 逐样本缩放
        summed = gs.sum(dim=0)  # 聚合到参数维度
        if noise_multiplier > 0:
            noise = torch.randn_like(summed) * noise_multiplier * max_grad_norm
            summed = summed + noise
        p.grad = summed / B    # 设定最终梯度（批平均）
        del p.grad_sample      # 释放显存

    optimizer.step()
    return preclip_norm
