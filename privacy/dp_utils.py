"""
Utilities for DP-SGD training including privacy accounting and data processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
import math

# 尝试导入 Opacus 的 RDPAccountant
try:
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
except Exception:
    OpacusRDPAccountant = None


# ============================================================
# Privacy Accountant
# ============================================================
class PrivacyAccountant:
    """Privacy accountant for DP-SGD using RDP (Renyi Differential Privacy)"""
    
    def __init__(self, noise_multiplier: float, batch_size: int, dataset_size: int):
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.sampling_rate = batch_size / dataset_size
        
    def compute_rdp(self, alpha: float) -> float:
        if self.noise_multiplier == 0:
            return float("inf")
        return alpha / (2 * self.noise_multiplier ** 2)
    
    def compute_epsilon(self, delta: float, steps: int) -> float:
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
        epsilon = self.compute_epsilon(delta, steps)
        return epsilon, delta


def solve_noise_from_epsilon(target_epsilon: float, delta: float, steps: int) -> float:
    """二分搜索求 noise multiplier"""
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        acc = PrivacyAccountant(mid, batch_size=1, dataset_size=1)
        eps = acc.compute_epsilon(delta, steps)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# ============================================================
# Opacus-based epsilon计算
# ============================================================
def compute_epsilon_opacus(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    """
    Compute epsilon using Opacus RDP accountant with given q and steps.
    """
    if OpacusRDPAccountant is None:
        # fallback 简单公式
        return (steps * (sample_rate ** 2)) / (2 * noise_multiplier ** 2) + math.log(1/delta)

    accountant = OpacusRDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta)


def solve_noise_from_epsilon_opacus(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """二分搜索，利用 Opacus 来解 noise multiplier"""
    lo, hi = 1e-3, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        eps = compute_epsilon_opacus(mid, sample_rate, steps, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# ============================================================
# Gradient clipping utilities
# ============================================================
class GradientClipper:
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        clip_coef = min(1.0, self.max_norm / (total_norm + 1e-6))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
        return total_norm


# ============================================================
# Data utilities
# ============================================================
class DataProcessor:
    """CIFAR-10 数据处理工具"""
    
    @staticmethod
    def get_cifar10_classes() -> Dict[str, List[int]]:
        return {
            "animal_classes": [2, 3, 4, 5, 6, 7],   # bird, cat, deer, dog, frog, horse
            "vehicle_classes": [0, 1, 8, 9],       # airplane, automobile, ship, truck
            "all_classes": list(range(10)),
        }
    
    @staticmethod
    def filter_dataset(dataset, target_classes: List[int]) -> torch.utils.data.Subset:
        indices = [idx for idx, (_, label) in enumerate(dataset) if label in target_classes]
        return torch.utils.data.Subset(dataset, indices)
    
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
# Accuracy
# ============================================================
def compute_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
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
