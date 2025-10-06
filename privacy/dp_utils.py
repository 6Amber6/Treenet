"""
Utilities for DP-SGD training including privacy accounting and data processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import math
try:
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
except Exception:
    OpacusRDPAccountant = None

from torch.func import functional_call, grad, vmap


# ---------------- Privacy Accountant ---------------- #

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
        return alpha / (2 * self.noise_multiplier**2)

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


# ---------------- Per-sample Gradients ---------------- #

def _per_sample_grad_images(model, params, buffers, x, y):
    """Compute per-sample gradients using functorch"""
    def loss_of_single(params_, buffers_, x_i, y_i):
        out = functional_call(model, params_, buffers_, (x_i.unsqueeze(0),))

        # ✅ 兼容 (logits, embeddings) 或 logits
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        return F.cross_entropy(logits, y_i.unsqueeze(0))

    grad_fn = grad(loss_of_single)
    grads_pytree = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, x, y)
    return grads_pytree


# ---------------- DP Optimizer ---------------- #

class DPOptimizerVec:
    """Vectorized DP-SGD optimizer"""

    def __init__(self, model, optimizer, noise_multiplier=1.0, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def dp_step_images(self, x, y):
        params = {k: v for k, v in self.model.named_parameters()}
        buffers = {k: v for k, v in self.model.named_buffers()}

        grads_ps = _per_sample_grad_images(self.model, params, buffers, x, y)

        # 逐样本梯度裁剪
        per_sample_norms = []
        for g in grads_ps.values():
            flat = g.reshape(g.shape[0], -1)
            norms = torch.norm(flat, dim=1)
            per_sample_norms.append(norms)
        total_norms = torch.sqrt(sum([n**2 for n in per_sample_norms]))
        clip_coeffs = (self.max_grad_norm / (total_norms + 1e-6)).clamp(max=1.0)

        clipped_grads = {}
        for name, g in grads_ps.items():
            clipped_grads[name] = g * clip_coeffs.view(-1, *([1] * (g.dim() - 1)))

        # 聚合梯度 + 高斯噪声
        final_grads = {}
        for name, g in clipped_grads.items():
            summed = g.mean(dim=0)
            if self.noise_multiplier > 0:
                noise = torch.normal(
                    0,
                    self.noise_multiplier * self.max_grad_norm / x.size(0),
                    size=summed.shape,
                    device=summed.device,
                )
                summed += noise
            final_grads[name] = summed

        # 应用梯度
        self.optimizer.zero_grad()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.grad = final_grads[n]
        self.optimizer.step()

        return total_norms.mean().item()


# ---------------- Data Processing ---------------- #

class DataProcessor:
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

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )

        class_mappings = DataProcessor.get_cifar10_classes()

        animal_train = DataProcessor.filter_dataset(train_dataset, class_mappings["animal_classes"])
        animal_test = DataProcessor.filter_dataset(test_dataset, class_mappings["animal_classes"])
        vehicle_train = DataProcessor.filter_dataset(train_dataset, class_mappings["vehicle_classes"])
        vehicle_test = DataProcessor.filter_dataset(test_dataset, class_mappings["vehicle_classes"])

        loaders = {
            "animal_train": torch.utils.data.DataLoader(animal_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "animal_test": torch.utils.data.DataLoader(animal_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "vehicle_train": torch.utils.data.DataLoader(vehicle_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "vehicle_test": torch.utils.data.DataLoader(vehicle_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            "full_train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            "full_test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        }
        return loaders


# ---------------- Accuracy Utils ---------------- #

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


def save_model(model: nn.Module, filepath: str):
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str):
    model.load_state_dict(torch.load(filepath, map_location="cpu"))
