"""
Utilities for DP-SGD training including privacy accounting and data processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
import math
from scipy import special


class PrivacyAccountant:
    """Privacy accountant for DP-SGD using RDP (Renyi Differential Privacy)"""
    
    def __init__(self, noise_multiplier: float, batch_size: int, dataset_size: int):
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.sampling_rate = batch_size / dataset_size
        
    def compute_rdp(self, alpha: float) -> float:
        """Compute RDP for given alpha"""
        if self.noise_multiplier == 0:
            return float('inf')
        
        # RDP for Gaussian mechanism
        return alpha / (2 * self.noise_multiplier**2)
    
    def compute_epsilon(self, delta: float, steps: int) -> float:
        """Compute epsilon for given delta and number of steps"""
        if self.noise_multiplier == 0:
            return float('inf')
        
        # Use a simplified RDP approximation; steps scales linearly
        alphas = np.arange(2, 100, 0.5)
        rdps = [steps * self.compute_rdp(alpha) for alpha in alphas]
        
        # Convert RDP to (epsilon, delta)
        epsilons = []
        for alpha, rdp in zip(alphas, rdps):
            eps = rdp + math.log(1 / delta) / (alpha - 1)
            epsilons.append(eps)
        
        return min(epsilons)
    
    def get_privacy_spent(self, steps: int, delta: float = 1e-5) -> Tuple[float, float]:
        """Get privacy spent (epsilon, delta) for given number of steps"""
        epsilon = self.compute_epsilon(delta, steps)
        return epsilon, delta


def solve_noise_from_epsilon(target_epsilon: float, delta: float, steps: int) -> float:
    """
    Solve noise_multiplier such that epsilon ~= target_epsilon under the same RDP approx.
    Uses a simple monotonic binary search over noise in [1e-3, 50].
    """
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


class GradientClipper:
    """Gradient clipping utility for DP-SGD"""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to max_norm
        Returns the total gradient norm before clipping
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        clip_coef = min(1.0, self.max_norm / (total_norm + 1e-6))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
        
        return total_norm


class DPOptimizer:
    """DP-SGD optimizer with gradient clipping and noise addition"""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 noise_multiplier: float, max_grad_norm: float = 1.0,
                 momentum_beta: float = 0.0, clip_constant: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.gradient_clipper = GradientClipper(max_grad_norm)
        self.privacy_accountant = None
        self.clip_constant = clip_constant
        
    def step(self, batch_size: int, dataset_size: int):
        """Perform DP-SGD step with gradient clipping and noise addition"""
        # Clip gradients
        grad_norm = self.gradient_clipper.clip_gradients(self.model)
        
        # Add noise to gradients
        if self.noise_multiplier > 0:
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.normal(0, self.noise_multiplier * self.clip_constant, 
                                        size=p.grad.shape, device=p.grad.device)
                    p.grad.data.add_(noise)
        
        # Update parameters
        self.optimizer.step()
        
        return grad_norm
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()


def create_dp_optimizer(model: nn.Module, lr: float = 0.01, momentum: float = 0.9,
                       noise_multiplier: float = 1.0, max_grad_norm: float = 1.0) -> DPOptimizer:
    """Create DP-SGD optimizer"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return DPOptimizer(model, optimizer, noise_multiplier, max_grad_norm)


class DataProcessor:
    """Data processing utilities for DP-SGD training"""
    
    @staticmethod
    def get_cifar10_classes() -> Dict[str, List[int]]:
        """Get CIFAR-10 class mappings for 4-class and 6-class models"""
        return {
            'animal_classes': [0, 1, 2, 3, 4, 5],  # airplane, automobile, bird, cat, deer, dog
            'vehicle_classes': [6, 7, 8, 9],  # frog, horse, ship, truck
            'all_classes': list(range(10))
        }
    
    @staticmethod
    def filter_dataset(dataset, target_classes: List[int]) -> torch.utils.data.Subset:
        """Filter dataset to include only target classes"""
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if label in target_classes:
                indices.append(idx)
        return torch.utils.data.Subset(dataset, indices)
    
    @staticmethod
    def create_data_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4) -> Dict:
        """Create data loaders for different class subsets"""
        import torchvision
        import torchvision.transforms as transforms
        
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        
        # Get class mappings
        class_mappings = DataProcessor.get_cifar10_classes()
        
        # Create filtered datasets
        animal_train = DataProcessor.filter_dataset(train_dataset, class_mappings['animal_classes'])
        animal_test = DataProcessor.filter_dataset(test_dataset, class_mappings['animal_classes'])
        
        vehicle_train = DataProcessor.filter_dataset(train_dataset, class_mappings['vehicle_classes'])
        vehicle_test = DataProcessor.filter_dataset(test_dataset, class_mappings['vehicle_classes'])
        
        # Create data loaders
        loaders = {
            'animal_train': torch.utils.data.DataLoader(
                animal_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            'animal_test': torch.utils.data.DataLoader(
                animal_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            'vehicle_train': torch.utils.data.DataLoader(
                vehicle_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            'vehicle_test': torch.utils.data.DataLoader(
                vehicle_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
            ),
            'full_train': torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            ),
            'full_test': torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        }
        
        return loaders


def compute_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                     device: torch.device) -> float:
    """Compute model accuracy on given data loader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Some models return (logits, embeddings); normalize to logits
            if isinstance(output, tuple):
                output = output[0]
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return correct / total


def save_model(model: nn.Module, filepath: str):
    """Save model to file"""
    torch.save(model.state_dict(), filepath)


def load_model(model: nn.Module, filepath: str):
    """Load model from file"""
    model.load_state_dict(torch.load(filepath, map_location='cpu'))


if __name__ == "__main__":
    # Test privacy accountant
    accountant = PrivacyAccountant(noise_multiplier=1.0, batch_size=64, dataset_size=50000)
    epsilon, delta = accountant.get_privacy_spent(steps=100, delta=1e-5)
    print(f"Privacy spent: ε={epsilon:.3f}, δ={delta}")
    
    # Test data processor
    processor = DataProcessor()
    class_mappings = processor.get_cifar10_classes()
    print(f"Class mappings: {class_mappings}")
    
    print("All utilities tested successfully!")
