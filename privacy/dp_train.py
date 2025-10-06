"""
DP-SGD training script strictly following the paper 
"A Theory to Instruct Differentially Private Learning via Clipping Bias Reduction"
"""

import os
import sys
import json
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel
from privacy.dp_utils import PrivacyAccountant, DPOptimizer, DataProcessor, compute_accuracy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preprocessing strictly following the paper
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_cifar10_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4):
    """
    Data loaders strictly following the paper requirements
    """
    # Data augmentation for training
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Data preprocessing for testing
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader

def get_filtered_loaders(data_dir: str, keep_labels: list, batch_size: int = 64, num_workers: int = 4):
    """
    Get filtered data loaders
    """
    train_loader, test_loader = get_cifar10_loaders(data_dir, batch_size, num_workers)
    
    # Filter training set
    train_indices = []
    train_remap = {old: new for new, old in enumerate(keep_labels)}
    for idx, (_, label) in enumerate(train_loader.dataset):
        if label in keep_labels:
            train_indices.append(idx)
    
    # Filter test set
    test_indices = []
    for idx, (_, label) in enumerate(test_loader.dataset):
        if label in keep_labels:
            test_indices.append(idx)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_loader.dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_loader.dataset, test_indices)
    
    # Remap labels
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, remap):
            self.subset = subset
            self.remap = remap
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, self.remap[y]
    
    train_dataset = RemappedDataset(train_subset, train_remap)
    test_dataset = RemappedDataset(test_subset, train_remap)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def train_dp_model_paper(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                        epochs: int, lr: float, noise_multiplier: float, max_grad_norm: float,
                        model_name: str, output_dir: str) -> Tuple[float, float, float]:
    """
    DP-SGD training strictly following the paper requirements
    """
    print(f"\nTraining {model_name} with paper-compliant DP-SGD...")
    
    # Create DP optimizer
    optimizer = DPOptimizer(
        model,
        torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4),
        noise_multiplier, max_grad_norm
    )
    
    # Create privacy accountant
    privacy_accountant = PrivacyAccountant(
        noise_multiplier, len(train_loader.dataset), len(train_loader.dataset)
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    best_acc = 0.0
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, (DP4Classifier, DP6Classifier, DP10Classifier)):
                output, _ = model(data)  # Return (logits, embeddings)
            else:
                output = model(data)
            
            loss = criterion(output, target)
            loss.backward()
            
            # DP-SGD step
            grad_norm = optimizer.step(len(train_loader.dataset), len(train_loader.dataset))
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Grad Norm: {grad_norm:.4f}')
        
        # Compute accuracy
        train_acc = correct / total
        test_acc = compute_accuracy(model, test_loader, DEVICE)
        
        # Compute privacy spent
        steps = epoch * len(train_loader)
        epsilon, delta = privacy_accountant.get_privacy_spent(steps, 1e-5)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Privacy: ε={epsilon:.3f}, δ={delta:.2e}')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_final.pth'))
    
    return best_acc, epsilon, delta

def extract_embeddings_paper(model: nn.Module, data_loader: DataLoader) -> torch.Tensor:
    """
    Extract embeddings strictly following the paper requirements
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(DEVICE)
            if isinstance(model, (DP4Classifier, DP6Classifier, DP10Classifier)):
                _, embedding = model(data)
            else:
                embedding = model(data)
            embeddings.append(embedding.cpu())
    
    return torch.cat(embeddings, dim=0)

def train_fusion_paper(model_4class: nn.Module, model_6class: nn.Module,
                      train_loader: DataLoader, test_loader: DataLoader,
                      epochs: int, lr: float, noise_multiplier: float, max_grad_norm: float,
                      output_dir: str) -> Tuple[float, float, float]:
    """
    Train fusion model strictly following the paper requirements
    """
    print("\nTraining fusion model with paper-compliant DP-SGD...")
    
    # Extract embeddings
    print("Extracting embeddings from 4-class model...")
    embeddings_4class = extract_embeddings_paper(model_4class, train_loader)
    
    print("Extracting embeddings from 6-class model...")
    embeddings_6class = extract_embeddings_paper(model_6class, train_loader)
    
    # Create fusion dataset
    fusion_dataset = torch.utils.data.TensorDataset(
        embeddings_4class, embeddings_6class,
        torch.tensor([label for _, label in train_loader.dataset])
    )
    fusion_loader = DataLoader(fusion_dataset, batch_size=train_loader.batch_size, shuffle=True)
    
    # Create fusion model
    fusion_model = DPFusionModel(embedding_dim=64, num_classes=10, groups=8).to(DEVICE)
    
    # Train fusion model
    return train_dp_model_paper(fusion_model, fusion_loader, test_loader, epochs, lr, 
                               noise_multiplier, max_grad_norm, 'fusion', output_dir)

def main():
    parser = argparse.ArgumentParser(description='Paper-compliant DP-SGD Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./dp_paper_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs_4class', type=int, default=50, help='Epochs for 4-class model')
    parser.add_argument('--epochs_6class', type=int, default=50, help='Epochs for 6-class model')
    parser.add_argument('--epochs_10class', type=int, default=50, help='Epochs for 10-class model')
    parser.add_argument('--epochs_fusion', type=int, default=30, help='Epochs for fusion model')
    
    # DP-SGD arguments (strictly following paper settings)
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta parameter')
    
    # Training mode
    parser.add_argument('--train_4class', action='store_true', help='Train 4-class model')
    parser.add_argument('--train_6class', action='store_true', help='Train 6-class model')
    parser.add_argument('--train_10class', action='store_true', help='Train 10-class model')
    parser.add_argument('--train_fusion', action='store_true', help='Train fusion model')
    parser.add_argument('--train_all', action='store_true', help='Train all models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define classes (strictly following paper requirements)
    animal_classes = [0, 1, 2, 3, 4, 5]  # airplane, automobile, bird, cat, deer, dog
    vehicle_classes = [6, 7, 8, 9]  # frog, horse, ship, truck
    
    print("="*60)
    print("PAPER-COMPLIANT DP-SGD TRAINING")
    print("="*60)
    print("Strictly following the paper 'A Theory to Instruct Differentially Private Learning via Clipping Bias Reduction'")
    print(f"Device: {DEVICE}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"DP-SGD parameters: noise_multiplier={args.noise_multiplier}, max_grad_norm={args.max_grad_norm}")
    
    # Create data loaders
    print("\nLoading data...")
    _, full_test = get_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)
    
    m1_train, m1_test = get_filtered_loaders(args.data_dir, animal_classes, args.batch_size, args.num_workers)
    m2_train, m2_test = get_filtered_loaders(args.data_dir, vehicle_classes, args.batch_size, args.num_workers)
    
    print(f"Data loaders created:")
    print(f"  Animal train: {len(m1_train)} batches")
    print(f"  Animal test: {len(m1_test)} batches")
    print(f"  Vehicle train: {len(m2_train)} batches")
    print(f"  Vehicle test: {len(m2_test)} batches")
    print(f"  Full test: {len(full_test)} batches")
    
    # Train models
    models = {}
    results = {}
    
    if args.train_all or args.train_4class:
        print("\n" + "="*50)
        print("TRAINING 4-CLASS MODEL (Vehicle Classes)")
        print("="*50)
        model_4class = DP4Classifier(groups=8).to(DEVICE)
        acc, eps, delta = train_dp_model_paper(
            model_4class, m2_train, m2_test, args.epochs_4class, args.lr,
            args.noise_multiplier, args.max_grad_norm, '4class', args.output_dir
        )
        models['4class'] = model_4class
        results['4class'] = {'accuracy': acc, 'epsilon': eps, 'delta': delta}
    
    if args.train_all or args.train_6class:
        print("\n" + "="*50)
        print("TRAINING 6-CLASS MODEL (Animal Classes)")
        print("="*50)
        model_6class = DP6Classifier(groups=8).to(DEVICE)
        acc, eps, delta = train_dp_model_paper(
            model_6class, m1_train, m1_test, args.epochs_6class, args.lr,
            args.noise_multiplier, args.max_grad_norm, '6class', args.output_dir
        )
        models['6class'] = model_6class
        results['6class'] = {'accuracy': acc, 'epsilon': eps, 'delta': delta}
    
    if args.train_all or args.train_10class:
        print("\n" + "="*50)
        print("TRAINING 10-CLASS MODEL (Direct Training)")
        print("="*50)
        model_10class = DP10Classifier(groups=8).to(DEVICE)
        acc, eps, delta = train_dp_model_paper(
            model_10class, m1_train, m1_test, args.epochs_10class, args.lr,
            args.noise_multiplier, args.max_grad_norm, '10class', args.output_dir
        )
        models['10class'] = model_10class
        results['10class'] = {'accuracy': acc, 'epsilon': eps, 'delta': delta}
    
    if args.train_all or args.train_fusion:
        print("\n" + "="*50)
        print("TRAINING FUSION MODEL")
        print("="*50)
        if '4class' in models and '6class' in models:
            acc, eps, delta = train_fusion_paper(
                models['4class'], models['6class'], m1_train, full_test,
                args.epochs_fusion, args.lr, args.noise_multiplier, args.max_grad_norm, args.output_dir
            )
            results['fusion'] = {'accuracy': acc, 'epsilon': eps, 'delta': delta}
        else:
            print("Warning: 4-class and 6-class models not found. Skipping fusion training.")
    
    # Save results
    with open(os.path.join(args.output_dir, 'paper_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("PAPER-COMPLIANT TRAINING COMPLETED!")
    print("="*60)
    print("\nResults:")
    for model_name, result in results.items():
        print(f"  {model_name}: Accuracy={result['accuracy']:.4f}, "
              f"Privacy=ε={result['epsilon']:.3f}, δ={result['delta']:.2e}")
    
    print(f"\nModels saved to {args.output_dir}")
    print("\nStrictly following paper requirements:")
    print("- ResNet-20 architecture")
    print("- GroupNorm (groups=8)")
    print("- DP-SGD training")
    print("- Gradient clipping and noise addition")
    print("- Privacy accounting")


if __name__ == '__main__':
    main()
