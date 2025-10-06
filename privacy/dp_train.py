"""
DP-SGD training script (patched, full)
- Correct class splits, DP accounting, fusion pipeline, and epsilon/delta handling
"""

import os
import sys
import json
import argparse
import math
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# Add parent directory to path (so 'privacy.*' can be imported)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel
from privacy.dp_utils import (
    DPOptimizer,
    compute_accuracy,
    compute_epsilon_opacus,
    solve_noise_from_epsilon_opacus,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(data_dir: str, batch_size: int = 64, num_workers: int = 4):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def get_filtered_loaders(data_dir: str, keep_labels: List[int], batch_size: int = 64, num_workers: int = 4):
    full_train, full_test = get_cifar10_loaders(data_dir, batch_size, num_workers)

    train_idx = []
    for i in range(len(full_train.dataset)):
        _, y = full_train.dataset[i]
        if int(y) in keep_labels: train_idx.append(i)

    test_idx = []
    for i in range(len(full_test.dataset)):
        _, y = full_test.dataset[i]
        if int(y) in keep_labels: test_idx.append(i)

    train_subset = torch.utils.data.Subset(full_train.dataset, train_idx)
    test_subset  = torch.utils.data.Subset(full_test.dataset,  test_idx)

    remap = {old: new for new, old in enumerate(keep_labels)}

    class RemapDS(torch.utils.data.Dataset):
        def __init__(self, subset, remap):
            self.subset = subset
            self.remap = remap
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, int(self.remap[int(y)])

    tr = RemapDS(train_subset, remap)
    te = RemapDS(test_subset, remap)

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def train_dp_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float,
    model_name: str,
    output_dir: str,
    clip_constant: float = 1.0,
) -> Tuple[float, float, float]:
    print(f"\nTraining {model_name} with DP-SGD...")
    optimizer = DPOptimizer(
        model,
        torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=5e-4),
        noise_multiplier, max_grad_norm, momentum_beta=0.0, clip_constant=clip_constant
    )
    criterion = nn.CrossEntropyLoss()
    model.train()

    best_acc = 0.0
    steps_so_far = 0

    for epoch in range(epochs):
        total, correct, run_loss = 0, 0, 0.0

        for bidx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            if isinstance(model, (DP4Classifier, DP6Classifier, DP10Classifier)):
                logits, _ = model(data)
            else:
                logits = model(data)

            loss = criterion(logits, target)
            loss.backward()

            # IMPORTANT: pass current batch size + dataset size
            grad_norm = optimizer.step(target.size(0), len(train_loader.dataset))

            # stats
            run_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            steps_so_far += 1
            if bidx % 50 == 0:
                bs = train_loader.batch_size if train_loader.batch_size else 1
                q = bs / max(1, len(train_loader.dataset))
                eps_mid = compute_epsilon_opacus(noise_multiplier, q, steps_so_far, delta)
                print(f"Epoch {epoch}, Batch {bidx}, Loss {loss.item():.4f}, "
                      f"GradNorm {grad_norm:.4f}, ε={eps_mid:.3f}, δ={delta:.1e}")

        train_acc = correct / max(1, total)
        test_acc = compute_accuracy(model, test_loader, DEVICE)

        bs = train_loader.batch_size if train_loader.batch_size else 1
        q = bs / max(1, len(train_loader.dataset))
        eps = compute_epsilon_opacus(noise_multiplier, q, steps_so_far, delta)

        print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, "
              f"Privacy ε={eps:.3f}, δ={delta:.1e}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_best.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_final.pth"))
    # 最后再算一次 ε
    bs = train_loader.batch_size if train_loader.batch_size else 1
    q = bs / max(1, len(train_loader.dataset))
    eps = compute_epsilon_opacus(noise_multiplier, q, steps_so_far, delta)
    return best_acc, eps, delta


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (embeddings[N,D], labels[N]) using classifier's embedding output."""
    model.eval()
    E, Y = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        if isinstance(model, (DP4Classifier, DP6Classifier, DP10Classifier)):
            _, emb = model(x)
        else:
            emb = model(x)  # 如果是纯 encoder
        E.append(emb.detach().cpu())
        Y.append(y.clone())
    return torch.cat(E, dim=0), torch.cat(Y, dim=0)


def train_fusion(
    model4: nn.Module,
    model6: nn.Module,
    full_train: DataLoader,
    full_test: DataLoader,
    epochs: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float,
    output_dir: str,
) -> Tuple[float, float, float]:
    """Fusion on FULL CIFAR-10 (labels 0..9)."""
    print("\nTraining fusion model with DP-SGD...")
    print("Extracting 4-class embeddings on FULL train...")
    emb4, lab4 = extract_embeddings(model4, full_train)

    print("Extracting 6-class embeddings on FULL train...")
    emb6, lab6 = extract_embeddings(model6, full_train)

    # 对齐长度（通常一致）
    N = min(len(emb4), len(emb6), len(lab4), len(lab6))
    emb4, emb6, labels = emb4[:N], emb6[:N], lab4[:N]

    # 断言标签对齐
    if not torch.equal(labels, lab6[:N]):
        print("Warning: labels misaligned between 4/6-class embeddings; aligning by order only.")

    fusion_inputs = torch.cat([emb4, emb6], dim=1)      # [N, D4+D6]
    fusion_dim = fusion_inputs.shape[1]

    fusion_ds = torch.utils.data.TensorDataset(fusion_inputs, labels)
    fusion_loader = DataLoader(fusion_ds, batch_size=full_train.batch_size, shuffle=True,
                               num_workers=0, pin_memory=torch.cuda.is_available())

    # 构建融合模型（embedding_dim=拼接后的维度）
    fusion_model = DPFusionModel(embedding_dim=fusion_dim, num_classes=10, groups=8).to(DEVICE)

    return train_dp_model(
        fusion_model, fusion_loader, full_test, epochs, lr,
        noise_multiplier, max_grad_norm, delta,
        "fusion", output_dir
    )


def estimate_steps(num_epochs: int, dataset_size: int, batch_size: int) -> int:
    return int(num_epochs * math.ceil(dataset_size / max(1, batch_size)))


def main():
    parser = argparse.ArgumentParser(description="Patched DP-SGD Training")

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./dp_patched_output')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)

    # Train
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--epochs_4class', type=int, default=50)
    parser.add_argument('--epochs_6class', type=int, default=50)
    parser.add_argument('--epochs_10class', type=int, default=50)
    parser.add_argument('--epochs_fusion', type=int, default=30)

    # DP
    parser.add_argument('--epsilon', type=float, default=None, help='If set, solve sigma per-task.')
    parser.add_argument('--noise_multiplier', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--clip_constant', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-5)

    # Modes
    parser.add_argument('--train_4class', action='store_true')
    parser.add_argument('--train_6class', action='store_true')
    parser.add_argument('--train_10class', action='store_true')
    parser.add_argument('--train_fusion', action='store_true')
    parser.add_argument('--train_all', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Correct CIFAR-10 splits
    # 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    animal_classes  = [2, 3, 4, 5, 6, 7]  # 6 Animals
    vehicle_classes = [0, 1, 8, 9]        # 4 Vehicles

    print("=" * 60)
    print("PATCHED DP-SGD TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"LR={args.lr}, batch_size={args.batch_size}")
    print(f"DP: sigma={args.noise_multiplier}, C={args.max_grad_norm}, delta={args.delta}")

    print("\nLoading data...")
    full_train, full_test = get_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)
    m6_train, m6_test = get_filtered_loaders(args.data_dir, animal_classes, args.batch_size, args.num_workers)
    m4_train, m4_test = get_filtered_loaders(args.data_dir, vehicle_classes, args.batch_size, args.num_workers)

    print("Data loaders:")
    print(f"  Animals(6) train/test batches:  {len(m6_train)}/{len(m6_test)}")
    print(f"  Vehicles(4) train/test batches: {len(m4_train)}/{len(m4_test)}")
    print(f"  Full train/test batches:        {len(full_train)}/{len(full_test)}")

    results: Dict[str, Dict[str, float]] = {}
    models: Dict[str, nn.Module] = {}

    # helper: per-loader q & steps and noise solving
    def q_steps(loader: DataLoader, epochs: int) -> Tuple[float, int]:
        n = len(loader.dataset)
        b = loader.batch_size if loader.batch_size else args.batch_size
        q = b / max(1, n)
        steps = estimate_steps(epochs, n, b)
        return q, steps

    def resolve_sigma(eps: float, loader: DataLoader, epochs: int) -> float:
        q, steps = q_steps(loader, epochs)
        sigma = solve_noise_from_epsilon_opacus(eps, q, steps, args.delta)
        print(f"Solved sigma from ε={eps}: q={q:.6f}, steps={steps} -> sigma={sigma:.4f}")
        return float(sigma)

    # 4-class (Vehicles)
    if args.train_all or args.train_4class:
        print("\n" + "="*50 + "\nTRAINING 4-CLASS (Vehicles)\n" + "="*50)
        model4 = DP4Classifier(groups=8).to(DEVICE)
        sigma4 = resolve_sigma(args.epsilon, m4_train, args.epochs_4class) if args.epsilon else args.noise_multiplier
        acc, eps, delt = train_dp_model(model4, m4_train, m4_test, args.epochs_4class, args.lr,
                                        sigma4, args.max_grad_norm, args.delta,
                                        "4class", args.output_dir, args.clip_constant)
        results['4class'] = {'accuracy': acc, 'epsilon': eps, 'delta': delt}
        models['4class'] = model4

    # 6-class (Animals)
    if args.train_all or args.train_6class:
        print("\n" + "="*50 + "\nTRAINING 6-CLASS (Animals)\n" + "="*50)
        model6 = DP6Classifier(groups=8).to(DEVICE)
        sigma6 = resolve_sigma(args.epsilon, m6_train, args.epochs_6class) if args.epsilon else args.noise_multiplier
        acc, eps, delt = train_dp_model(model6, m6_train, m6_test, args.epochs_6class, args.lr,
                                        sigma6, args.max_grad_norm, args.delta,
                                        "6class", args.output_dir, args.clip_constant)
        results['6class'] = {'accuracy': acc, 'epsilon': eps, 'delta': delt}
        models['6class'] = model6

    # 10-class (Full)
    if args.train_all or args.train_10class:
        print("\n" + "="*50 + "\nTRAINING 10-CLASS (Full CIFAR-10)\n" + "="*50)
        model10 = DP10Classifier(groups=8).to(DEVICE)
        sigma10 = resolve_sigma(args.epsilon, full_train, args.epochs_10class) if args.epsilon else args.noise_multiplier
        acc, eps, delt = train_dp_model(model10, full_train, full_test, args.epochs_10class, args.lr,
                                        sigma10, args.max_grad_norm, args.delta,
                                        "10class", args.output_dir, args.clip_constant)
        results['10class'] = {'accuracy': acc, 'epsilon': eps, 'delta': delt}
        models['10class'] = model10

    # Fusion (Full)
    if args.train_all or args.train_fusion:
        print("\n" + "="*50 + "\nTRAINING FUSION\n" + "="*50)
        if '4class' in models and '6class' in models:
            sigmaF = resolve_sigma(args.epsilon, full_train, args.epochs_fusion) if args.epsilon else args.noise_multiplier
            acc, eps, delt = train_fusion(models['4class'], models['6class'], full_train, full_test,
                                          args.epochs_fusion, args.lr, sigmaF, args.max_grad_norm, args.delta,
                                          args.output_dir)
            results['fusion'] = {'accuracy': acc, 'epsilon': eps, 'delta': delt}
        else:
            print("Warning: need both 4-class and 6-class models before fusion; skipping.")

    # Save results
    with open(os.path.join(args.output_dir, "training_results_patched.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("PATCHED TRAINING COMPLETED!")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: Acc={v['accuracy']:.4f}, ε={v['epsilon']:.3f}, δ={v['delta']:.2e}")
    print(f"\nModels saved to {args.output_dir}")


if __name__ == "__main__":
    main()
