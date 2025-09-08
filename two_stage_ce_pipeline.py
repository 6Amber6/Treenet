# two_stage_ce_pipeline.py
# Method 2: Train two specialist models (M1: 6 classes, M2: 4 classes),
# extract penultimate (second-last) embeddings from each, concatenate them,
# and train a small head G to do 10-class classification.

import os
import sys
from typing import Tuple, List

# --- Make the 'core' package importable (Plan B) ---
# add the local 'adversarial_robustness_pytorch' directory to sys.path,
# so imports like `from core.models.resnet import ...` work.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

# After path injection, we can import from the local package as a top-level module `core`.
from core.models.resnet import LightResnet, BasicBlock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CIFAR-10 normalization (critical to prevent divergence/NaNs) ---
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# CIFAR-10 label partitions for the two specialist models
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
VEHICLE_CLASSES = [0, 1, 8, 9]       # airplane, automobile, ship, truck


def build_lightresnet20(num_classes: int) -> LightResnet:
    """Build a LightResNet-20 (2-2-2 blocks) classifier."""
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


def get_cifar10_dataloaders(data_dir: str, batch_size: int = 256, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Full CIFAR-10 loaders (with standard augments for train, none for test)."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def filter_and_remap_indices(dataset: torchvision.datasets.CIFAR10, keep_labels: List[int]):
    """Return indices of samples with labels in keep_labels, and a mapping old->new (0..K-1)."""
    indices = [i for i, (_, y) in enumerate(dataset) if y in keep_labels]
    remap = {old: new for new, old in enumerate(keep_labels)}
    return indices, remap


class RemappedSubset(torch.utils.data.Dataset):
    """A subset wrapper that remaps labels to a contiguous range."""
    def __init__(self, base: torchvision.datasets.CIFAR10, indices: List[int], remap: dict):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        y = self.remap[int(y)]
        return x, y


def build_filtered_loaders(data_dir: str, keep_labels: List[int], batch_size: int = 256,
                           train: bool = True, num_workers: int = 2) -> DataLoader:
    """Loader for a filtered subset (either train or test) with normalized transforms."""
    transform = (T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]) if train else T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]))

    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    indices, remap = filter_and_remap_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)

    pin = torch.cuda.is_available()
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin)
    return loader


def train_classifier(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                     epochs: int = 50, lr: float = 0.05) -> None:
    """
    Train a classifier with a stability-first setup:
    - Lower LR (0.05 vs 0.1)
    - Turn off Nesterov
    - Skip non-finite losses
    - Gradient clipping
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            # Skip non-finite losses to avoid polluting parameters with NaNs.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                print("Non-finite loss encountered. Skipping batch.")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients to prevent explosions/divergence.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = eval_classifier(model, test_loader)
            print(f"Epoch {epoch:03d} | Train Loss {(running_loss/total):.4f} | Test Acc {acc:.4f}")


@torch.no_grad()
def eval_classifier(model: nn.Module, loader: DataLoader) -> float:
    """Compute top-1 accuracy on a given loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def register_penultimate_hook(model: LightResnet):
    """
    Register a forward hook on model.fc to capture its INPUT,
    which is the penultimate embedding vector.
    """
    feats = {}
    def hook_fn(module, input, output):
        feats["feat"] = input[0].detach()
    handle = model.fc.register_forward_hook(hook_fn)
    return feats, handle


@torch.no_grad()
def extract_embeddings(model_m1: nn.Module, model_m2: nn.Module, loader: DataLoader):
    """
    Pass each batch through M1 and M2, grab penultimate embeddings from each,
    concatenate them along feature dimension, and collect original labels (0..9).
    """
    model_m1.eval()
    model_m2.eval()

    feats_m1, h1 = register_penultimate_hook(model_m1)
    feats_m2, h2 = register_penultimate_hook(model_m2)

    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        _ = model_m1(x); f1 = feats_m1["feat"].cpu()
        _ = model_m2(x); f2 = feats_m2["feat"].cpu()
        all_feats.append(torch.cat([f1, f2], dim=1))
        all_labels.append(y)

    h1.remove(); h2.remove()
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


class HeadG(nn.Module):
    """A small MLP head that maps concatenated embeddings -> 10 logits."""
    def __init__(self, in_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def train_head_g(X_train: torch.Tensor, y_train: torch.Tensor,
                 X_test: torch.Tensor, y_test: torch.Tensor,
                 epochs: int = 50, lr: float = 1e-3, batch_size: int = 512) -> HeadG:
    """Train the 10-class head on top of concatenated embeddings."""
    model_g = HeadG(in_dim=X_train.size(1), num_classes=10).to(DEVICE)
    opt = torch.optim.Adam(model_g.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    test_ds  = torch.utils.data.TensorDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        model_g.train()
        total, correct, run_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model_g(xb)
            loss = crit(logits, yb)

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                print("Non-finite loss in G. Skipping batch.")
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=1.0)
            opt.step()

            run_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        if epoch % 5 == 0 or epoch == 1:
            acc = eval_classifier(model_g, test_loader)
            print(f"[G] Epoch {epoch:03d} | Train Loss {(run_loss/total):.4f} | Test Acc {acc:.4f}")
    return model_g


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Two-stage pipeline: train M1/M2, extract embeddings, train G (10-class).")
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-10 data directory')
    parser.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2')
    parser.add_argument('--epochs-g', type=int, default=50, help='epochs for head G')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=2, help='dataloader workers (use 0 if issues)')
    parser.add_argument('--save-dir', type=str, default='./log_ce_optimization_two_stage')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print('Preparing data ...')
    full_train_loader, full_test_loader = get_cifar10_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    m1_train_loader = build_filtered_loaders(args.data_dir, ANIMAL_CLASSES, batch_size=args.batch_size, train=True,  num_workers=args.num_workers)
    m1_test_loader  = build_filtered_loaders(args.data_dir, ANIMAL_CLASSES, batch_size=args.batch_size, train=False, num_workers=args.num_workers)
    m2_train_loader = build_filtered_loaders(args.data_dir, VEHICLE_CLASSES, batch_size=args.batch_size, train=True,  num_workers=args.num_workers)
    m2_test_loader  = build_filtered_loaders(args.data_dir, VEHICLE_CLASSES, batch_size=args.batch_size, train=False, num_workers=args.num_workers)

    print('Training M1 (6-class animal) ...')
    m1 = build_lightresnet20(num_classes=len(ANIMAL_CLASSES))
    train_classifier(m1, m1_train_loader, m1_test_loader, epochs=args.epochs_m)
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(args.save_dir, 'M1_animal_6cls.pt'))

    print('Training M2 (4-class vehicle) ...')
    m2 = build_lightresnet20(num_classes=len(VEHICLE_CLASSES))
    train_classifier(m2, m2_train_loader, m2_test_loader, epochs=args.epochs_m)
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(args.save_dir, 'M2_vehicle_4cls.pt'))

    print('Extracting embeddings for G ...')
    X_train, y_train = extract_embeddings(m1, m2, full_train_loader)
    X_test,  y_test  = extract_embeddings(m1, m2, full_test_loader)
    X_train = X_train.to(DEVICE); y_train = y_train.to(DEVICE)
    X_test  = X_test.to(DEVICE);  y_test  = y_test.to(DEVICE)

    print('Training G (10-class) ...')
    model_g = train_head_g(X_train, y_train, X_test, y_test, epochs=args.epochs_g)
    torch.save({'model_state_dict': model_g.state_dict()}, os.path.join(args.save_dir, 'G_head_10cls.pt'))
    print('Done. Saved models to', args.save_dir)


if __name__ == '__main__':
    main()
