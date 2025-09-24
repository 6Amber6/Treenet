"""
Two-Stage CE Optimization - Clean Accuracy Only (+ Min-Max on G)
Pipeline:
  1) Train M1 (6-class animal) with Cross-Entropy
  2) Train M2 (4-class vehicle) with Cross-Entropy
  3) Extract penultimate embeddings from M1 & M2
  4) Min-Max scale the concatenated embeddings using TRAIN-set statistics
  5) Train G (10-class) on the scaled embeddings
  6) Report clean accuracies for M1, M2, and G
"""

import os
import sys
import json
import shutil
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as T

# ----- Make local 'core' package importable (Plan B) -----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.models.resnet import LightResnet, BasicBlock
from core.utils import Logger, parser_train, seed
from core import animal_classes, vehicle_classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
#   Models
# --------------------------
def build_lightresnet20(num_classes: int) -> LightResnet:
    """Build LightResNet-20 model."""
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
    """10-class classifier head for concatenated embeddings."""
    def __init__(self, in_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------
#   Data
# --------------------------
def get_full_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """Full CIFAR-10 loaders with basic transforms (no normalization)."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.Compose([T.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def filter_and_remap_indices(dataset: torchvision.datasets.CIFAR10, keep_labels: List[int]):
    """Filter dataset indices and create label remapping."""
    indices = [i for i, (_, y) in enumerate(dataset) if y in keep_labels]
    remap = {old: new for new, old in enumerate(keep_labels)}
    return indices, remap


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        y = self.remap[int(y)]
        return x, y


def build_filtered_loaders(data_dir: str, keep_labels: List[int], batch_size: int,
                           train: bool, num_workers: int = 4):
    """Build filtered loaders for specific classes (no normalization)."""
    transform = (T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]) if train else T.Compose([T.ToTensor()]))

    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    indices, remap = filter_and_remap_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    pin = torch.cuda.is_available()
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin)
    return loader


# --------------------------
#   Train / Eval
# --------------------------
def train_model_clean(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                      epochs: int, lr: float = 0.1, logger: Logger = None, tag: str = ""):
    """Train a model with clean Cross-Entropy loss."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            clean_acc = eval_clean(model, test_loader)
            train_acc = correct / max(total, 1)
            msg = f'{tag} Epoch {epoch:03d} | Train Loss {(running_loss/total):.4f} | Train Acc {train_acc:.4f} | Test Acc {clean_acc:.4f}'
            logger.log(msg) if logger else print(msg)


@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate clean accuracy."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def register_penultimate_hook(model: LightResnet):
    """Register a forward hook on fc to capture its input (penultimate features)."""
    feats = {}

    def hook_fn(module, input, output):
        feats["feat"] = input[0].detach()

    handle = model.fc.register_forward_hook(hook_fn)
    return feats, handle


@torch.no_grad()
def extract_embeddings(model_m1: nn.Module, model_m2: nn.Module, dataloader: DataLoader):
    """Extract penultimate embeddings from M1 and M2 models."""
    model_m1.eval()
    model_m2.eval()

    feats_m1, h1 = register_penultimate_hook(model_m1)
    feats_m2, h2 = register_penultimate_hook(model_m2)

    all_feats, all_labels = [], []
    for x, y in dataloader:
        x = x.to(DEVICE)
        _ = model_m1(x)  # fills feats_m1["feat"]
        f1 = feats_m1["feat"].cpu()
        _ = model_m2(x)  # fills feats_m2["feat"]
        f2 = feats_m2["feat"].cpu()
        all_feats.append(torch.cat([f1, f2], dim=1))
        all_labels.append(y)

    h1.remove()
    h2.remove()
    X = torch.cat(all_feats, dim=0)
    y = torch.cat(all_labels, dim=0)
    return X, y


# --------------------------
#   Min-Max Scaler for embeddings
# --------------------------
@torch.no_grad()
def fit_minmax_scaler(X_train: torch.Tensor, eps: float = 1e-6):
    """
    Compute per-dimension min and max on TRAIN embeddings.
    Return: X_scaled, x_min, x_scale where scale = 1/(x_max - x_min + eps)
    """
    x_min = X_train.min(dim=0, keepdim=True).values
    x_max = X_train.max(dim=0, keepdim=True).values
    x_scale = 1.0 / (x_max - x_min + eps)
    X_scaled = (X_train - x_min) * x_scale
    X_scaled.clamp_(0.0, 1.0)
    return X_scaled, x_min, x_scale


@torch.no_grad()
def transform_minmax(X: torch.Tensor, x_min: torch.Tensor, x_scale: torch.Tensor):
    Y = (X - x_min) * x_scale
    return torch.clamp(Y, 0.0, 1.0)


# --------------------------
#   Train G on scaled embeddings
# --------------------------
def train_g_clean(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,
                  epochs: int, lr: float = 1e-3, logger: Logger = None):
    """Train model G with clean Cross-Entropy loss on (scaled) embeddings."""
    model_g = HeadG(in_dim=X_train.size(1), num_classes=10).to(DEVICE)
    optimizer = torch.optim.Adam(model_g.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    for epoch in range(1, epochs + 1):
        model_g.train()
        total, correct, run_loss = 0, 0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model_g(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_g.parameters(), max_norm=5.0)
            optimizer.step()

            run_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            clean_acc = eval_clean(model_g, test_loader)
            train_acc = correct / max(total, 1)
            msg = f'G Epoch {epoch:03d} | Train Loss {(run_loss/total):.4f} | Train Acc {train_acc:.4f} | Test Acc {clean_acc:.4f}'
            logger.log(msg) if logger else print(msg)

    return model_g


# --------------------------
#   Main
# --------------------------
def main():
    # Parse arguments
    parse = parser_train()
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 training')
    parse.add_argument('--epochs-g', type=int, default=50, help='epochs for G training')
    parse.add_argument('--lr-m', type=float, default=0.1, help='learning rate for M1/M2')
    parse.add_argument('--lr-g', type=float, default=1e-3, help='learning rate for G')
    args = parse.parse_args()

    # Setup directories and logging
    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Seeds & device
    logger.log(f'Using device: {DEVICE}')
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Load data
    workers = getattr(args, 'workers', 4) if hasattr(args, 'workers') else 4
    full_train_loader, full_test_loader = get_full_cifar10_loaders(DATA_DIR, args.batch_size, num_workers=workers)

    # Filtered loaders for submodels
    m1_train_loader = build_filtered_loaders(DATA_DIR, animal_classes, args.batch_size, train=True,  num_workers=workers)
    m1_test_loader  = build_filtered_loaders(DATA_DIR, animal_classes, args.batch_size, train=False, num_workers=workers)
    m2_train_loader = build_filtered_loaders(DATA_DIR, vehicle_classes, args.batch_size, train=True,  num_workers=workers)
    m2_test_loader  = build_filtered_loaders(DATA_DIR, vehicle_classes, args.batch_size, train=False, num_workers=workers)

    logger.log(f'Animal classes: {animal_classes} (M1: {len(animal_classes)} classes)')
    logger.log(f'Vehicle classes: {vehicle_classes} (M2: {len(vehicle_classes)} classes)')
    logger.log(f'M1 train/test samples: {len(m1_train_loader.dataset)}/{len(m1_test_loader.dataset)}')
    logger.log(f'M2 train/test samples: {len(m2_train_loader.dataset)}/{len(m2_test_loader.dataset)}')

    # ------------------ Stage 1: Train M1/M2 ------------------
    logger.log("Starting Two-Stage CE (Clean Only, with Min-Max for G)...")

    # Train M1 (6-class animal)
    logger.log(f"Training M1 (6-class animal) for {args.epochs_m} epochs...")
    m1 = build_lightresnet20(num_classes=len(animal_classes))
    train_model_clean(m1, m1_train_loader, m1_test_loader, args.epochs_m, args.lr_m, logger, "[M1]")

    # Train M2 (4-class vehicle)
    logger.log(f"Training M2 (4-class vehicle) for {args.epochs_m} epochs...")
    m2 = build_lightresnet20(num_classes=len(vehicle_classes))
    train_model_clean(m2, m2_train_loader, m2_test_loader, args.epochs_m, args.lr_m, logger, "[M2]")

    # ------------------ Stage 2: Extract Embeddings ------------------
    logger.log("Extracting embeddings for G...")
    X_train, y_train = extract_embeddings(m1, m2, full_train_loader)
    X_test, y_test   = extract_embeddings(m1, m2, full_test_loader)

    # Convert to float32 explicitly (just in case)
    X_train = X_train.float()
    X_test  = X_test.float()

    # ------------------ NEW: Min-Max scaling on embeddings ------------------
    logger.log("Fitting Min-Max scaler on training embeddings and transforming train/test...")
    X_train_scaled, x_min, x_scale = fit_minmax_scaler(X_train, eps=1e-6)
    X_test_scaled = transform_minmax(X_test, x_min, x_scale)

    # Move to device for training head G
    X_train_scaled = X_train_scaled.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test_scaled = X_test_scaled.to(DEVICE)
    y_test = y_test.to(DEVICE)

    logger.log(f"Embedding dim (after concat): {X_train_scaled.shape[1]} (M1: 64 + M2: 64)")

    # ------------------ Stage 3: Train G on scaled embeddings ------------------
    logger.log(f"Training G (10-class) for {args.epochs_g} epochs on Min-Max scaled embeddings...")
    model_g = train_g_clean(X_train_scaled, y_train, X_test_scaled, y_test, args.epochs_g, args.lr_g, logger)

    # ------------------ Final Evaluation ------------------
    logger.log('\n=== Final Evaluation ===')

    logger.log('M1 (Animal 6-class):')
    clean_acc_m1 = eval_clean(m1, m1_test_loader)
    logger.log(f'  Clean Acc: {clean_acc_m1:.4f}')

    logger.log('M2 (Vehicle 4-class):')
    clean_acc_m2 = eval_clean(m2, m2_test_loader)
    logger.log(f'  Clean Acc: {clean_acc_m2:.4f}')

    logger.log('G (10-class ensemble on scaled embeddings):')
    test_ds = TensorDataset(X_test_scaled, y_test)
    test_loader_g = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    clean_acc_g = eval_clean(model_g, test_loader_g)
    logger.log(f'  Clean Acc: {clean_acc_g:.4f}')

    # Save models and scaler params
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(LOG_DIR, 'M1_animal_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(LOG_DIR, 'M2_vehicle_4cls.pt'))
    torch.save({'model_state_dict': model_g.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'min': x_min.cpu(), 'scale': x_scale.cpu()}, os.path.join(LOG_DIR, 'minmax_scaler.pt'))

    logger.log(f'\nDone. Saved models & scaler to {LOG_DIR}')
    logger.log(f'Final Results: M1={clean_acc_m1:.4f}, M2={clean_acc_m2:.4f}, G={clean_acc_g:.4f}')


if __name__ == '__main__':
    main()
