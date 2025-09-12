"""
Two-Stage Fusion with Min–Max (Adversarial) Training at the 10-class Head

Stage A (clean):
  - Train M1 = M_6 on animal classes (6-way) with CE
  - Train M2 = M_4 on vehicle classes (4-way) with CE

Stage B (fusion):
  - Freeze M1/M2
  - Build Fusion: x -> penult(M1(x)) || penult(M2(x)) -> HeadG(10)
  - Train HeadG either:
      (clean)  :  min_w   L( M10(w, [M6(x),       M4(x)      ]), y )
      (adv PGD):  min_w max_δ L( M10(w, [M6(x+δ), M4(x+δ)]), y ),  ||δ||_∞ <= eps

Notes:
  * We DO NOT normalize images (ToTensor only), so PGD uses eps/step in [0,1] scale (e.g., 8/255, 2/255).
  * For adversarial training we generate δ on the IMAGE (x) and backprop through frozen M1/M2 into δ only.
  * M1/M2 params are frozen; only HeadG is updated in Stage B.
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

# ----- Plan B: make local 'core' package importable -----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

# ----- Imports from your codebase -----
from core.models.resnet import LightResnet, BasicBlock
from core.utils import Logger, seed
from core import animal_classes, vehicle_classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
#   Models & Heads
# --------------------------
def build_lightresnet20(num_classes: int) -> LightResnet:
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
    """10-class classifier head for concatenated embeddings."""
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class FusionForHead(nn.Module):
    """
    A wrapper that takes (frozen) M1 and M2 and exposes a forward(x) -> logits(10)
    by capturing their penultimate features (inputs to fc) *with gradients* (no detach),
    concatenating them, and passing to a head.
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1.eval()
        self.m2 = m2.eval()
        for p in self.m1.parameters():
            p.requires_grad = False
        for p in self.m2.parameters():
            p.requires_grad = False

        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        def hook_m1(module, inp, out): feats["m1"] = inp[0]   # no detach -> keep grad wrt x
        def hook_m2(module, inp, out): feats["m2"] = inp[0]

        h1 = self.m1.fc.register_forward_hook(hook_m1)
        h2 = self.m2.fc.register_forward_hook(hook_m2)
        _ = self.m1(x)
        _ = self.m2(x)
        h1.remove(); h2.remove()

        f1 = feats["m1"]
        f2 = feats["m2"]
        z = torch.cat([f1, f2], dim=1)
        return self.head(z)


# --------------------------
#   Data
# --------------------------
def get_full_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),            # NO Normalize: PGD eps/step are in [0,1]
    ])
    transform_test = T.Compose([T.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def filter_and_remap_indices(dataset: torchvision.datasets.CIFAR10, keep_labels: List[int]):
    indices = [i for i, (_, y) in enumerate(dataset) if y in keep_labels]
    remap = {old: new for new, old in enumerate(keep_labels)}  # maps original class id -> 0..k-1
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
#   Utils: train/eval
# --------------------------
@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_model_ce(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   epochs: int, lr: float = 0.1, logger: Logger = None, tag: str = ""):
    """Clean CE training (for M1 and M2)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, run_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            run_loss += float(loss.item()) * x.size(0)
            with torch.no_grad():
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            test_acc = eval_clean(model, test_loader)
            msg = f'{tag} Epoch {epoch:03d} | Train Loss {(run_loss/max(total,1)):.4f} | Train Acc {(correct/max(total,1)):.4f} | Test Acc {test_acc:.4f}'
            print(msg) if logger is None else logger.log(msg)


# --------------------------
#   PGD inner maximization (on image)
# --------------------------
@torch.enable_grad()
def pgd_linf_attack(model: nn.Module,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    eps: float = 8/255,
                    step_size: float = 2/255,
                    steps: int = 10) -> torch.Tensor:
    """
    Standard Linf-PGD on the IMAGE. Works even if model has frozen submodules.
    Inputs/outputs in [0,1].
    """
    model.eval()  # eval helps stabilize BN of frozen backbones
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    x_adv.requires_grad_(True)

    for _ in range(steps):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        # project back to Linf ball & image bounds
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = x_adv.clamp(0.0, 1.0)
        x_adv.requires_grad_(True)

    return x_adv.detach()


# --------------------------
#   Train HeadG (clean or min–max)
# --------------------------
def infer_head_input_dim(m1: LightResnet, m2: LightResnet) -> int:
    """Probe penultimate dims of M1/M2 to set HeadG input size."""
    m1.eval(); m2.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 32, 32, device=DEVICE)
        feats = {}
        def hook_m1(module, inp, out): feats["m1"] = inp[0]
        def hook_m2(module, inp, out): feats["m2"] = inp[0]
        h1 = m1.fc.register_forward_hook(hook_m1)
        h2 = m2.fc.register_forward_hook(hook_m2)
        _ = m1(dummy); _ = m2(dummy)
        h1.remove(); h2.remove()
        d = feats["m1"].shape[1] + feats["m2"].shape[1]
    return d


def train_head_fusion_clean(fusion: FusionForHead,
                            train_loader: DataLoader,
                            test_loader: DataLoader,
                            epochs: int,
                            lr: float = 1e-3,
                            logger: Logger = None):
    """Clean training of HeadG (fusion model forward)."""
    # Only head params require grad
    params = [p for p in fusion.head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        fusion.train()
        total, correct, run_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = fusion(x)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), max_norm=5.0)
            optimizer.step()

            run_loss += float(loss.item()) * x.size(0)
            with torch.no_grad():
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = eval_clean(fusion, test_loader)
            msg = f'[G-clean] Epoch {epoch:03d} | Train Loss {(run_loss/max(total,1)):.4f} | Train Acc {(correct/max(total,1)):.4f} | Test Acc {acc:.4f}'
            print(msg) if logger is None else logger.log(msg)


def train_head_fusion_minmax(fusion: FusionForHead,
                             train_loader: DataLoader,
                             test_loader: DataLoader,
                             epochs: int,
                             lr: float = 1e-3,
                             eps: float = 8/255,
                             step_size: float = 2/255,
                             steps: int = 10,
                             logger: Logger = None):
    """
    Min–max training: for each batch craft x_adv by PGD wrt fusion,
    then update only HeadG on CE(fusion(x_adv), y).
    """
    params = [p for p in fusion.head.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        fusion.train()
        total, correct, run_loss = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # ---- inner max: PGD on image, gradients flow through frozen M1/M2 into x ----
            x_adv = pgd_linf_attack(fusion, x, y, eps=eps, step_size=step_size, steps=steps)

            # ---- outer min: update HeadG only ----
            optimizer.zero_grad(set_to_none=True)
            logits_adv = fusion(x_adv)
            loss = ce(logits_adv, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), max_norm=5.0)
            optimizer.step()

            run_loss += float(loss.item()) * x.size(0)
            with torch.no_grad():
                preds = logits_adv.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc_clean = eval_clean(fusion, test_loader)
            msg = f'[G-adv]  Epoch {epoch:03d} | Train Loss {(run_loss/max(total,1)):.4f} | Train Acc {(correct/max(total,1)):.4f} | Test Clean Acc {acc_clean:.4f}'
            print(msg) if logger is None else logger.log(msg)


# --------------------------
#   Main
# --------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Two-Stage Fusion with Min–Max on the 10-way head")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--log-dir',  type=str, default='./log_minmax_fusion')
    parser.add_argument('--desc',     type=str, default='run1')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 (clean CE)')
    parser.add_argument('--epochs-g', type=int, default=50, help='epochs for G')
    parser.add_argument('--lr-m', type=float, default=0.1,  help='LR for M1/M2 (SGD)')
    parser.add_argument('--lr-g', type=float, default=1e-3, help='LR for HeadG (Adam)')
    parser.add_argument('--mode',  type=str, default='adv', choices=['clean','adv'],
                        help='Train G on clean (min only) or adversarial (min–max)')
    parser.add_argument('--eps', type=float, default=8/255, help='PGD eps (Linf)')
    parser.add_argument('--step', type=float, default=2/255, help='PGD step size')
    parser.add_argument('--steps', type=int, default=10, help='PGD steps')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Folders & logger
    run_dir = os.path.join(args.log_dir, args.desc)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, 'log.txt'))

    # Seed/device
    seed(args.seed)
    logger.log(f'Using device: {DEVICE}')

    # Full CIFAR-10 loaders (10-way labels)
    full_train_loader, full_test_loader = get_full_cifar10_loaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers
    )

    # Subset loaders for M1 (animals) & M2 (vehicles)
    m1_train_loader = build_filtered_loaders(args.data_dir, animal_classes, args.batch_size, train=True,  num_workers=args.num_workers)
    m1_test_loader  = build_filtered_loaders(args.data_dir, animal_classes, args.batch_size, train=False, num_workers=args.num_workers)
    m2_train_loader = build_filtered_loaders(args.data_dir, vehicle_classes, args.batch_size, train=True,  num_workers=args.num_workers)
    m2_test_loader  = build_filtered_loaders(args.data_dir, vehicle_classes, args.batch_size, train=False, num_workers=args.num_workers)

    # ---- Stage A: Train submodels clean ----
    m1 = build_lightresnet20(num_classes=len(animal_classes))
    m2 = build_lightresnet20(num_classes=len(vehicle_classes))
    logger.log(f'Training M1 (6-class) for {args.epochs_m} epochs (CE)...')
    train_model_ce(m1, m1_train_loader, m1_test_loader, args.epochs_m, args.lr_m, logger, tag='[M1]')
    logger.log(f'Training M2 (4-class) for {args.epochs_m} epochs (CE)...')
    train_model_ce(m2, m2_train_loader, m2_test_loader, args.epochs_m, args.lr_m, logger, tag='[M2]')

    # Optional: save submodels
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(run_dir, 'M1_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(run_dir, 'M2_4cls.pt'))

    # ---- Stage B: Build fusion & train HeadG ----
    in_dim = infer_head_input_dim(m1, m2)
    head = HeadG(in_dim=in_dim, num_classes=10).to(DEVICE)
    fusion = FusionForHead(m1, m2, head).to(DEVICE)

    # Train G (clean or min–max)
    if args.mode == 'clean':
        logger.log(f'Training G clean for {args.epochs_g} epochs ...')
        train_head_fusion_clean(fusion, full_train_loader, full_test_loader, args.epochs_g, lr=args.lr_g, logger=logger)
    else:
        logger.log(f'Training G min–max (PGD-AT) for {args.epochs_g} epochs ...')
        train_head_fusion_minmax(fusion, full_train_loader, full_test_loader, args.epochs_g,
                                 lr=args.lr_g, eps=args.eps, step_size=args.step, steps=args.steps, logger=logger)

    # Final clean eval
    acc_m1 = eval_clean(m1, m1_test_loader)
    acc_m2 = eval_clean(m2, m2_test_loader)
    acc_g  = eval_clean(fusion, full_test_loader)
    logger.log(f'Final Clean Acc: M1={acc_m1:.4f} | M2={acc_m2:.4f} | G={acc_g:.4f}')

    # Save head
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(run_dir, 'G_head_10cls.pt'))
    logger.log(f'Saved to: {run_dir}')


if __name__ == '__main__':
    main()
