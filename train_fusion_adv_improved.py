# train_fusion_adv_improved.py
"""
Improved Two-Stage CE + Full Adversarial Training (All components participate in delta gradients)

Pipeline:
  1) Train M1 (6-class animals) with CE (clean only)
  2) Train M2 (4-class vehicles) with CE (clean only)
  3) Build Fusion model:
      x -> M1(x) -> 6-class logits
      x -> M2(x) -> 4-class logits
      x -> [penult(M1(x)) || penult(M2(x))] -> HeadG(10) -> 10-class logits
  4) Adversarial training on FULL fusion model (TRADES/MART/CE):
     - ALL parameters (M1 + M2 + HeadG) are trainable and participate in delta gradients
     - TRADES is driven by the 10-class fusion head
     - Small masked auxiliary CE for M1/M2 stabilizes clean accuracy
  5) Evaluation uses a wrapper that exposes ONLY the fusion (10-class) logits
"""

import os
import sys
import json
import shutil
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

# ---------- import local 'core' from Treenet repo ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.models.resnet import LightResnet, BasicBlock
from core.utils import Logger, parser_train, seed
from core.attacks import create_attack
from core.utils.trades import trades_loss  # kept for reference
from core.utils.mart import mart_loss
from core import animal_classes, vehicle_classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# Models
# ------------------------------------------------
def build_lightresnet20(num_classes: int) -> LightResnet:
    """Small ResNet used throughout the repo (works well for CIFAR-10)."""
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
    """Simple 2-layer MLP used as fusion head to produce 10-class logits."""
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class ImprovedFusionModel(nn.Module):
    """
    Fusion model where ALL components (M1, M2, HeadG) are trainable.
    We register forward hooks on (m1.fc / m2.fc) to access penultimate features.
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1  # 6-class animal classifier
        self.m2 = m2  # 4-class vehicle classifier
        self.head = head  # 10-class fusion classifier

        # Store penultimate features passed into final FC layers of m1/m2
        self._feats = {}
        self._h1 = self.m1.fc.register_forward_hook(
            lambda m, inp, out: self._save_feat("m1", inp)
        )
        self._h2 = self.m2.fc.register_forward_hook(
            lambda m, inp, out: self._save_feat("m2", inp)
        )

    def _save_feat(self, key, inp_tuple):
        # inp_tuple[0] is the input feature to the final fc layer: shape [B, D]
        self._feats[key] = inp_tuple[0]

    def forward(self, x):
        # Branch 1 (animals, 6-class) & Branch 2 (vehicles, 4-class)
        m1_logits = self.m1(x)
        m2_logits = self.m2(x)

        # Concatenate penultimate features and feed to fusion head (10-class)
        z = torch.cat([self._feats["m1"], self._feats["m2"]], dim=1)
        fusion_logits = self.head(z)

        return m1_logits, m2_logits, fusion_logits

    def remove_hooks(self):
        self._h1.remove()
        self._h2.remove()


# ------------------------------------------------
# Data
# ------------------------------------------------
def _build_cifar10(data_dir, train: bool, num_workers=4, batch_size=128):
    """Standard CIFAR-10 loader with light augmentation for train."""
    tfm = (T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]) if train else T.Compose([T.ToTensor()]))
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return ds, loader


def _filter_indices(ds: torchvision.datasets.CIFAR10, keep: List[int]):
    """Collect indices where the label is in 'keep', and build a map old->new indices."""
    idx = [i for i, (_, y) in enumerate(ds) if y in keep]
    remap = {old: new for new, old in enumerate(keep)}
    return idx, remap


class RemappedSubset(torch.utils.data.Dataset):
    """A subset that remaps labels to a compact range starting at 0."""
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.remap[int(y)]


def build_filtered_loader(data_dir, keep_labels, batch_size, train, num_workers=4):
    """Build loader that keeps only labels in 'keep_labels', labels remapped to 0..K-1."""
    ds, _ = _build_cifar10(data_dir, train=train, num_workers=num_workers, batch_size=batch_size)
    indices, remap = _filter_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader


# ------------------------------------------------
# Clean training for submodels
# ------------------------------------------------
def train_clean_classifier(model, train_loader, test_loader, epochs, lr, logger, tag):
    """Plain CE + MultiStepLR for M1/M2 clean training."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    milestone1 = max(epochs // 2, 1)
    milestone2 = max(epochs * 3 // 4, 1)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[milestone1, milestone2], gamma=0.1)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        sch.step()

        if ep % 5 == 0 or ep == 1:
            acc = eval_clean(model, test_loader)
            logger.log(f'{tag} Epoch {ep:03d} | Train Loss {(loss_sum/max(seen,1)):.4f} | '
                       f'Train Acc {(correct/max(seen,1)):.4f} | Test Acc {acc:.4f}')


# ------------------------------------------------
# Evaluation helpers (fusion-head-only)
# ------------------------------------------------
@torch.no_grad()
def eval_clean(model, loader) -> float:
    """
    Clean evaluation using a wrapper that always returns the fusion (10-class) logits.
    If 'model' is a plain classifier (M1/M2), its direct logits are used.
    """
    class FusionWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            return out[-1] if isinstance(out, (tuple, list)) else out

    m = FusionWrapper(model).eval()

    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = m(x)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


def make_eval_attack(model, args):
    """
    Build an evaluation attack that targets ONLY the fusion (10-class) head,
    while gradients still flow through the entire fusion model.
    """
    crit  = nn.CrossEntropyLoss()
    eps   = getattr(args, 'attack_eps', 8/255)
    step  = getattr(args, 'attack_step', 2/255)
    iters = getattr(args, 'attack_iter', 10)
    attack_name = getattr(args, 'attack', 'linf-pgd')

    class FusionWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            return out[-1] if isinstance(out, (tuple, list)) else out

    return create_attack(FusionWrapper(model), crit, attack_name, eps, iters, step)


def eval_adv(model, loader, attack) -> float:
    """Adversarial evaluation: craft x_adv w.r.t. fusion logits; then measure fusion acc."""
    class FusionWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            return out[-1] if isinstance(out, (tuple, list)) else out

    m = FusionWrapper(model).eval()

    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            logits = m(x_adv)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


@torch.no_grad()
def bn_calibration(model, loader, max_batches=200):
    """
    Recalibrate BatchNorm running stats using clean data.
    Call after CE warmup and periodically during adversarial training (e.g. every 15 epochs).
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None
            m.reset_running_stats()
    seen = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        model(x)
        seen += 1
        if seen >= max_batches:
            break
    model.eval()


# ------------------------------------------------
# Auxiliary: masked CE for M1/M2 (stabilizes clean)
# ------------------------------------------------
def masked_aux_ce(m1_logits: torch.Tensor, m2_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute masked auxiliary CE losses:
      - M1 (6-class) is trained ONLY on labels from animal_classes (mapped to 0..5)
      - M2 (4-class) is trained ONLY on labels from vehicle_classes (mapped to 0..3)
    Returns a scalar loss tensor (0 if no valid samples for a head).
    """
    device = y.device
    loss = torch.tensor(0.0, device=device)

    # Build 10->subset index lookups; invalid = -1
    animal_map  = torch.full((10,), -1, dtype=torch.long, device=device)
    vehicle_map = torch.full((10,), -1, dtype=torch.long, device=device)
    for i, c in enumerate(animal_classes):
        animal_map[c] = i
    for i, c in enumerate(vehicle_classes):
        vehicle_map[c] = i

    # M1 auxiliary CE
    if m1_logits is not None:
        y1 = animal_map[y]           # [-1 or 0..5]
        mask1 = y1.ge(0)
        if mask1.any():
            loss = loss + F.cross_entropy(m1_logits[mask1], y1[mask1])

    # M2 auxiliary CE
    if m2_logits is not None:
        y2 = vehicle_map[y]          # [-1 or 0..3]
        mask2 = y2.ge(0)
        if mask2.any():
            loss = loss + F.cross_entropy(m2_logits[mask2], y2[mask2])

    return loss


# ------------------------------------------------
# TRADES (fusion-head-driven) + masked aux CE
# ------------------------------------------------
def improved_trades_loss(model: ImprovedFusionModel,
                         x_natural, y, optimizer,
                         step_size=0.003, epsilon=0.031, perturb_steps=10,
                         beta=4.5, aux_w=0.05):
    """
    TRADES guided by the fusion head (10-class) + small masked CE on M1/M2.
    ALL parameters participate in delta gradient computation.
    """
    class FusionWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            return out[-1] if isinstance(out, (tuple, list)) else out

    logits_only = FusionWrapper(model)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = x_natural.size(0)

    # ------ generate adversarial example (freeze BN stats) ------
    logits_only.eval()
    x_adv = (x_natural.detach() + 0.001 * torch.randn_like(x_natural)).clamp(0, 1)

    with torch.no_grad():
        p_nat = F.softmax(logits_only(x_natural), dim=1)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = logits_only(x_adv)
        loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), p_nat)
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = (x_adv.detach() + step_size * torch.sign(grad)).clamp(0, 1)
        # Project back to L_inf ball
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)

    # ------ training step ------
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Fusion CE (clean) + TRADES KL(adv||clean)
    logits_nat = logits_only(x_natural)
    logits_adv = logits_only(x_adv)

    loss_nat = F.cross_entropy(logits_nat, y)
    loss_rob = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1)
    )

    # Masked auxiliary CE for M1/M2 on their respective label subsets
    m1_nat, m2_nat, _ = model(x_natural)
    loss_aux = masked_aux_ce(m1_nat, m2_nat, y)

    loss = loss_nat + beta * loss_rob + aux_w * loss_aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss


# ------------------------------------------------
# Full-fusion training loop (staged LRs)
# ------------------------------------------------
def build_full_optimizer(fusion_model: ImprovedFusionModel, base_lr: float, weight_decay=5e-4):
    """
    Parameter groups: HeadG (fast), M1/M2 (slower). Helps preserve pre-trained backbones.
    """
    param_groups = [
        {'params': fusion_model.head.parameters(), 'lr': base_lr},        # Head fast
        {'params': fusion_model.m1.parameters(),   'lr': base_lr * 0.3},  # M1 slow
        {'params': fusion_model.m2.parameters(),   'lr': base_lr * 0.3},  # M2 slow
    ]
    opt = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return opt


def train_improved_fusion_adversarial(fusion_model: ImprovedFusionModel,
                                      train_loader, test_loader,
                                      args, logger,
                                      aux_w: float = 0.05):
    """
    Train ALL components (M1 + M2 + HeadG) with:
      - TRADES (improved, fusion-head-driven) + small masked aux CE
      - or MART (via wrapper)
      - or CE (warmup)
    Scheduler: Cosine Annealing over 'epochs_g' (works well with staged LRs).
    BN calibration is applied every 15 epochs (and after warmup outside this function).
    """
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    opt = build_full_optimizer(fusion_model, base_lr=args.lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()
    eval_atk = make_eval_attack(fusion_model, args)

    for ep in range(1, args.epochs_g + 1):
        fusion_model.train()
        seen, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                loss_main = improved_trades_loss(
                    fusion_model, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                    aux_w=aux_w,
                )
            elif args.trainer == 'mart':
                # Route MART to fusion-head-only while optimizing the full model
                class FusionWrapper(nn.Module):
                    def __init__(self, base):
                        super().__init__()
                        self.base = base
                    def forward(self, x):
                        out = self.base(x)
                        return out[-1] if isinstance(out, (tuple, list)) else out

                wrapped_model = FusionWrapper(fusion_model)
                loss_val = mart_loss(
                    wrapped_model, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_val[0] if isinstance(loss_val, (tuple, list)) else loss_val
            else:
                # CE warmup on the fusion head only
                opt.zero_grad(set_to_none=True)
                # forward once to get fusion logits
                _, _, fusion_logits = fusion_model(x)
                loss_main = ce(fusion_logits, y)
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 5.0)
                opt.step()

            loss_sum += float(loss_main.detach().item()) * x.size(0)

            # Clean accuracy on the fly (fusion head)
            with torch.no_grad():
                _, _, fusion_logits = fusion_model(x)
                correct += (fusion_logits.argmax(1) == y).sum().item()
                seen += y.size(0)

        sch.step()

        # Optional BN calibration during adversarial training improves stability
        if ep % 15 == 1 and args.trainer != 'ce':
            bn_calibration(fusion_model, train_loader, max_batches=200)

        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(fusion_model, test_loader)
            adv_acc = eval_adv(fusion_model, test_loader, eval_atk)
            logger.log(f'[Improved-Fusion-{args.trainer}] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # Reuse repo's parser_train (defines --data-dir, --log-dir, --desc, --data, and attack args)
    parse = parser_train()
    # Add only NEW flags for this script
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 clean training')
    parse.add_argument('--epochs-g', type=int, default=50, help='total epochs for fusion adversarial training')
    parse.add_argument('--lr-m', type=float, default=0.1, help='LR for M1/M2 clean training')
    parse.add_argument('--trainer', type=str, default='trades', choices=['trades', 'mart', 'ce'],
                       help='objective for fusion stage (trades/mart/ce)')
    parse.add_argument('--aux-w', type=float, default=0.05, help='weight for masked auxiliary CE on M1/M2')

    args = parse.parse_args()

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger.log(f'Using device: {DEVICE}')
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Full 10-class loaders
    _, full_train_loader = _build_cifar10(DATA_DIR, train=True,
                                          num_workers=getattr(args, 'workers', 4),
                                          batch_size=args.batch_size)
    _, full_test_loader = _build_cifar10(DATA_DIR, train=False,
                                         num_workers=getattr(args, 'workers', 4),
                                         batch_size=args.batch_size)

    # Filtered loaders (6 animal classes / 4 vehicle classes)
    m1_train_loader = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=True,
                                            num_workers=getattr(args, 'workers', 4))
    m1_test_loader = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=False,
                                           num_workers=getattr(args, 'workers', 4))
    m2_train_loader = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=True,
                                            num_workers=getattr(args, 'workers', 4))
    m2_test_loader = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=False,
                                           num_workers=getattr(args, 'workers', 4))

    # -------------------------
    # Stage 1: clean training
    # -------------------------
    min_epochs_m = max(args.epochs_m, 50)
    logger.log(f"Training M1 (6-class) for {min_epochs_m} epochs (CE)...")
    m1 = build_lightresnet20(num_classes=len(animal_classes))
    train_clean_classifier(m1, m1_train_loader, m1_test_loader, min_epochs_m, args.lr_m, logger, '[M1]')

    logger.log(f"Training M2 (4-class) for {min_epochs_m} epochs (CE)...")
    m2 = build_lightresnet20(num_classes=len(vehicle_classes))
    train_clean_classifier(m2, m2_train_loader, m2_test_loader, min_epochs_m, args.lr_m, logger, '[M2]')

    m1_acc = eval_clean(m1, m1_test_loader)
    m2_acc = eval_clean(m2, m2_test_loader)
    logger.log(f'[M1] Clean Test Acc: {m1_acc:.4f}')
    logger.log(f'[M2] Clean Test Acc: {m2_acc:.4f}')

    # Optional small extra fine-tuning if submodel accuracy is low
    if m1_acc < 0.80:
        logger.log(f"[M1] Acc {m1_acc:.4f} is low; fine-tuning +20 epochs @ lr*0.1...")
        train_clean_classifier(m1, m1_train_loader, m1_test_loader, 20, args.lr_m * 0.1, logger, '[M1-extra]')
        m1_acc = eval_clean(m1, m1_test_loader)
        logger.log(f'[M1] Updated Clean Test Acc: {m1_acc:.4f}')

    if m2_acc < 0.85:
        logger.log(f"[M2] Acc {m2_acc:.4f} is low; fine-tuning +20 epochs @ lr*0.1...")
        train_clean_classifier(m2, m2_train_loader, m2_test_loader, 20, args.lr_m * 0.1, logger, '[M2-extra]')
        m2_acc = eval_clean(m2, m2_test_loader)
        logger.log(f'[M2] Updated Clean Test Acc: {m2_acc:.4f}')

    # -------------------------
    # Stage 2: fusion + adversarial training
    # -------------------------
    # infer concat dim from fc.in_features of both backbones
    penult_dim = int(m1.fc.in_features + m2.fc.in_features)
    logger.log(f'Inferred penultimate concat dim: {penult_dim}')
    head = HeadG(in_dim=penult_dim, num_classes=10).to(DEVICE)

    fusion_model = ImprovedFusionModel(m1, m2, head).to(DEVICE)
    logger.log('Improved fusion model: staged learning rates')
    logger.log(f'Total trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)}')

    # CE warmup stabilizes head + BN before adversarial training
    logger.log("Warmup (CE) for improved fusion: 5 epochs...")
    warm_args = argparse.Namespace(**vars(args))
    warm_args.trainer = 'ce'
    warm_args.epochs_g = 5
    train_improved_fusion_adversarial(fusion_model, full_train_loader, full_test_loader, warm_args, logger, aux_w=args.aux_w)

    # BN calibration right after warmup helps a lot
    bn_calibration(fusion_model, full_train_loader, max_batches=200)

    # Full adversarial training
    logger.log(f"Improved adversarial training ({args.trainer}) for {args.epochs_g} epochs...")
    logger.log(f"Head LR: {args.lr}, M1/M2 LR: {args.lr * 0.3}")
    train_improved_fusion_adversarial(fusion_model, full_train_loader, full_test_loader, args, logger, aux_w=args.aux_w)

    # Final evaluation
    clean_g = eval_clean(fusion_model, full_test_loader)
    adv_g = eval_adv(fusion_model, full_test_loader, make_eval_attack(fusion_model, args))
    logger.log(f'[Improved-Fusion] Final Test Clean: {clean_g:.4f} | Final Test Adv: {adv_g:.4f}')

    # Save checkpoints
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(LOG_DIR, 'M1_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(LOG_DIR, 'M2_4cls.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'model_state_dict': fusion_model.state_dict()}, os.path.join(LOG_DIR, 'Improved_Fusion.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
