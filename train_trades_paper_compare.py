#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRADES (paper-faithful) comparison script

Implements the exact TRADES loss from:
Zhang et al., "Theoretically Principled Trade-off between Robustness and Accuracy" (ICML 2019)

Loss:
  total = CE(f(x), y) + beta * KL( f(x) || f(x_adv) ),
where x_adv is obtained by maximizing the KL term within an L_inf ball (PGD on KL).

Two modes in one file:
  --mode fusion : two submodels (M1: 6-class animals, M2: 4-class vehicles) + fusion head (10-class).
                   Adversarial training is applied ONLY on the fusion 10-class head
                   (all parameters are trainable). No auxiliary losses.
  --mode plain  : a single 10-class LightResnet trained with the SAME TRADES loss.

Pipeline:
  1) Build data loaders for CIFAR-10
  2) (fusion only) Clean-train M1/M2 with CE
  3) Short CE warmup for the model to stabilize BN/head
  4) BN calibration on clean data
  5) Full adversarial training with TRADES (paper-faithful)
  6) Report clean & adversarial test accuracy

This script depends on your repo's "core" module (LightResnet, attacks, parser_train, etc.).
"""

import os
import sys
import json
import shutil
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

# --- import local 'core' from your repo ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.models.resnet import LightResnet, BasicBlock
from core.utils import Logger, parser_train, seed
from core.attacks import create_attack
from core import animal_classes, vehicle_classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# Models
# ------------------------------------------------
def build_lightresnet20(num_classes: int) -> nn.Module:
    """Small ResNet used throughout the repo (works well for CIFAR-10)."""
    return LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE).to(DEVICE)


class HeadG(nn.Module):
    """2-layer MLP as fusion head to produce 10-class logits."""
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


class FullFusionModel(nn.Module):
    """
    Fusion model:
      - M1: 6-class (animals)
      - M2: 4-class (vehicles)
      - HeadG: 10-class fusion head on concatenated penultimate features
    We use forward hooks to capture penultimate features (inputs to m1.fc/m2.fc).
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.head = head

        self._feats = {}
        self._h1 = self.m1.fc.register_forward_hook(lambda m, inp, out: self._save_feat("m1", inp))
        self._h2 = self.m2.fc.register_forward_hook(lambda m, inp, out: self._save_feat("m2", inp))

    def _save_feat(self, key, inp_tuple):
        # inp_tuple[0] is the input to the final fc layer: shape [B, D]
        self._feats[key] = inp_tuple[0]

    def forward(self, x):
        m1_logits = self.m1(x)  # 6-way
        m2_logits = self.m2(x)  # 4-way
        z = torch.cat([self._feats["m1"], self._feats["m2"]], dim=1)
        fusion_logits = self.head(z)  # 10-way
        return m1_logits, m2_logits, fusion_logits

    def remove_hooks(self):
        self._h1.remove()
        self._h2.remove()


# ------------------------------------------------
# Data
# ------------------------------------------------
def build_cifar10(data_dir: str, train: bool, num_workers=4, batch_size=128) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    """Standard CIFAR-10 with light augmentation for train."""
    tfm = (T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]) if train else T.Compose([T.ToTensor()]))
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
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
    ds, _ = build_cifar10(data_dir, train=train, num_workers=num_workers, batch_size=batch_size)
    indices, remap = _filter_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader


# ------------------------------------------------
# Eval helpers
# ------------------------------------------------
class FusionWrapper(nn.Module):
    """Wrap any model to expose ONLY a 10-class logits tensor (for eval/attacks)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        out = self.base(x)
        return out[-1] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader) -> float:
    """
    Clean accuracy of the 10-class logits.
      - For fusion model: use the fusion head
      - For plain model: use its single 10-class head
    """
    m = FusionWrapper(model).eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = m(x)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


def make_eval_attack(model: nn.Module, args):
    """
    Build an evaluation attack that crafts perturbations w.r.t. the 10-class logits.
    Uses the exact same attack hyper-parameters passed via CLI.
    """
    crit  = nn.CrossEntropyLoss()
    eps   = getattr(args, 'attack_eps', 8/255)
    step  = getattr(args, 'attack_step', 2/255)
    iters = getattr(args, 'attack_iter', 10)
    attack_name = getattr(args, 'attack', 'linf-pgd')
    return create_attack(FusionWrapper(model), crit, attack_name, eps, iters, step)


def eval_adv(model: nn.Module, loader: DataLoader, attack) -> float:
    """Adversarial accuracy measured on x_adv crafted for fusion/plain head."""
    m = FusionWrapper(model).eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv, _ = attack.perturb(x, y)  # attack uses gradients w.r.t. m (fusion/plain head)
        with torch.no_grad():
            logits = m(x_adv)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


@torch.no_grad()
def bn_calibration(model: nn.Module, loader: DataLoader, max_batches=200):
    """
    Recalibrate BatchNorm running stats using clean data.
    Call after CE warmup and optionally during adversarial training.
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
# Clean training for submodels (fusion mode)
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
# TRADES (paper-faithful) loss
# ------------------------------------------------
def trades_loss_paper(model: nn.Module,
                      x_natural, y, optimizer,
                      step_size=0.003, epsilon=0.031, perturb_steps=10,
                      beta=6.0):
    """
    Paper-faithful TRADES on the exposed 10-class head:
      total = CE(f(x), y) + beta * KL( f(x) || f(x_adv) )

    Implementation details:
      - We compute probabilities with softmax() and log-probabilities with log_softmax()
      - Inner maximization: PGD on the KL term, *ascending* the KL
      - We freeze BN running stats during inner maximization (eval mode)
      - We detach the "clean" distribution for the outer KL to avoid gradient leakage
    """
    head_only = FusionWrapper(model)        # expose only 10-class logits
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = x_natural.size(0)

    # ----- generate adversarial example (inner max) -----
    head_only.eval()  # freeze BN stats during PGD
    x_adv = (x_natural.detach() + 0.001 * torch.randn_like(x_natural)).clamp(0, 1)

    with torch.no_grad():
        p_nat = F.softmax(head_only(x_natural), dim=1)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = head_only(x_adv)
        # KL( p_nat || p_adv )  -> maximize w.r.t. x_adv
        loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), p_nat)
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        # PGD step + L_inf projection + clip
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)
        x_adv = x_adv.clamp(0, 1)

    # ----- parameter update (outer min) -----
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits_nat = head_only(x_natural)
    logits_adv = head_only(x_adv)

    loss_nat = F.cross_entropy(logits_nat, y)
    loss_rob = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1)
    )
    loss = loss_nat + beta * loss_rob

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss


# ------------------------------------------------
# Train loops
# ------------------------------------------------
def train_plain10(model: nn.Module,
                  train_loader: DataLoader,
                  test_loader: DataLoader,
                  args,
                  logger: Logger,
                  warmup_epochs: int = 5):
    """Plain 10-class model trained with CE warmup + TRADES (paper-faithful)."""
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=weight_decay, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)

    ce = nn.CrossEntropyLoss()
    eval_atk = make_eval_attack(model, args)

    # --- Warmup ---
    logger.log(f"Warmup (CE) for plain-10: {warmup_epochs} epochs...")
    for ep in range(1, warmup_epochs + 1):
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
            clean_acc = eval_clean(model, test_loader)
            adv_acc = eval_adv(model, test_loader, eval_atk)
            logger.log(f'[Plain-10-ce] Epoch {ep:03d} | Train Loss {(loss_sum/max(seen,1)):.4f} | '
                       f'Train Acc {(correct/max(seen,1)):.4f} | Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')

    # BN calibration helps before adversarial phase
    bn_calibration(model, train_loader, max_batches=200)

    # --- TRADES phase ---
    logger.log(f"Plain-10 adversarial training (TRADES paper) for {args.epochs_g} epochs...")
    for ep in range(1, args.epochs_g + 1):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss_main = trades_loss_paper(
                model, x, y, optimizer=opt,
                beta=args.beta,
                step_size=getattr(args, 'attack_step', 2/255),
                epsilon=getattr(args, 'attack_eps', 8/255),
                perturb_steps=getattr(args, 'attack_iter', 10),
            )
            loss_sum += float(loss_main.detach().item()) * x.size(0)
            with torch.no_grad():
                logits = FusionWrapper(model)(x)
                correct += (logits.argmax(1) == y).sum().item()
                seen += y.size(0)
        sch.step()
        if ep % 15 == 1:
            bn_calibration(model, train_loader, max_batches=200)
        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(model, test_loader)
            adv_acc = eval_adv(model, test_loader, eval_atk)
            logger.log(f'[Plain-10-trades] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')


def train_fusion(model_fusion: FullFusionModel,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 args,
                 logger: Logger,
                 warmup_epochs: int = 5):
    """Fusion training with CE warmup + TRADES (paper-faithful). No aux losses."""
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    # Single LR for all params to keep parity with plain-10 setup
    opt = torch.optim.SGD(model_fusion.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=weight_decay, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)

    ce = nn.CrossEntropyLoss()
    eval_atk = make_eval_attack(model_fusion, args)

    # --- Warmup on fusion head only ---
    logger.log(f"Warmup (CE) for fusion: {warmup_epochs} epochs...")
    for ep in range(1, warmup_epochs + 1):
        model_fusion.train()
        seen, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            # forward once to get fusion logits
            _, _, fusion_logits = model_fusion(x)
            loss = ce(fusion_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_fusion.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item()) * x.size(0)
            correct += (fusion_logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        sch.step()
        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(model_fusion, test_loader)
            adv_acc = eval_adv(model_fusion, test_loader, eval_atk)
            logger.log(f'[Fusion-ce] Epoch {ep:03d} | Train Loss {(loss_sum/max(seen,1)):.4f} | '
                       f'Train Acc {(correct/max(seen,1)):.4f} | Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')

    # BN calibration helps before adversarial phase
    bn_calibration(model_fusion, train_loader, max_batches=200)

    # --- TRADES phase on fusion head ---
    logger.log(f"Fusion adversarial training (TRADES paper) for {args.epochs_g} epochs...")
    for ep in range(1, args.epochs_g + 1):
        model_fusion.train()
        seen, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss_main = trades_loss_paper(
                model_fusion, x, y, optimizer=opt,
                beta=args.beta,
                step_size=getattr(args, 'attack_step', 2/255),
                epsilon=getattr(args, 'attack_eps', 8/255),
                perturb_steps=getattr(args, 'attack_iter', 10),
            )
            loss_sum += float(loss_main.detach().item()) * x.size(0)
            with torch.no_grad():
                _, _, fusion_logits = model_fusion(x)
                correct += (fusion_logits.argmax(1) == y).sum().item()
                seen += y.size(0)
        sch.step()
        if ep % 15 == 1:
            bn_calibration(model_fusion, train_loader, max_batches=200)
        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(model_fusion, test_loader)
            adv_acc = eval_adv(model_fusion, test_loader, eval_atk)
            logger.log(f'[Fusion-trades] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # Reuse repo's parser_train to keep CLI consistent
    parse = parser_train()
    parse.add_argument('--mode', type=str, default='plain', choices=['plain', 'fusion'],
                       help='plain: single 10-class model; fusion: M1+M2+HeadG 10-class fusion')
    parse.add_argument('--epochs-m', type=int, default=50,
                       help='(fusion only) epochs for M1/M2 clean training')
    parse.add_argument('--epochs-g', type=int, default=50,
                       help='epochs for TRADES adversarial phase')
    parse.add_argument('--lr-m', type=float, default=0.1,
                       help='(fusion only) LR for M1/M2 clean training')
    parse.add_argument('--warmup-epochs', type=int, default=5,
                       help='CE warmup epochs before adversarial phase')

    args = parse.parse_args()

    # Paths & logging
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

    # Data
    _, train_loader = build_cifar10(DATA_DIR, train=True,
                                    num_workers=getattr(args, 'workers', 4),
                                    batch_size=args.batch_size)
    _, test_loader  = build_cifar10(DATA_DIR, train=False,
                                    num_workers=getattr(args, 'workers', 4),
                                    batch_size=args.batch_size)

    if args.mode == 'plain':
        # Plain 10-class baseline
        model = build_lightresnet20(num_classes=10)
        logger.log(f'Plain 10-class model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        train_plain10(model, train_loader, test_loader, args, logger, warmup_epochs=args.warmup_epochs)
        # Final evaluation & save
        clean_acc = eval_clean(model, test_loader)
        adv_acc   = eval_adv(model, test_loader, make_eval_attack(model, args))
        logger.log(f'[Plain-10] Final Test Clean: {clean_acc:.4f} | Final Test Adv: {adv_acc:.4f}')
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(LOG_DIR, 'Plain10_TRADES_Paper.pt'))
        logger.log(f'Saved model to {LOG_DIR}')
        return

    # --- fusion mode ---
    # Build subset loaders for M1 (animals) / M2 (vehicles)
    m1_train_loader = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=True,
                                            num_workers=getattr(args, 'workers', 4))
    m1_test_loader  = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=False,
                                            num_workers=getattr(args, 'workers', 4))
    m2_train_loader = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=True,
                                            num_workers=getattr(args, 'workers', 4))
    m2_test_loader  = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=False,
                                            num_workers=getattr(args, 'workers', 4))

    # Stage-1: clean train M1/M2
    epochs_m = max(args.epochs_m, 50)
    logger.log(f"Training M1 (6-class animals) for {epochs_m} epochs (CE)...")
    m1 = build_lightresnet20(num_classes=len(animal_classes))
    train_clean_classifier(m1, m1_train_loader, m1_test_loader, epochs_m, args.lr_m, logger, '[M1]')

    logger.log(f"Training M2 (4-class vehicles) for {epochs_m} epochs (CE)...")
    m2 = build_lightresnet20(num_classes=len(vehicle_classes))
    train_clean_classifier(m2, m2_train_loader, m2_test_loader, epochs_m, args.lr_m, logger, '[M2]')

    m1_acc = eval_clean(m1, m1_test_loader)
    m2_acc = eval_clean(m2, m2_test_loader)
    logger.log(f'[M1] Clean Test Acc: {m1_acc:.4f}')
    logger.log(f'[M2] Clean Test Acc: {m2_acc:.4f}')

    # Stage-2: build fusion and adversarially train fusion head with TRADES
    penult_dim = int(m1.fc.in_features + m2.fc.in_features)
    logger.log(f'Inferred penultimate concat dim: {penult_dim}')
    head = HeadG(in_dim=penult_dim, num_classes=10).to(DEVICE)
    fusion_model = FullFusionModel(m1, m2, head).to(DEVICE)
    logger.log(f'Fusion model params: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)}')

    train_fusion(fusion_model, train_loader, test_loader, args, logger, warmup_epochs=args.warmup_epochs)

    # Final evaluation & save
    clean_g = eval_clean(fusion_model, test_loader)
    adv_g   = eval_adv(fusion_model, test_loader, make_eval_attack(fusion_model, args))
    logger.log(f'[Fusion] Final Test Clean: {clean_g:.4f} | Final Test Adv: {adv_g:.4f}')

    torch.save({'model_state_dict': m1.state_dict()},   os.path.join(LOG_DIR, 'M1_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()},   os.path.join(LOG_DIR, 'M2_4cls.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'model_state_dict': fusion_model.state_dict()}, os.path.join(LOG_DIR, 'Fusion_TRADES_Paper.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
