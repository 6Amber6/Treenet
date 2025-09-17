"""
Plain 10-class CIFAR-10 + TRADES (baseline)
-------------------------------------------
Implements Zhang et al. (ICML'19) TRADES objective on a single 10-class model:

  Loss = CE( f(x), y ) + beta * KL( p(x) || p(x_adv) )

where x_adv is obtained by maximizing KL( p(x) || p(x_adv) ) within an L_inf ball
via PGD. No auxiliary heads, no masking — a clean baseline to compare against
your fusion model trained with the same attack hyper-parameters.

This script uses the same repo utilities as your fusion code:
- core.utils.parser_train / Logger / seed
- core.models.resnet.LightResnet
- core.attacks.create_attack  (for evaluation only)
"""

import os
import sys
import json
import shutil
import argparse
from typing import Tuple

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
from core.utils.mart import mart_loss  # kept for optional comparison

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# Model
# ------------------------------------------------
def build_lightresnet20_10() -> nn.Module:
    """LightResnet-20 variant used in your repo, set to 10 classes."""
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=10, device=DEVICE)
    return model.to(DEVICE)


# ------------------------------------------------
# Data
# ------------------------------------------------
def build_cifar10(data_dir: str, train: bool, num_workers=4, batch_size=128) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    """Standard CIFAR-10; light aug on train, plain ToTensor on test."""
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


# ------------------------------------------------
# Eval helpers (model outputs 10-class logits directly)
# ------------------------------------------------
@torch.no_grad()
def eval_clean(model: nn.Module, loader: DataLoader) -> float:
    """Clean accuracy of the 10-class head."""
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


def make_eval_attack(model: nn.Module, args):
    """
    Build an evaluation attack that crafts perturbations w.r.t. the 10-class logits.
    Uses exactly the same attack hyper-parameters as training flags.
    """
    crit  = nn.CrossEntropyLoss()
    eps   = getattr(args, 'attack_eps', 8/255)
    step  = getattr(args, 'attack_step', 2/255)
    iters = getattr(args, 'attack_iter', 10)
    attack_name = getattr(args, 'attack', 'linf-pgd')
    return create_attack(model, crit, attack_name, eps, iters, step)


def eval_adv(model: nn.Module, loader: DataLoader, attack) -> float:
    """Adversarial accuracy measured on x_adv crafted for this model."""
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # Attack internally uses model gradients (do NOT wrap this in no_grad)
        x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            logits = model(x_adv)
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
# TRADES loss (paper-authentic) + CE warmup
# ------------------------------------------------
def trades_loss_plain(model: nn.Module,
                      x_natural, y, optimizer,
                      step_size=0.003, epsilon=0.031, perturb_steps=10,
                      beta=6.0):
    """
    TRADES on a plain 10-class model (Zhang et al., ICML'19):

      total = CE( f(x), y ) + beta * KL( p(x) || p(x_adv) )

    Implementation details:
      - Inner PGD maximizes KL( p(x) || p(x_adv) ) w.r.t x_adv in L_inf-ball.
      - We freeze BN running stats when crafting x_adv: model.eval().
      - Then switch back to model.train() for the parameter update.
      - KL is computed as KL(target || input) using torch.nn.KLDivLoss with
        input = log_softmax, target = softmax.
    """
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = x_natural.size(0)

    # ----- generate adversarial example -----
    model.eval()  # keep BN stats stable during inner maximization
    x_adv = (x_natural.detach() + 0.001 * torch.randn_like(x_natural)).clamp(0, 1)

    with torch.no_grad():
        p_nat = F.softmax(model(x_natural), dim=1)  # p(x)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = model(x_adv)
        # maximize KL( p(x) || p(x_adv) )
        loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), p_nat)
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        # PGD step + L_inf projection + clip
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)
        x_adv = x_adv.clamp(0, 1)

    # ----- parameter update -----
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    loss_nat = F.cross_entropy(logits_nat, y)
    # KL( p(x) || p(x_adv) )
    loss_rob = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat.detach(), dim=1)
    )
    loss = loss_nat + beta * loss_rob

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss


def train_plain10(model: nn.Module,
                  train_loader: DataLoader,
                  test_loader: DataLoader,
                  args,
                  logger: Logger,
                  warmup_epochs: int = 5):
    """
    Train a 10-class model with:
      - CE warmup (warmup_epochs)
      - CosineAnnealingLR
      - TRADES (or MART) as selected by --trainer
    """
    weight_decay = getattr(args, 'weight_decay', 5e-4)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=weight_decay, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)

    ce = nn.CrossEntropyLoss()
    eval_atk = make_eval_attack(model, args)

    # -------- Warmup (CE only) --------
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
            adv_acc = eval_adv(model, test_loader, eval_atk)  # ~0 during warmup
            logger.log(f'[Plain-10-ce] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')

    # BN calibration before adversarial phase
    bn_calibration(model, train_loader, max_batches=200)

    # -------- Adversarial phase --------
    logger.log(f"Plain-10 adversarial training ({args.trainer}) for {args.epochs_g} epochs...")
    for ep in range(1, args.epochs_g + 1):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                loss_main = trades_loss_plain(
                    model, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
            elif args.trainer == 'mart':
                # Optional MART baseline (not used if you select --trainer trades)
                loss_val = mart_loss(
                    model, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_val[0] if isinstance(loss_val, (tuple, list)) else loss_val
            else:
                # CE-only branch (rarely used for this baseline)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss_main = ce(logits, y)
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            loss_sum += float(loss_main.detach().item()) * x.size(0)
            with torch.no_grad():
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                seen += y.size(0)

        sch.step()

        # Optional: periodic BN calibration (e.g., every 15 epochs)
        if ep % 15 == 1 and args.trainer != 'ce':
            bn_calibration(model, train_loader, max_batches=200)

        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(model, test_loader)
            adv_acc = eval_adv(model, test_loader, eval_atk)
            logger.log(f'[Plain-10-{args.trainer}] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # Keep CLI consistent with your repo (parser_train defines --data/--attack-*)
    parse = parser_train()
    parse.add_argument('--epochs-g', type=int, default=50, help='epochs for adversarial phase')
    parse.add_argument('--trainer', type=str, default='trades', choices=['trades', 'mart', 'ce'],
                       help='objective (trades/mart/ce) for adversarial phase')
    parse.add_argument('--warmup-epochs', type=int, default=5, help='CE warmup epochs before adversarial phase')

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

    # Seed & device
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

    # Model
    model = build_lightresnet20_10()
    logger.log(f'Plain 10-class model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # Train
    train_plain10(model, train_loader, test_loader, args, logger, warmup_epochs=args.warmup_epochs)

    # Final evaluation
    clean_acc = eval_clean(model, test_loader)
    adv_acc   = eval_adv(model, test_loader, make_eval_attack(model, args))
    logger.log(f'[Plain-10] Final Test Clean: {clean_acc:.4f} | Final Test Adv: {adv_acc:.4f}')

    # Save checkpoint
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(LOG_DIR, 'Plain10_TRADES.pt'))
    logger.log(f'Saved model to {LOG_DIR}')


if __name__ == '__main__':
    main()
