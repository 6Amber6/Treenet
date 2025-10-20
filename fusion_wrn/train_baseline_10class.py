# train_wrn10_adv_baseline.py
# Baseline: Single WRN-28-10 on CIFAR-10 with TRADES / MART (no 4/6 split), matching fusion settings

import os
import sys
import json
import shutil
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ------------------------------ Repo Paths ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
ARP_ROOT = os.path.join(REPO_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

# Core utilities from your repo
from core.models.wideresnet import wideresnet
from core.utils import Logger, parser_train, seed
from core.attacks import create_attack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------ Data ------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def _build_cifar10(data_dir, train: bool, num_workers=4, batch_size=128):
    if train:
        tfm = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            # Cutout-style regularization that helps TRADES a bit
            T.RandomErasing(p=1.0, scale=(0.05, 0.10), ratio=(0.5, 2.0), value=0),
        ])
    else:
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=False if num_workers == 0 else True)
    return ds, loader

# ------------------------------ Model -----------------------------------
def build_wrn_28_10(num_classes: int = 10):
    model = wideresnet('wideresnet-28-10', num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)

# --------------------------- EMA helper ---------------------------------
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.backup = {}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n].clone()
    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]

# --------------------------- Utils --------------------------------------
def freeze_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

def unfreeze_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()


# --------------------------- MART Loss ----------------------------------
def mart_loss(logits_adv: torch.Tensor, logits_nat: torch.Tensor, y: torch.Tensor, lam: float = 5.0):
    """
    Misclassification Aware Adversarial Training loss
    Consistent with fusion implementation: adv CE + margin hinge + KL
    """
    ce_adv = F.cross_entropy(logits_adv, y, reduction='none')
    prob_nat = F.softmax(logits_nat, dim=1)
    top2_values, top2_idx = prob_nat.topk(2, dim=1)
    y_hat = torch.where(top2_idx[:, 0] == y, top2_idx[:, 1], top2_idx[:, 0])

    logp = F.log_softmax(logits_nat, dim=1)
    margin = logp[torch.arange(y.size(0), device=y.device), y] - \
             logp[torch.arange(y.size(0), device=y.device), y_hat]
    loss_miscls = lam * F.relu(1.0 - margin)

    kl = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1),
        reduction='batchmean'
    )
    return (ce_adv + loss_miscls).mean() + kl

# ----------------------- Adversarial Step (TRADES/MART) -----------------
def adv_step(model: nn.Module, x_natural, y, optimizer,
             step_size_pix=2/255, epsilon_pix=8/255, perturb_steps=12,
             beta=8.0, use_mart: bool=False, label_smoothing: float=0.0):
    """
    Single model adversarial training step; PGD in pixel space, then normalize input.
    """
    model_logits_only = model  # baseline directly outputs logits
    model_logits_only.eval()

    # Denormalize back to pixel space [0,1]
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    x_orig = x_natural * std + mean

    with torch.no_grad():
        p_nat = F.softmax(model_logits_only(x_natural), dim=1)

    # PGD perturbation in pixel space
    x_adv = x_orig + 1e-3 * torch.randn_like(x_orig)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        # Renormalize before forward pass
        x_adv_norm = (x_adv - mean) / std
        logits_adv = model_logits_only(x_adv_norm)
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_nat, reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + step_size_pix * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_orig - epsilon_pix), x_orig + epsilon_pix)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Normalize back to input distribution
    x_adv_norm = (x_adv - mean) / std

    # Parameter update
    model.train()
    optimizer.zero_grad(set_to_none=True)

    f_nat = model(x_natural)
    f_adv = model(x_adv_norm)

    if label_smoothing > 0.0:
        loss_nat = F.cross_entropy(f_nat, y, label_smoothing=label_smoothing)
    else:
        loss_nat = F.cross_entropy(f_nat, y)

    if use_mart:
        loss_rob = mart_loss(f_adv, f_nat, y)
        loss = loss_nat + loss_rob
    else:
        loss_rob = F.kl_div(F.log_softmax(f_adv, dim=1),
                            F.softmax(f_nat.detach(), dim=1), reduction='batchmean')
        loss = loss_nat + beta * loss_rob

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())

# --------------------------- Eval --------------------------------------
@torch.no_grad()
def eval_clean(model, loader) -> float:
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)

def make_eval_attack(model, args):
    crit = nn.CrossEntropyLoss()
    attack_type = getattr(args, 'attack', None) or 'linf-pgd'
    attack_eps = getattr(args, 'attack_eps', None) or 8/255
    attack_iter = getattr(args, 'attack_iter', None) or 20  # Strong evaluation: PGD-20
    attack_step = getattr(args, 'attack_step', None) or 2/255
    return create_attack(model, crit, attack_type, attack_eps, attack_iter, attack_step)

def eval_adv(model, loader, attack) -> float:
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv, _ = attack.perturb(x, y)
        logits = model(x_adv)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)

# --------------------------- Training Loop ------------------------------
def train_baseline(model, train_loader, test_loader, args, logger):
    # Optimizer and scheduler
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    ema = EMA(model, decay=getattr(args, 'ema_decay', 0.9995))

    # CE warmup first (consistent with fusion: max 10 epochs)
    warmup_epochs = min(10, args.epochs)
    for ep in range(1, warmup_epochs + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            total_loss += loss.item(); num_batches += 1
        # Don't step scheduler during warmup
        ema.apply_to(model); clean = eval_clean(model, test_loader); ema.restore(model)
        logger.log(f'[WRN10-CE] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Clean {clean:.4f}')

    # Adversarial phase: disable Dropout (WRN built-in dropout still controlled by train(), here ensure BN stats frozen for stable PGD)
    freeze_bn(model)
    atk_eval = make_eval_attack(model, args)

    # If MART, do some initialization alignment (optional but aligned with fusion)
    if getattr(args, 'use_mart', False):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.reset_running_stats()
        # Appropriately reduce learning rate
        for pg in opt.param_groups:
            pg['lr'] *= 0.1
        logger.log(f'[MART-Init] Reset BN stats and reduced LR to {opt.param_groups[0]["lr"]:.6f}')

    method_name = 'MART' if getattr(args, 'use_mart', False) else 'TRADES'
    unfroze_bn = False

    for ep in range(warmup_epochs + 1, args.epochs + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        # Default parameters
        base_attack_step = getattr(args, 'attack_step', None) or 2/255
        base_attack_eps  = getattr(args, 'attack_eps', None)  or 8/255
        base_attack_iter = getattr(args, 'attack_iter', None) or 12
        beta_value       = getattr(args, 'beta', None) or 8.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # MART early stage gradual attack strength increase (aligned with fusion)
            if getattr(args, 'use_mart', False) and (ep - warmup_epochs) <= 10:
                attack_step = base_attack_step * 0.5
                attack_eps  = base_attack_eps * 0.5
                attack_iter = max(7, base_attack_iter // 2)
            else:
                attack_step = base_attack_step
                attack_eps  = base_attack_eps
                attack_iter = base_attack_iter

            loss = adv_step(model, x, y, optimizer=opt,
                            step_size_pix=attack_step,
                            epsilon_pix=attack_eps,
                            perturb_steps=attack_iter,
                            beta=beta_value,
                            use_mart=getattr(args, 'use_mart', False),
                            label_smoothing=getattr(args, 'label_smoothing', 0.0))
            ema.update(model)
            total_loss += float(loss); num_batches += 1

        sch.step()

        # Unfreeze BN after 40 epochs in adversarial phase to let stats adapt (aligned with fusion)
        if not unfroze_bn and (ep - warmup_epochs) >= 40:
            unfreeze_bn(model)
            unfroze_bn = True

        # EMA evaluation
        ema.apply_to(model)
        clean = eval_clean(model, test_loader)
        adv   = eval_adv(model, test_loader, atk_eval)
        ema.restore(model)

        if getattr(args, 'use_mart', False) and (ep - warmup_epochs) <= 10:
            logger.log(f'[WRN10-{method_name}] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} '
                       f'| Test Clean {clean:.4f} | Test Adv {adv:.4f} '
                       f'| Attack: eps={attack_eps:.5f}, step={attack_step:.5f}, iter={attack_iter}')
        else:
            logger.log(f'[WRN10-{method_name}] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} '
                       f'| Test Clean {clean:.4f} | Test Adv {adv:.4f}')

# ------------------------------ Main -----------------------------------
def main():
    parse = parser_train()
    # Unified configuration: consistent with fusion code style
    parse.add_argument('--epochs', type=int, default=120, help='total epochs for baseline WRN-10 (CE warmup + ADV)')
    parse.add_argument('--ema-decay', type=float, default=0.9995, help='EMA decay')
    parse.add_argument('--use-mart', action='store_true', help='use MART (else TRADES)')
    parse.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing on natural CE')
    parse.add_argument('--workers', type=int, default=4, help='DataLoader workers (set 0 if deadlock in container)')

    args = parse.parse_args()

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc + ('_MART' if args.use_mart else '_TRADES'))
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.log(f'Using device: {DEVICE}')
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Dataloaders (consistent transforms / workers)
    _, train_loader = _build_cifar10(DATA_DIR, train=True,
                                     num_workers=getattr(args, 'workers', 4),
                                     batch_size=args.batch_size)
    _, test_loader  = _build_cifar10(DATA_DIR, train=False,
                                     num_workers=getattr(args, 'workers', 4),
                                     batch_size=args.batch_size)

    # Model
    model = build_wrn_28_10(num_classes=10)

    # Train
    method_name = 'MART' if args.use_mart else 'TRADES'
    logger.log(f'Baseline WRN-28-10 10-class with {method_name}')
    logger.log(f'epochs={args.epochs} lr={args.lr} batch_size={args.batch_size} '
               f'attack={getattr(args,"attack","linf-pgd")} eps={getattr(args,"attack_eps",8/255)} '
               f'step={getattr(args,"attack_step",2/255)} iter={getattr(args,"attack_iter",12)}')
    train_baseline(model, train_loader, test_loader, args, logger)

    # Final eval & save
    atk = make_eval_attack(model, args)
    clean = eval_clean(model, test_loader)
    adv   = eval_adv(model, test_loader, atk)
    logger.log(f'[WRN10-Baseline] Final Test Clean {clean:.4f} | Adv {adv:.4f}')

    torch.save({'model_state_dict': model.state_dict()}, os.path.join(LOG_DIR, 'WRN28x10_Baseline.pt'))
    logger.log(f'Saved model to {LOG_DIR}')

if __name__ == '__main__':
    main()
