# train_baseline_10class_adv_improved.py
# Direct 10-class baseline with MART/TRADES switching for comparison with fusion model
# All parameters kept identical to train_fusion_wrn_adv_improved.py

import os
import sys
import json
import shutil
import argparse
from typing import List, Optional

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
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return ds, loader

# ------------------------------ Models ----------------------------------
def build_wrn_28_10(num_classes: int):
    model = wideresnet('wideresnet-28-10', num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)

# --------------------------- EMA helper ---------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
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

# --------------------------- Utilities ----------------------------------
def freeze_backbone_bn(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

def unfreeze_backbone_bn(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()

# --------------------------- MART Loss Implementation -------------------
def mart_loss(logits_adv: torch.Tensor, logits_nat: torch.Tensor, y: torch.Tensor, lam: float = 5.0):
    """
    MART loss implementation: Misclassification Aware Adversarial Robustness Training
    Combines adversarial CE with margin-based misclassification penalty and KL divergence
    """
    # Cross-entropy on adversarial examples
    ce_adv = F.cross_entropy(logits_adv, y, reduction='none')
    
    # Misclassification-aware margin loss
    prob_nat = F.softmax(logits_nat, dim=1)
    top2_values, top2_idx = prob_nat.topk(2, dim=1)
    y_hat = torch.where(top2_idx[:, 0] == y, top2_idx[:, 1], top2_idx[:, 0])
    
    # Margin between true class and second-best class
    margin = F.log_softmax(logits_nat, dim=1)[torch.arange(y.size(0), device=y.device), y] - \
             F.log_softmax(logits_nat, dim=1)[torch.arange(y.size(0), device=y.device), y_hat]
    loss_miscls = lam * F.relu(1.0 - margin)  # hinge loss on margin
    
    # KL divergence between adversarial and natural predictions
    kl = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_nat.detach(), dim=1),
        reduction='batchmean'
    )
    
    return (ce_adv + loss_miscls).mean() + kl

# --------------------------- Adversarial Training Step ------------------
def adv_baseline_step(model, x_natural, y, optimizer,
                     step_size=2/255, epsilon=8/255, perturb_steps=12,
                     beta=8.0, use_mart=False, label_smoothing=0.0):
    """
    Unified adversarial training step supporting both TRADES and MART for baseline 10-class model
    """
    # Craft adversarial examples with eval() so BN/dropout frozen
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x_natural), dim=1)

    # PGD in normalized space
    x_adv = (x_natural.detach() + 1e-3 * torch.randn_like(x_natural)).clamp(-5.0, 5.0)
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = model(x_adv)
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_nat, reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)

    # Training update
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    
    # Natural loss with optional label smoothing
    if label_smoothing > 0.0:
        loss_nat = F.cross_entropy(logits_nat, y, label_smoothing=label_smoothing)
    else:
        loss_nat = F.cross_entropy(logits_nat, y)
    
    # Robust loss: MART or TRADES
    if use_mart:
        loss_rob = mart_loss(logits_adv, logits_nat, y)
        loss = loss_nat + loss_rob
    else:
        loss_rob = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat.detach(), dim=1), reduction='batchmean')
        loss = loss_nat + beta * loss_rob

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.detach()

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
    
    # Use defaults for attack parameters if not specified
    attack_type = getattr(args, 'attack', None) or 'linf-pgd'
    attack_eps = getattr(args, 'attack_eps', None) or 8/255
    attack_iter = getattr(args, 'attack_iter', None) or 20  # strong eval: PGD-20
    attack_step = getattr(args, 'attack_step', None) or 2/255
    
    return create_attack(model, crit,
                         attack_type, attack_eps, attack_iter, attack_step)

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

# --------------------------- Training Loops ----------------------------
def train_ce(model, train_loader, test_loader, epochs, lr, logger, tag, ema=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs // 2, int(epochs * 0.75)], gamma=0.1)
    for ep in range(1, epochs + 1):
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
            if ema: ema.update(model)
            total_loss += loss.item(); num_batches += 1
        sch.step()
        if ep % 5 == 0 or ep == 1:
            if ema: ema.apply_to(model)
            acc = eval_clean(model, test_loader)
            if ema: ema.restore(model)
            logger.log(f'{tag} Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Acc {acc:.4f}')

def train_baseline(model, train_loader, test_loader, args, logger):
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)
    ema = EMA(model, decay=getattr(args, 'ema_decay', 0.999))

    # CE warmup
    warmup_epochs = min(10, args.epochs_g)
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
        sch.step()
        ema.apply_to(model)
        clean = eval_clean(model, test_loader)
        ema.restore(model)
        logger.log(f'[WRN-Baseline-CE] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Clean {clean:.4f}')

    # Freeze BN stats for adversarial training
    freeze_backbone_bn(model)

    atk_eval = make_eval_attack(model, args)
    unfroze_bn = False
    method_name = 'MART' if getattr(args, 'use_mart', False) else 'TRADES'
    
    # MART-specific initialization
    if getattr(args, 'use_mart', False):
        # Reset BN stats for MART phase
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.reset_running_stats()
        
        # Lower learning rate for MART phase
        for param_group in opt.param_groups:
            param_group['lr'] *= 0.1
        logger.log(f'[MART-Init] Reset BN stats and reduced LR to {opt.param_groups[0]["lr"]:.6f}')
    
    for ep in range(warmup_epochs + 1, args.epochs_g + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Use default beta=8.0 if not specified
            beta_value = getattr(args, 'beta', None) or 8.0
            
            # Use defaults for attack parameters if not specified
            base_attack_step = getattr(args, 'attack_step', None) or 2/255
            base_attack_eps = getattr(args, 'attack_eps', None) or 8/255
            base_attack_iter = getattr(args, 'attack_iter', None) or 12
            
            # MART: Gradual attack strength increase for first 10 epochs
            if getattr(args, 'use_mart', False) and (ep - warmup_epochs) <= 10:
                attack_step = base_attack_step * 0.5  # 1/255
                attack_eps = base_attack_eps * 0.5   # 4/255
                attack_iter = max(7, base_attack_iter // 2)  # 7 iterations
            else:
                attack_step = base_attack_step
                attack_eps = base_attack_eps
                attack_iter = base_attack_iter
            
            loss = adv_baseline_step(model, x, y, optimizer=opt,
                                   step_size=attack_step,
                                   epsilon=attack_eps,
                                   perturb_steps=attack_iter,
                                   beta=beta_value,
                                   use_mart=getattr(args, 'use_mart', False),
                                   label_smoothing=getattr(args, 'label_smoothing', 0.0))
            ema.update(model)
            total_loss += float(loss); num_batches += 1
        sch.step()

        # Unfreeze BN after 40 adversarial epochs to let stats adapt
        if not unfroze_bn and (ep - warmup_epochs) >= 40:
            unfreeze_backbone_bn(model)
            unfroze_bn = True

        # EMA eval
        ema.apply_to(model)
        clean = eval_clean(model, test_loader)
        adv   = eval_adv(model, test_loader, atk_eval)
        ema.restore(model)

        # Log attack parameters for MART
        if getattr(args, 'use_mart', False) and (ep - warmup_epochs) <= 10:
            logger.log(f'[WRN-Baseline-{method_name}] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Clean {clean:.4f} | Test Adv {adv:.4f} | Attack: eps={attack_eps:.3f}, step={attack_step:.3f}, iter={attack_iter}')
        else:
            logger.log(f'[WRN-Baseline-{method_name}] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Clean {clean:.4f} | Test Adv {adv:.4f}')

# ------------------------------ Main -----------------------------------
def main():
    parse = parser_train()

    # Training parameters (same as fusion model)
    parse.add_argument('--epochs-g', type=int, default=120, help="epochs for baseline model")
    parse.add_argument('--ema-decay', type=float, default=0.999, help="EMA decay for baseline model")
    
    # Adversarial training method (same as fusion model)
    parse.add_argument('--use-mart', action='store_true', help='use MART robust loss instead of TRADES')
    parse.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing on natural CE')
    
    # Note: --beta, --attack, --attack-eps, --attack-step, --attack-iter are already defined in parser_train()
    # We'll use appropriate defaults in the code if they are None

    args = parse.parse_args()

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    
    # Add method suffix to log directory
    method_suffix = '_MART' if args.use_mart else '_TRADES'
    LOG_DIR = LOG_DIR + method_suffix + '_baseline'
    
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.log(f'Using device: {DEVICE}')
    logger.log(f'Training mode: baseline 10-class WRN-28-10')
    logger.log(f'Method: {"MART" if args.use_mart else "TRADES"}')
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # ----------------- Dataloaders -----------------
    _, full_train = _build_cifar10(DATA_DIR, train=True,
                                   num_workers=getattr(args, 'workers', 4),
                                   batch_size=args.batch_size)
    _, full_test  = _build_cifar10(DATA_DIR, train=False,
                                   num_workers=getattr(args, 'workers', 4),
                                   batch_size=args.batch_size)

    # ----------------- Train baseline model -----------------
    logger.log(f'Training baseline WRN-28-10 (10-class) for {args.epochs_g} epochs')
    model = build_wrn_28_10(num_classes=10)

    method_name = 'MART' if args.use_mart else 'TRADES'
    logger.log(f'Starting baseline training with {method_name} (WRN-28-10)')
    train_baseline(model, full_train, full_test, args, logger)

    # ----------------- Final Eval & Save -----------------
    atk = make_eval_attack(model, args)
    clean = eval_clean(model, full_test)
    adv   = eval_adv(model, full_test, atk)
    logger.log(f'[WRN-Baseline] Final Test Clean {clean:.4f} | Adv {adv:.4f}')

    # Save model
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(LOG_DIR, 'Baseline_WRN.pt'))
    logger.log(f'Saved baseline model to {LOG_DIR}')


if __name__ == '__main__':
    main()
