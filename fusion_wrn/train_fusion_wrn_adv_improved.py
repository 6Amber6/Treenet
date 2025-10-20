# train_fusion_wrn_adv_improved.py
# Enhanced WRN-28-10 fusion with MART/TRADES switching, auto-save submodels, and flexible training modes

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
from core import animal_classes, vehicle_classes

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


def _filter_indices(ds: torchvision.datasets.CIFAR10, keep: List[int]):
    idx = [i for i, (_, y) in enumerate(ds) if y in keep]
    remap = {old: new for new, old in enumerate(keep)}
    return idx, remap


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, self.remap[int(y)]


def build_filtered_loader(data_dir, keep_labels, batch_size, train, num_workers=4):
    ds, _ = _build_cifar10(data_dir, train=train, num_workers=num_workers, batch_size=batch_size)
    indices, remap = _filter_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return loader

# ------------------------------ Models ----------------------------------
class WRNHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),          # disabled during TRADES
            nn.Linear(512, num_classes),
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)


def build_wrn_28_10(num_classes: int):
    model = wideresnet('wideresnet-28-10', num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class FusionWRN(nn.Module):
    """x -> M1(6c), M2(4c), concat penult -> Head(10c)"""
    def __init__(self, m1: nn.Module, m2: nn.Module, head: WRNHead):
        super().__init__()
        self.m1, self.m2, self.head = m1, m2, head
        self._feats = {}
        self._h1 = self.m1.fc.register_forward_hook(lambda m, inp, out: self._save('m1', inp))
        self._h2 = self.m2.fc.register_forward_hook(lambda m, inp, out: self._save('m2', inp))
    def _save(self, k, inp): self._feats[k] = inp[0]
    def forward(self, x):
        m1_logits = self.m1(x)
        m2_logits = self.m2(x)
        z = torch.cat([self._feats['m1'], self._feats['m2']], dim=1)
        fusion_logits = self.head(z)
        return m1_logits, m2_logits, fusion_logits

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
def set_head_dropout_prob(model: FusionWRN, p: float):
    for module in model.head.modules():
        if isinstance(module, nn.Dropout):
            module.p = p

def freeze_backbone_bn(model: FusionWRN):
    for backbone in (model.m1, model.m2):
        for module in backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

def unfreeze_backbone_bn(model: FusionWRN):
    for backbone in (model.m1, model.m2):
        for module in backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()

def masked_aux_ce(m1_logits: torch.Tensor, m2_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    device = y.device
    loss = torch.tensor(0.0, device=device)
    if y.numel() == 0:
        return loss
    if len(animal_classes) > 0:
        idx1 = torch.tensor(animal_classes, device=device)
        map1 = torch.full((10,), -1, dtype=torch.long, device=device)
        map1[idx1] = torch.arange(len(idx1), device=device)
        mask1 = map1[y] >= 0
        if mask1.any():
            loss = loss + F.cross_entropy(m1_logits[mask1], map1[y[mask1]])
    if len(vehicle_classes) > 0:
        idx2 = torch.tensor(vehicle_classes, device=device)
        map2 = torch.full((10,), -1, dtype=torch.long, device=device)
        map2[idx2] = torch.arange(len(idx2), device=device)
        mask2 = map2[y] >= 0
        if mask2.any():
            loss = loss + F.cross_entropy(m2_logits[mask2], map2[y[mask2]])
    return loss

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
def adv_fusion_step(model: FusionWRN, x_natural, y, optimizer,
                   step_size=2/255, epsilon=8/255, perturb_steps=12,
                   beta=8.0, aux_w=0.02, use_mart=False, label_smoothing=0.0):
    """
    Unified adversarial training step supporting both TRADES and MART
    """
    class LogitsOnly(nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return self.base(x)[-1]

    logits_model = LogitsOnly(model).to(DEVICE)

    # Craft adversarial examples with eval() so BN/dropout frozen
    logits_model.eval()
    with torch.no_grad():
        p_nat = F.softmax(logits_model(x_natural), dim=1)

    # PGD in normalized space
    x_adv = (x_natural.detach() + 1e-3 * torch.randn_like(x_natural)).clamp(-5.0, 5.0)
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = logits_model(x_adv)
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_nat, reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)

    # Training update
    model.train()
    optimizer.zero_grad(set_to_none=True)
    m1_nat, m2_nat, f_nat = model(x_natural)
    _,      _,     f_adv = model(x_adv)
    
    # Natural loss with optional label smoothing
    if label_smoothing > 0.0:
        loss_nat = F.cross_entropy(f_nat, y, label_smoothing=label_smoothing)
    else:
        loss_nat = F.cross_entropy(f_nat, y)
    
    # Robust loss: MART or TRADES
    if use_mart:
        loss_rob = mart_loss(f_adv, f_nat, y)
        loss = loss_nat + loss_rob + masked_aux_ce(m1_nat, m2_nat, y) * aux_w
    else:
        loss_rob = F.kl_div(F.log_softmax(f_adv, dim=1), F.softmax(f_nat.detach(), dim=1), reduction='batchmean')
        loss = loss_nat + beta * loss_rob + masked_aux_ce(m1_nat, m2_nat, y) * aux_w

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
        out = model(x)
        logits = out[-1] if isinstance(out, (tuple, list)) else out
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)

def make_eval_attack(model, args):
    class FusionWrapper(nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return self.base(x)[-1]
    crit = nn.CrossEntropyLoss()
    return create_attack(FusionWrapper(model), crit,
                         getattr(args, 'attack', 'linf-pgd'),
                         getattr(args, 'attack_eps', 8/255),
                         getattr(args, 'attack_iter', 20),   # strong eval: PGD-20
                         getattr(args, 'attack_step', 2/255))

def eval_adv(model, loader, attack) -> float:
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv, _ = attack.perturb(x, y)
        _, _, f_logits = model(x_adv)
        correct += (f_logits.argmax(1) == y).sum().item()
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

def train_fusion(model: FusionWRN, train_loader, test_loader, args, logger):
    params = [
        {'params': model.head.parameters(), 'lr': args.lr * 1.0},   # e.g., 0.1
        {'params': model.m1.parameters(),   'lr': args.lr * 0.2},   # e.g., 0.02
        {'params': model.m2.parameters(),   'lr': args.lr * 0.2},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)
    ema = EMA(model, decay=getattr(args, 'ema_decay', 0.999))

    # CE warmup (dropout ON)
    warmup_epochs = min(10, args.epochs_g)
    for ep in range(1, warmup_epochs + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            _, _, f_logits = model(x)
            loss = F.cross_entropy(f_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            total_loss += loss.item(); num_batches += 1
        sch.step()
        ema.apply_to(model)
        clean = eval_clean(model, test_loader)
        ema.restore(model)
        logger.log(f'[WRN-Fusion-CE] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Clean {clean:.4f}')

    # Disable head dropout for adversarial training, freeze BN stats
    set_head_dropout_prob(model, 0.0)
    freeze_backbone_bn(model)

    atk_eval = make_eval_attack(model, args)
    unfroze_bn = False
    method_name = 'MART' if getattr(args, 'use_mart', False) else 'TRADES'
    
    for ep in range(warmup_epochs + 1, args.epochs_g + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = adv_fusion_step(model, x, y, optimizer=opt,
                                 step_size=getattr(args, 'attack_step', 2/255),
                                 epsilon=getattr(args, 'attack_eps', 8/255),
                                 perturb_steps=getattr(args, 'attack_iter', 12),
                                 beta=getattr(args, 'beta', 8.0),
                                 aux_w=getattr(args, 'aux_w', 0.02),
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

        logger.log(f'[WRN-Fusion-{method_name}] Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} | Test Clean {clean:.4f} | Test Adv {adv:.4f}')

# --------------------------- Model Loading/Saving ------------------------
def save_submodels(m1, m2, save_dir, logger):
    """Save submodels M1 and M2"""
    m1_path = os.path.join(save_dir, 'M1_WRN.pt')
    m2_path = os.path.join(save_dir, 'M2_WRN.pt')
    
    torch.save({'model_state_dict': m1.state_dict()}, m1_path)
    torch.save({'model_state_dict': m2.state_dict()}, m2_path)
    
    logger.log(f'Saved M1 to {m1_path}')
    logger.log(f'Saved M2 to {m2_path}')

def load_submodels(save_dir, logger):
    """Load submodels M1 and M2"""
    m1_path = os.path.join(save_dir, 'M1_WRN.pt')
    m2_path = os.path.join(save_dir, 'M2_WRN.pt')
    
    if not os.path.exists(m1_path) or not os.path.exists(m2_path):
        raise FileNotFoundError(f"Submodel files not found in {save_dir}")
    
    m1 = build_wrn_28_10(num_classes=len(animal_classes))
    m2 = build_wrn_28_10(num_classes=len(vehicle_classes))
    
    m1.load_state_dict(torch.load(m1_path, map_location=DEVICE)['model_state_dict'])
    m2.load_state_dict(torch.load(m2_path, map_location=DEVICE)['model_state_dict'])
    
    logger.log(f'Loaded M1 from {m1_path}')
    logger.log(f'Loaded M2 from {m2_path}')
    
    return m1, m2

# ------------------------------ Main -----------------------------------
def main():
    parse = parser_train()

    # Training mode control
    parse.add_argument('--train-mode', type=str, choices=['all', 'submodels', 'fusion'], default='all',
                       help='Training mode: all=full pipeline, submodels=only M1/M2, fusion=only fusion')
    parse.add_argument('--submodel-dir', type=str, default=None,
                       help='Directory containing saved submodels (required for fusion-only mode)')
    
    # Training parameters
    parse.add_argument('--epochs-m', type=int, default=100, help="epochs for submodels M1/M2")
    parse.add_argument('--epochs-g', type=int, default=120, help="epochs for fusion model")
    parse.add_argument('--lr-m', type=float, default=0.1, help="learning rate for submodels")
    parse.add_argument('--aux_w', type=float, default=0.02, help="weight for auxiliary CE loss")
    parse.add_argument('--ema-decay', type=float, default=0.999, help="EMA decay for fusion model")
    
    # Adversarial training method
    parse.add_argument('--use-mart', action='store_true', help='use MART robust loss instead of TRADES')
    parse.add_argument('--beta', type=float, default=8.0, help='TRADES beta (ignored if MART)')
    parse.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing on natural CE')
    
    # Attack parameters
    parse.add_argument('--attack', type=str, default='linf-pgd', help='attack type for evaluation')
    parse.add_argument('--attack-eps', type=float, default=8/255, help='attack epsilon')
    parse.add_argument('--attack-step', type=float, default=2/255, help='attack step size')
    parse.add_argument('--attack-iter', type=int, default=12, help='attack iterations')

    args = parse.parse_args()

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    
    # Add method suffix to log directory
    method_suffix = '_MART' if args.use_mart else '_TRADES'
    LOG_DIR = LOG_DIR + method_suffix
    
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.log(f'Using device: {DEVICE}')
    logger.log(f'Training mode: {args.train_mode}')
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

    m1_train = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=True)
    m1_test  = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=False)
    m2_train = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=True)
    m2_test  = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=False)

    # ----------------- Train submodels (if needed) -----------------
    if args.train_mode in ['all', 'submodels']:
        logger.log(f'Training M1 (WRN-28-10, 6-class) for {args.epochs_m} epochs (CE)')
        m1 = build_wrn_28_10(num_classes=len(animal_classes))
        train_ce(m1, m1_train, m1_test, args.epochs_m, args.lr_m, logger, '[M1]')

        logger.log(f'Training M2 (WRN-28-10, 4-class) for {args.epochs_m} epochs (CE)')
        m2 = build_wrn_28_10(num_classes=len(vehicle_classes))
        train_ce(m2, m2_train, m2_test, args.epochs_m, args.lr_m, logger, '[M2]')

        a_acc = eval_clean(m1, m1_test)
        v_acc = eval_clean(m2, m2_test)
        logger.log(f'[M1] Clean Test Acc: {a_acc:.4f}')
        logger.log(f'[M2] Clean Test Acc: {v_acc:.4f}')
        
        # Auto-save submodels
        save_submodels(m1, m2, LOG_DIR, logger)
        
        if args.train_mode == 'submodels':
            logger.log('Submodel training completed. Exiting.')
            return
    else:
        # Load pre-trained submodels
        submodel_dir = args.submodel_dir or LOG_DIR
        logger.log(f'Loading submodels from {submodel_dir}')
        m1, m2 = load_submodels(submodel_dir, logger)

    # ----------------- Fusion training -----------------
    in_dim = int(m1.fc.in_features + m2.fc.in_features)
    head = WRNHead(in_dim=in_dim, num_classes=10, p_drop=0.2).to(DEVICE)
    fusion = FusionWRN(m1, m2, head).to(DEVICE)

    method_name = 'MART' if args.use_mart else 'TRADES'
    logger.log(f'Starting fusion training with {method_name} (WRN-28-10 backbones)')
    train_fusion(fusion, full_train, full_test, args, logger)

    # ----------------- Final Eval & Save -----------------
    atk = make_eval_attack(fusion, args)
    clean = eval_clean(fusion, full_test)
    adv   = eval_adv(fusion, full_test, atk)
    logger.log(f'[WRN-Fusion] Final Test Clean {clean:.4f} | Adv {adv:.4f}')

    # Save all models
    torch.save({'model_state_dict': m1.state_dict()},   os.path.join(LOG_DIR, 'M1_WRN.pt'))
    torch.save({'model_state_dict': m2.state_dict()},   os.path.join(LOG_DIR, 'M2_WRN.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'Head_WRN.pt'))
    torch.save({'model_state_dict': fusion.state_dict()}, os.path.join(LOG_DIR, 'Fusion_WRN.pt'))
    logger.log(f'Saved all models to {LOG_DIR}')


if __name__ == '__main__':
    main()
