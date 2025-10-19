# fusion_trades_robust_mart.py
# CIFAR-10 WRN-28-10 fusion + TRADES/MART with stronger PGD and normalized-space eps/step
# Default robust setup: PGD-20, step=0.01, eps=8/255; EMA decay=0.9995
import os
import sys
import json
import shutil
import argparse
from typing import List, Tuple, Optional

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
from core import animal_classes, vehicle_classes

# Try to import MART loss from repo; fall back to a compatible implementation
def _import_mart():
    try:
        from core.losses import mart_loss  # preferred if present
        return mart_loss
    except Exception:
        pass
    try:
        from adversarial_robustness_pytorch.core.losses import mart_loss
        return mart_loss
    except Exception:
        pass

    # Fallback: a simple MART-style loss (miscls-weighted CE + KL with hardening)
    # Note: This fallback is a reasonable approximation if the repo variant differs slightly.
    def mart_loss(logits_adv: torch.Tensor, logits_nat: torch.Tensor, y: torch.Tensor, lam: float = 5.0):
        # Cross-entropy on adversarial with misclassification awareness
        ce_adv = F.cross_entropy(logits_adv, y, reduction='none')
        # Confidence difference to penalize overconfident wrong classes
        prob_nat = F.softmax(logits_nat, dim=1)
        top2_values, top2_idx = prob_nat.topk(2, dim=1)
        y_hat = torch.where(top2_idx[:, 0] == y, top2_idx[:, 1], top2_idx[:, 0])
        # large margin between true and second-best is encouraged on natural
        margin = F.log_softmax(logits_nat, dim=1)[torch.arange(y.size(0), device=y.device), y] - \
                 F.log_softmax(logits_nat, dim=1)[torch.arange(y.size(0), device=y.device), y_hat]
        loss_miscls = lam * F.relu(1.0 - margin)  # hinge on margin
        # KL between adv and nat (stop grad on nat)
        kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_nat.detach(), dim=1),
            reduction='batchmean'
        )
        return (ce_adv + loss_miscls).mean() + kl
    return mart_loss

mart_loss = _import_mart()

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
            nn.Dropout(p_drop),
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
        # robust hook: support fc or linear as final layer name
        last1 = getattr(self.m1, 'fc', getattr(self.m1, 'linear', None))
        last2 = getattr(self.m2, 'fc', getattr(self.m2, 'linear', None))
        assert last1 is not None and last2 is not None, "Cannot find final FC layer on WRN (fc/linear)."
        self._h1 = last1.register_forward_hook(lambda m, inp, out: self._save('m1', inp))
        self._h2 = last2.register_forward_hook(lambda m, inp, out: self._save('m2', inp))
    def _save(self, k, inp): self._feats[k] = inp[0]
    def forward(self, x):
        m1_logits = self.m1(x)
        m2_logits = self.m2(x)
        z = torch.cat([self._feats['m1'], self._feats['m2']], dim=1)
        fusion_logits = self.head(z)
        return m1_logits, m2_logits, fusion_logits

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

# --------------------------- PGD helper (normalized space) -------------
def _pixel_to_normed_step_and_eps(step_pix: float, eps_pix: float, device: torch.device):
    """
    Convert pixel-space step/eps to normalized-space tensors by dividing by std per-channel.
    """
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    step = torch.tensor(step_pix, dtype=torch.float32, device=device).view(1, 1, 1, 1) / std
    eps = torch.tensor(eps_pix, dtype=torch.float32, device=device).view(1, 1, 1, 1) / std
    return step, eps

# --------------------------- TRADES/MART Step ---------------------------
def adv_fusion_step(model: FusionWRN, x_natural, y, optimizer,
                    step_size_pix=2/255, epsilon_pix=8/255, perturb_steps=12,
                    beta=8.0, aux_w=0.02, use_mart: bool=False, label_smoothing: float=0.0):
    """
    One adversarial training step for FusionWRN. If use_mart=True, swap TRADES robust term with MART loss.
    """
    class LogitsOnly(nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x): return self.base(x)[-1]

    logits_model = LogitsOnly(model).to(DEVICE)

    # Prepare normalized-space step/eps
    step_t, eps_t = _pixel_to_normed_step_and_eps(step_size_pix, epsilon_pix, DEVICE)

    # Craft with eval() so BN/dropout frozen for stability
    logits_model.eval()
    with torch.no_grad():
        p_nat = F.softmax(logits_model(x_natural), dim=1)

    # PGD in normalized space with random start
    x_adv = (x_natural.detach() + 1e-3 * torch.randn_like(x_natural)).clamp(-5.0, 5.0)
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = logits_model(x_adv)
        # TRADES inner loss: KL(adv || nat)
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_nat, reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + step_t * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x_natural + eps_t), x_natural - eps_t)

    # Update
    model.train()
    optimizer.zero_grad(set_to_none=True)

    m1_nat, m2_nat, f_nat = model(x_natural)
    _,      _,     f_adv = model(x_adv)

    # natural CE (optionally with label smoothing)
    if label_smoothing and label_smoothing > 0.0:
        loss_nat = F.cross_entropy(f_nat, y, label_smoothing=label_smoothing)
    else:
        loss_nat = F.cross_entropy(f_nat, y)

    if use_mart:
        # MART robust term replaces TRADES KL, typically includes CE on adv with miscls weighting
        loss_rob = mart_loss(f_adv, f_nat, y)
        loss = loss_nat + loss_rob + masked_aux_ce(m1_nat, m2_nat, y) * aux_w
    else:
        # TRADES robust term
        loss_rob = F.kl_div(F.log_softmax(f_adv, dim=1), F.softmax(f_nat.detach(), dim=1), reduction='batchmean')
        loss = loss_nat + beta * loss_rob + masked_aux_ce(m1_nat, m2_nat, y) * aux_w

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
    return create_attack(
        FusionWrapper(model), crit,
        getattr(args, 'attack', 'linf-pgd'),
        getattr(args, 'attack_eps', 8/255),
        getattr(args, 'attack_iter', 20),   # strong eval: PGD-20
        getattr(args, 'attack_step', 0.01)
    )

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
def train_ce(model, train_loader, test_loader, epochs, lr, logger, tag, ema: Optional[EMA]=None):
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
        {'params': model.head.parameters(), 'lr': args.lr * 1.0},
        {'params': model.m1.parameters(),   'lr': args.lr * 0.2},
        {'params': model.m2.parameters(),   'lr': args.lr * 0.2},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)
    ema = EMA(model, decay=args.ema_decay)

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

    # Disable head dropout for TRADES/MART, freeze BN stats, stronger attack
    set_head_dropout_prob(model, 0.0)
    freeze_backbone_bn(model)

    atk_eval = make_eval_attack(model, args)
    unfroze_bn = False
    for ep in range(warmup_epochs + 1, args.epochs_g + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = adv_fusion_step(
                model, x, y, optimizer=opt,
                step_size_pix=getattr(args, 'attack_step', 0.01),
                epsilon_pix=getattr(args, 'attack_eps', 8/255),
                perturb_steps=getattr(args, 'attack_iter', 20),
                beta=getattr(args, 'beta', 8.0),
                aux_w=getattr(args, 'aux_w', 0.02),
                use_mart=getattr(args, 'use_mart', False),
                label_smoothing=getattr(args, 'label_smoothing', 0.0),
            )
            ema.update(model)
            total_loss += float(loss); num_batches += 1
        sch.step()

        # Unfreeze BN after 40 TRADES/MART epochs to let stats adapt
        if not unfroze_bn and (ep - warmup_epochs) >= 40:
            unfreeze_backbone_bn(model)
            unfroze_bn = True

        # EMA eval
        ema.apply_to(model)
        clean = eval_clean(model, test_loader)
        adv   = eval_adv(model, test_loader, atk_eval)
        ema.restore(model)

        logger.log(f'[WRN-Fusion-{"MART" if args.use_mart else "TRADES"}] '
                   f'Epoch {ep:03d} | Train Loss {total_loss/max(num_batches,1):.4f} '
                   f'| Test Clean {clean:.4f} | Test Adv {adv:.4f}')

# ------------------------------ Main -----------------------------------
def main():
    parse = parser_train()

    # training lengths & lrs
    parse.add_argument('--epochs-m', type=int, default=100, help="epochs for submodels M1/M2 (CE)")
    parse.add_argument('--epochs-g', type=int, default=120, help="epochs for fusion model (TRADES/MART)")
    parse.add_argument('--lr-m', type=float, default=0.1, help="learning rate for submodels (CE)")
    parse.add_argument('--aux_w', type=float, default=0.02, help="weight for auxiliary CE loss")
    parse.add_argument('--ema-decay', type=float, default=0.9995, help="EMA decay for fusion model")


    # TRADES / MART choices
    parse.add_argument('--beta', type=float, default=8.0, help='TRADES beta (ignored if MART)')
    parse.add_argument('--use-mart', action='store_true', help='use MART robust loss instead of TRADES')
    parse.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing on natural CE')

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

    # ----------------- Train submodels -----------------
    logger.log(f'Training M1 (WRN-28-10, 6-class) for {args.epochs_m} epochs (CE)')
    m1 = build_wrn_28_10(num_classes=len(animal_classes))
    train_ce(m1, m1_train, m1_test, args.epochs_m, args.lr_m, logger, '[M1]', ema=None)

    logger.log(f'Training M2 (WRN-28-10, 4-class) for {args.epochs_m} epochs (CE)')
    m2 = build_wrn_28_10(num_classes=len(vehicle_classes))
    train_ce(m2, m2_train, m2_test, args.epochs_m, args.lr_m, logger, '[M2]', ema=None)

    a_acc = eval_clean(m1, m1_test)
    v_acc = eval_clean(m2, m2_test)
    logger.log(f'[M1] Clean Test Acc: {a_acc:.4f}')
    logger.log(f'[M2] Clean Test Acc: {v_acc:.4f}')

    # ----------------- Fusion training -----------------
    in_dim = int(m1.fc.in_features if hasattr(m1, 'fc') else m1.linear.in_features) + \
             int(m2.fc.in_features if hasattr(m2, 'fc') else m2.linear.in_features)
    head = WRNHead(in_dim=in_dim, num_classes=10, p_drop=0.2).to(DEVICE)
    fusion = FusionWRN(m1, m2, head).to(DEVICE)

    logger.log('Starting fusion training with {} (WRN-28-10 backbones)'.format('MART' if args.use_mart else 'TRADES'))
    train_fusion(fusion, full_train, full_test, args, logger)

    # ----------------- Final Eval & Save -----------------
    atk = make_eval_attack(fusion, args)
    clean = eval_clean(fusion, full_test)
    adv   = eval_adv(fusion, full_test, atk)
    logger.log(f'[WRN-Fusion] Final Test Clean {clean:.4f} | Adv {adv:.4f}')

    torch.save({'model_state_dict': m1.state_dict()},      os.path.join(LOG_DIR, 'M1_WRN.pt'))
    torch.save({'model_state_dict': m2.state_dict()},      os.path.join(LOG_DIR, 'M2_WRN.pt'))
    torch.save({'model_state_dict': head.state_dict()},    os.path.join(LOG_DIR, 'Head_WRN.pt'))
    torch.save({'model_state_dict': fusion.state_dict()},  os.path.join(LOG_DIR, 'Fusion_WRN.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
