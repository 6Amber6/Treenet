import os
import sys
import json
import shutil
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# Paths and imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
ARP_ROOT = os.path.join(REPO_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.models.wideresnet import wideresnet
from core.utils import Logger, parser_train, seed
from core.attacks import create_attack
from core.utils.mart import mart_loss
from core import animal_classes, vehicle_classes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------ Data ---------------------------------
def _build_cifar10(data_dir, train: bool, num_workers=4, batch_size=128):
    tfm = (T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]) if train else T.Compose([T.ToTensor()]))
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


# ------------------------------ Models ---------------------------------
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
    def forward(self, x):
        return self.net(x)


def build_wrn_28_10(num_classes: int):
    # depth=28, widen=10
    model = wideresnet('wideresnet-28-10', num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class FusionWRN(nn.Module):
    """
    Fusion wrapper: x -> M1(6c), M2(4c), concat penult (before fc) -> Head(10c)
    """
    def __init__(self, m1: nn.Module, m2: nn.Module, head: WRNHead):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.head = head
        self._feats = {}
        # Wideresnet penultimate feature is input to fc
        self._h1 = self.m1.fc.register_forward_hook(lambda m, inp, out: self._save('m1', inp))
        self._h2 = self.m2.fc.register_forward_hook(lambda m, inp, out: self._save('m2', inp))
    def _save(self, k, inp):
        self._feats[k] = inp[0]
    def forward(self, x):
        m1_logits = self.m1(x)
        m2_logits = self.m2(x)
        z = torch.cat([self._feats['m1'], self._feats['m2']], dim=1)
        fusion_logits = self.head(z)
        return m1_logits, m2_logits, fusion_logits


# --------------------------- Train & Eval ------------------------------
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
    crit = nn.CrossEntropyLoss()
    eps = getattr(args, 'attack_eps', 8/255)
    step = getattr(args, 'attack_step', 2/255)
    iters = getattr(args, 'attack_iter', 10)
    attack_name = getattr(args, 'attack', 'linf-pgd')

    class FusionWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return self.base(x)[-1]
    return create_attack(FusionWrapper(model), crit, attack_name, eps, iters, step)


def masked_aux_ce(m1_logits: torch.Tensor, m2_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, device=y.device)
    if y.numel() == 0:
        return loss
    if len(animal_classes) > 0:
        mask1 = torch.isin(y, torch.tensor(animal_classes, device=y.device))
        if mask1.any():
            y1 = torch.stack([torch.tensor(animal_classes.index(int(t.item())), device=y.device) for t in y[mask1]])
            loss = loss + F.cross_entropy(m1_logits[mask1], y1)
    if len(vehicle_classes) > 0:
        mask2 = torch.isin(y, torch.tensor(vehicle_classes, device=y.device))
        if mask2.any():
            y2 = torch.stack([torch.tensor(vehicle_classes.index(int(t.item())), device=y.device) for t in y[mask2]])
            loss = loss + F.cross_entropy(m2_logits[mask2], y2)
    return loss


def trades_fusion_step(model: FusionWRN, x_natural, y, optimizer,
                       step_size=0.00784314, epsilon=0.03137255, perturb_steps=12,
                       beta=6.0, aux_w=0.05):
    class LogitsOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return self.base(x)[-1]

    logits_model = LogitsOnly(model)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = x_natural.size(0)

    # generate adversarial example (freeze BN stats during attack)
    logits_model.eval()
    x_adv = (x_natural.detach() + 0.001 * torch.randn_like(x_natural)).clamp(0, 1)
    with torch.no_grad():
        p_nat = F.softmax(logits_model(x_natural), dim=1)
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = logits_model(x_adv)
        loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1), p_nat)
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = (x_adv.detach() + step_size * torch.sign(grad)).clamp(0, 1)
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)

    # training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    m1_nat, m2_nat, f_nat = model(x_natural)
    m1_adv, m2_adv, f_adv = model(x_adv)

    loss_nat = F.cross_entropy(f_nat, y)
    loss_rob = (1.0 / batch_size) * criterion_kl(F.log_softmax(f_adv, dim=1), F.softmax(f_nat.detach(), dim=1))
    loss_aux = masked_aux_ce(m1_nat, m2_nat, y) * aux_w

    loss = loss_nat + beta * loss_rob + loss_aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss


def train_ce(model, train_loader, test_loader, epochs, lr, logger, tag):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[epochs // 2, int(epochs * 0.75)], gamma=0.1)
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
        if ep % 5 == 0 or ep == 1:
            acc = eval_clean(model, test_loader)
            logger.log(f'{tag} Epoch {ep:03d} | Test Acc {acc:.4f}')


def train_fusion(model: FusionWRN, train_loader, test_loader, args, logger):
    # Staged LRs: Head LR high; M1/M2 lower
    params = [
        {'params': model.head.parameters(), 'lr': args.lr},
        {'params': model.m1.parameters(), 'lr': args.lr * 0.1},
        {'params': model.m2.parameters(), 'lr': args.lr * 0.1},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_g, eta_min=1e-6)

    # CE warmup 5 epochs
    for ep in range(1, min(5, args.epochs_g) + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            _, _, f_logits = model(x)
            loss = F.cross_entropy(f_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
        clean = eval_clean(model, test_loader)
        logger.log(f'[WRN-Fusion-CE] Epoch {ep:03d} | Test Clean {clean:.4f}')

    # TRADES
    atk = make_eval_attack(model, args)
    for ep in range(6, args.epochs_g + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _ = trades_fusion_step(model, x, y, optimizer=opt,
                                   step_size=getattr(args, 'attack_step', 2/255),
                                   epsilon=getattr(args, 'attack_eps', 8/255),
                                   perturb_steps=getattr(args, 'attack_iter', 12),
                                   beta=args.beta, aux_w=0.05)
        sch.step()
        clean = eval_clean(model, test_loader)
        adv = eval_adv(model, test_loader, atk)
        logger.log(f'[WRN-Fusion-TRADES] Epoch {ep:03d} | Test Clean {clean:.4f} | Test Adv {adv:.4f}')


def eval_adv(model, loader, attack) -> float:
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            _, _, f_logits = model(x_adv)
        correct += (f_logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


# ------------------------------ Main ----------------------------------
def main():
    parse = parser_train()
    parse.add_argument('--epochs-m', type=int, default=100)
    parse.add_argument('--epochs-g', type=int, default=100)
    parse.add_argument('--lr-m', type=float, default=0.1)
    parse.add_argument('--trainer', type=str, default='trades', choices=['trades'])
    args = parse.parse_args()

    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.log(f'Using device: {DEVICE}')
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Loaders
    _, full_train = _build_cifar10(DATA_DIR, train=True, num_workers=getattr(args, 'workers', 4), batch_size=args.batch_size)
    _, full_test = _build_cifar10(DATA_DIR, train=False, num_workers=getattr(args, 'workers', 4), batch_size=args.batch_size)
    m1_train = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=True, num_workers=getattr(args, 'workers', 4))
    m1_test = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=False, num_workers=getattr(args, 'workers', 4))
    m2_train = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=True, num_workers=getattr(args, 'workers', 4))
    m2_test = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=False, num_workers=getattr(args, 'workers', 4))

    # Stage 1: train submodels (WRN-28-10)
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

    # Stage 2: fusion
    in_dim = int(m1.fc.in_features + m2.fc.in_features)
    head = WRNHead(in_dim=in_dim, num_classes=10, p_drop=0.2).to(DEVICE)
    fusion = FusionWRN(m1, m2, head).to(DEVICE)

    logger.log('Starting fusion training with TRADES (WRN-28-10 backbones)')
    train_fusion(fusion, full_train, full_test, args, logger)

    # Final eval
    atk = make_eval_attack(fusion, args)
    clean = eval_clean(fusion, full_test)
    adv = eval_adv(fusion, full_test, atk)
    logger.log(f'[WRN-Fusion] Final Test Clean {clean:.4f} | Adv {adv:.4f}')

    # Save
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(LOG_DIR, 'M1_WRN.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(LOG_DIR, 'M2_WRN.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'Head_WRN.pt'))
    torch.save({'model_state_dict': fusion.state_dict()}, os.path.join(LOG_DIR, 'Fusion_WRN.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
