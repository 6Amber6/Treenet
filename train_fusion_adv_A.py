# train_fusion_adv_A.py
"""
Two-Stage CE (+ Adversarial Training with partial-unfreeze, TRADES/MART)

Pipeline
  1) Train M1 (6-class animal) with CE (clean only)
  2) Train M2 (4-class vehicle) with CE (clean only)
  3) Build Fusion model:
       x -> [penult(M1(x)) || penult(M2(x))] -> HeadG -> logits (10-class)
     - Freeze all of M1/M2 EXCEPT the last residual stage (partial unfreeze)
     - Train {HeadG + last_stage(M1) + last_stage(M2)} with TRADES/MART
  4) Report clean & adversarial accuracy of fusion model
"""

import os
import sys
import json
import shutil
import argparse
from typing import List

import torch
import torch.nn as nn
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
from core.utils.trades import trades_loss
from core.utils.mart import mart_loss
from core import animal_classes, vehicle_classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# Models
# ------------------------------------------------
def build_lightresnet20(num_classes: int) -> LightResnet:
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
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


class FusionHead(nn.Module):
    """
    x -> penult(M1(x)) || penult(M2(x)) -> HeadG -> logits
    Note: do NOT wrap forward in no_grad so adversarial attacks can backprop to pixels.
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.head = head

        # Forward hooks to capture the input of the final fc (penultimate features)
        self._feats = {}
        self._h1 = self.m1.fc.register_forward_hook(lambda m, inp, out: self._save_feat("m1", inp))
        self._h2 = self.m2.fc.register_forward_hook(lambda m, inp, out: self._save_feat("m2", inp))

    def _save_feat(self, key, inp_tuple):
        # inp_tuple[0] shape: [B, D] = features before final fc
        self._feats[key] = inp_tuple[0]

    def forward(self, x):
        _ = self.m1(x)
        _ = self.m2(x)
        z = torch.cat([self._feats["m1"], self._feats["m2"]], dim=1)
        return self.head(z)

    def remove_hooks(self):
        self._h1.remove()
        self._h2.remove()


# ------------------------------------------------
# Data
# ------------------------------------------------
def _build_cifar10(data_dir, train: bool, num_workers=4, batch_size=128):
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
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return loader


# ------------------------------------------------
# Train & Eval
# ------------------------------------------------
def train_clean_classifier(model, train_loader, test_loader, epochs, lr, logger, tag):
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
    """Use unified evaluation attack. Defaults (from args): attack type/eps/step/iter come from parser_train."""
    crit = nn.CrossEntropyLoss()
    attack = getattr(args, 'attack', 'linf-pgd')
    eps = getattr(args, 'attack_eps', 8/255)
    step = getattr(args, 'attack_step', 2/255)
    iters = getattr(args, 'attack_iter', 10)
    return create_attack(model, crit, attack, eps, iters, step)


def eval_adv(model, loader, attack) -> float:
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # Do NOT wrap in no_grad: attack needs gradients wrt x internally
        x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            logits = model(x_adv)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


# -------------------- Scheme A helpers --------------------
def get_last_stage(model: nn.Module) -> nn.Module:
    """Return the last residual stage. Tries layer4 -> layer3 -> layer2."""
    for name in ['layer4', 'layer3', 'layer2']:
        m = getattr(model, name, None)
        if isinstance(m, nn.Module):
            return m
    raise AttributeError("Cannot find last residual stage (layer4/layer3/layer2) in LightResnet")


def set_partial_unfreeze_and_modes(m1: nn.Module, m2: nn.Module):
    """Freeze all params, then unfreeze only the last residual stage and set its mode to train()."""
    for p in m1.parameters(): p.requires_grad = False
    for p in m2.parameters(): p.requires_grad = False
    m1.eval(); m2.eval()

    last1 = get_last_stage(m1)
    last2 = get_last_stage(m2)
    for p in last1.parameters(): p.requires_grad = True
    for p in last2.parameters(): p.requires_grad = True
    last1.train()
    last2.train()
    return last1, last2


@torch.no_grad()
def infer_penult_dim(m1: nn.Module, m2: nn.Module, sample_loader: DataLoader) -> int:
    """Infer penultimate concat feature dimension from a sample batch."""
    feats = {}

    def hk1(m, inp, out): feats['m1'] = inp[0]
    def hk2(m, inp, out): feats['m2'] = inp[0]

    h1 = m1.fc.register_forward_hook(hk1)
    h2 = m2.fc.register_forward_hook(hk2)

    x, _ = next(iter(sample_loader))
    x = x.to(DEVICE)
    _ = m1(x)
    _ = m2(x)
    h1.remove(); h2.remove()

    assert 'm1' in feats and 'm2' in feats, "Failed to capture penultimate features"
    d = feats['m1'].shape[1] + feats['m2'].shape[1]
    return int(d)


def build_optimizer_for_A(fusion: FusionHead, lr_head=1e-3, lr_stage=5e-4, weight_decay=5e-4):
    """Optimizer for Scheme A: head + last stage of each submodel."""
    last1 = get_last_stage(fusion.m1)
    last2 = get_last_stage(fusion.m2)
    params = [
        {'params': fusion.head.parameters(), 'lr': lr_head},
        {'params': last1.parameters(), 'lr': lr_stage},
        {'params': last2.parameters(), 'lr': lr_stage},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 45], gamma=0.1)
    return opt, sch


def train_fusion_adversarial_A(fusion: FusionHead,
                               train_loader, test_loader,
                               args, logger):
    """Train head + last stages with TRADES/MART (or CE for warmup)."""
    lr_head = min(args.lr, 1e-3)
    lr_stage = min(args.lr * 0.5, 5e-4)
    opt, sch = build_optimizer_for_A(fusion, lr_head=lr_head, lr_stage=lr_stage, weight_decay=5e-4)

    ce = nn.CrossEntropyLoss()
    eval_atk = make_eval_attack(fusion, args)

    for ep in range(1, args.epochs_g + 1):
        fusion.train()
        # ensure only the last stages are in train() mode
        get_last_stage(fusion.m1).train()
        get_last_stage(fusion.m2).train()

        seen, correct, loss_sum = 0, 0, 0.0
        did_update = False

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                # EXPLICIT update: zero_grad -> loss.backward -> step
                opt.zero_grad(set_to_none=True)
                loss_val = trades_loss(
                    fusion, x, y,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_val[0] if isinstance(loss_val, (tuple, list)) else loss_val
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(get_last_stage(fusion.m1).parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(get_last_stage(fusion.m2).parameters(), 5.0)
                opt.step()
                did_update = True

            elif args.trainer == 'mart':
                opt.zero_grad(set_to_none=True)
                loss_val = mart_loss(
                    fusion, x, y,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_val[0] if isinstance(loss_val, (tuple, list)) else loss_val
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(get_last_stage(fusion.m1).parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(get_last_stage(fusion.m2).parameters(), 5.0)
                opt.step()
                did_update = True

            else:
                # CE warmup branch (already correct)
                opt.zero_grad(set_to_none=True)
                logits = fusion(x)
                loss_main = ce(logits, y)
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(get_last_stage(fusion.m1).parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(get_last_stage(fusion.m2).parameters(), 5.0)
                opt.step()
                did_update = True

            loss_sum += float(loss_main.detach().item()) * x.size(0)

            with torch.no_grad():
                preds = fusion(x).argmax(1)
                correct += (preds == y).sum().item()
                seen += y.size(0)

        # step LR only if we actually updated this epoch
        if did_update:
            sch.step()

        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(fusion, test_loader)
            adv_acc = eval_adv(fusion, test_loader, eval_atk)
            logger.log(f'[Fusion-{args.trainer}-A] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # Reuse repo's parser_train (already defines --attack, --attack_eps, --attack_step, --attack_iter, etc.)
    parse = parser_train()
    # Add only NEW flags (avoid duplicates)
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 clean training')
    parse.add_argument('--epochs-g', type=int, default=50, help='epochs for fusion adversarial training')
    parse.add_argument('--lr-m', type=float, default=0.1, help='LR for M1/M2 clean training')
    parse.add_argument('--trainer', type=str, default='trades', choices=['trades', 'mart', 'ce'],
                       help='objective for fusion stage (trades/mart/ce)')

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

    # Filtered loaders
    m1_train_loader = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=True,
                                            num_workers=getattr(args, 'workers', 4))
    m1_test_loader = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=False,
                                           num_workers=getattr(args, 'workers', 4))
    m2_train_loader = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=True,
                                            num_workers=getattr(args, 'workers', 4))
    m2_test_loader = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=False,
                                           num_workers=getattr(args, 'workers', 4))

    # Stage 1: clean training for submodels
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

    # Optional: extra fine-tuning if accuracy is low
    if m1_acc < 0.80:
        logger.log(f"M1 acc {m1_acc:.4f} too low, +20 epochs fine-tune...")
        train_clean_classifier(m1, m1_train_loader, m1_test_loader, 20, args.lr_m * 0.1, logger, '[M1-extra]')
        m1_acc = eval_clean(m1, m1_test_loader)
        logger.log(f'[M1] Updated Clean Test Acc: {m1_acc:.4f}')

    if m2_acc < 0.85:
        logger.log(f"M2 acc {m2_acc:.4f} too low, +20 epochs fine-tune...")
        train_clean_classifier(m2, m2_train_loader, m2_test_loader, 20, args.lr_m * 0.1, logger, '[M2-extra]')
        m2_acc = eval_clean(m2, m2_test_loader)
        logger.log(f'[M2] Updated Clean Test Acc: {m2_acc:.4f}')

    # Stage 2: Fusion + Scheme A adversarial training
    # Infer penultimate concat dimension and build head
    penult_dim = infer_penult_dim(m1, m2, full_train_loader)
    logger.log(f'Inferred penultimate concat dim: {penult_dim}')
    head = HeadG(in_dim=penult_dim, num_classes=10).to(DEVICE)

    # Build fusion model
    fusion = FusionHead(m1, m2, head).to(DEVICE)

    # Partial unfreeze (only last residual stage from each backbone)
    last1, last2 = set_partial_unfreeze_and_modes(fusion.m1, fusion.m2)
    logger.log(f'Partial-unfreeze last stages: '
               f'M1[{last1.__class__.__name__}], M2[{last2.__class__.__name__}]')

    # CE warmup (head + last stages)
    logger.log("Warmup (CE) for fusion: 5 epochs (head + last stages)...")
    warm_args = argparse.Namespace(**vars(args))
    warm_args.trainer = 'ce'
    warm_args.epochs_g = 5
    train_fusion_adversarial_A(fusion, full_train_loader, full_test_loader, warm_args, logger)

    # Adversarial training with TRADES/MART
    logger.log(f"Adversarial training ({args.trainer}) with eps={args.attack_eps}, "
               f"step={args.attack_step}, iters={args.attack_iter} for {args.epochs_g} epochs...")
    train_fusion_adversarial_A(fusion, full_train_loader, full_test_loader, args, logger)

    clean_g = eval_clean(fusion, full_test_loader)
    adv_g = eval_adv(fusion, full_test_loader, make_eval_attack(fusion, args))
    logger.log(f'[Fusion-A] Final Test Clean: {clean_g:.4f} | Final Test Adv: {adv_g:.4f}')

    # Save checkpoints
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(LOG_DIR, 'M1_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(LOG_DIR, 'M2_4cls.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'model_state_dict': fusion.state_dict()}, os.path.join(LOG_DIR, 'Fusion_partial_unfreeze.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
