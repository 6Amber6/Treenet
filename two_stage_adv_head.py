"""
Two-Stage CE (+ Adversarial Training on G with existing TRADES/MART)

Pipeline
  1) Train M1 (6-class animal) with CE (clean only)
  2) Train M2 (4-class vehicle) with CE (clean only)
  3) Build a FusionHead model: x -> [penult(M1(x)) || penult(M2(x))] -> G(x)
     - Freeze M1/M2 parameters
     - Train ONLY G with min-max objective using EXISTING losses:
       * TRADES (default) or MART (select by --trainer)
  4) Report clean & (optionally) adversarial accuracy of G

Notes
- We DO NOT wrap adversarial example creation in torch.no_grad.
- We keep M1/M2 params requires_grad=False, but forward through them so
  gradients wrt x still flow for the attack.
"""

import os
import sys
import json
import shutil
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

# ---------- Plan B: import local 'core' ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

# ---------- use existing repo modules ----------
from core.models.resnet import LightResnet, BasicBlock
from core.utils import Logger, parser_train, seed
from core.attacks import create_attack            # for eval adversary
from core.utils.trades import trades_loss         # existing min-max training
from core.utils.mart import mart_loss
from core.utils.context import ctx_eval           # ONLY switches to eval(), NO no_grad()
from core import animal_classes, vehicle_classes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# Models
# ------------------------------------------------
def build_lightresnet20(num_classes: int) -> LightResnet:
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
    """10-class classifier head taking concatenated penultimate features."""
    def __init__(self, in_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """
    Frozen-feature fusion head:
      x -> penult(M1(x)) || penult(M2(x)) -> HeadG -> logits(10)

    M1/M2 parameters are frozen (requires_grad=False) but we DO NOT run in
    torch.no_grad during forward, so gradients wrt input x still propagate,
    which is required by PGD/TRADES/MART to craft x+delta.
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.head = head

        # freeze submodels
        for p in self.m1.parameters():
            p.requires_grad = False
        for p in self.m2.parameters():
            p.requires_grad = False

    def forward(self, x):
        feats = {}

        def hook_m1(module, inp, out): feats["m1"] = inp[0]   # don't detach
        def hook_m2(module, inp, out): feats["m2"] = inp[0]

        h1 = self.m1.fc.register_forward_hook(hook_m1)
        h2 = self.m2.fc.register_forward_hook(hook_m2)

        # Forward through frozen backbones (eval mode outside)
        _ = self.m1(x)
        _ = self.m2(x)

        h1.remove(); h2.remove()

        z = torch.cat([feats["m1"], feats["m2"]], dim=1)  # [B, 128]
        return self.head(z)

# ------------------------------------------------
# Data
# ------------------------------------------------
def _build_cifar10(data_dir, train: bool, num_workers=4, batch_size=128):
    tfm = (T.Compose([T.RandomCrop(32, padding=4),
                      T.RandomHorizontalFlip(),
                      T.ToTensor()])
           if train else T.Compose([T.ToTensor()]))
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
        self.base = base; self.indices = indices; self.remap = remap
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
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 45], gamma=0.1)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
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
    crit = nn.CrossEntropyLoss()
    # stronger eval PGD (no no_grad wrapper!)
    if args.attack in ['linf-pgd', 'l2-pgd']:
        iters = max(20, 2*args.attack_iter)
        return create_attack(model, crit, args.attack, args.attack_eps, iters, args.attack_step)
    elif args.attack in ['fgsm', 'linf-df']:
        return create_attack(model, crit, 'linf-pgd', 8/255, 20, 2/255)
    elif args.attack in ['fgm', 'l2-df']:
        return create_attack(model, crit, 'l2-pgd', 128/255, 20, 15/255)
    else:
        return create_attack(model, crit, args.attack, args.attack_eps, args.attack_iter, args.attack_step)

def eval_adv(model, loader, attack) -> float:
    """
    Evaluate adversarial accuracy.
    Important: do NOT use torch.no_grad() during perturbation.
    """
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # no no_grad here; attack will backward wrt x
        x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            logits = model(x_adv)
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)

def train_head_adversarial(fusion: FusionHead,
                           train_loader, test_loader,
                           args, logger):
    """
    Train ONLY the head G in min-max fashion using existing TRADES/MART.
    We pass optimizer into trades_loss/mart_loss; they handle zero_grad/step.
    """
    # put frozen backbones in eval mode (BN/Dropout stable), but keep graph
    fusion.m1.eval()
    fusion.m2.eval()

    # Only G's parameters are trainable
    params = [p for p in fusion.parameters() if p.requires_grad]
    assert all([p.requires_grad for p in fusion.head.parameters()]) and len(params) == sum(p.numel() for p in fusion.head.parameters())

    opt = torch.optim.Adam(fusion.head.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 45], gamma=0.1)

    eval_atk = make_eval_attack(fusion, args)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs_g+1):
        fusion.train()
        # keep backbones eval: no BN updates, no dropout
        fusion.m1.eval()
        fusion.m2.eval()

        seen, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                # loss tuple-like, first one is CE(clean) + beta*KL(...)
                loss_all = trades_loss(
                    fusion, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_all[0] if isinstance(loss_all, (tuple, list)) else loss_all

            elif args.trainer == 'mart':
                loss_all = mart_loss(
                    fusion, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_all[0] if isinstance(loss_all, (tuple, list)) else loss_all

            else:  # plain CE (no adversary)
                opt.zero_grad(set_to_none=True)
                logits = fusion(x)
                loss_main = ce(logits, y)
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), 5.0)
                opt.step()

            loss_sum += float(loss_main.detach().item()) * x.size(0)

            with torch.no_grad():
                preds = fusion(x).argmax(1)
                correct += (preds == y).sum().item()
                seen += y.size(0)

        sch.step()

        if ep % 5 == 0 or ep == 1:
            clean_acc = eval_clean(fusion, test_loader)
            adv_acc   = eval_adv(fusion, test_loader, eval_atk)
            logger.log(f'[G-{args.trainer}] Epoch {ep:03d} | Train Loss {(loss_sum/max(seen,1)):.4f} | '
                       f'Train Acc {(correct/max(seen,1)):.4f} | Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # Use shared parser (already defines --beta/--attack/--attack-eps/--attack-iter/--attack-step/--workers etc.)
    parse = parser_train()
    # extra knobs (do not re-add --beta to avoid conflicts)
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 clean training')
    parse.add_argument('--epochs-g', type=int, default=50, help='epochs for G adversarial training')
    parse.add_argument('--lr-m', type=float, default=0.1, help='LR for M1/M2 clean training')
    parse.add_argument('--lr',  type=float, default=1e-3, help='LR for G (head) adversarial training')
    parse.add_argument('--trainer', type=str, default='trades', choices=['trades', 'mart', 'ce'],
                       help='objective for G: trades/mart (adv) or ce (clean)')
    args = parse.parse_args()

    # Paths & logger
    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR  = os.path.join(args.log_dir, args.desc)
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

    # Full loaders (10-class)
    _, full_train_loader = _build_cifar10(DATA_DIR, train=True,  num_workers=getattr(args, 'workers', 4), batch_size=args.batch_size)
    _, full_test_loader  = _build_cifar10(DATA_DIR, train=False, num_workers=getattr(args, 'workers', 4), batch_size=args.batch_size)

    # Filtered loaders for submodels
    m1_train_loader = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=True,  num_workers=getattr(args, 'workers', 4))
    m1_test_loader  = build_filtered_loader(DATA_DIR, animal_classes, args.batch_size, train=False, num_workers=getattr(args, 'workers', 4))
    m2_train_loader = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=True,  num_workers=getattr(args, 'workers', 4))
    m2_test_loader  = build_filtered_loader(DATA_DIR, vehicle_classes, args.batch_size, train=False, num_workers=getattr(args, 'workers', 4))

    logger.log(f'Animal classes: {animal_classes} | Vehicle classes: {vehicle_classes}')

    # -------- Stage 1: clean train M1/M2 --------
    logger.log(f"Training M1 (6-class) for {args.epochs_m} epochs (clean CE)...")
    m1 = build_lightresnet20(num_classes=len(animal_classes))
    train_clean_classifier(m1, m1_train_loader, m1_test_loader, args.epochs_m, args.lr_m, logger, '[M1]')

    logger.log(f"Training M2 (4-class) for {args.epochs_m} epochs (clean CE)...")
    m2 = build_lightresnet20(num_classes=len(vehicle_classes))
    train_clean_classifier(m2, m2_train_loader, m2_test_loader, args.epochs_m, args.lr_m, logger, '[M2]')

    # Final clean acc of submodels
    acc_m1 = eval_clean(m1, m1_test_loader)
    acc_m2 = eval_clean(m2, m2_test_loader)
    logger.log(f'[M1] Clean Test Acc: {acc_m1:.4f}')
    logger.log(f'[M2] Clean Test Acc: {acc_m2:.4f}')

    # -------- Stage 2: adversarial train ONLY G --------
    head = HeadG(in_dim=128, num_classes=10).to(DEVICE)
    fusion = FusionHead(m1, m2, head).to(DEVICE)

    # Put backbones to eval (BN frozen), head training mode during training loop
    fusion.m1.eval()
    fusion.m2.eval()

    logger.log(f"Adversarial training G with {args.trainer.upper()} for {args.epochs_g} epochs ...")
    train_head_adversarial(fusion, full_train_loader, full_test_loader, args, logger)

    # Final report
    clean_g = eval_clean(fusion, full_test_loader)
    adv_g   = eval_adv(fusion, full_test_loader, make_eval_attack(fusion, args))
    logger.log(f'[G] Final Test Clean: {clean_g:.4f} | Final Test Adv: {adv_g:.4f}')

    # Save weights
    torch.save({'model_state_dict': m1.state_dict()},   os.path.join(LOG_DIR, 'M1_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()},   os.path.join(LOG_DIR, 'M2_4cls.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'model_state_dict': fusion.state_dict()}, os.path.join(LOG_DIR, 'Fusion_frozen_backbones.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
