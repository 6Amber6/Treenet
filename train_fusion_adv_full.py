# train_fusion_adv_full.py
"""
Two-Stage CE + Full Adversarial Training
(All components participate in delta gradients; stable clean accuracy)

Pipeline:
  1) Train M1 (6-class animals) with CE (clean only)
  2) Train M2 (4-class vehicles) with CE (clean only)
  3) Build Fusion model:
      x -> M1(x) -> 6-class logits
      x -> M2(x) -> 4-class logits
      x -> [penult(M1(x)) || penult(M2(x))] -> HeadG(10) -> 10-class logits
  4) Full adversarial training (TRADES / MART):
     - ALL parameters (M1 + M2 + HeadG) are trainable and participate in delta gradients
     - To stabilize clean accuracy, we add small masked auxiliary CE on M1/M2
       (only when labels fall in their class groups)
  5) Evaluations report fusion (10-class) performance for clean & adversarial.
"""

import os
import sys
import json
import shutil
import argparse
from typing import List, Tuple

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
from core.utils.trades import trades_loss  # kept for reference (we use custom TRADES below)
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


class FullFusionModel(nn.Module):
    """
    Full fusion model where ALL components participate in adversarial training:
    - M1: 6-class animal classifier
    - M2: 4-class vehicle classifier
    - HeadG: 10-class fusion classifier
    Forward returns a tuple: (m1_logits, m2_logits, fusion_logits).
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.head = head

        # Forward hooks to capture penultimate features (inputs to final fc of each backbone)
        self._feats = {}
        self._h1 = self.m1.fc.register_forward_hook(lambda m, inp, out: self._save_feat("m1", inp))
        self._h2 = self.m2.fc.register_forward_hook(lambda m, inp, out: self._save_feat("m2", inp))

    def _save_feat(self, key, inp_tuple):
        # inp_tuple[0] shape: [B, D] = features BEFORE final fc
        self._feats[key] = inp_tuple[0]

    def forward(self, x):
        # M1 forward (6-class)
        m1_logits = self.m1(x)
        # M2 forward (4-class)
        m2_logits = self.m2(x)
        # Fusion forward (10-class)
        z = torch.cat([self._feats["m1"], self._feats["m2"]], dim=1)
        fusion_logits = self.head(z)
        return m1_logits, m2_logits, fusion_logits

    def remove_hooks(self):
        self._h1.remove()
        self._h2.remove()


class FusionAdapter(nn.Module):
    """
    Adapter that exposes ONLY fusion (10-class) logits for attacks / MART.
    Backprop still flows through the full fusion model.
    """
    def __init__(self, full: FullFusionModel):
        super().__init__()
        self.full = full

    def forward(self, x):
        return self.full(x)[-1]  # fusion logits


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
# Train & Eval utils
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
    """
    Works for both: single-head models (logits) and fusion model (tuple).
    Uses the fusion (10-class) head for evaluation when present.
    """
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        logits = out[-1] if isinstance(out, tuple) else out
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


def make_eval_attack(full_fusion_model: FullFusionModel, args):
    """
    Build evaluation attack that targets ONLY the 10-class fusion head,
    but gradients flow through the entire fusion model.
    """
    crit = nn.CrossEntropyLoss()
    # Use adapter so attack sees logits only
    head_only = FusionAdapter(full_fusion_model)
    attack = create_attack(
        head_only, crit,
        getattr(args, 'attack', 'linf-pgd'),
        getattr(args, 'attack_eps', 8/255),
        getattr(args, 'attack_iter', 10),
        getattr(args, 'attack_step', 2/255),
    )
    return attack


def eval_adv(model, loader, attack) -> float:
    """
    Evaluate adversarial accuracy on the fusion (10-class) head.
    """
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # Attack targets the fusion head adapter; it still backprops through the full model
        x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            out = model(x_adv)
            logits = out[-1] if isinstance(out, tuple) else out
        correct += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return correct / max(tot, 1)


@torch.no_grad()
def bn_calibration(model, loader, max_batches=200):
    """
    Recalibrate BatchNorm running stats (useful after strong adversarial phases).
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
def masked_aux_ce(m1_logits: torch.Tensor, m2_logits: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Compute masked auxiliary CE losses:
    - M1 (6-class) is trained ONLY on labels from animal_classes (mapped to 0..5)
    - M2 (4-class) is trained ONLY on labels from vehicle_classes (mapped to 0..3)
    Returns (loss_aux, n1, n2) where n1/n2 are number of samples used for each head.
    """
    import torch.nn.functional as F
    device = y.device

    loss_aux = torch.tensor(0.0, device=device)
    n1 = n2 = 0

    # Build label maps: 10 -> {0..5} for animals, {0..3} for vehicles, -1 otherwise
    animal_map = torch.full((10,), -1, dtype=torch.long, device=device)
    vehicle_map = torch.full((10,), -1, dtype=torch.long, device=device)
    for i, c in enumerate(animal_classes):
        animal_map[c] = i
    for i, c in enumerate(vehicle_classes):
        vehicle_map[c] = i

    if m1_logits is not None:
        sel1 = animal_map[y] != -1
        if sel1.any():
            y1 = animal_map[y[sel1]]
            loss_aux = loss_aux + F.cross_entropy(m1_logits[sel1], y1)
            n1 = int(sel1.sum().item())

    if m2_logits is not None:
        sel2 = vehicle_map[y] != -1
        if sel2.any():
            y2 = vehicle_map[y[sel2]]
            loss_aux = loss_aux + F.cross_entropy(m2_logits[sel2], y2)
            n2 = int(sel2.sum().item())

    return loss_aux, n1, n2


# ------------------------------------------------
# Custom TRADES (fusion-head-driven) + masked aux CE
# ------------------------------------------------
def custom_trades_loss(model: FullFusionModel,
                       x_natural, y, optimizer,
                       step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                       aux_w=0.05):
    """
    TRADES guided by the fusion head (10-class) + small masked CE on M1/M2.
    ALL parameters participate in delta gradients.

    - Fusion drives robustness (KL on adv vs nat fusion logits).
    - M1/M2 get weak auxiliary CE only for their respective label subsets (stabilizes features).
    """
    import torch.nn.functional as F
    from torch.autograd import Variable

    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)

    # Initial adv seed
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).to(x_natural.device).detach()

    # Natural predictions (fusion only) for KL target
    with torch.no_grad():
        _, _, fusion_nat = model(x_natural)
        p_natural = F.softmax(fusion_nat, dim=1)

    # PGD on fusion logits
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            _, _, fusion_adv = model(x_adv)
            loss_kl = criterion_kl(F.log_softmax(fusion_adv, dim=1), p_natural)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    # Forward (clean & adv)
    m1_nat, m2_nat, fusion_nat = model(x_natural)
    _, _, fusion_adv = model(x_adv)

    # Fusion clean CE
    loss_nat = F.cross_entropy(fusion_nat, y)

    # Robust KL (TRADES)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(fusion_adv, dim=1),
        F.softmax(fusion_nat.detach(), dim=1)
    )

    # Masked auxiliary CE for M1/M2 (small weight)
    loss_aux, _, _ = masked_aux_ce(m1_nat, m2_nat, y)
    loss = loss_nat + beta * loss_robust + aux_w * loss_aux

    loss.backward()
    optimizer.step()
    return loss


# ------------------------------------------------
# Optimizer (parameter groups: head fast, backbones slow)
# ------------------------------------------------
def build_full_optimizer(fusion_model: FullFusionModel, base_lr: float, weight_decay=5e-4):
    return torch.optim.SGD(
        [
            {'params': fusion_model.head.parameters(), 'lr': base_lr},        # HeadG: fast
            {'params': fusion_model.m1.parameters(),   'lr': base_lr * 0.3}, # M1: slower
            {'params': fusion_model.m2.parameters(),   'lr': base_lr * 0.3}, # M2: slower
        ],
        momentum=0.9, weight_decay=weight_decay, nesterov=True
    )


# ------------------------------------------------
# Full-fusion training loop
# ------------------------------------------------
def train_full_fusion_adversarial(fusion_model: FullFusionModel,
                                  train_loader, test_loader,
                                  args, logger,
                                  aux_w: float = 0.05):
    """
    Train ALL components (M1 + M2 + HeadG) with TRADES/MART/CE.
    - TRADES: custom implementation with masked aux CE
    - MART: use repo's MART but via FusionAdapter so it sees only fusion logits
    - CE: plain CE on fusion logits (for warmup)
    """
    opt = build_full_optimizer(fusion_model, base_lr=args.lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[35, 45], gamma=0.1)
    ce = nn.CrossEntropyLoss()
    eval_atk = make_eval_attack(fusion_model, args)

    for ep in range(1, args.epochs_g + 1):
        fusion_model.train()
        seen, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                loss_main = custom_trades_loss(
                    fusion_model, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                    aux_w=aux_w,
                )

            elif args.trainer == 'mart':
                # Route MART to fusion-head-only while optimizing the full model
                head_only = FusionAdapter(fusion_model)
                loss_val = mart_loss(
                    head_only, x, y, optimizer=opt,
                    beta=args.beta,
                    step_size=getattr(args, 'attack_step', 2/255),
                    epsilon=getattr(args, 'attack_eps', 8/255),
                    perturb_steps=getattr(args, 'attack_iter', 10),
                )
                loss_main = loss_val[0] if isinstance(loss_val, (tuple, list)) else loss_val

            else:
                # CE warmup on the fusion head only
                opt.zero_grad(set_to_none=True)
                fusion_logits = FusionAdapter(fusion_model)(x)
                loss_main = ce(fusion_logits, y)
                loss_main.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 5.0)
                opt.step()

            loss_sum += float(loss_main.detach().item()) * x.size(0)

            with torch.no_grad():
                logits = FusionAdapter(fusion_model)(x)
                correct += (logits.argmax(1) == y).sum().item()
                seen += y.size(0)

        sch.step()

        if ep % 5 == 0 or ep == 1:
            # (Optional) light BN calibration every 15 epochs can help
            if ep % 15 == 1 and args.trainer != 'ce':
                bn_calibration(fusion_model, train_loader, max_batches=200)

            clean_acc = eval_clean(fusion_model, test_loader)
            adv_acc = eval_adv(fusion_model, test_loader, eval_atk)
            logger.log(f'[Full-Fusion-{args.trainer}] Epoch {ep:03d} | '
                       f'Train Loss {(loss_sum/max(seen,1)):.4f} | Train Acc {(correct/max(seen,1)):.4f} | '
                       f'Test Clean {clean_acc:.4f} | Test Adv {adv_acc:.4f}')


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # Use repo's parser_train (defines --data-dir/--log-dir/--desc/--data and attack args)
    parse = parser_train()
    # Add only NEW flags
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 clean training')
    parse.add_argument('--epochs-g', type=int, default=50, help='total epochs for full fusion adversarial training')
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

    # Stage 2: Full fusion adversarial training (ALL parameters trainable)
    # Infer penultimate concat dim from fc.in_features of both backbones
    penult_dim = int(m1.fc.in_features + m2.fc.in_features)
    logger.log(f'Inferred penultimate concat dim: {penult_dim}')
    head = HeadG(in_dim=penult_dim, num_classes=10).to(DEVICE)

    fusion_model = FullFusionModel(m1, m2, head).to(DEVICE)
    logger.log(f'Full fusion model: ALL parameters trainable')
    logger.log(f'Total trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)}')

    # CE warmup (helps stabilize head & BN)
    logger.log("Warmup (CE) for full fusion: 5 epochs (ALL components)...")
    warm_args = argparse.Namespace(**vars(args))
    warm_args.trainer = 'ce'
    warm_args.epochs_g = 5
    train_full_fusion_adversarial(fusion_model, full_train_loader, full_test_loader, warm_args, logger, aux_w=args.aux_w)

    # Light BN calibration after warmup
    bn_calibration(fusion_model, full_train_loader, max_batches=200)

    # Full adversarial training
    logger.log(f"Full adversarial training ({args.trainer}) with eps={args.attack_eps}, "
               f"step={args.attack_step}, iters={args.attack_iter} for {args.epochs_g} epochs...")
    logger.log("ALL components (M1 + M2 + HeadG) participate in delta gradient computation")
    train_full_fusion_adversarial(fusion_model, full_train_loader, full_test_loader, args, logger, aux_w=args.aux_w)

    # Final evaluation (head-only attack)
    clean_g = eval_clean(fusion_model, full_test_loader)
    adv_g = eval_adv(fusion_model, full_test_loader, make_eval_attack(fusion_model, args))
    logger.log(f'[Full-Fusion] Final Test Clean: {clean_g:.4f} | Final Test Adv: {adv_g:.4f}')

    # Save checkpoints
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(LOG_DIR, 'M1_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(LOG_DIR, 'M2_4cls.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'model_state_dict': fusion_model.state_dict()}, os.path.join(LOG_DIR, 'Full_Fusion_All_Trainable.pt'))
    logger.log(f'Saved models to {LOG_DIR}')


if __name__ == '__main__':
    main()
