"""
Pipeline:
  1) Train M1 (6-class animal) with adversarial training (TRADES/MART/CE depending on parser flags)
  2) Train M2 (4-class vehicle) with adversarial training
  3) Build FusionModel that concatenates penultimate embeddings of M1 and M2 -> Head G (10-class)
     and train it END-TO-END with the same adversarial training method on the full 10-class data
  4) Report clean & adversarial accuracies for M1, M2, and FusionModel
"""

import os
import sys
import json
import shutil
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as T

# ----- Make local 'core' package importable (Plan B path injection) -----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

# ----- Imports from your codebase -----
from core.data import get_data_info, load_data
from core.models.resnet import LightResnet, BasicBlock
from core.attacks import create_attack
from core.utils.context import ctx_noparamgrad_and_eval
from core.metrics import accuracy
from core.utils import format_time, Logger, parser_train, seed
from core.utils.trades import trades_loss
from core.utils.mart import mart_loss
from core import animal_classes, vehicle_classes

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 normalization (needed for custom filtered loaders)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# --------------------------
#   Models & Heads
# --------------------------
def build_lightresnet20(num_classes: int) -> LightResnet:
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class HeadG(nn.Module):
    """Small MLP head mapping concatenated embeddings -> 10 logits."""
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


class FusionModel(nn.Module):
    """
    End-to-end fusion: x -> penult(M1(x)) || penult(M2(x)) -> Head G -> logits(10)
    extract penultimate features using a forward hook on each sub-model's fc layer.
    """
    def __init__(self, m1: LightResnet, m2: LightResnet, head: HeadG):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.head = head

    @torch.no_grad()
    def _penult(self, model: LightResnet, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        def hook_fn(module, inp, out):
            feats["feat"] = inp[0].detach()
        h = model.fc.register_forward_hook(hook_fn)
        _ = model(x)  # forward to populate feats
        h.remove()
        return feats["feat"]

    def forward(self, x):
        # want gradients to flow through M1/M2 into G, so do NOT no_grad here.
        feats = {}
        def hook_m1(module, inp, out): feats["m1"] = inp[0]
        def hook_m2(module, inp, out): feats["m2"] = inp[0]
        h1 = self.m1.fc.register_forward_hook(hook_m1)
        h2 = self.m2.fc.register_forward_hook(hook_m2)

        # Forward both sub-models
        _ = self.m1(x)
        _ = self.m2(x)

        h1.remove(); h2.remove()
        f1 = feats["m1"]
        f2 = feats["m2"]
        z = torch.cat([f1, f2], dim=1)  # [B, d1+d2] -> here each is 64-d penult typically (total 128)
        return self.head(z)


# --------------------------
#   Dataloaders
# --------------------------
def filter_and_remap_indices(dataset: torchvision.datasets.CIFAR10, keep_labels: List[int]):
    indices = [i for i, (_, y) in enumerate(dataset) if y in keep_labels]
    remap = {old: new for new, old in enumerate(keep_labels)}
    return indices, remap


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices, remap):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        y = self.remap[int(y)]
        return x, y


def build_filtered_loaders(data_dir: str, keep_labels: List[int], batch_size: int, train: bool, num_workers: int = 4):
    """Filtered loaders with the same normalization as the main pipeline."""
    transform = (T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]) if train else T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]))
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    indices, remap = filter_and_remap_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    pin = torch.cuda.is_available()
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin)
    return loader


# --------------------------
#   Training / Eval utils
# --------------------------
def init_attack_for_model(model: nn.Module, args):
    """
    Create train-time attack and stronger eval-time attack for the given model.
    We use parser_train() flags: --attack, --attack-eps, --attack-iter, --attack-step
    """
    crit = nn.CrossEntropyLoss()
    atk = create_attack(model, crit, args.attack, args.attack_eps, args.attack_iter, args.attack_step, rand_init_type='uniform')
    # For eval, typically stronger/longer PGD:
    if args.attack in ['linf-pgd', 'l2-pgd']:
        eval_atk = create_attack(model, crit, args.attack, args.attack_eps, max(20, 2*args.attack_iter), args.attack_step)
    elif args.attack in ['fgsm', 'linf-df']:
        eval_atk = create_attack(model, crit, 'linf-pgd', 8/255, 20, 2/255)
    elif args.attack in ['fgm', 'l2-df']:
        eval_atk = create_attack(model, crit, 'l2-pgd', 128/255, 20, 15/255)
    else:
        eval_atk = atk
    return atk, eval_atk


def train_one_model(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    train_loader: DataLoader,
                    test_loader: DataLoader,
                    args,
                    logger: Logger,
                    tag: str):
    """
    Train a single classifier (M1 or M2) with TRADES/MART/CE depending on args.
    Adversarial examples are crafted in IMAGE space using the model itself.
    """
    atk, eval_atk = init_attack_for_model(model, args)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs_m + 1):
        model.train()
        total, correct, run_loss = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                # trades_loss internally crafts adversarial examples for input x
                # and returns CE(clean,y) + beta * KL(clean||adv)
                loss = trades_loss(model, x, y, optimizer, beta=args.beta)
            elif args.trainer == 'mart':
                loss = mart_loss(model, x, y, optimizer, beta=args.beta)
            else:
                # plain CE training
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.trainer in ['trades', 'mart']:
                # trades_loss/mart_loss already zero_grad/backward/step inside (typical in many repos)
                pass

            run_loss += loss.item() * x.size(0)

            with torch.no_grad():
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        if scheduler is not None:
            scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            clean_acc, adv_acc = eval_clean_and_adv(model, test_loader, eval_atk)
            logger.log(f'{tag} Epoch {epoch:03d} | Train Loss {(run_loss/max(total,1)):.4f} | Clean Acc {clean_acc:.4f} | Adv Acc {adv_acc:.4f}')


@torch.no_grad()
def eval_clean_and_adv(model: nn.Module, loader: DataLoader, attack) -> Tuple[float, float]:
    """Evaluate clean and adversarial accuracy using the provided attack (image-space)."""
    model.eval()
    clean_correct, adv_correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # clean
        logits = model(x)
        clean_correct += (logits.argmax(1) == y).sum().item()

        # adv
        with ctx_noparamgrad_and_eval(model):
            x_adv, _ = attack.perturb(x, y)
        logits_adv = model(x_adv)
        adv_correct += (logits_adv.argmax(1) == y).sum().item()

        total += y.size(0)

    return clean_correct / max(total, 1), adv_correct / max(total, 1)


def train_fusion(model: FusionModel,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 args,
                 logger: Logger):
    """
    Train FusionModel end-to-end with TRADES/MART/CE.
    Attacks are generated in IMAGE space against FusionModel.
    """
    atk, eval_atk = init_attack_for_model(model, args)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs_g + 1):
        model.train()
        total, correct, run_loss = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if args.trainer == 'trades':
                loss = trades_loss(model, x, y, optimizer, beta=args.beta)
            elif args.trainer == 'mart':
                loss = mart_loss(model, x, y, optimizer, beta=args.beta)
            else:
                logits = model(x)
                loss = ce(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.trainer in ['trades', 'mart']:
                pass

            run_loss += loss.item() * x.size(0)
            with torch.no_grad():
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        if scheduler is not None:
            scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            clean_acc, adv_acc = eval_clean_and_adv(model, test_loader, eval_atk)
            logger.log(f'[Fusion] Epoch {epoch:03d} | Train Loss {(run_loss/max(total,1)):.4f} | Clean Acc {clean_acc:.4f} | Adv Acc {adv_acc:.4f}')


# --------------------------
#   Main
# --------------------------
def main():
    # Use the shared parser from your repo; it already defines many useful flags,
    # including --beta, --attack, --attack-eps, --attack-iter, --attack-step, etc.
    parse = parser_train()

    # Add only our extra knobs (do NOT re-add --beta to avoid conflict)
    parse.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 training')
    parse.add_argument('--epochs-g', type=int, default=50, help='epochs for Fusion (G) training')
    parse.add_argument('--trainer', type=str, default='trades', choices=['trades', 'mart', 'ce'],
                       help='adversarial training objective for all stages')
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

    # Info & seeds
    info = get_data_info(DATA_DIR)
    logger.log(f'Using device: {DEVICE}')
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Full 10-class loaders from your core.data loader (already normalized)
    train_dataset, test_dataset, full_train_loader, full_test_loader = load_data(
        DATA_DIR, args.batch_size, args.batch_size_validation,
        use_augmentation=args.augment, shuffle_train=True,
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction
    )
    del train_dataset, test_dataset  # not needed further

    # Filtered loaders for M1/M2 (with our own normalized transforms)
    workers = getattr(args, 'workers', 4)
    m1_train_loader = build_filtered_loaders(DATA_DIR, animal_classes, args.batch_size, train=True, num_workers=workers)
    m1_test_loader  = build_filtered_loaders(DATA_DIR, animal_classes, args.batch_size, train=False, num_workers=workers)
    m2_train_loader = build_filtered_loaders(DATA_DIR, vehicle_classes, args.batch_size, train=True, num_workers=workers)
    m2_test_loader  = build_filtered_loaders(DATA_DIR, vehicle_classes, args.batch_size, train=False, num_workers=workers)

    # ------------------ Stage 1: Train M1/M2 ------------------
    m1 = build_lightresnet20(num_classes=len(animal_classes))
    m2 = build_lightresnet20(num_classes=len(vehicle_classes))

    # Optimizers & schedulers for M1/M2 (robust-friendly defaults)
    m1_opt = torch.optim.SGD(m1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=False)
    m2_opt = torch.optim.SGD(m2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=False)
    # You can also swap to cosine + warmup if desired; keep MultiStep for consistency with earlier code
    m1_sch = torch.optim.lr_scheduler.MultiStepLR(m1_opt, milestones=[75, 90], gamma=0.1)
    m2_sch = torch.optim.lr_scheduler.MultiStepLR(m2_opt, milestones=[75, 90], gamma=0.1)

    logger.log(f'Training M1 (6-class animal) for {args.epochs_m} epochs with {args.trainer.upper()}...')
    train_one_model(m1, m1_opt, m1_sch, m1_train_loader, m1_test_loader, args, logger, tag='[M1]')

    logger.log(f'Training M2 (4-class vehicle) for {args.epochs_m} epochs with {args.trainer.upper()}...')
    train_one_model(m2, m2_opt, m2_sch, m2_train_loader, m2_test_loader, args, logger, tag='[M2]')

    # Eval submodels (clean & adv)
    m1_atk, m1_eval_atk = init_attack_for_model(m1, args)
    m2_atk, m2_eval_atk = init_attack_for_model(m2, args)
    c1, a1 = eval_clean_and_adv(m1, m1_test_loader, m1_eval_atk)
    c2, a2 = eval_clean_and_adv(m2, m2_test_loader, m2_eval_atk)
    logger.log(f'[M1] Clean: {c1:.4f} | Adv: {a1:.4f}')
    logger.log(f'[M2] Clean: {c2:.4f} | Adv: {a2:.4f}')

    # ------------------ Stage 2: Fusion end-to-end ------------------
    head = HeadG(in_dim=128, num_classes=10).to(DEVICE)
    fusion = FusionModel(m1, m2, head).to(DEVICE)

    # Fusion optimizer/scheduler
    # Tip: often better to fine-tune with a slightly smaller LR; but we keep args.lr for consistency
    fusion_opt = torch.optim.SGD(fusion.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=False)
    fusion_sch = torch.optim.lr_scheduler.MultiStepLR(fusion_opt, milestones=[75, 90], gamma=0.1)

    logger.log(f'Training Fusion (M1+M2+G) for {args.epochs_g} epochs with {args.trainer.upper()}...')
    train_fusion(fusion, fusion_opt, fusion_sch, full_train_loader, full_test_loader, args, logger)

    # Final eval on Fusion
    f_atk, f_eval_atk = init_attack_for_model(fusion, args)
    cf, af = eval_clean_and_adv(fusion, full_test_loader, f_eval_atk)
    logger.log(f'[Fusion] Final Clean: {cf:.4f} | Final Adv: {af:.4f}')

    # Save weights
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(LOG_DIR, 'M1_animal_6cls.pt'))
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(LOG_DIR, 'M2_vehicle_4cls.pt'))
    torch.save({'model_state_dict': head.state_dict()}, os.path.join(LOG_DIR, 'G_head_10cls.pt'))
    torch.save({'model_state_dict': fusion.state_dict()}, os.path.join(LOG_DIR, 'Fusion_end2end.pt'))
    logger.log(f'Saved models to {LOG_DIR}')

    # Optional: wandb summary
    try:
        wandb.init(project="two-stage-ce", name=args.desc, reinit=True)
        wandb.summary["m1_clean_acc"] = c1
        wandb.summary["m1_adv_acc"] = a1
        wandb.summary["m2_clean_acc"] = c2
        wandb.summary["m2_adv_acc"] = a2
        wandb.summary["fusion_clean_acc"] = cf
        wandb.summary["fusion_adv_acc"] = af
        wandb.finish()
    except:
        pass


if __name__ == '__main__':
    main()
