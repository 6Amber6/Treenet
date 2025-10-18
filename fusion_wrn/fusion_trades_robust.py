# fusion_trades_robust.py
# CIFAR-10 WRN-28-10 fusion + TRADES + Diffusion-Augmented training
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
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------ Repo Paths ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
ARP_ROOT = os.path.join(REPO_ROOT, "adversarial_robustness_pytorch")
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.models.wideresnet import wideresnet
from core.utils import Logger, seed
from core.attacks import create_attack
from core import animal_classes, vehicle_classes
from dp_utils_diffusion import build_diffusion_augmented_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ------------------------------ CIFAR Loader ------------------------------
def _build_cifar10(data_dir, train=True, num_workers=4, batch_size=128):
    tfm = (
        T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                T.RandomErasing(p=1.0, scale=(0.05, 0.10), ratio=(0.5, 2.0), value=0),
            ]
        )
        if train
        else T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    )
    ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=tfm
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


def build_wrn_28_10(num_classes: int):
    model = wideresnet("wideresnet-28-10", num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


class FusionWRN(nn.Module):
    """x -> M1(4c), M2(6c), concat penult -> Head(10c)"""

    def __init__(self, m1, m2, head):
        super().__init__()
        self.m1, self.m2, self.head = m1, m2, head
        self._feats = {}
        self.m1.fc.register_forward_hook(lambda m, inp, out: self._save("m1", inp))
        self.m2.fc.register_forward_hook(lambda m, inp, out: self._save("m2", inp))

    def _save(self, k, inp):
        self._feats[k] = inp[0]

    def forward(self, x):
        m1_logits = self.m1(x)
        m2_logits = self.m2(x)
        z = torch.cat([self._feats["m1"], self._feats["m2"]], dim=1)
        return m1_logits, m2_logits, self.head(z)


# --------------------------- EMA ----------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[n].clone()

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]


# --------------------------- Helper funcs -------------------------------
def set_head_dropout_prob(model: FusionWRN, p: float):
    for m in model.head.modules():
        if isinstance(m, nn.Dropout):
            m.p = p


def freeze_backbone_bn(model: FusionWRN):
    for bb in (model.m1, model.m2):
        for m in bb.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()


def unfreeze_backbone_bn(model: FusionWRN):
    for bb in (model.m1, model.m2):
        for m in bb.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()


# --------------------------- Loss helpers -------------------------------
def masked_aux_ce(m1_logits, m2_logits, y):
    device = y.device
    loss = torch.tensor(0.0, device=device)
    if y.numel() == 0:
        return loss
    # m1 = 4-class vehicle, m2 = 6-class animal
    if len(vehicle_classes) > 0:
        idx1 = torch.tensor(vehicle_classes, device=device)
        map1 = torch.full((10,), -1, dtype=torch.long, device=device)
        map1[idx1] = torch.arange(len(idx1), device=device)
        mask1 = map1[y] >= 0
        if mask1.any():
            loss += F.cross_entropy(m1_logits[mask1], map1[y[mask1]])
    if len(animal_classes) > 0:
        idx2 = torch.tensor(animal_classes, device=device)
        map2 = torch.full((10,), -1, dtype=torch.long, device=device)
        map2[idx2] = torch.arange(len(idx2), device=device)
        mask2 = map2[y] >= 0
        if mask2.any():
            loss += F.cross_entropy(m2_logits[mask2], map2[y[mask2]])
    return loss


# --------------------------- TRADES Step -------------------------------
def trades_fusion_step(
    model,
    x_natural,
    y,
    optimizer,
    step_size=2 / 255,
    epsilon=8 / 255,
    perturb_steps=12,
    beta=8.0,
    aux_w=0.02,
):
    class LogitsOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            return self.base(x)[-1]

    logits_model = LogitsOnly(model).to(DEVICE)
    logits_model.eval()
    with torch.no_grad():
        p_nat = F.softmax(logits_model(x_natural), dim=1)
    x_adv = (x_natural.detach() + 1e-3 * torch.randn_like(x_natural)).clamp(-5, 5)
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        logits_adv = logits_model(x_adv)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1), p_nat, reduction="batchmean"
        )
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x_natural + epsilon), x_natural - epsilon)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    m1_nat, m2_nat, f_nat = model(x_natural)
    _, _, f_adv = model(x_adv)
    loss_nat = F.cross_entropy(f_nat, y)
    loss_rob = F.kl_div(
        F.log_softmax(f_adv, dim=1),
        F.softmax(f_nat.detach(), dim=1),
        reduction="batchmean",
    )
    loss_aux = masked_aux_ce(m1_nat, m2_nat, y) * aux_w
    loss = loss_nat + beta * loss_rob + loss_aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.detach()


# --------------------------- Eval --------------------------------------
@torch.no_grad()
def eval_clean(model, loader):
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
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            return self.base(x)[-1]

    crit = nn.CrossEntropyLoss()
    return create_attack(
        FusionWrapper(model),
        crit,
        getattr(args, "attack", "linf-pgd"),
        getattr(args, "attack_eps", 8 / 255),
        getattr(args, "attack_iter", 20),
        getattr(args, "attack_step", 2 / 255),
    )


def eval_adv(model, loader, attack) -> float:
    """
    Evaluate model adversarial accuracy using a given attack (e.g. PGD).
    Unlike eval_clean, this function must keep gradient computation ON
    because the attack needs to call loss.backward().
    """
    model.eval()
    tot, correct = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.enable_grad():
            x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            model_output = model(x_adv)
            if isinstance(model_output, tuple) and len(model_output) == 3:
                # FusionWRN model returns (m1_logits, m2_logits, fusion_logits)
                _, _, f_logits = model_output
            else:
                # Individual WRN model returns only logits
                f_logits = model_output
            correct += (f_logits.argmax(1) == y).sum().item()
            tot += y.size(0)

    acc = correct / max(tot, 1)
    return acc


def plot_training_curves(train_losses, train_accs, test_accs, test_adv_accs, save_path):
    """Plot training curves for loss and accuracy"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Clean Accuracy
    ax2.plot(train_accs, label='Train Clean Acc', color='green')
    ax2.plot(test_accs, label='Test Clean Acc', color='red')
    ax2.set_title('Clean Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Adversarial Accuracy
    ax3.plot(test_adv_accs, label='Test Adv Acc', color='orange')
    ax3.set_title('Adversarial Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Clean vs Adversarial
    ax4.plot(test_accs, label='Test Clean Acc', color='red')
    ax4.plot(test_adv_accs, label='Test Adv Acc', color='orange')
    ax4.set_title('Clean vs Adversarial Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def train_ce(
    model, train_loader, test_loader, epochs, lr, logger, tag, args=None, ema=None
):
    """Standard CE training + optional adversarial eval"""
    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=getattr(args, "weight_decay", 5e-4),
        nesterov=True,
    )
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[epochs // 2, int(epochs * 0.75)], gamma=0.1
    )

    # ✅ PGD attack setup for evaluation
    from core.attacks import create_attack

    crit = nn.CrossEntropyLoss()
    atk_eval = create_attack(
        model,
        crit,
        getattr(args, "attack", "linf-pgd"),
        getattr(args, "attack_eps", 8 / 255),
        getattr(args, "attack_iter", 10),
        getattr(args, "attack_step", 2 / 255),
    )

    # 📊 Record metrics for plotting
    train_losses, train_accs, test_accs, test_adv_accs = [], [], [], []

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
            if ema:
                ema.update(model)
            total_loss += loss.item()
            num_batches += 1

        sch.step()
        
        # 📊 Calculate train accuracy
        train_acc = eval_clean(model, train_loader)
        train_losses.append(total_loss / max(num_batches, 1))
        train_accs.append(train_acc)

        # ✅ Every 5 epochs, evaluate both Clean and Adversarial accuracy
        if ep % 5 == 0 or ep == 1:
            if ema:
                ema.apply_to(model)
            clean_acc = eval_clean(model, test_loader)
            adv_acc = eval_adv(model, test_loader, atk_eval)
            if ema:
                ema.restore(model)
            
            test_accs.append(clean_acc)
            test_adv_accs.append(adv_acc)

            logger.log(
                f"{tag} Epoch {ep:03d} | Loss {total_loss/max(num_batches,1):.4f} | Train Acc {train_acc:.4f} | Clean {clean_acc:.4f} | Adv {adv_acc:.4f}"
            )
        else:
            # For epochs without full evaluation, use previous values
            test_accs.append(test_accs[-1] if test_accs else 0.0)
            test_adv_accs.append(test_adv_accs[-1] if test_adv_accs else 0.0)
    
    # 📊 Plot training curves
    plot_path = f"./logs_diffusion/{args.desc if args else 'debug'}/{tag}_training_curves.png"
    plot_training_curves(train_losses, train_accs, test_accs, test_adv_accs, plot_path)


def train_fusion(model, train_loader, test_loader, args, logger):
    params = [
        {"params": model.head.parameters(), "lr": args.lr},
        {"params": model.m1.parameters(), "lr": args.lr * 0.2},
        {"params": model.m2.parameters(), "lr": args.lr * 0.2},
    ]
    opt = torch.optim.SGD(params, momentum=0.9,
                          weight_decay=getattr(args, "weight_decay", 5e-4),
                          nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs_g, eta_min=1e-6
    )
    ema = EMA(model, decay=0.999)
    warmup_epochs = min(10, args.epochs_g)
    
    # 📊 Record metrics for plotting
    train_losses, train_accs, test_accs, test_adv_accs = [], [], [], []
    
    for ep in range(1, warmup_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            _, _, f_logits = model(x)
            loss = F.cross_entropy(f_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model)
            total_loss += loss.item()
            n += 1
        sch.step()
        
        # 📊 Calculate train accuracy
        ema.apply_to(model)
        train_acc = eval_clean(model, train_loader)
        clean = eval_clean(model, test_loader)
        ema.restore(model)
        
        train_losses.append(total_loss / max(n, 1))
        train_accs.append(train_acc)
        test_accs.append(clean)
        test_adv_accs.append(0.0)  # No adversarial training in warmup
        
        logger.log(
            f"[Fusion-CE] Epoch {ep:03d} | Train Loss {total_loss/max(n,1):.4f} | Train Acc {train_acc:.4f} | Clean {clean:.4f}"
        )
    
    set_head_dropout_prob(model, 0.0)
    freeze_backbone_bn(model)
    atk_eval = make_eval_attack(model, args)
    unfroze_bn = False
    
    for ep in range(warmup_epochs + 1, args.epochs_g + 1):
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = trades_fusion_step(
                model,
                x,
                y,
                opt,
                step_size=args.attack_step,
                epsilon=args.attack_eps,
                perturb_steps=args.attack_iter,
                beta=args.beta,
                aux_w=args.aux_w,
            )
            ema.update(model)
            total_loss += float(loss)
            n += 1
        sch.step()
        if not unfroze_bn and (ep - warmup_epochs) >= 40:
            unfreeze_backbone_bn(model)
            unfroze_bn = True
        
        # 📊 Calculate train accuracy
        ema.apply_to(model)
        train_acc = eval_clean(model, train_loader)
        clean = eval_clean(model, test_loader)
        adv = eval_adv(model, test_loader, atk_eval)
        ema.restore(model)
        
        train_losses.append(total_loss / max(n, 1))
        train_accs.append(train_acc)
        test_accs.append(clean)
        test_adv_accs.append(adv)
        
        logger.log(
            f"[Fusion-TRADES] Epoch {ep:03d} | Train Loss {total_loss/max(n,1):.4f} | Train Acc {train_acc:.4f} | Clean {clean:.4f} | Adv {adv:.4f}"
        )
    
    # 📊 Plot training curves
    plot_path = f"./logs_diffusion/{args.desc}/Fusion_training_curves.png"
    plot_training_curves(train_losses, train_accs, test_accs, test_adv_accs, plot_path)


# ------------------------------ Main -----------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fusion WRN TRADES + Diffusion Augmentation"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="dataset root directory"
    )
    parser.add_argument("--data", type=str, default="cifar10", help="dataset name")
    parser.add_argument(
        "--log-dir", type=str, default="./logs_diffusion", help="log directory"
    )
    parser.add_argument(
        "--desc", type=str, required=True, help="description for experiment folder"
    )
    
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs-m", type=int, default=100)
    parser.add_argument("--epochs-g", type=int, default=120)
    parser.add_argument("--lr-m", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--aux_w", type=float, default=0.02)
    parser.add_argument("--attack", type=str, default="linf-pgd")
    parser.add_argument("--attack-eps", type=float, default=8 / 255)
    parser.add_argument("--attack-step", type=float, default=2 / 255)
    parser.add_argument("--attack-iter", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--diff-fraction",
        type=float,
        default=0.7,
        help="fraction of diffusion data used for training (default 0.7)",
    )
    parser.add_argument('--skip-submodels', action='store_true',
                        help='Skip training 4/6-class submodels and load pretrained weights.')
    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_dir, args.data)
    DIFF_DIR = "./data/edm_cifar10_1M"
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = Logger(os.path.join(LOG_DIR, "log-train.log"))
    with open(os.path.join(LOG_DIR, "args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.log(f"Using device: {DEVICE}")
    seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # ---------------- Dataloaders ----------------
    _, full_train = _build_cifar10(DATA_DIR, True, 4, args.batch_size)
    _, full_test = _build_cifar10(DATA_DIR, False, 4, args.batch_size)

    # model1 = 4-class vehicles, model2 = 6-class animals
    m1_train = build_diffusion_augmented_loader(
        DATA_DIR,
        DIFF_DIR,
        vehicle_classes,
        args.batch_size,
        train=True,
        diff_fraction=args.diff_fraction,
    )
    m1_test = build_diffusion_augmented_loader(
        DATA_DIR,
        DIFF_DIR,
        vehicle_classes,
        args.batch_size,
        train=False,
        diff_fraction=args.diff_fraction,
    )
    m2_train = build_diffusion_augmented_loader(
        DATA_DIR,
        DIFF_DIR,
        animal_classes,
        args.batch_size,
        train=True,
        diff_fraction=args.diff_fraction,
    )
    m2_test = build_diffusion_augmented_loader(
        DATA_DIR,
        DIFF_DIR,
        animal_classes,
        args.batch_size,
        train=False,
        diff_fraction=args.diff_fraction,
    )

    # ---------------- Train submodels ----------------
    # ----------------- Train or Load Submodels -----------------
    if not args.skip_submodels:
        logger.log(f"Training M1 (WRN-28-10, 4-class vehicles)")
        m1 = build_wrn_28_10(num_classes=len(vehicle_classes))
        train_ce(
            m1, m1_train, m1_test, args.epochs_m, args.lr_m, logger, "[M1]", args=args
        )
        torch.save(
            {"model_state_dict": m1.state_dict()}, os.path.join(LOG_DIR, "M1_WRN.pt")
        )

        logger.log(f"Training M2 (WRN-28-10, 6-class animals)")
        m2 = build_wrn_28_10(num_classes=len(animal_classes))
        train_ce(
            m2, m2_train, m2_test, args.epochs_m, args.lr_m, logger, "[M2]", args=args
        )
        torch.save(
            {"model_state_dict": m2.state_dict()}, os.path.join(LOG_DIR, "M2_WRN.pt")
        )
    else:
        logger.log("🟢 Skipping submodel training, loading pretrained weights...")
        base_dir = os.path.join(args.log_dir, "wrn28x10_diff70")
        m1_path = os.path.join(base_dir, "M1_WRN.pt")
        m2_path = os.path.join(base_dir, "M2_WRN.pt")
        m1 = build_wrn_28_10(num_classes=len(vehicle_classes))
        m2 = build_wrn_28_10(num_classes=len(animal_classes))
        m1.load_state_dict(torch.load(m1_path)["model_state_dict"])
        m2.load_state_dict(torch.load(m2_path)["model_state_dict"])
        logger.log("✅ Loaded M1_WRN.pt and M2_WRN.pt successfully.")

    a_acc = eval_clean(m1, m1_test)
    v_acc = eval_clean(m2, m2_test)
    logger.log(f"[M1-4class] Clean Test Acc: {a_acc:.4f}")
    logger.log(f"[M2-6class] Clean Test Acc: {v_acc:.4f}")

    # ---------------- Fusion training ----------------
    in_dim = int(m1.fc.in_features + m2.fc.in_features)
    head = WRNHead(in_dim, num_classes=10, p_drop=0.2).to(DEVICE)
    fusion = FusionWRN(m1, m2, head).to(DEVICE)

    logger.log("Starting fusion training (TRADES)...")
    train_fusion(fusion, full_train, full_test, args, logger)

    atk = make_eval_attack(fusion, args)
    clean = eval_clean(fusion, full_test)
    adv = eval_adv(fusion, full_test, atk)
    logger.log(f"[WRN-Fusion] Final Clean {clean:.4f} | Adv {adv:.4f}")

    torch.save(
        {"model_state_dict": m1.state_dict()}, os.path.join(LOG_DIR, "M1_WRN.pt")
    )
    torch.save(
        {"model_state_dict": m2.state_dict()}, os.path.join(LOG_DIR, "M2_WRN.pt")
    )
    torch.save(
        {"model_state_dict": fusion.state_dict()},
        os.path.join(LOG_DIR, "Fusion_WRN.pt"),
    )
    logger.log(f"Saved models to {LOG_DIR}")


if __name__ == "__main__":
    main()
