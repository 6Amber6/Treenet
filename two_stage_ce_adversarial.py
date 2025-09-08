import os
import sys
import copy
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

# Ensure subpackage 'core' is importable when models/treeresnet references it
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARP_ROOT = os.path.join(PROJECT_ROOT, 'adversarial_robustness_pytorch')
if ARP_ROOT not in sys.path:
    sys.path.insert(0, ARP_ROOT)

from core.models.resnet import LightResnet, BasicBlock
from core.attacks import create_attack
from core.utils.context import ctx_noparamgrad_and_eval
from core.metrics import accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class ids
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
VEHICLE_CLASSES = [0, 1, 8, 9]       # airplane, automobile, ship, truck


def build_lightresnet20(num_classes: int) -> LightResnet:
    model = LightResnet(BasicBlock, [2, 2, 2], num_classes=num_classes, device=DEVICE)
    return model.to(DEVICE)


def get_cifar10_dataloaders(data_dir: str, batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def filter_and_remap_indices(dataset: torchvision.datasets.CIFAR10, keep_labels: List[int]) -> Tuple[List[int], dict]:
    indices = [i for i, (_, y) in enumerate(dataset) if y in keep_labels]
    remap = {old: new for new, old in enumerate(keep_labels)}
    return indices, remap


class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, base: torchvision.datasets.CIFAR10, indices: List[int], remap: dict):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        y = self.remap[int(y)]
        return x, y


def build_filtered_loaders(data_dir: str, keep_labels: List[int], batch_size: int = 256, train: bool = True) -> DataLoader:
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ]) if train else T.Compose([T.ToTensor()])

    ds = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    indices, remap = filter_and_remap_indices(ds, keep_labels)
    sub = RemappedSubset(ds, indices, remap)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)
    return loader


def trades_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0):
    """TRADES adversarial training loss"""
    model.eval()
    batch_size = len(x_natural)
    
    # Generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(DEVICE).detach()
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                              F.softmax(model(x_natural), dim=1),
                              reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = x_adv.detach()

    # Calculate TRADES loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    
    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                          F.softmax(logits_natural, dim=1),
                          reduction='batchmean')
    
    loss = loss_natural + beta * loss_robust
    return loss


def train_classifier_adversarial(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                                epochs: int = 50, lr: float = 0.1, beta: float = 6.0) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)

    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # TRADES adversarial training
            loss = trades_loss(model, x, y, optimizer, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            with torch.no_grad():
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            clean_acc, adv_acc = eval_classifier_adversarial(model, test_loader)
            print(f"Epoch {epoch:03d} | Train Loss {(running_loss/total):.4f} | Clean Acc {clean_acc:.4f} | Adv Acc {adv_acc:.4f}")


def eval_classifier_adversarial(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    
    # Clean accuracy
    correct_clean, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct_clean += (preds == y).sum().item()
            total += y.size(0)
    clean_acc = correct_clean / max(total, 1)
    
    # Adversarial accuracy with PGD attack
    attack = create_attack(model, nn.CrossEntropyLoss(), 'linf-pgd', 8/255, 20, 2/255)
    correct_adv, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with ctx_noparamgrad_and_eval(model):
            x_adv, _ = attack.perturb(x, y)
        with torch.no_grad():
            logits = model(x_adv)
            preds = logits.argmax(dim=1)
            correct_adv += (preds == y).sum().item()
            total += y.size(0)
    adv_acc = correct_adv / max(total, 1)
    
    return clean_acc, adv_acc


def register_penultimate_hook(model: LightResnet):
    """Register a forward hook on fc to capture its input (penultimate features)."""
    feats = {}

    def hook_fn(module, input, output):
        feats["feat"] = input[0].detach()

    handle = model.fc.register_forward_hook(hook_fn)
    return feats, handle


@torch.no_grad()
def extract_embeddings(model_m1: nn.Module, model_m2: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    model_m1.eval()
    model_m2.eval()

    feats_m1, h1 = register_penultimate_hook(model_m1)
    feats_m2, h2 = register_penultimate_hook(model_m2)

    all_feats, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        _ = model_m1(x)  # fills feats_m1["feat"]
        f1 = feats_m1["feat"].cpu()
        _ = model_m2(x)  # fills feats_m2["feat"]
        f2 = feats_m2["feat"].cpu()
        all_feats.append(torch.cat([f1, f2], dim=1))
        all_labels.append(y)

    h1.remove()
    h2.remove()
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


class HeadG(nn.Module):
    def __init__(self, in_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head_g_adversarial(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,
                            epochs: int = 50, lr: float = 1e-3, batch_size: int = 512, beta: float = 6.0) -> HeadG:
    model_g = HeadG(in_dim=X_train.size(1), num_classes=10).to(DEVICE)
    opt = torch.optim.Adam(model_g.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        model_g.train()
        total, correct, run_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # TRADES adversarial training for embeddings
            loss = trades_loss(model_g, xb, yb, opt, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            run_loss += loss.item() * xb.size(0)
            with torch.no_grad():
                logits = model_g(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        if epoch % 5 == 0 or epoch == 1:
            clean_acc, adv_acc = eval_classifier_adversarial(model_g, test_loader)
            print(f"[G] Epoch {epoch:03d} | Train Loss {(run_loss/total):.4f} | Clean Acc {clean_acc:.4f} | Adv Acc {adv_acc:.4f}")

    return model_g


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Two-stage CE optimization with adversarial training")
    parser.add_argument('--data-dir', type=str, default='./data', help='CIFAR-10 data directory')
    parser.add_argument('--epochs-m', type=int, default=50, help='epochs for M1/M2 training')
    parser.add_argument('--epochs-g', type=int, default=50, help='epochs for G training')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--save-dir', type=str, default='./log_ce_optimization_two_stage_adv')
    parser.add_argument('--beta', type=float, default=6.0, help='TRADES beta parameter')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Build loaders
    print('Preparing data ...')
    # Full loaders for embedding extraction and G training
    full_train_loader, full_test_loader = get_cifar10_dataloaders(args.data_dir, batch_size=args.batch_size)
    # Filtered loaders for M1 (6-class animal) and M2 (4-class vehicle)
    m1_train_loader = build_filtered_loaders(args.data_dir, ANIMAL_CLASSES, batch_size=args.batch_size, train=True)
    m1_test_loader = build_filtered_loaders(args.data_dir, ANIMAL_CLASSES, batch_size=args.batch_size, train=False)
    m2_train_loader = build_filtered_loaders(args.data_dir, VEHICLE_CLASSES, batch_size=args.batch_size, train=True)
    m2_test_loader = build_filtered_loaders(args.data_dir, VEHICLE_CLASSES, batch_size=args.batch_size, train=False)

    # Train M1 (6-class animal) with adversarial training
    print('Training M1 (6-class animal) with TRADES ...')
    m1 = build_lightresnet20(num_classes=len(ANIMAL_CLASSES))
    train_classifier_adversarial(m1, m1_train_loader, m1_test_loader, epochs=args.epochs_m, beta=args.beta)
    torch.save({'model_state_dict': m1.state_dict()}, os.path.join(args.save_dir, 'M1_animal_6cls_adv.pt'))

    # Train M2 (4-class vehicle) with adversarial training
    print('Training M2 (4-class vehicle) with TRADES ...')
    m2 = build_lightresnet20(num_classes=len(VEHICLE_CLASSES))
    train_classifier_adversarial(m2, m2_train_loader, m2_test_loader, epochs=args.epochs_m, beta=args.beta)
    torch.save({'model_state_dict': m2.state_dict()}, os.path.join(args.save_dir, 'M2_vehicle_4cls_adv.pt'))

    # Extract embeddings for full train/test sets
    print('Extracting embeddings for G ...')
    X_train, y_train = extract_embeddings(m1, m2, full_train_loader)
    X_test, y_test = extract_embeddings(m1, m2, full_test_loader)
    # Move to device for training head G
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # Train head G with adversarial training
    print('Training G (10-class) with TRADES ...')
    model_g = train_head_g_adversarial(X_train, y_train, X_test, y_test, epochs=args.epochs_g, beta=args.beta)
    torch.save({'model_state_dict': model_g.state_dict()}, os.path.join(args.save_dir, 'G_head_10cls_adv.pt'))

    # Final evaluation
    print('\n=== Final Evaluation ===')
    print('M1 (Animal 6-class):')
    clean_acc_m1, adv_acc_m1 = eval_classifier_adversarial(m1, m1_test_loader)
    print(f'  Clean Acc: {clean_acc_m1:.4f} | Adv Acc: {adv_acc_m1:.4f}')
    
    print('M2 (Vehicle 4-class):')
    clean_acc_m2, adv_acc_m2 = eval_classifier_adversarial(m2, m2_test_loader)
    print(f'  Clean Acc: {clean_acc_m2:.4f} | Adv Acc: {adv_acc_m2:.4f}')
    
    print('G (10-class ensemble):')
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader_g = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    clean_acc_g, adv_acc_g = eval_classifier_adversarial(model_g, test_loader_g)
    print(f'  Clean Acc: {clean_acc_g:.4f} | Adv Acc: {adv_acc_g:.4f}')

    print(f'\nDone. Saved models to {args.save_dir}')


if __name__ == '__main__':
    main()
