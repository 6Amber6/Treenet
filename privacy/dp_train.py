"""
Per-sample DP-SGD training with vectorized per-sample gradients (torch.func.vmap)
- Compare: 4-class (vehicles), 6-class (animals), 10-class baseline, and Fusion
"""

import os, sys, json, math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel
from privacy.dp_utils import DPOptimizer, compute_accuracy, compute_epsilon_opacus, solve_noise_from_epsilon_opacus

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                    pin_memory=torch.cuda.is_available())
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=torch.cuda.is_available())
    return tr, te


def get_filtered_loaders(data_dir: str, keep_labels: List[int], batch_size: int, num_workers: int):
    full_train, full_test = get_cifar10_loaders(data_dir, batch_size, num_workers)

    def subset(ds, keep):
        idx = []
        for i in range(len(ds)):
            _, y = ds[i]
            if int(y) in keep:
                idx.append(i)
        return torch.utils.data.Subset(ds, idx)

    tr_sub = subset(full_train.dataset, keep_labels)
    te_sub = subset(full_test.dataset,  keep_labels)

    remap = {old: new for new, old in enumerate(keep_labels)}

    class RemapDS(torch.utils.data.Dataset):
        def __init__(self, base, remap): self.base, self.remap = base, remap
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, int(self.remap[int(y)])

    tr = RemapDS(tr_sub, remap)
    te = RemapDS(te_sub, remap)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                           pin_memory=torch.cuda.is_available())
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=torch.cuda.is_available())
    return tr_loader, te_loader


def train_dp(model: nn.Module,
             train_loader: DataLoader,
             test_loader: DataLoader,
             epochs: int,
             lr: float,
             noise_multiplier: float,
             max_grad_norm: float,
             delta: float,
             model_name: str,
             outdir: str):

    print(f"\nTraining {model_name} with vectorized per-sample DP-SGD...")
    base_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    dp_opt = DPOptimizer(model, base_opt, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)

    best = 0.0
    steps = 0
    model.train()

    for ep in range(epochs):
        correct, total = 0, 0
        for b, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # forward once for metrics (not used for grads)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss_mean = F.cross_entropy(logits, y, reduction='mean').item()

            # DP-SGD step (vectorized per-sample grads)
            dp_opt.zero_grad()
            pre = dp_opt.dp_step_images(x, y)
            steps += 1

            with torch.no_grad():
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

            if b % 50 == 0:
                q = (train_loader.batch_size or 1) / max(1, len(train_loader.dataset))
                eps = compute_epsilon_opacus(noise_multiplier, q, steps, delta)
                print(f"Epoch {ep}, Batch {b}, Loss(mean) {loss_mean:.4f}, PreClip||g|| {pre:.2f}, "
                      f"ε={eps:.3f}, δ={delta:.1e}")

        train_acc = correct / max(1, total)
        test_acc  = compute_accuracy(model, test_loader, DEVICE)
        q = (train_loader.batch_size or 1) / max(1, len(train_loader.dataset))
        eps = compute_epsilon_opacus(noise_multiplier, q, steps, delta)
        print(f"Epoch {ep}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, Privacy ε={eps:.3f}, δ={delta:.1e}")

        if test_acc > best:
            best = test_acc
            torch.save(model.state_dict(), os.path.join(outdir, f"{model_name}_best.pth"))

    torch.save(model.state_dict(), os.path.join(outdir, f"{model_name}_final.pth"))
    return best, eps, delta


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader):
    model.eval()
    E, Y = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        emb = out[1] if isinstance(out, tuple) else out
        E.append(emb.detach().cpu())
        Y.append(y.clone())
    return torch.cat(E, 0), torch.cat(Y, 0)


def train_fusion(model4, model6, full_train, full_test,
                 epochs, lr, noise_multiplier, max_grad_norm, delta, outdir):
    print("\nTraining FUSION with vectorized per-sample DP-SGD...")

    emb4, y4 = extract_embeddings(model4, full_train)
    emb6, y6 = extract_embeddings(model6, full_train)
    N = min(len(emb4), len(emb6), len(y4), len(y6))
    emb4, emb6, labels = emb4[:N], emb6[:N], y4[:N]

    class FuseDS(torch.utils.data.Dataset):
        def __len__(self): return N
        def __getitem__(self, i): return emb4[i], emb6[i], labels[i]

    fds = FuseDS()
    floader = DataLoader(fds, batch_size=full_train.batch_size, shuffle=True, num_workers=0)

    fusion = DPFusionModel(embedding_dim=emb4.shape[1], num_classes=10, groups=8).to(DEVICE)
    base_opt = torch.optim.SGD(fusion.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    dp_opt = DPOptimizer(fusion, base_opt, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)

    best = 0.0
    steps = 0
    fusion.train()

    for ep in range(epochs):
        correct, total = 0, 0
        for b, (e4, e6, y) in enumerate(floader):
            e4, e6, y = e4.to(DEVICE), e6.to(DEVICE), y.to(DEVICE)

            # metrics forward
            logits = fusion(e4, e6)
            loss_mean = F.cross_entropy(logits, y, reduction='mean').item()

            # DP-SGD step
            dp_opt.zero_grad()
            pre = dp_opt.dp_step_fusion(e4, e6, y)
            steps += 1

            with torch.no_grad():
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()

            if b % 50 == 0:
                q = (floader.batch_size or 1) / max(1, len(fds))
                eps = compute_epsilon_opacus(noise_multiplier, q, steps, delta)
                print(f"[Fusion] Epoch {ep}, Batch {b}, Loss(mean) {loss_mean:.4f}, "
                      f"PreClip||g|| {pre:.2f}, ε={eps:.3f}")

        train_acc = correct / max(1, total)
        test_acc  = compute_accuracy(fusion, full_test, DEVICE)
        q = (floader.batch_size or 1) / max(1, len(fds))
        eps = compute_epsilon_opacus(noise_multiplier, q, steps, delta)
        print(f"[Fusion] Epoch {ep}: Train {train_acc:.4f}, Test {test_acc:.4f}, ε={eps:.3f}")

        if test_acc > best:
            best = test_acc
            torch.save(fusion.state_dict(), os.path.join(outdir, "fusion_best.pth"))

    torch.save(fusion.state_dict(), os.path.join(outdir, "fusion_final.pth"))
    return best, eps, delta


def estimate_steps(epochs: int, n: int, b: int) -> int:
    return int(epochs * math.ceil(n / max(1, b)))


def main():
    import argparse
    parser = argparse.ArgumentParser("Vectorized per-sample DP-SGD (hierarchical vs baseline)")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results_all')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--epochs_4class', type=int, default=50)
    parser.add_argument('--epochs_6class', type=int, default=50)
    parser.add_argument('--epochs_10class', type=int, default=50)
    parser.add_argument('--epochs_fusion', type=int, default=30)

    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--noise_multiplier', type=float, default=1.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-5)

    parser.add_argument('--train_4class', action='store_true')
    parser.add_argument('--train_6class', action='store_true')
    parser.add_argument('--train_10class', action='store_true')
    parser.add_argument('--train_fusion', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    animals  = [2,3,4,5,6,7]
    vehicles = [0,1,8,9]

    print("="*60)
    print("PER-SAMPLE DP-SGD TRAINING (hierarchical vs baseline)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"DP params: sigma={args.noise_multiplier}, C={args.max_grad_norm}, delta={args.delta}")
    print(f"LR={args.lr}, batch_size={args.batch_size}")

    full_train, full_test = get_cifar10_loaders(args.data_dir, args.batch_size, args.num_workers)
    m6_train, m6_test = get_filtered_loaders(args.data_dir, animals,  args.batch_size, args.num_workers)
    m4_train, m4_test = get_filtered_loaders(args.data_dir, vehicles, args.batch_size, args.num_workers)

    def q_steps(loader, epochs):
        n = len(loader.dataset); b = loader.batch_size or args.batch_size
        return b / max(1, n), estimate_steps(epochs, n, b)

    def sigma_for(loader, epochs):
        if args.epsilon is None:
            return args.noise_multiplier
        q, steps = q_steps(loader, epochs)
        s = solve_noise_from_epsilon_opacus(args.epsilon, q, steps, args.delta)
        print(f"solve sigma: ε={args.epsilon}, q={q:.6f}, steps={steps} -> σ={s:.4f}")
        return s

    results: Dict[str, Dict[str, float]] = {}
    models: Dict[str, nn.Module] = {}

    # 4-class (Vehicles)
    if args.train_all or args.train_4class:
        model4 = DP4Classifier(groups=8).to(DEVICE)
        sigma4 = sigma_for(m4_train, args.epochs_4class)
        acc, eps, delt = train_dp(model4, m4_train, m4_test, args.epochs_4class, args.lr,
                                  sigma4, args.max_grad_norm, args.delta, "4class", args.output_dir)
        results["4class"] = {"accuracy": acc, "epsilon": eps, "delta": delt}
        models["4class"] = model4

    # 6-class (Animals)
    if args.train_all or args.train_6class:
        model6 = DP6Classifier(groups=8).to(DEVICE)
        sigma6 = sigma_for(m6_train, args.epochs_6class)
        acc, eps, delt = train_dp(model6, m6_train, m6_test, args.epochs_6class, args.lr,
                                  sigma6, args.max_grad_norm, args.delta, "6class", args.output_dir)
        results["6class"] = {"accuracy": acc, "epsilon": eps, "delta": delt}
        models["6class"] = model6

    # 10-class baseline
    if args.train_all or args.train_10class:
        model10 = DP10Classifier(groups=8).to(DEVICE)
        sigma10 = sigma_for(full_train, args.epochs_10class)
        acc, eps, delt = train_dp(model10, full_train, full_test, args.epochs_10class, args.lr,
                                  sigma10, args.max_grad_norm, args.delta, "10class", args.output_dir)
        results["10class"] = {"accuracy": acc, "epsilon": eps, "delta": delt}

    # Fusion
    if args.train_all or args.train_fusion:
        if "4class" in models and "6class" in models:
            sigmaF = sigma_for(full_train, args.epochs_fusion)
            acc, eps, delt = train_fusion(models["4class"], models["6class"], full_train, full_test,
                                          args.epochs_fusion, args.lr, sigmaF, args.max_grad_norm, args.delta,
                                          args.output_dir)
            results["fusion"] = {"accuracy": acc, "epsilon": eps, "delta": delt}
        else:
            print("Warning: need both 4/6-class models before fusion; skipping")

    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k:8s} | Acc={v['accuracy']:.4f} | ε={v['epsilon']:.3f} | δ={v['delta']:.2e}")


if __name__ == "__main__":
    main()
