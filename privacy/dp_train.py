"""
Training script for DP-SGD with hierarchical (4-class + 6-class fusion) vs baseline (10-class).
Implements per-sample gradient computation with vectorized approach.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch import optim

from privacy.dp_utils import (
    compute_accuracy,
    compute_epsilon_opacus,
    solve_noise_from_epsilon_opacus,
    DataProcessor,
    dp_step_images,
)

from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel


# ------------------------------
# Helper: remap targets to match subset labels
# ------------------------------
def remap_targets(y, dataset):
    """
    Ensure that labels match the number of classes in the current model.
    If dataset has a custom .targets attribute (set in dp_utils), use it.
    """
    if hasattr(dataset, "targets"):
        # if Subset, get targets from .dataset
        if isinstance(dataset.targets, torch.Tensor):
            return dataset.targets[y]
        elif isinstance(dataset.targets, list):
            return torch.tensor(dataset.targets)[y]
    return y


# ------------------------------
# Training loop with per-sample DP-SGD
# ------------------------------
def train_dp(model, train_loader, test_loader, epochs, lr, noise_multiplier, max_grad_norm, delta, device):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    steps = 0
    eps = 0.0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            # remap labels if needed
            y = remap_targets(y, train_loader.dataset).to(device)

            optimizer.zero_grad()

            # DP-SGD step
            preclip_norm = dp_step_images(model, optimizer, x, y, noise_multiplier, max_grad_norm)

            steps += 1
            eps = compute_epsilon_opacus(
                noise_multiplier,
                train_loader.batch_size / len(train_loader.dataset),
                steps,
                delta,
            )

            if batch_idx % 50 == 0:
                loss_val = F.cross_entropy(model(x)[0], y).item()
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Loss {loss_val:.4f}, PreClip||g|| {preclip_norm:.2f}, "
                      f"ε={eps:.3f}, δ={delta:.1e}")

        # Epoch end: compute accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, "
              f"Privacy ε={eps:.3f}, δ={delta:.1e}")

    return train_acc, eps, delta


# ------------------------------
# Main function
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_all")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs_4class", type=int, default=50)
    parser.add_argument("--epochs_6class", type=int, default=50)
    parser.add_argument("--epochs_10class", type=int, default=50)
    parser.add_argument("--epochs_fusion", type=int, default=30)
    parser.add_argument("--noise_multiplier", type=float, default=1.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--train_4class", action="store_true")
    parser.add_argument("--train_6class", action="store_true")
    parser.add_argument("--train_10class", action="store_true")
    parser.add_argument("--train_fusion", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("PER-SAMPLE DP-SGD TRAINING (hierarchical vs baseline)")
    print("============================================================")
    print(f"Device: {device}")
    print(f"DP params: sigma={args.noise_multiplier}, C={args.max_grad_norm}, delta={args.delta}")
    print(f"LR={args.lr}, batch_size={args.batch_size}")

    # Load CIFAR-10 datasets
    loaders = DataProcessor.create_data_loaders(args.data_dir, batch_size=args.batch_size)
    m4_train, m4_test = loaders["vehicle_train"], loaders["vehicle_test"]
    m6_train, m6_test = loaders["animal_train"], loaders["animal_test"]
    m10_train, m10_test = loaders["full_train"], loaders["full_test"]

    # ---------------- training ----------------
    if args.train_all or args.train_4class:
        print("\nTraining 4class with DP-SGD...")
        model4 = DP4Classifier()
        train_dp(model4, m4_train, m4_test, args.epochs_4class, args.lr,
                 args.noise_multiplier, args.max_grad_norm, args.delta, device)

    if args.train_all or args.train_6class:
        print("\nTraining 6class with DP-SGD...")
        model6 = DP6Classifier()
        train_dp(model6, m6_train, m6_test, args.epochs_6class, args.lr,
                 args.noise_multiplier, args.max_grad_norm, args.delta, device)

    if args.train_all or args.train_10class:
        print("\nTraining 10class (baseline) with DP-SGD...")
        model10 = DP10Classifier()
        train_dp(model10, m10_train, m10_test, args.epochs_10class, args.lr,
                 args.noise_multiplier, args.max_grad_norm, args.delta, device)

    if args.train_all or args.train_fusion:
        print("\nTraining Fusion model (4class + 6class embeddings)...")
        model4 = DP4Classifier().to(device)
        model6 = DP6Classifier().to(device)
        fusion = DPFusionModel()

        # Freeze sub-models
        model4.eval()
        model6.eval()
        fusion.to(device)
        optimizer = optim.SGD(fusion.parameters(), lr=args.lr, momentum=0.9)

        for epoch in range(args.epochs_fusion):
            fusion.train()
            for (x4, y4), (x6, y6) in zip(m4_train, m6_train):
                x4, y4 = x4.to(device), remap_targets(y4, m4_train.dataset).to(device)
                x6, y6 = x6.to(device), remap_targets(y6, m6_train.dataset).to(device)

                with torch.no_grad():
                    _, emb4 = model4(x4)
                    _, emb6 = model6(x6)

                optimizer.zero_grad()
                logits = fusion(emb4, emb6)
                loss = F.cross_entropy(logits, y4[:logits.size(0)])
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: Fusion Loss {loss.item():.4f}")

        print("Fusion training done.")


if __name__ == "__main__":
    main()
