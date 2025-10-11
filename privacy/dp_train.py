"""
Training script for DP-SGD with hierarchical (4-class + 6-class fusion) vs baseline (10-class).
Implements per-sample gradient computation with DP noise.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch import optim
from opacus.grad_sample import GradSampleModule  # ✅ 新增：用于逐样本梯度

from privacy.dp_utils import (
    compute_accuracy,
    compute_epsilon_opacus,
    DataProcessor,
    dp_step_images,
)
from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel


# ------------------------------
# Label remapping helper (安全版)
# ------------------------------
def remap_targets(y, dataset, device):
    """
    仅当 dataset 显式提供 class_map: dict{old_label->new_label} 时才做映射；
    否则直接返回原始 y，避免把标签当索引误用。
    """
    class_map = getattr(dataset, "class_map", None)
    if class_map is not None:
        y_cpu = y.detach().cpu().tolist()
        y_new = torch.tensor([class_map.get(int(v), int(v)) for v in y_cpu], dtype=torch.long)
        return y_new.to(device)
    return y.to(device)


# ------------------------------
# Training loop with per-sample DP-SGD
# ------------------------------
def train_dp(model, train_loader, test_loader, epochs, lr,
             noise_multiplier, max_grad_norm, delta, device):
    # ✅ 关键修复：包一层，才能得到 per-sample grads (.grad_sample)
    model = GradSampleModule(model).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    steps = 0
    eps = 0.0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # remap targets（仅在提供 class_map 时生效）
            y = remap_targets(y, train_loader.dataset, device)

            optimizer.zero_grad()

            # ✅ DP-SGD（逐样本裁剪逻辑在 dp_utils.dp_step_images 内）
            preclip_norm = dp_step_images(model, optimizer, x, y, noise_multiplier, max_grad_norm)

            steps += 1
            eps = compute_epsilon_opacus(
                noise_multiplier,
                train_loader.batch_size / len(train_loader.dataset),
                steps,
                delta,
            )

            if batch_idx % 50 == 0:
                # 维持你原有的日志格式
                logits, _ = model(x)
                loss_val = F.cross_entropy(logits, y).item()
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss_val:.4f}, "
                      f"PreClip||g|| {preclip_norm:.2f}, ε={eps:.3f}, δ={delta:.1e}")

        # Epoch end: accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)
        print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, "
              f"Privacy ε={eps:.3f}, δ={delta:.1e}")

    return train_acc, eps, delta


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results_all")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01")
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

    # Data
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

        model4.eval()
        model6.eval()
        fusion.to(device)
        optimizer = optim.SGD(fusion.parameters(), lr=args.lr, momentum=0.9)

        for epoch in range(args.epochs_fusion):
            fusion.train()
            for (x4, y4), (x6, y6) in zip(m4_train, m6_train):
                x4, y4 = x4.to(device), y4.to(device)
                x6, y6 = x6.to(device), y6.to(device)

                # 只有在提供 class_map 时才 remap
                y4 = remap_targets(y4, m4_train.dataset, device)
                y6 = remap_targets(y6, m6_train.dataset, device)

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
