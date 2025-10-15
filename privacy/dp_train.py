# # dp_train.py
# # -*- coding: utf-8 -*-
# """
# End-to-end DP-SGD training script:

# 1) Train 4-class (vehicles) with DP-SGD for T1 iterations
# 2) Train 6-class (animals)  with DP-SGD for T1 iterations  (T1 = T2)
# 3) Freeze 4/6 backbones, build Fusion10 head; DP-SGD on full CIFAR-10 for T3 iterations

# Single sigma is computed once from (epsilon, delta, q, T1+T3) and
# shared across all phases, matching the teacher's requirement.
# Expected-batch-size normalization is used throughout.

# You can also train a 10-class baseline by setting --baseline_only.
# """

# import argparse
# from typing import Dict
# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn

# from dp_models import Classifier4, Classifier6, Classifier10, Fusion10
# from dp_utils import dp_step_images, compute_accuracy, build_split_loaders, estimate_sigma


# def _cycle(loader: DataLoader):
#     while True:
#         for batch in loader:
#             yield batch


# def train_dp(model: nn.Module,
#              loader: DataLoader,
#              test_loader: DataLoader,
#              steps: int,
#              lr: float,
#              sigma: float,
#              C: float,
#              device: torch.device,
#              expected_batchsize: int,
#              train_head_only: bool = False) -> nn.Module:
#     model = model.to(device)
#     if train_head_only and hasattr(model, "trainable_parameters"):
#         params = model.trainable_parameters()
#     else:
#         params = (p for p in model.parameters() if p.requires_grad)

#     optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
#     it = _cycle(loader)

#     for step in range(1, steps + 1):
#         x, y = next(it)
#         x, y = x.to(device), y.to(device)
#         dp_step_images(model, optimizer, x, y, sigma, C, expected_batchsize)
#         if step % 100 == 0:
#             acc = compute_accuracy(model, test_loader, device)
#             print(f"[DP] Step {step}/{steps} | Test Acc = {acc*100:.2f}%")
#     return model


# def main():
#     parser = argparse.ArgumentParser()

#     # privacy / schedule
#     parser.add_argument("--sampling_rate", type=float, default=0.10, help="q, Poisson subsampling rate (constant across phases)")
#     parser.add_argument("--epsilon", type=float, default=6.0)
#     parser.add_argument("--delta", type=float, default=1e-5)
#     parser.add_argument("--T1", type=int, default=1500, help="Iterations for 4- and 6-class")
#     parser.add_argument("--T3", type=int, default=1500, help="Iterations for Fusion10 or Baseline10")

#     # training
#     parser.add_argument("--data_dir", type=str, default="./data")
#     parser.add_argument("--lr", type=float, default=0.05)
#     parser.add_argument("--max_grad_norm", type=float, default=1.0)
#     parser.add_argument("--num_workers", type=int, default=2)
#     parser.add_argument("--seed", type=int, default=1)
#     parser.add_argument("--batchsize_full", type=int, default=None, help="override b10 if desired")

#     # modes
#     parser.add_argument("--train_all", action="store_true", help="Train 4,6, then Fusion10")
#     parser.add_argument("--baseline_only", action="store_true", help="Train 10-class baseline only")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     # loaders with q-based batch sizes per subset
#     loaders = build_split_loaders(q=args.sampling_rate,
#                                   data_dir=args.data_dir,
#                                   batchsize_full=args.batchsize_full,
#                                   num_workers=args.num_workers,
#                                   seed=args.seed)
#     b4, b6, b10 = loaders["b4"], loaders["b6"], loaders["b10"]
#     n4, n6, n10 = loaders["n4"], loaders["n6"], loaders["n10"]
#     print(f"q={args.sampling_rate:.4f} | b4={b4}, b6={b6}, b10={b10} | n4={n4}, n6={n6}, n10={n10}")

#     # single sigma for (q, T1+T3, epsilon, delta)
#     total_steps = args.T1 + args.T3
#     sigma = estimate_sigma(epsilon=args.epsilon, delta=args.delta, q=args.sampling_rate, total_steps=total_steps)
#     print(f"Computed sigma={sigma:.4f} for eps={args.epsilon}, delta={args.delta}, q={args.sampling_rate}, total_steps={total_steps}")

#     # training plan
#     if args.baseline_only:
#         print("\n[Training 10-class Baseline with DP-SGD]")
#         model10 = Classifier10().to(device)
#         model10 = train_dp(model10, loaders["full_train"], loaders["full_test"],
#                            args.T3, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b10)
#         acc10 = compute_accuracy(model10, loaders["full_test"], device)
#         print(f"[Baseline10] Final Test Acc = {acc10*100:.2f}%")
#         return

#     # default: train all phases
#     print("\n[Training 4-class (vehicles)]")
#     model4 = Classifier4().to(device)
#     model4 = train_dp(model4, loaders["vehicle_train"], loaders["vehicle_test"],
#                       args.T1, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b4)

#     print("\n[Training 6-class (animals)]")
#     model6 = Classifier6().to(device)
#     model6 = train_dp(model6, loaders["animal_train"], loaders["animal_test"],
#                       args.T1, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b6)

#     print("\n[Training Fused 10-class Head with DP-SGD]")
#     fusion = Fusion10(model4, model6, hidden=128).to(device)
#     # only train the head under DP
#     fusion = train_dp(fusion, loaders["full_train"], loaders["full_test"],
#                       args.T3, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b10,
#                       train_head_only=True)

#     acc_fused = compute_accuracy(fusion, loaders["full_test"], device)
#     print(f"[Fusion10] Final Test Acc = {acc_fused*100:.2f}%")

#     # Optional: also train a baseline for comparison in the same run
#     print("\n[Training 10-class Baseline with DP-SGD for Comparison]")
#     model10 = Classifier10().to(device)
#     model10 = train_dp(model10, loaders["full_train"], loaders["full_test"],
#                        args.T3, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b10)
#     acc10 = compute_accuracy(model10, loaders["full_test"], device)
#     print(f"[Baseline10] Final Test Acc = {acc10*100:.2f}%")

#     if acc_fused >= acc10:
#         print("[Result] Fusion10 >= Baseline10 ✅")
#     else:
#         print("[Result] Fusion10 < Baseline10 ⚠️  Consider tuning lr/T1/T3 or increasing Fusion head width.")

# if __name__ == "__main__":
#     main()



# dp_train.py
# -*- coding: utf-8 -*-
import argparse
import torch
from torch.utils.data import DataLoader

from dp_models import MultiHeadShared, Baseline10
from dp_utils import dp_step_images, compute_accuracy, build_split_loaders, estimate_sigma

def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

def train_phase(model, phase, loader, test_loader, steps, lr, sigma, C, device, expected_batchsize, log_prefix):
    # 设置阶段 & 可训练参数
    model.set_phase(phase)
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 0, f"No trainable params in phase={phase}"
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    it = _cycle(loader)

    for step in range(1, steps + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        # 直接用 model.forward（已按 phase 路由好）
        dp_step_images(model, optimizer, x, y, sigma, C, expected_batchsize)

        if step % 100 == 0:
            acc = compute_accuracy(model, test_loader, device)
            print(f"{log_prefix} Step {step}/{steps} | Test Acc = {acc*100:.2f}%")
    return model

def train_baseline10(steps, lr, sigma, C, device, loaders):
    model = Baseline10().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    it = _cycle(loaders["full_train"])
    for step in range(1, steps + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        dp_step_images(model, optimizer, x, y, sigma, C, loaders["b10"])
        if step % 100 == 0:
            acc = compute_accuracy(model, loaders["full_test"], device)
            print(f"[DP] Step {step}/{steps} | Test Acc = {acc*100:.2f}%")
    acc = compute_accuracy(model, loaders["full_test"], device)
    print(f"[Baseline10] Final Test Acc = {acc*100:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_rate", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=6.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--T1", type=int, default=1000)
    parser.add_argument("--T3", type=int, default=1000)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lr_fusion", type=float, default=0.5, help="LR for fusion head phase only")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batchsize_full", type=int, default=None)
    parser.add_argument("--train_all", action="store_true", help="Run 4->6->Fusion10 pipeline")
    parser.add_argument("--baseline_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_split_loaders(
        q=args.sampling_rate,
        data_dir=args.data_dir,
        batchsize_full=args.batchsize_full,
        num_workers=args.num_workers,
        seed=args.seed
    )
    b4, b6, b10 = loaders["b4"], loaders["b6"], loaders["b10"]
    n4, n6, n10 = loaders["n4"], loaders["n6"], loaders["n10"]
    print(f"q={args.sampling_rate:.4f} | b4={b4}, b6={b6}, b10={b10} | n4={n4}, n6={n6}, n10={n10}")

    sigma = estimate_sigma(
        epsilon=args.epsilon,
        delta=args.delta,
        q=args.sampling_rate,
        total_steps=args.T1 + args.T3
    )
    print(f"Computed sigma={sigma:.4f} for eps={args.epsilon}, delta={args.delta}, q={args.sampling_rate}, total_steps={args.T1 + args.T3}")

    if args.baseline_only:
        train_baseline10(args.T3, args.lr, sigma, args.max_grad_norm, device, loaders)
        return

    if not args.train_all:
        print("Nothing to do: pass --train_all or --baseline_only")
        return

    # ---------- Multi-phase training ----------
    model = MultiHeadShared().to(device)

    print("\n[Phase 1] Train backbone + head4 (4-class)")
    train_phase(model, phase="4",
                loader=loaders["vehicle_train"], test_loader=loaders["vehicle_test"],
                steps=args.T1, lr=args.lr, sigma=sigma, C=args.max_grad_norm,
                device=device, expected_batchsize=b4, log_prefix="[DP/4-class]")

    print("\n[Phase 2] Train head6 + partial backbone (6-class)")
    train_phase(model, phase="6",
            loader=loaders["animal_train"], test_loader=loaders["animal_test"],
            steps=args.T1, lr=1.5, sigma=sigma, C=args.max_grad_norm,
            device=device, expected_batchsize=b6, log_prefix="[DP/6-class]")

    print("\n[Phase 3] Train fusion head only (10-class)")
    train_phase(model, phase="fusion",
                loader=loaders["full_train"], test_loader=loaders["full_test"],
                steps=args.T3, lr=args.lr_fusion, sigma=sigma, C=args.max_grad_norm,
                device=device, expected_batchsize=b10, log_prefix="[DP/Fusion10]")

    # Final eval on fusion
    model.set_phase("fusion")
    acc_fused = compute_accuracy(model, loaders["full_test"], device)
    print(f"[Fusion10] Final Test Acc = {acc_fused*100:.2f}%")

    print("\n[Baseline] Train independent 10-class DP-SGD for comparison")
    acc_base = train_baseline10(args.T3, args.lr, sigma, args.max_grad_norm, device, loaders)

    if acc_fused >= acc_base:
        print("[Result] Fusion10 >= Baseline10 ✅")
    else:
        print("[Result] Fusion10 < Baseline10 ⚠️  Try: --lr_fusion 0.3 and/or --T3 1500; increase fusion hidden dim.")
        print("        You can edit FusionHead(hidden=256→384/512) in dp_models.py")

if __name__ == "__main__":
    main()
