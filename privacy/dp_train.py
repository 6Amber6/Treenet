# """
# Training script for DP-SGD with hierarchical (4-class + 6-class fusion) vs baseline (10-class).
# Implements per-sample gradient computation with DP noise.
# """

# import argparse
# import os
# import torch
# import torch.nn.functional as F
# from torch import optim
# from opacus.grad_sample import GradSampleModule
# from opacus.utils.batch_memory_manager import BatchMemoryManager

# from privacy.dp_utils import (
#     compute_accuracy,
#     compute_epsilon_opacus,
#     DataProcessor,
#     dp_step_images,
#     get_std,   # ✅ 新增导入
# )
# from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel


# # ------------------------------
# # Label remapping helper (安全版)
# # ------------------------------
# def remap_targets(y, dataset, device):
#     """
#     Handle label remapping for different dataset types.
#     For filtered datasets, labels are already properly mapped to 0..N-1.
#     """
#     # Check if this is a filtered dataset with remapped targets
#     if hasattr(dataset, 'targets') and hasattr(dataset, 'indices'):
#         # This is a filtered dataset, labels are already remapped
#         return y.to(device)
    
#     # For other cases, return as-is
#     return y.to(device)


# # ------------------------------
# # Training loop with per-sample DP-SGD
# # ------------------------------
# def train_dp(model, train_loader, test_loader, total_iterations, lr,
#              noise_multiplier, max_grad_norm, delta, device, sampling_rate=0.05):
#     """
#     DP-SGD training with fixed sampling rate and total iterations.
    
#     Args:
#         model: Model to train
#         train_loader: Training data loader
#         test_loader: Test data loader  
#         total_iterations: Total number of DP-SGD iterations (T_1 + T_3)
#         lr: Learning rate
#         noise_multiplier: Noise multiplier σ
#         max_grad_norm: Clipping constant C
#         delta: Privacy parameter δ
#         device: Device to use
#         sampling_rate: Sampling rate q (default 0.05)
#     """
#     model = model.to(device)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#     # Calculate batch size from sampling rate
#     dataset_size = len(train_loader.dataset)
#     batch_size = int(sampling_rate * dataset_size)
    
#     # Create new data loader with calculated batch size
#     from torch.utils.data import DataLoader
#     new_train_loader = DataLoader(
#         train_loader.dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=train_loader.num_workers,
#         pin_memory=train_loader.pin_memory
#     )
    
#     print(f"Training with sampling_rate={sampling_rate:.3f}, batch_size={batch_size}, total_iterations={total_iterations}")
    
#     steps = 0
#     eps = 0.0
    
#     # Training loop with fixed total iterations
#     for iteration in range(total_iterations):
#         model.train()
        
#         # Get next batch
#         if not hasattr(train_dp, 'batch_iter'):
#             train_dp.batch_iter = iter(new_train_loader)
        
#         try:
#             x, y = next(train_dp.batch_iter)
#         except StopIteration:
#             # Reset iterator if we run out of data
#             train_dp.batch_iter = iter(new_train_loader)
#             x, y = next(train_dp.batch_iter)
        
#         x, y = x.to(device), y.to(device)
        
#         # Remap targets
#         y = remap_targets(y, train_loader.dataset, device)
        
#         optimizer.zero_grad()
        
#         # DP-SGD step
#         preclip_norm = dp_step_images(model, optimizer, x, y, noise_multiplier, max_grad_norm)
        
#         steps += 1
        
#         # Compute privacy spent
#         eps = compute_epsilon_opacus(
#             noise_multiplier,
#             sampling_rate,
#             steps,
#             delta,
#         )
        
#         if iteration % 50 == 0:
#             # Compute loss for logging
#             with torch.no_grad():
#                 output = model(x)
#                 if isinstance(output, tuple):
#                     logits = output[0]
#                 else:
#                     logits = output
#                 loss_val = F.cross_entropy(logits, y).item()
#             print(f"Iteration {iteration}, Loss {loss_val:.4f}, "
#                   f"GradNorm {preclip_norm:.2f}, ε={eps:.3f}, δ={delta:.1e}")
    
#     # Final evaluation
#     train_acc = compute_accuracy(model, train_loader, device)
#     test_acc = compute_accuracy(model, test_loader, device)
#     print(f"Final: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, "
#           f"Privacy ε={eps:.3f}, δ={delta:.1e}")
    
#     return train_acc, eps, delta


# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     parser = argparse.ArgumentParser(description="DP-SGD Training with Fixed Sampling Rate")
#     parser.add_argument("--data_dir", type=str, default="./data")
#     parser.add_argument("--output_dir", type=str, default="./results_dp")
#     parser.add_argument("--sampling_rate", type=float, default=0.05, help="Sampling rate q (default: 0.05)")
#     parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
#     parser.add_argument("--T1", type=int, default=1000, help="Iterations for 4-class and 6-class models (T_1 = T_2)")
#     parser.add_argument("--T3", type=int, default=1000, help="Iterations for 10-class model")
#     parser.add_argument("--noise_multiplier", type=float, default=None, help="Noise multiplier σ (auto-computed if not provided)")
#     parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Clipping constant C")
#     parser.add_argument("--delta", type=float, default=1e-5, help="Privacy parameter δ")
#     parser.add_argument("--epsilon", type=float, default=8.0, help="Target privacy budget ε")
#     parser.add_argument("--train_all", action="store_true", help="Train all models")
#     parser.add_argument("--train_4class", action="store_true", help="Train 4-class model")
#     parser.add_argument("--train_6class", action="store_true", help="Train 6-class model")
#     parser.add_argument("--train_10class", action="store_true", help="Train 10-class model")
#     parser.add_argument("--train_fusion", action="store_true", help="Train fusion model")
#     args = parser.parse_args()

#     # Calculate total iterations and noise multiplier
#     total_iterations = args.T1 + args.T3  # T_1 + T_3
    
#     if args.noise_multiplier is None:
#         # Auto-compute noise multiplier based on (epsilon, delta, total_iterations, sampling_rate)
#         args.noise_multiplier = get_std(
#             q=args.sampling_rate,
#             total_iterations=total_iterations,
#             epsilon=args.epsilon,
#             delta=args.delta,
#             verbose=True
#         )

#     os.makedirs(args.output_dir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("============================================================")
#     print("DP-SGD TRAINING WITH FIXED SAMPLING RATE")
#     print("============================================================")
#     print(f"Device: {device}")
#     print(f"Sampling rate: {args.sampling_rate}")
#     print(f"T_1 (4-class, 6-class): {args.T1} iterations")
#     print(f"T_3 (10-class): {args.T3} iterations")
#     print(f"Total iterations: {total_iterations}")
#     print(f"DP params: σ={args.noise_multiplier:.4f}, C={args.max_grad_norm}, δ={args.delta}")
#     print(f"Target ε: {args.epsilon}")

#     # Data loaders (use any batch size for data loading, will be recalculated based on sampling rate)
#     loaders = DataProcessor.create_data_loaders(args.data_dir, batch_size=256)
#     m4_train, m4_test = loaders["vehicle_train"], loaders["vehicle_test"]
#     m6_train, m6_test = loaders["animal_train"], loaders["animal_test"]
#     m10_train, m10_test = loaders["full_train"], loaders["full_test"]

#     # Training 4-class model (T_1 iterations)
#     if args.train_all or args.train_4class:
#         print(f"\n{'='*50}")
#         print("TRAINING 4-CLASS MODEL (T_1 iterations)")
#         print(f"{'='*50}")
#         model4 = DP4Classifier()
#         train_dp(model4, m4_train, m4_test, args.T1, args.lr,
#                  args.noise_multiplier, args.max_grad_norm, args.delta, device, args.sampling_rate)

#     # Training 6-class model (T_2 = T_1 iterations)
#     if args.train_all or args.train_6class:
#         print(f"\n{'='*50}")
#         print("TRAINING 6-CLASS MODEL (T_2 = T_1 iterations)")
#         print(f"{'='*50}")
#         model6 = DP6Classifier()
#         train_dp(model6, m6_train, m6_test, args.T1, args.lr,
#                  args.noise_multiplier, args.max_grad_norm, args.delta, device, args.sampling_rate)

#     # Training 10-class model (T_3 iterations)
#     if args.train_all or args.train_10class:
#         print(f"\n{'='*50}")
#         print("TRAINING 10-CLASS MODEL (T_3 iterations)")
#         print(f"{'='*50}")
#         model10 = DP10Classifier()
#         train_dp(model10, m10_train, m10_test, args.T3, args.lr,
#                  args.noise_multiplier, args.max_grad_norm, args.delta, device, args.sampling_rate)


#     # Training fusion model (non-DP)
#     if args.train_all or args.train_fusion:
#         print(f"\n{'='*50}")
#         print("TRAINING FUSION MODEL (Non-DP)")
#         print(f"{'='*50}")
#         model4 = DP4Classifier().to(device)
#         model6 = DP6Classifier().to(device)
#         fusion = DPFusionModel()

#         model4.eval()
#         model6.eval()
#         fusion.to(device)
#         optimizer = optim.SGD(fusion.parameters(), lr=args.lr, momentum=0.9)

#         # Simple fusion training (non-DP)
#         for epoch in range(30):  # Fixed epochs for fusion
#             fusion.train()
#             for (x4, y4), (x6, y6) in zip(m4_train, m6_train):
#                 x4, y4 = x4.to(device), y4.to(device)
#                 x6, y6 = x6.to(device), y6.to(device)

#                 y4 = remap_targets(y4, m4_train.dataset, device)
#                 y6 = remap_targets(y6, m6_train.dataset, device)

#                 with torch.no_grad():
#                     _, emb4 = model4(x4)
#                     _, emb6 = model6(x6)

#                 optimizer.zero_grad()
#                 logits = fusion(emb4, emb6)
#                 loss = F.cross_entropy(logits, y4[:logits.size(0)])
#                 loss.backward()
#                 optimizer.step()

#             if epoch % 10 == 0:
#                 print(f"Fusion Epoch {epoch}: Loss {loss.item():.4f}")

#         print("Fusion training completed.")


# if __name__ == "__main__":
#     main()



# dp_train.py
# -*- coding: utf-8 -*-
"""
End-to-end DP-SGD training script:

1) Train 4-class (vehicles) with DP-SGD for T1 iterations
2) Train 6-class (animals)  with DP-SGD for T1 iterations  (T1 = T2)
3) Freeze 4/6 backbones, build Fusion10 head; DP-SGD on full CIFAR-10 for T3 iterations

Single sigma is computed once from (epsilon, delta, q, T1+T3) and
shared across all phases, matching the teacher's requirement.
Expected-batch-size normalization is used throughout.

You can also train a 10-class baseline by setting --baseline_only.
"""

import argparse
from typing import Dict
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dp_models import Classifier4, Classifier6, Classifier10, Fusion10
from dp_utils import dp_step_images, compute_accuracy, build_split_loaders, estimate_sigma


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train_dp(model: nn.Module,
             loader: DataLoader,
             test_loader: DataLoader,
             steps: int,
             lr: float,
             sigma: float,
             C: float,
             device: torch.device,
             expected_batchsize: int,
             train_head_only: bool = False) -> nn.Module:
    model = model.to(device)
    if train_head_only and hasattr(model, "trainable_parameters"):
        params = model.trainable_parameters()
    else:
        params = (p for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    it = _cycle(loader)

    for step in range(1, steps + 1):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        dp_step_images(model, optimizer, x, y, sigma, C, expected_batchsize)
        if step % 100 == 0:
            acc = compute_accuracy(model, test_loader, device)
            print(f"[DP] Step {step}/{steps} | Test Acc = {acc*100:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser()

    # privacy / schedule
    parser.add_argument("--sampling_rate", type=float, default=0.10, help="q, Poisson subsampling rate (constant across phases)")
    parser.add_argument("--epsilon", type=float, default=6.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--T1", type=int, default=1500, help="Iterations for 4- and 6-class")
    parser.add_argument("--T3", type=int, default=1500, help="Iterations for Fusion10 or Baseline10")

    # training
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batchsize_full", type=int, default=None, help="override b10 if desired")

    # modes
    parser.add_argument("--train_all", action="store_true", help="Train 4,6, then Fusion10")
    parser.add_argument("--baseline_only", action="store_true", help="Train 10-class baseline only")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # loaders with q-based batch sizes per subset
    loaders = build_split_loaders(q=args.sampling_rate,
                                  data_dir=args.data_dir,
                                  batchsize_full=args.batchsize_full,
                                  num_workers=args.num_workers,
                                  seed=args.seed)
    b4, b6, b10 = loaders["b4"], loaders["b6"], loaders["b10"]
    n4, n6, n10 = loaders["n4"], loaders["n6"], loaders["n10"]
    print(f"q={args.sampling_rate:.4f} | b4={b4}, b6={b6}, b10={b10} | n4={n4}, n6={n6}, n10={n10}")

    # single sigma for (q, T1+T3, epsilon, delta)
    total_steps = args.T1 + args.T3
    sigma = estimate_sigma(epsilon=args.epsilon, delta=args.delta, q=args.sampling_rate, total_steps=total_steps)
    print(f"Computed sigma={sigma:.4f} for eps={args.epsilon}, delta={args.delta}, q={args.sampling_rate}, total_steps={total_steps}")

    # training plan
    if args.baseline_only:
        print("\n[Training 10-class Baseline with DP-SGD]")
        model10 = Classifier10().to(device)
        model10 = train_dp(model10, loaders["full_train"], loaders["full_test"],
                           args.T3, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b10)
        acc10 = compute_accuracy(model10, loaders["full_test"], device)
        print(f"[Baseline10] Final Test Acc = {acc10*100:.2f}%")
        return

    # default: train all phases
    print("\n[Training 4-class (vehicles)]")
    model4 = Classifier4().to(device)
    model4 = train_dp(model4, loaders["vehicle_train"], loaders["vehicle_test"],
                      args.T1, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b4)

    print("\n[Training 6-class (animals)]")
    model6 = Classifier6().to(device)
    model6 = train_dp(model6, loaders["animal_train"], loaders["animal_test"],
                      args.T1, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b6)

    print("\n[Training Fused 10-class Head with DP-SGD]")
    fusion = Fusion10(model4, model6, hidden=128).to(device)
    # only train the head under DP
    fusion = train_dp(fusion, loaders["full_train"], loaders["full_test"],
                      args.T3, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b10,
                      train_head_only=True)

    acc_fused = compute_accuracy(fusion, loaders["full_test"], device)
    print(f"[Fusion10] Final Test Acc = {acc_fused*100:.2f}%")

    # Optional: also train a baseline for comparison in the same run
    print("\n[Training 10-class Baseline with DP-SGD for Comparison]")
    model10 = Classifier10().to(device)
    model10 = train_dp(model10, loaders["full_train"], loaders["full_test"],
                       args.T3, args.lr, sigma, args.max_grad_norm, device, expected_batchsize=b10)
    acc10 = compute_accuracy(model10, loaders["full_test"], device)
    print(f"[Baseline10] Final Test Acc = {acc10*100:.2f}%")

    if acc_fused >= acc10:
        print("[Result] Fusion10 >= Baseline10 ✅")
    else:
        print("[Result] Fusion10 < Baseline10 ⚠️  Consider tuning lr/T1/T3 or increasing Fusion head width.")

if __name__ == "__main__":
    main()
