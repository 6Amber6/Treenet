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



# dp_models.py (shared-backbone multi-head + fusion)
# -*- coding: utf-8 -*-
from typing import Tuple, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- ResNet20 with GroupNorm (same as before) ----------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlockGN(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, groups=8):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.gn1 = nn.GroupNorm(groups, planes)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(groups, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetGN_Features(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.in_planes = 16
        self.groups = groups
        self.conv1 = conv3x3(3, 16, 1)
        self.gn1 = nn.GroupNorm(groups, 16)
        self.layer1 = self._make_layer(BasicBlockGN, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlockGN, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlockGN, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.feat_dim = 64

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, groups=self.groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # [B, feat_dim]
        return out  # features only

# ---------- Heads ----------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, f):
        return self.fc(f)

class FusionHead(nn.Module):
    """
    Fusion head taking concatenation of:
      - backbone features (feat_dim)
      - logits_4 (4)
      - logits_6 (6)
    Then MLP -> 10 classes
    """
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = feat_dim + 4 + 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 10),
        )
    def forward(self, fused):
        return self.net(fused)

# ---------- Multi-Head Model ----------
class MultiHeadShared(nn.Module):
    def __init__(self, groups=8, fusion_hidden=256):
        super().__init__()
        self.backbone = ResNetGN_Features(groups=groups)
        d = self.backbone.feat_dim
        self.head4 = LinearHead(d, 4)
        self.head6 = LinearHead(d, 6)
        self.fusion = FusionHead(d, hidden=fusion_hidden)

    # forward helpers
    def forward_4(self, x):
        f = self.backbone(x)
        logits4 = self.head4(f)
        return logits4, f

    def forward_6(self, x):
        f = self.backbone(x)
        logits6 = self.head6(f)
        return logits6, f

    def forward_fusion10(self, x):
        # IMPORTANT: head4 & head6 used as frozen feature-to-logits adapters
        with torch.no_grad():
            f = self.backbone(x)  # backbone is frozen during fusion phase
            logits4 = self.head4(f)
            logits6 = self.head6(f)
        fused = torch.cat([f, logits4, logits6], dim=1)
        logits10 = self.fusion(fused)
        return logits10, f

    # utilities for training phases
    def params_backbone_and_head4(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
            yield p
        for p in self.head4.parameters():
            p.requires_grad = True
            yield p
        for p in self.head6.parameters():
            p.requires_grad = False
        for p in self.fusion.parameters():
            p.requires_grad = False

    def params_head6_only(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head4.parameters():
            p.requires_grad = False
        for p in self.head6.parameters():
            p.requires_grad = True
            yield p
        for p in self.fusion.parameters():
            p.requires_grad = False

    def params_fusion_only(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head4.parameters():
            p.requires_grad = False
        for p in self.head6.parameters():
            p.requires_grad = False
        for p in self.fusion.parameters():
            p.requires_grad = True
            yield p

# ---------- A simple 10-class baseline model for comparison ----------
class Baseline10(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.backbone = ResNetGN_Features(groups=groups)
        self.fc = nn.Linear(self.backbone.feat_dim, 10)
    def forward(self, x):
        f = self.backbone(x)
        logits = self.fc(f)
        return logits, f
