# """
# DP-SGD CNN models for CIFAR-10 with GroupNorm.
# We provide:
# - ResNet20GN backbone (no BatchNorm; use GroupNorm)
# - Heads for 4-class, 6-class, 10-class
# - A Fusion10 model that concatenates frozen (penultimate) features from
#   the 4-class and 6-class models and trains a small DP head on top.
# """

# from typing import Tuple, Optional, List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # --------------------
# # ResNet-20 with GroupNorm
# # --------------------

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# class BasicBlockGN(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1, groups=8):
#         super().__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.gn1 = nn.GroupNorm(groups, planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.gn2 = nn.GroupNorm(groups, planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(groups, planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.gn1(self.conv1(x)))
#         out = self.gn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNetGN(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, groups=8):
#         super().__init__()
#         self.in_planes = 16
#         self.groups = groups

#         self.conv1 = conv3x3(3, 16, 1)
#         self.gn1 = nn.GroupNorm(groups, 16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#         self.feat_dim = 64 * block.expansion
#         self.fc = nn.Linear(self.feat_dim, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for s in strides:
#             layers.append(BasicBlockGN(self.in_planes, planes, s, groups=self.groups))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def features(self, x):
#         out = F.relu(self.gn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         return out

#     def forward(self, x):
#         f = self.features(x)
#         logits = self.fc(f)
#         return logits, f


# def ResNet20GN(num_classes=10, groups=8):
#     return ResNetGN(BasicBlockGN, [3,3,3], num_classes=num_classes, groups=groups)


# # --------------------
# # Heads / Wrapper Models
# # --------------------

# class Classifier4(nn.Module):
#     """4-class vehicle classifier"""
#     def __init__(self):
#         super().__init__()
#         self.backbone = ResNet20GN(num_classes=4)

#     def forward(self, x):
#         return self.backbone(x)


# class Classifier6(nn.Module):
#     """6-class animal classifier"""
#     def __init__(self):
#         super().__init__()
#         self.backbone = ResNet20GN(num_classes=6)

#     def forward(self, x):
#         return self.backbone(x)


# class Classifier10(nn.Module):
#     """10-class baseline classifier"""
#     def __init__(self):
#         super().__init__()
#         self.backbone = ResNet20GN(num_classes=10)

#     def forward(self, x):
#         return self.backbone(x)


# class Fusion10(nn.Module):
#     """
#     Fusion model for 10 classes:
#     - Take two trained models: 4-class and 6-class
#     - Freeze their backbones
#     - Concatenate their penultimate features
#     - Train a small linear head with DP-SGD for 10 classes
#     """
#     def __init__(self, model4: Classifier4, model6: Classifier6, hidden: int = 128):
#         super().__init__()
#         # freeze
#         for p in model4.parameters():
#             p.requires_grad = False
#         for p in model6.parameters():
#             p.requires_grad = False

#         self.m4 = model4
#         self.m6 = model6
#         feat_dim4 = self.m4.backbone.feat_dim
#         feat_dim6 = self.m6.backbone.feat_dim
#         in_dim = feat_dim4 + feat_dim6

#         # a light projection + classifier
#         self.head = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, 10),
#         )

#     def forward(self, x):
#         _, f4 = self.m4.backbone(x)
#         _, f6 = self.m6.backbone(x)
#         f = torch.cat([f4, f6], dim=1)
#         logits = self.head(f)
#         return logits, f

#     def trainable_parameters(self):
#         # only train the fusion head
#         return self.head.parameters()





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
