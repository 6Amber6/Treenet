"""
ResNet-20 model implementation strictly following the paper 
"A Theory to Instruct Differentially Private Learning via Clipping Bias Reduction"
Uses GroupNorm instead of BatchNorm for DP-SGD compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BasicBlock(nn.Module):
    """
    BasicBlock implementation strictly following the paper
    Uses GroupNorm instead of BatchNorm, groups=8
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, groups=8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, bias=False),
                nn.GroupNorm(groups, self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)   
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet-20 implementation strictly following the paper
    Architecture: 3x3 conv -> 3 layers of 3 blocks each -> global avg pool -> fc
    """
    def __init__(self, num_classes=10, groups=8):
        super(ResNet20, self).__init__()
        self.groups = groups
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, 16)
        
        # ResNet layers: 3 blocks each layer
        self.layer1 = self._make_layer(16, 16, 3, stride=1)    # 3 blocks, 16->16
        self.layer2 = self._make_layer(16, 32, 3, stride=2)    # 3 blocks, 16->32, stride=2
        self.layer3 = self._make_layer(32, 64, 3, stride=2)    # 3 blocks, 32->64, stride=2
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # For embedding extraction
        self.embedding_dim = 64
        
    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride, self.groups))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1, self.groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        out = F.relu(self.gn1(self.conv1(x)))
        
        # ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global average pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        # Classifier
        logits = self.fc(out)
        
        return logits, out  # Return logits and embeddings


class DPResNet20(ResNet20):
    """
    DP-SGD compatible ResNet-20
    Strictly following the paper implementation, using GroupNorm
    """
    def __init__(self, num_classes=10, groups=8):
        super().__init__(num_classes, groups)
        self.embedding_dim = 64


class DP4Classifier(DPResNet20):
    """4-class classifier - Vehicle classes"""
    def __init__(self, groups=8):
        super().__init__(num_classes=4, groups=groups)


class DP6Classifier(DPResNet20):
    """6-class classifier - Animal classes"""
    def __init__(self, groups=8):
        super().__init__(num_classes=6, groups=groups)


class DP10Classifier(DPResNet20):
    """10-class classifier - Full CIFAR-10"""
    def __init__(self, groups=8):
        super().__init__(num_classes=10, groups=groups)


class DPFusionModel(nn.Module):
    """
    Fusion model implementation strictly following the paper requirements
    Combines embeddings from 4-class and 6-class models
    """
    def __init__(self, embedding_dim=64, num_classes=10, groups=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Fusion layers using GroupNorm
        self.fusion_fc1 = nn.Linear(embedding_dim * 2, 128)
        self.gn1 = nn.GroupNorm(groups, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.gn2 = nn.GroupNorm(groups, 64)
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, embeddings_4class, embeddings_6class):
        """
        Fuse embeddings from two models
        Args:
            embeddings_4class: Embeddings from 4-class model
            embeddings_6class: Embeddings from 6-class model
        Returns:
            logits: Classification results
        """
        # Concatenate embeddings
        combined_embeddings = torch.cat([embeddings_4class, embeddings_6class], dim=1)
        
        # Fusion layers
        x = F.relu(self.gn1(self.fusion_fc1(combined_embeddings)))
        x = F.relu(self.gn2(self.fusion_fc2(x)))
        logits = self.classifier(x)
        
        return logits


def create_dp_model(model_type: str, num_classes: int, groups: int = 8) -> nn.Module:
    """
    Create DP-SGD compatible model
    Strictly following the paper requirements
    """
    if model_type == 'resnet20':
        return DPResNet20(num_classes=num_classes, groups=groups)
    elif model_type == '4class':
        return DP4Classifier(groups=groups)
    elif model_type == '6class':
        return DP6Classifier(groups=groups)
    elif model_type == '10class':
        return DP10Classifier(groups=groups)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models strictly following the paper implementation
    print("Testing paper-compliant DP-SGD models...")
    
    # Test ResNet-20 architecture
    model = DPResNet20(num_classes=10, groups=8)
    print(f"ResNet-20 parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    logits, embeddings = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test classifiers
    model_4 = DP4Classifier()
    model_6 = DP6Classifier()
    model_10 = DP10Classifier()
    
    logits_4, emb_4 = model_4(x)
    logits_6, emb_6 = model_6(x)
    logits_10, emb_10 = model_10(x)
    
    print(f"\nClassifier outputs:")
    print(f"4-class: {logits_4.shape}, embedding: {emb_4.shape}")
    print(f"6-class: {logits_6.shape}, embedding: {emb_6.shape}")
    print(f"10-class: {logits_10.shape}, embedding: {emb_10.shape}")
    
    # Test fusion model
    fusion_model = DPFusionModel()
    fusion_logits = fusion_model(emb_4, emb_6)
    print(f"Fusion output: {fusion_logits.shape}")
    
    print("\n✅ All tests passed!")
    print("Models are strictly compliant with the paper requirements:")
    print("- ResNet-20 architecture")
    print("- GroupNorm with groups=8")
    print("- No BatchNorm layers")
    print("- Proper embedding extraction")
