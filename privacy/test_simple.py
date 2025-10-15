#!/usr/bin/env python3
"""
Simple test script to debug the hierarchical CNN training issues.
"""

import torch
import torch.nn.functional as F
from torch import optim
import sys
import os

# Add current directory to path
sys.path.append('/workspace/Treenet')

from privacy.dp_utils import DataProcessor, compute_accuracy
from privacy.dp_models import DP4Classifier, DP6Classifier
from privacy.dp_train_improved import AttentionFusion, HierarchicalLoss

def test_simple_training():
    """Test simple training without DP-SGD to isolate the issue."""
    print("🔍 Testing simple training without DP-SGD...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create models
    model_4 = DP4Classifier().to(device)
    model_6 = DP6Classifier().to(device)
    fusion_model = AttentionFusion().to(device)
    
    # Create optimizers
    optimizer_4 = optim.SGD(model_4.parameters(), lr=0.1)
    optimizer_6 = optim.SGD(model_6.parameters(), lr=0.1)
    optimizer_fusion = optim.SGD(fusion_model.parameters(), lr=0.1)
    
    # Create data loaders
    loaders = DataProcessor.create_data_loaders("./data", sampling_rate=0.1)
    train_loader = loaders["full_train"]
    test_loader = loaders["full_test"]
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test a few batches
    for step in range(5):
        print(f"\n--- Step {step} ---")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 1:  # Only test first batch
                break
                
            x, y = x.to(device), y.to(device)
            print(f"Batch shape: {x.shape}, Labels: {y[:5]}")
            
            # Forward pass
            logits_4, emb_4 = model_4(x)
            logits_6, emb_6 = model_6(x)
            fusion_logits = fusion_model(emb_4, emb_6)
            
            print(f"4-class logits shape: {logits_4.shape}")
            print(f"6-class logits shape: {logits_6.shape}")
            print(f"Fusion logits shape: {fusion_logits.shape}")
            
            # Create auxiliary targets
            aux_4_targets = create_auxiliary_targets_simple(y, 'vehicle')
            aux_6_targets = create_auxiliary_targets_simple(y, 'animal')
            
            print(f"Original labels: {y[:5]}")
            print(f"4-class targets: {aux_4_targets[:5]}")
            print(f"6-class targets: {aux_6_targets[:5]}")
            
            # Compute losses
            loss_4 = F.cross_entropy(logits_4, aux_4_targets)
            loss_6 = F.cross_entropy(logits_6, aux_6_targets)
            loss_fusion = F.cross_entropy(fusion_logits, y)
            
            print(f"4-class loss: {loss_4.item():.4f}")
            print(f"6-class loss: {loss_6.item():.4f}")
            print(f"Fusion loss: {loss_fusion.item():.4f}")
            
            # Backward pass
            optimizer_4.zero_grad()
            optimizer_6.zero_grad()
            optimizer_fusion.zero_grad()
            
            loss_4.backward()
            loss_6.backward()
            loss_fusion.backward()
            
            optimizer_4.step()
            optimizer_6.step()
            optimizer_fusion.step()
            
            # Test accuracy
            model_4.eval()
            model_6.eval()
            fusion_model.eval()
            
            with torch.no_grad():
                test_acc = compute_accuracy(fusion_model, test_loader, device)
                print(f"Test accuracy: {test_acc*100:.2f}%")
            
            model_4.train()
            model_6.train()
            fusion_model.train()
            
            break  # Only test first batch

def create_auxiliary_targets_simple(original_labels, target_type):
    """Create auxiliary targets for testing."""
    if target_type == 'vehicle':
        # Map to vehicle classes: [0, 1, 8, 9] -> [0, 1, 2, 3]
        mapping = {0: 0, 1: 1, 8: 2, 9: 3}
        aux_targets = torch.zeros_like(original_labels)
        for orig, new in mapping.items():
            aux_targets[original_labels == orig] = new
        return aux_targets
    elif target_type == 'animal':
        # Map to animal classes: [2, 3, 4, 5, 6, 7] -> [0, 1, 2, 3, 4, 5]
        mapping = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        aux_targets = torch.zeros_like(original_labels)
        for orig, new in mapping.items():
            aux_targets[original_labels == orig] = new
        return aux_targets
    else:
        return original_labels

if __name__ == "__main__":
    test_simple_training()
