#!/usr/bin/env python3
"""
Quick test to identify the issue
"""

import sys
import os
import time

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def test_script_imports():
    """Test script imports"""
    print("Testing script imports...")
    try:
        # Add path
        sys.path.insert(0, '/Users/litong/workspace/Treenet/adversarial_robustness_pytorch')
        
        # Test imports one by one
        print("  - Importing core.models.wideresnet...")
        from core.models.wideresnet import wideresnet
        print("  ✓ wideresnet imported")
        
        print("  - Importing core.utils...")
        from core.utils import Logger, parser_train, seed
        print("  ✓ core.utils imported")
        
        print("  - Importing core.attacks...")
        from core.attacks import create_attack
        print("  ✓ core.attacks imported")
        
        print("  - Importing core...")
        from core import animal_classes, vehicle_classes
        print("  ✓ core imported")
        
        return True
    except Exception as e:
        print(f"✗ Script imports failed: {e}")
        return False

def test_data_directory():
    """Test data directory"""
    print("Testing data directory...")
    data_dir = './data'
    if not os.path.exists(data_dir):
        print(f"✗ Data directory {data_dir} does not exist")
        return False
    
    cifar_dir = os.path.join(data_dir, 'cifar10')
    if not os.path.exists(cifar_dir):
        print(f"✗ CIFAR-10 directory {cifar_dir} does not exist")
        return False
    
    print(f"✓ Data directory exists: {data_dir}")
    return True

def test_minimal_training():
    """Test minimal training setup"""
    print("Testing minimal training setup...")
    try:
        import torch
        import torch.nn as nn
        sys.path.insert(0, '/Users/litong/workspace/Treenet/adversarial_robustness_pytorch')
        from core.models.wideresnet import wideresnet
        
        # Create minimal model
        model = wideresnet('wideresnet-28-10', num_classes=10, device='cpu')
        print("✓ Model created")
        
        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        print(f"✓ Forward pass: {x.shape} -> {y.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Minimal training failed: {e}")
        return False

def main():
    """Run quick tests"""
    print("=" * 50)
    print("Quick Training Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Script Imports", test_script_imports),
        ("Data Directory", test_data_directory),
        ("Minimal Training", test_minimal_training),
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✓ {name} PASSED")
        else:
            print(f"✗ {name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Training should work.")
    else:
        print("❌ Some tests failed. Check the issues above.")
    
    return 0

if __name__ == '__main__':
    main()
