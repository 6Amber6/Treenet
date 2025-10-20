#!/usr/bin/env python3
"""
Debug version of training to identify where it gets stuck
"""

import os
import sys
import subprocess
import time
import signal
from datetime import datetime

def run_with_timeout(cmd, timeout=300):  # 5 minute timeout
    """Run command with timeout and detailed logging"""
    print(f"Running command: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output
        start_time = time.time()
        output_lines = []
        error_lines = []
        
        while True:
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                output_lines.extend(stdout.split('\n'))
                error_lines.extend(stderr.split('\n'))
                break
            
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"⏰ Timeout after {timeout} seconds")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False, output_lines, error_lines + ["TIMEOUT"]
            
            # Read output
            try:
                line = process.stdout.readline()
                if line:
                    print(f"OUT: {line.strip()}")
                    output_lines.append(line.strip())
                
                line = process.stderr.readline()
                if line:
                    print(f"ERR: {line.strip()}")
                    error_lines.append(line.strip())
            except:
                pass
            
            time.sleep(0.1)
        
        return process.returncode == 0, output_lines, error_lines
        
    except Exception as e:
        return False, [], [f"Exception: {e}"]

def test_imports():
    """Test if we can import required modules"""
    print("Testing imports...")
    
    try:
        import torch
        print("✓ torch imported successfully")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device count: {torch.cuda.device_count()}")
            print(f"  - Current device: {torch.cuda.current_device()}")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        import torchvision
        print("✓ torchvision imported successfully")
    except ImportError as e:
        print(f"✗ torchvision import failed: {e}")
        return False
    
    try:
        # Test our script imports
        sys.path.insert(0, '/Users/litong/workspace/Treenet/adversarial_robustness_pytorch')
        from core.models.wideresnet import wideresnet
        print("✓ wideresnet imported successfully")
    except ImportError as e:
        print(f"✗ wideresnet import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if data loading works"""
    print("Testing data loading...")
    
    try:
        import torch
        import torchvision
        from torchvision import datasets, transforms
        
        # Test CIFAR-10 loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        print(f"✓ CIFAR-10 dataset loaded: {len(dataset)} samples")
        
        # Test data loading
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
        
        batch = next(iter(dataloader))
        print(f"✓ DataLoader works: batch shape {batch[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test if model creation works"""
    print("Testing model creation...")
    
    try:
        import torch
        import torch.nn as nn
        sys.path.insert(0, '/Users/litong/workspace/Treenet/adversarial_robustness_pytorch')
        from core.models.wideresnet import wideresnet
        
        # Test WRN creation
        model = wideresnet('wideresnet-28-10', num_classes=10, device='cpu')
        print(f"✓ WRN model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        print(f"✓ Forward pass works: input {x.shape} -> output {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Main debug function"""
    print("=" * 60)
    print("Training Debug Analysis")
    print("=" * 60)
    
    # Test 1: Imports
    print("\n1. Testing imports...")
    if not test_imports():
        print("❌ Import test failed - check PyTorch installation")
        return 1
    
    # Test 2: Data loading
    print("\n2. Testing data loading...")
    if not test_data_loading():
        print("❌ Data loading test failed - check data directory")
        return 1
    
    # Test 3: Model creation
    print("\n3. Testing model creation...")
    if not test_model_creation():
        print("❌ Model creation test failed - check model code")
        return 1
    
    # Test 4: Run training with timeout
    print("\n4. Testing training script...")
    cmd = [
        sys.executable, 'train_fusion_wrn_adv_improved.py',
        '--desc', 'debug_test',
        '--data-dir', './data',
        '--log-dir', './logs_debug',
        '--data', 'cifar10',
        '--batch-size', '32',  # Smaller batch size
        '--epochs-m', '1',     # Just 1 epoch
        '--epochs-g', '1',     # Just 1 epoch
        '--lr', '0.1',
        '--lr-m', '0.1',
        '--aux_w', '0.02',
        '--beta', '8.0',
        '--attack', 'linf-pgd',
        '--attack-eps', '8/255',
        '--attack-step', '2/255',
        '--attack-iter', '12',
        '--ema-decay', '0.999',
        '--train-mode', 'all',
        '--seed', '42'
    ]
    
    success, output, error = run_with_timeout(cmd, timeout=120)  # 2 minute timeout
    
    print(f"\nTraining result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Output lines: {len(output)}")
    print(f"Error lines: {len(error)}")
    
    if error:
        print("\nLast 10 error lines:")
        for line in error[-10:]:
            print(f"  {line}")
    
    if output:
        print("\nLast 10 output lines:")
        for line in output[-10:]:
            print(f"  {line}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
