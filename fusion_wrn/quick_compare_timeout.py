#!/usr/bin/env python3
"""
Quick comparison with timeout and better error handling
"""

import os
import sys
import subprocess
import time
import signal
from datetime import datetime

def run_with_timeout(cmd, timeout=60, description=""):
    """Run command with timeout"""
    print(f"\n{'='*50}")
    print(f"Running {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return_code = process.returncode
            
            print(f"Completed in {timeout} seconds")
            print(f"Return code: {return_code}")
            
            if return_code == 0:
                print("✓ SUCCESS")
                return True, stdout, stderr
            else:
                print("✗ FAILED")
                print(f"STDERR: {stderr}")
                return False, stdout, stderr
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT after {timeout} seconds")
            process.kill()
            return False, "", "TIMEOUT"
            
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False, "", str(e)

def extract_metrics(stdout):
    """Extract final metrics from output"""
    lines = stdout.split('\n')
    final_clean = None
    final_adv = None
    
    for line in lines:
        if '[WRN-Fusion] Final Test Clean' in line and 'Adv' in line:
            try:
                # Extract clean accuracy
                clean_part = line.split('Clean ')[1].split(' |')[0]
                final_clean = float(clean_part)
                
                # Extract adversarial accuracy
                adv_part = line.split('Adv ')[1]
                final_adv = float(adv_part)
            except:
                pass
    
    return final_clean, final_adv

def main():
    """Quick comparison with timeout"""
    print("Quick MART vs TRADES Comparison (with timeout)")
    print("=" * 60)
    print("Using 1 epoch for submodels, 2 epochs for fusion")
    print("Timeout: 60 seconds per method")
    print("This is for debugging - use full training for real results")
    
    timestamp = datetime.now().strftime('%H%M%S')
    
    # Base command with minimal epochs
    base_cmd = [
        sys.executable, 'train_fusion_wrn_adv_improved.py',
        '--data-dir', './data',
        '--log-dir', './logs_quick',
        '--data', 'cifar10',
        '--batch-size', '64',      # Smaller batch size
        '--epochs-m', '1',         # Just 1 epoch for submodels
        '--epochs-g', '2',         # Just 2 epochs for fusion
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
    
    # Test TRADES
    trades_cmd = base_cmd + ['--desc', f'debug_{timestamp}_TRADES']
    trades_success, trades_stdout, trades_stderr = run_with_timeout(
        trades_cmd, timeout=60, description="TRADES (1+2 epochs)"
    )
    
    # Test MART
    mart_cmd = base_cmd + ['--desc', f'debug_{timestamp}_MART', '--use-mart']
    mart_success, mart_stdout, mart_stderr = run_with_timeout(
        mart_cmd, timeout=60, description="MART (1+2 epochs)"
    )
    
    # Results
    print(f"\n{'='*60}")
    print("QUICK COMPARISON RESULTS")
    print(f"{'='*60}")
    
    print(f"{'Method':<10} {'Status':<10} {'Clean Acc':<12} {'Adv Acc':<12}")
    print("-" * 50)
    
    # TRADES results
    trades_clean, trades_adv = extract_metrics(trades_stdout)
    trades_clean_str = f"{trades_clean:.4f}" if trades_clean else "N/A"
    trades_adv_str = f"{trades_adv:.4f}" if trades_adv else "N/A"
    trades_status = "✓" if trades_success else "✗"
    print(f"{'TRADES':<10} {trades_status:<10} {trades_clean_str:<12} {trades_adv_str:<12}")
    
    # MART results
    mart_clean, mart_adv = extract_metrics(mart_stdout)
    mart_clean_str = f"{mart_clean:.4f}" if mart_clean else "N/A"
    mart_adv_str = f"{mart_adv:.4f}" if mart_adv else "N/A"
    mart_status = "✓" if mart_success else "✗"
    print(f"{'MART':<10} {mart_status:<10} {mart_clean_str:<12} {mart_adv_str:<12}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if not trades_success and not mart_success:
        print("❌ Both methods failed - check environment setup")
        print("\nCommon issues:")
        print("1. PyTorch not installed or wrong version")
        print("2. CUDA/GPU issues")
        print("3. Data directory missing")
        print("4. Memory issues")
        print("\nTry running: python quick_test.py")
    elif trades_success and mart_success:
        print("🎉 Both methods completed successfully!")
        if trades_adv and mart_adv:
            improvement = mart_adv - trades_adv
            print(f"📊 MART vs TRADES: {improvement:+.4f} difference")
    else:
        print("⚠️  One method failed - check the error messages above")
    
    print(f"\n📁 Logs saved to: ./logs_quick/")
    print("💡 For full comparison, use: python compare_mart_trades.py")

if __name__ == '__main__':
    main()
