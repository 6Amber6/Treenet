#!/usr/bin/env python3
"""
Quick comparison script for MART vs TRADES
Uses shorter training for quick comparison
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_quick_training(method, desc):
    """Run quick training with minimal epochs"""
    print(f"\n{'='*50}")
    print(f"Quick {method} training (5 epochs each stage)")
    print(f"{'='*50}")
    
    # Quick parameters for testing
    cmd = [
        sys.executable, 'train_fusion_wrn_adv_improved.py',
        '--data-dir', './data',
        '--log-dir', './logs_quick',
        '--data', 'cifar10',
        '--batch-size', '128',
        '--epochs-m', '5',        # Quick submodel training
        '--epochs-g', '10',       # Quick fusion training
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
        '--seed', '42',
        '--desc', f'{desc}_{method}'
    ]
    
    if method == 'MART':
        cmd.append('--use-mart')
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        duration = time.time() - start_time
        
        print(f"\n{method} completed in {duration:.1f} seconds")
        
        if result.returncode == 0:
            print(f"✓ {method} training successful")
            return True, result.stdout
        else:
            print(f"✗ {method} training failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"✗ {method} training timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ {method} training failed: {e}")
        return False, str(e)

def extract_metrics(stdout):
    """Extract final metrics from training output"""
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
    """Quick comparison of MART vs TRADES"""
    print("Quick MART vs TRADES Comparison")
    print("=" * 50)
    print("Using 5 epochs for submodels, 10 epochs for fusion")
    print("This is for quick testing - use full training for real results")
    
    timestamp = datetime.now().strftime('%H%M%S')
    
    # Run TRADES
    print(f"\n🚀 Running TRADES...")
    trades_success, trades_output = run_quick_training('TRADES', f'quick_{timestamp}')
    
    # Run MART
    print(f"\n🚀 Running MART...")
    mart_success, mart_output = run_quick_training('MART', f'quick_{timestamp}')
    
    # Compare results
    print(f"\n{'='*60}")
    print("QUICK COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if trades_success and mart_success:
        trades_clean, trades_adv = extract_metrics(trades_output)
        mart_clean, mart_adv = extract_metrics(mart_output)
        
        print(f"{'Method':<10} {'Clean Acc':<12} {'Adv Acc':<12} {'Status':<10}")
        print("-" * 50)
        trades_clean_str = f"{trades_clean:.4f}" if trades_clean else "N/A"
        trades_adv_str = f"{trades_adv:.4f}" if trades_adv else "N/A"
        mart_clean_str = f"{mart_clean:.4f}" if mart_clean else "N/A"
        mart_adv_str = f"{mart_adv:.4f}" if mart_adv else "N/A"
        print(f"{'TRADES':<10} {trades_clean_str:<12} {trades_adv_str:<12} {'✓':<10}")
        print(f"{'MART':<10} {mart_clean_str:<12} {mart_adv_str:<12} {'✓':<10}")
        
        if trades_adv and mart_adv:
            improvement = mart_adv - trades_adv
            improvement_pct = (improvement / trades_adv) * 100 if trades_adv > 0 else 0
            
            print(f"\n📊 MART vs TRADES:")
            print(f"   Adversarial Accuracy Difference: {improvement:+.4f} ({improvement_pct:+.1f}%)")
            
            if improvement > 0:
                print(f"   🎉 MART is better by {improvement:.4f}")
            elif improvement < 0:
                print(f"   📈 TRADES is better by {-improvement:.4f}")
            else:
                print(f"   🤝 Equal performance")
    else:
        print("❌ One or both training runs failed")
        if not trades_success:
            print("   TRADES failed")
        if not mart_success:
            print("   MART failed")
    
    print(f"\n📁 Logs saved to: ./logs_quick/")
    print("💡 For full comparison, use: python compare_mart_trades.py")

if __name__ == '__main__':
    main()
