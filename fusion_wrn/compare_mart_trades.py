#!/usr/bin/env python3
"""
Compare MART vs TRADES with identical parameters
Based on the original training configuration
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def run_training(method, desc, base_params):
    """Run training with specified method and parameters"""
    print(f"\n{'='*60}")
    print(f"Starting {method} training...")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, 'train_fusion_wrn_adv_improved.py'
    ] + base_params
    
    if method == 'MART':
        cmd.extend(['--use-mart'])
        desc_suffix = f"{desc}_MART"
    else:
        desc_suffix = f"{desc}_TRADES"
    
    cmd.extend(['--desc', desc_suffix])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Description: {desc_suffix}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{method} training completed in {duration:.2f} seconds")
        
        if result.returncode == 0:
            print(f"✓ {method} training successful")
            return True, duration, result.stdout, result.stderr
        else:
            print(f"✗ {method} training failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False, duration, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ {method} training timed out after 1 hour")
        return False, 3600, "", "Timeout"
    except Exception as e:
        print(f"✗ {method} training failed with exception: {e}")
        return False, 0, "", str(e)

def extract_final_metrics(log_content):
    """Extract final clean and adversarial accuracy from log"""
    lines = log_content.split('\n')
    final_clean = None
    final_adv = None
    
    for line in lines:
        if '[WRN-Fusion] Final Test Clean' in line:
            # Extract clean accuracy
            try:
                clean_part = line.split('Clean ')[1].split(' |')[0]
                final_clean = float(clean_part)
            except:
                pass
        elif '[WRN-Fusion] Final Test Clean' in line and 'Adv' in line:
            # Extract adversarial accuracy
            try:
                adv_part = line.split('Adv ')[1]
                final_adv = float(adv_part)
            except:
                pass
    
    return final_clean, final_adv

def compare_results(mart_results, trades_results):
    """Compare and display results"""
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"{'Method':<10} {'Status':<10} {'Duration':<12} {'Clean Acc':<12} {'Adv Acc':<12} {'Improvement':<15}")
    print("-" * 80)
    
    # TRADES results
    trades_success, trades_duration, trades_stdout, trades_stderr = trades_results
    trades_clean, trades_adv = extract_final_metrics(trades_stdout)
    
    trades_clean_str = f"{trades_clean:.4f}" if trades_clean else "N/A"
    trades_adv_str = f"{trades_adv:.4f}" if trades_adv else "N/A"
    print(f"{'TRADES':<10} {'✓' if trades_success else '✗':<10} {trades_duration:.1f}s{'':<8} "
          f"{trades_clean_str:<12} {trades_adv_str:<12} {'Baseline':<15}")
    
    # MART results
    mart_success, mart_duration, mart_stdout, mart_stderr = mart_results
    mart_clean, mart_adv = extract_final_metrics(mart_stdout)
    
    mart_clean_str = f"{mart_clean:.4f}" if mart_clean else "N/A"
    mart_adv_str = f"{mart_adv:.4f}" if mart_adv else "N/A"
    print(f"{'MART':<10} {'✓' if mart_success else '✗':<10} {mart_duration:.1f}s{'':<8} "
          f"{mart_clean_str:<12} {mart_adv_str:<12} ", end="")
    
    # Calculate improvement
    if trades_success and mart_success and trades_adv and mart_adv:
        improvement = mart_adv - trades_adv
        improvement_pct = (improvement / trades_adv) * 100
        print(f"{improvement:+.4f} ({improvement_pct:+.1f}%)")
    else:
        print("N/A")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if trades_success and mart_success:
        if mart_adv and trades_adv:
            improvement = mart_adv - trades_adv
            improvement_pct = (improvement / trades_adv) * 100
            
            if improvement > 0:
                print(f"🎉 MART outperforms TRADES by {improvement:.4f} ({improvement_pct:.1f}%)")
                print(f"   MART Adv Acc: {mart_adv:.4f}")
                print(f"   TRADES Adv Acc: {trades_adv:.4f}")
            elif improvement < 0:
                print(f"📊 TRADES outperforms MART by {-improvement:.4f} ({-improvement_pct:.1f}%)")
                print(f"   TRADES Adv Acc: {trades_adv:.4f}")
                print(f"   MART Adv Acc: {mart_adv:.4f}")
            else:
                print("🤝 MART and TRADES perform equally")
        else:
            print("⚠️  Could not extract final metrics for comparison")
    else:
        print("❌ One or both training runs failed")
        if not trades_success:
            print("   TRADES training failed")
        if not mart_success:
            print("   MART training failed")

def main():
    """Main comparison function"""
    print("MART vs TRADES Comparison")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Base parameters (matching original configuration)
    base_params = [
        '--data-dir', './data',
        '--log-dir', './logs_comparison',
        '--data', 'cifar10',
        '--batch-size', '128',
        '--epochs-m', '100',      # Submodel epochs
        '--epochs-g', '120',       # Fusion epochs
        '--lr', '0.1',            # Fusion learning rate
        '--lr-m', '0.1',          # Submodel learning rate
        '--aux_w', '0.02',        # Auxiliary loss weight
        '--beta', '8.0',          # TRADES beta
        '--attack', 'linf-pgd',
        '--attack-eps', '8/255',
        '--attack-step', '2/255',
        '--attack-iter', '12',
        '--ema-decay', '0.999',
        '--train-mode', 'all',
        '--seed', '42'
    ]
    
    # Generate unique experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_desc = f'comparison_{timestamp}'
    
    print(f"Base parameters: {base_params}")
    print(f"Base description: {base_desc}")
    
    # Run TRADES training
    print(f"\n🚀 Starting TRADES training...")
    trades_results = run_training('TRADES', base_desc, base_params)
    
    # Run MART training
    print(f"\n🚀 Starting MART training...")
    mart_results = run_training('MART', base_desc, base_params)
    
    # Compare results
    compare_results(mart_results, trades_results)
    
    # Save comparison results
    results = {
        'timestamp': timestamp,
        'base_params': base_params,
        'trades_results': {
            'success': trades_results[0],
            'duration': trades_results[1],
            'clean_acc': extract_final_metrics(trades_results[2])[0],
            'adv_acc': extract_final_metrics(trades_results[2])[1]
        },
        'mart_results': {
            'success': mart_results[0],
            'duration': mart_results[1],
            'clean_acc': extract_final_metrics(mart_results[2])[0],
            'adv_acc': extract_final_metrics(mart_results[2])[1]
        }
    }
    
    # Save results to JSON
    results_file = f'comparison_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")
    print(f"📁 Logs saved to: ./logs_comparison/")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
