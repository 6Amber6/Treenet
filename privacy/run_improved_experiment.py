#!/usr/bin/env python3
"""
Simplified script to run the improved hierarchical CNN experiment.
This script demonstrates the key improvements and provides easy-to-use commands.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS!")
        if result.stdout:
            print("OUTPUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    print("🚀 IMPROVED HIERARCHICAL CNN EXPERIMENT")
    print("This script will demonstrate the improvements over the original implementation.")
    
    # Check if we're in the right directory
    if not os.path.exists("privacy/dp_train_improved.py"):
        print("❌ Error: Please run this script from the Treenet root directory")
        sys.exit(1)
    
    # Test 1: Quick comparison with small iterations
    print("\n📋 TEST 1: Quick Comparison (Small Scale)")
    print("This will train both models with minimal iterations for demonstration.")
    
    cmd1 = """python3 privacy/dp_train_improved.py \\
        --data_dir ./data \\
        --output_dir ./results_quick \\
        --sampling_rate 0.05 \\
        --T1 50 \\
        --T3 50 \\
        --epsilon 8.0 \\
        --lr 1.0 \\
        --compare"""
    
    success1 = run_command(cmd1, "Quick comparison test")
    
    if success1:
        print("\n✅ Quick test completed successfully!")
        print("Check ./results_quick/comparison_results.txt for results")
    else:
        print("\n❌ Quick test failed. Let's try a simpler approach...")
        
        # Test 2: Individual model training
        print("\n📋 TEST 2: Individual Model Training")
        
        # Train hierarchical model only
        cmd2_hier = """python3 privacy/dp_train_improved.py \\
            --data_dir ./data \\
            --output_dir ./results_hier \\
            --sampling_rate 0.05 \\
            --T1 100 \\
            --epsilon 8.0 \\
            --lr 1.0 \\
            --train_hierarchical"""
        
        success2_hier = run_command(cmd2_hier, "Train hierarchical model only")
        
        # Train baseline model only
        cmd2_base = """python3 privacy/dp_train_improved.py \\
            --data_dir ./data \\
            --output_dir ./results_base \\
            --sampling_rate 0.05 \\
            --T3 100 \\
            --epsilon 8.0 \\
            --lr 1.0 \\
            --train_baseline"""
        
        success2_base = run_command(cmd2_base, "Train baseline model only")
        
        if success2_hier and success2_base:
            print("\n✅ Individual training completed!")
            print("You can now compare the results manually.")
    
    # Test 3: Full experiment (if user wants)
    print("\n📋 TEST 3: Full Experiment (Optional)")
    print("For a complete comparison, you can run:")
    print("""
    # Full hierarchical vs baseline comparison
    python3 privacy/dp_train_improved.py \\
        --data_dir ./data \\
        --output_dir ./results_full \\
        --sampling_rate 0.05 \\
        --T1 1000 \\
        --T3 1000 \\
        --epsilon 8.0 \\
        --lr 1.0 \\
        --compare
    """)
    
    print("\n🎯 KEY IMPROVEMENTS IN THIS VERSION:")
    print("1. ✅ Attention-based fusion mechanism")
    print("2. ✅ Hierarchical loss function with auxiliary losses")
    print("3. ✅ Proper label mapping for auxiliary tasks")
    print("4. ✅ End-to-end joint training")
    print("5. ✅ Residual connections in fusion layers")
    print("6. ✅ Dropout for regularization")
    print("7. ✅ Comprehensive evaluation and comparison")
    
    print("\n📊 EXPECTED RESULTS:")
    print("- Hierarchical CNN should outperform baseline by 2-5%")
    print("- Better feature learning through auxiliary tasks")
    print("- More robust fusion through attention mechanism")
    print("- Improved generalization through hierarchical structure")

if __name__ == "__main__":
    main()
