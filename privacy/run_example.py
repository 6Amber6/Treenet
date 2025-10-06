#!/usr/bin/env python3
"""
Example script demonstrating DP-SGD training for the fusion architecture
This script shows how to use the privacy module for training and evaluation
"""

import os
import sys
import argparse
import torch
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.dp_train import DPTrainer
from privacy.dp_eval import DPEvaluator
from privacy.dp_utils import DataProcessor


def run_training_example():
    """Run a complete training example"""
    print("="*60)
    print("DP-SGD TRAINING EXAMPLE")
    print("="*60)
    
    # Training arguments
    class Args:
        def __init__(self):
            self.data_dir = './data'
            self.output_dir = './privacy_output'
            self.batch_size = 64
            self.num_workers = 4
            self.lr = 0.01
            self.epochs_4class = 20  # Reduced for example
            self.epochs_6class = 20
            self.epochs_10class = 20
            self.epochs_fusion = 15
            self.noise_multiplier = 1.0
            self.max_grad_norm = 1.0
            self.delta = 1e-5
            self.groups = 8
            self.train_all = True
    
    args = Args()
    
    # Create trainer
    trainer = DPTrainer(args)
    
    # Create data loaders
    print("Loading data...")
    data_loaders = DataProcessor.create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    
    print(f"Data loaders created:")
    for name, loader in data_loaders.items():
        print(f"  {name}: {len(loader)} batches")
    
    # Train models
    print("\nStarting training...")
    
    # Train 4-class model
    print("\n" + "="*50)
    print("TRAINING 4-CLASS MODEL")
    print("="*50)
    trainer.train_4class_model(data_loaders['animal_train'], data_loaders['animal_test'])
    
    # Train 6-class model
    print("\n" + "="*50)
    print("TRAINING 6-CLASS MODEL")
    print("="*50)
    trainer.train_6class_model(data_loaders['vehicle_train'], data_loaders['vehicle_test'])
    
    # Train 10-class model
    print("\n" + "="*50)
    print("TRAINING 10-CLASS MODEL")
    print("="*50)
    trainer.train_10class_model(data_loaders['full_train'], data_loaders['full_test'])
    
    # Train fusion model
    print("\n" + "="*50)
    print("TRAINING FUSION MODEL")
    print("="*50)
    trainer.train_fusion_model(data_loaders['full_train'], data_loaders['full_test'])
    
    # Save results
    trainer.save_training_results()
    
    print("\nTraining completed successfully!")
    return args.output_dir


def run_evaluation_example(model_dir):
    """Run evaluation example"""
    print("\n" + "="*60)
    print("DP-SGD EVALUATION EXAMPLE")
    print("="*60)
    
    # Evaluation arguments
    class EvalArgs:
        def __init__(self, model_dir):
            self.data_dir = './data'
            self.model_dir = model_dir
            self.output_dir = './evaluation_output'
            self.batch_size = 64
            self.num_workers = 4
            self.groups = 8
    
    eval_args = EvalArgs(model_dir)
    
    # Create evaluator
    evaluator = DPEvaluator(eval_args)
    
    # Run evaluation
    evaluator.run_evaluation()
    
    print("\nEvaluation completed successfully!")


def demonstrate_privacy_analysis():
    """Demonstrate privacy analysis capabilities"""
    print("\n" + "="*60)
    print("PRIVACY ANALYSIS DEMONSTRATION")
    print("="*60)
    
    from privacy.dp_utils import PrivacyAccountant
    
    # Demonstrate privacy accounting
    print("Privacy Accounting Examples:")
    print("-" * 40)
    
    # Different noise multipliers
    noise_multipliers = [0.5, 1.0, 2.0]
    batch_size = 64
    dataset_size = 50000
    steps = 100
    
    for noise_mult in noise_multipliers:
        accountant = PrivacyAccountant(noise_mult, batch_size, dataset_size)
        epsilon, delta = accountant.get_privacy_spent(steps, delta=1e-5)
        print(f"Noise Multiplier: {noise_mult}")
        print(f"  Privacy Spent: ε={epsilon:.3f}, δ={delta:.2e}")
        print()


def demonstrate_model_architecture():
    """Demonstrate model architecture"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE DEMONSTRATION")
    print("="*60)
    
    from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel
    
    # Create models
    model_4 = DP4Classifier()
    model_6 = DP6Classifier()
    model_10 = DP10Classifier()
    fusion_model = DPFusionModel()
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    
    print("Model Architecture:")
    print("-" * 40)
    print(f"4-class model: {sum(p.numel() for p in model_4.parameters()):,} parameters")
    print(f"6-class model: {sum(p.numel() for p in model_6.parameters()):,} parameters")
    print(f"10-class model: {sum(p.numel() for p in model_10.parameters()):,} parameters")
    print(f"Fusion model: {sum(p.numel() for p in fusion_model.parameters()):,} parameters")
    
    # Test forward pass
    logits_4, emb_4 = model_4(x)
    logits_6, emb_6 = model_6(x)
    logits_10, emb_10 = model_10(x)
    
    print(f"\nForward Pass Test:")
    print(f"4-class output: {logits_4.shape}, embedding: {emb_4.shape}")
    print(f"6-class output: {logits_6.shape}, embedding: {emb_6.shape}")
    print(f"10-class output: {logits_10.shape}, embedding: {emb_10.shape}")
    
    # Test fusion model
    fusion_logits = fusion_model(emb_4, emb_6)
    print(f"Fusion output: {fusion_logits.shape}")


def main():
    """Main function to run the complete example"""
    parser = argparse.ArgumentParser(description='DP-SGD Example')
    parser.add_argument('--skip_training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--demo_only', action='store_true', help='Run demonstrations only')
    
    args = parser.parse_args()
    
    print("DP-SGD Fusion Architecture Example")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not args.demo_only:
        # Run training
        if not args.skip_training:
            model_dir = run_training_example()
        else:
            model_dir = './privacy_output'
            print("Skipping training phase...")
        
        # Run evaluation
        if not args.skip_evaluation:
            run_evaluation_example(model_dir)
        else:
            print("Skipping evaluation phase...")
    
    # Run demonstrations
    demonstrate_model_architecture()
    demonstrate_privacy_analysis()
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the output directories for results")
    print("2. Examine the training curves and privacy analysis")
    print("3. Experiment with different noise multipliers")
    print("4. Try different privacy budgets")


if __name__ == '__main__':
    main()
