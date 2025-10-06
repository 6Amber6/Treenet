"""
Evaluation script for DP-SGD trained models
Includes privacy analysis and model performance evaluation
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privacy.dp_models import DP4Classifier, DP6Classifier, DP10Classifier, DPFusionModel
from privacy.dp_utils import DataProcessor, compute_accuracy, load_model


class DPEvaluator:
    """Evaluator for DP-SGD trained models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = DataProcessor()
        
        # Load models
        self.models = {}
        self.load_models()
        
        # Load data
        self.data_loaders = DataProcessor.create_data_loaders(
            args.data_dir, args.batch_size, args.num_workers
        )
    
    def load_models(self):
        """Load trained models"""
        model_paths = {
            '4class': os.path.join(self.args.model_dir, '4class_final.pth'),
            '6class': os.path.join(self.args.model_dir, '6class_final.pth'),
            '10class': os.path.join(self.args.model_dir, '10class_final.pth'),
            'fusion': os.path.join(self.args.model_dir, 'fusion_final.pth')
        }
        
        # Initialize models
        self.models['4class'] = DP4Classifier(groups=self.args.groups).to(self.device)
        self.models['6class'] = DP6Classifier(groups=self.args.groups).to(self.device)
        self.models['10class'] = DP10Classifier(groups=self.args.groups).to(self.device)
        self.models['fusion'] = DPFusionModel(embedding_dim=64, num_classes=10, groups=self.args.groups).to(self.device)
        
        # Load weights
        for model_name, model in self.models.items():
            if os.path.exists(model_paths[model_name]):
                load_model(model, model_paths[model_name])
                print(f"Loaded {model_name} model from {model_paths[model_name]}")
            else:
                print(f"Warning: {model_name} model not found at {model_paths[model_name]}")
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader, model_name: str) -> Dict:
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name} model...")
        
        # Compute accuracy
        accuracy = compute_accuracy(model, data_loader, self.device)
        
        # Compute per-class accuracy
        per_class_acc = self.compute_per_class_accuracy(model, data_loader)
        
        # Compute confidence statistics
        confidence_stats = self.compute_confidence_stats(model, data_loader)
        
        results = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'confidence_stats': confidence_stats
        }
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        return results
    
    def compute_per_class_accuracy(self, model: nn.Module, data_loader: DataLoader) -> Dict:
        """Compute per-class accuracy"""
        model.eval()
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if isinstance(model, (DP4Classifier, DP6Classifier, DP10Classifier)):
                    output, _ = model(data)
                else:
                    output = model(data)
                
                pred = output.argmax(dim=1)
                
                for i in range(target.size(0)):
                    label = target[i].item()
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    class_total[label] += 1
                    if pred[i] == target[i]:
                        class_correct[label] += 1
        
        per_class_acc = {}
        for class_id in class_correct:
            per_class_acc[class_id] = class_correct[class_id] / class_total[class_id]
        
        return per_class_acc
    
    def compute_confidence_stats(self, model: nn.Module, data_loader: DataLoader) -> Dict:
        """Compute confidence statistics"""
        model.eval()
        confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if isinstance(model, (DP4Classifier, DP6Classifier, DP10Classifier)):
                    output, _ = model(data)
                else:
                    output = model(data)
                
                probs = torch.softmax(output, dim=1)
                max_probs, pred = torch.max(probs, dim=1)
                
                for i in range(target.size(0)):
                    confidence = max_probs[i].item()
                    confidences.append(confidence)
                    
                    if pred[i] == target[i]:
                        correct_confidences.append(confidence)
                    else:
                        incorrect_confidences.append(confidence)
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'mean_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
            'mean_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0
        }
    
    def evaluate_fusion_model(self) -> Dict:
        """Evaluate fusion model using embeddings from 4-class and 6-class models"""
        print("\nEvaluating fusion model...")
        
        # Extract embeddings from test data
        self.models['4class'].eval()
        self.models['6class'].eval()
        
        embeddings_4class = []
        embeddings_6class = []
        targets = []
        
        with torch.no_grad():
            for data, target in self.data_loaders['full_test']:
                data, target = data.to(self.device), target.to(self.device)
                
                _, emb_4 = self.models['4class'](data)
                _, emb_6 = self.models['6class'](data)
                
                embeddings_4class.append(emb_4.cpu())
                embeddings_6class.append(emb_6.cpu())
                targets.append(target.cpu())
        
        embeddings_4class = torch.cat(embeddings_4class, dim=0)
        embeddings_6class = torch.cat(embeddings_6class, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Create fusion dataset
        fusion_dataset = torch.utils.data.TensorDataset(embeddings_4class, embeddings_6class, targets)
        fusion_loader = DataLoader(fusion_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        # Evaluate fusion model
        return self.evaluate_model(self.models['fusion'], fusion_loader, 'fusion')
    
    def analyze_privacy_spent(self) -> Dict:
        """Analyze privacy spent during training"""
        history_file = os.path.join(self.args.model_dir, 'training_history.json')
        
        if not os.path.exists(history_file):
            print("Warning: Training history not found")
            return {}
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        privacy_analysis = {}
        for model_name, model_history in history.items():
            if 'privacy_spent' in model_history and model_history['privacy_spent']:
                final_privacy = model_history['privacy_spent'][-1]
                privacy_analysis[model_name] = {
                    'final_epsilon': final_privacy[0],
                    'final_delta': final_privacy[1],
                    'privacy_curve': model_history['privacy_spent']
                }
        
        return privacy_analysis
    
    def plot_training_curves(self):
        """Plot training curves"""
        history_file = os.path.join(self.args.model_dir, 'training_history.json')
        
        if not os.path.exists(history_file):
            print("Warning: Training history not found")
            return
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Plot accuracy curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model_history) in enumerate(history.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            epochs = range(len(model_history['train_acc']))
            
            ax.plot(epochs, model_history['train_acc'], label='Train Acc', marker='o')
            ax.plot(epochs, model_history['test_acc'], label='Test Acc', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{model_name.upper()} Model')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'training_curves.png'))
        plt.close()
        
        # Plot privacy curves
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for model_name, model_history in history.items():
            if 'privacy_spent' in model_history and model_history['privacy_spent']:
                epsilons = [p[0] for p in model_history['privacy_spent']]
                epochs = range(len(epsilons))
                ax.plot(epochs, epsilons, label=f'{model_name} (ε)', marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Privacy Spent (ε)')
        ax.set_title('Privacy Spent Over Training')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'privacy_curves.png'))
        plt.close()
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("="*60)
        print("DP-SGD MODEL EVALUATION")
        print("="*60)
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Evaluate individual models
        results = {}
        
        # 4-class model
        if '4class' in self.models:
            results['4class'] = self.evaluate_model(
                self.models['4class'], self.data_loaders['animal_test'], '4class'
            )
        
        # 6-class model
        if '6class' in self.models:
            results['6class'] = self.evaluate_model(
                self.models['6class'], self.data_loaders['vehicle_test'], '6class'
            )
        
        # 10-class model
        if '10class' in self.models:
            results['10class'] = self.evaluate_model(
                self.models['10class'], self.data_loaders['full_test'], '10class'
            )
        
        # Fusion model
        if 'fusion' in self.models:
            results['fusion'] = self.evaluate_fusion_model()
        
        # Privacy analysis
        privacy_analysis = self.analyze_privacy_spent()
        results['privacy'] = privacy_analysis
        
        # Save results
        with open(os.path.join(self.args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, model_results in results.items():
            if model_name == 'privacy':
                continue
            print(f"\n{model_name.upper()} Model:")
            print(f"  Accuracy: {model_results['accuracy']:.4f}")
            print(f"  Mean Confidence: {model_results['confidence_stats']['mean_confidence']:.4f}")
        
        if privacy_analysis:
            print(f"\nPrivacy Analysis:")
            for model_name, privacy_info in privacy_analysis.items():
                print(f"  {model_name}: ε={privacy_info['final_epsilon']:.3f}, δ={privacy_info['final_delta']:.2e}")
        
        print(f"\nResults saved to {self.args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DP-SGD Model Evaluation')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='./privacy_output', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    # Model arguments
    parser.add_argument('--groups', type=int, default=8, help='Number of groups for GroupNorm')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = DPEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
