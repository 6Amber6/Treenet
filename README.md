# CE Optimization: Multi-Model Ensemble for Adversarial Robustness

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project implements a **Multi-Model Ensemble approach with Confidence-Based Fusion** for improving adversarial robustness on CIFAR-10. The core idea is to replace a single-path CNN with a tree-structured/parallel convolutional neural network that can simultaneously improve both clean accuracy and adversarial robustness when combined with adversarial training methods like TRADES/MART.

## Key Innovation

### Multi-Model Ensemble Architecture
- **7-class Animal Model (M1)**: Specialized classifier for animal categories (bird, cat, deer, dog, frog, horse + unknown)
- **5-class Vehicle Model (M2)**: Specialized classifier for vehicle categories (airplane, automobile, ship, truck + unknown)

### Confidence-Based Fusion Strategy
```
Final Prediction = Animal_Confidence × Animal_Logits + Vehicle_Confidence × Vehicle_Logits
```

Where confidence is calculated as: `confidence = 1 - P(unknown_class)`

## Architecture

### Model Structure
```python
class LightTreeResNet_Unknown(nn.Module):
    def __init__(self, block, subroot_num_blocks, num_classes=10, device='cpu'):
        super().__init__()
        # Two specialized sub-models
        self.subroot_animal = LightResnet(block, subroot_num_blocks, num_classes=7, device=device)
        self.subroot_vehicle = LightResnet(block, subroot_num_blocks, num_classes=5, device=device)
```

### Training Strategy
1. **Pre-training Phase**: Train M1 and M2 independently
2. **Joint Fine-tuning Phase**: Optimize ensemble with weighted loss function

### Loss Function
```python
Total Loss = α₁ × Fusion_Loss + α₂ × Animal_Loss + α₃ × Vehicle_Loss
```

## 🚀 Quick Start

### Environment Setup 

1. **Basic Preparation**
```bash
# Switch to a persistent directory
mkdir -p /workspace && cd /workspace

# Install common tools
apt-get update -y && apt-get install -y git tmux htop unzip
```

2. **Clone Repository**
```bash
cd /workspace
git clone https://github.com/your-username/CE-optimization-solution.git
cd CE-optimization-solution
```

3. **Create Python Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -V     # Verify version
pip -V
pip install --upgrade pip wheel setuptools
```

4. **Install PyTorch (CUDA 12.1)**
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio
```

5. **Install Dependencies**
```bash
# Core dependencies
pip install pandas tqdm matplotlib tensorboard

# Disable wandb (optional)
pip install wandb
export WANDB_MODE=disabled
export WANDB_SILENT=true

# AutoAttack for evaluation
pip install git+https://github.com/fra31/auto-attack.git
```

6. **Prepare Data Directory**
```bash
mkdir -p ./data
```

### Training Pipeline

#### Step 1: Pre-train Sub-models (First Time Only)
```bash
WANDB_MODE=disabled WANDB_SILENT=true \
python adversarial_robustness_pytorch/train-parx.py \
  --data-dir ./data \
  --log-dir ./log_ce_optimization \
  --desc ce_pretrain_once \
  --data cifar10 \
  --batch-size 512 \
  --model lighttreeresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6 \
  --train_submodels True \
  --unknown_classes True
```

#### Step 2: Joint Fine-tuning (Main Training)
```bash
WANDB_MODE=disabled WANDB_SILENT=true \
python adversarial_robustness_pytorch/train-parx.py \
  --data-dir ./data \
  --log-dir ./log_ce_optimization \
  --desc ce_ft_beta6_baseline \
  --data cifar10 \
  --batch-size 512 \
  --model lighttreeresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6 \
  --unknown_classes True \
  --strategy constant
```

## Key Parameters

### Model Parameters
- `--model lighttreeresnet20`: Tree-structured ResNet-20
- `--unknown_classes True`: Enable unknown class handling
- `--batch-size 512`: Training batch size

### Training Parameters
- `--num-adv-epochs 100`: Number of adversarial training epochs
- `--beta 6`: TRADES loss weight
- `--adv-eval-freq 10`: Adversarial evaluation frequency

### CE Optimization Parameters
- `--alpha1 1.0`: Fusion loss weight
- `--alpha2 1.0`: Animal model loss weight  
- `--alpha3 1.0`: Vehicle model loss weight
- `--strategy constant`: Weight update strategy (constant/linear/decay)

## Advanced Configuration

### Weight Update Strategies
```python
# Constant weights
--strategy constant

# Linear decay
--strategy linear

# Exponential decay  
--strategy decay --decay_factor 0.98
```

### Adversarial Training Methods
- **TRADES**: `--beta 6` (default)
- **MART**: `--mart True --beta 6`
- **PGD**: `--beta None`

### Evaluation Attacks
- **PGD-L∞**: `--attack linf-pgd --attack-eps 8/255`
- **PGD-L2**: `--attack l2-pgd --attack-eps 128/255`
- **FGSM**: `--attack fgsm --attack-eps 8/255`

## 📈 Monitoring & Evaluation

### Training Metrics
- **Clean Accuracy**: Standard classification accuracy
- **Adversarial Accuracy**: Robustness against adversarial attacks
- **Sub-category Accuracy**: Animal/Vehicle specific performance
- **Alpha Weights**: Dynamic weight evolution
- **Loss Decomposition**: Individual loss components

### Evaluation Commands
```bash
# AutoAttack evaluation
python adversarial_robustness_pytorch/eval-aa.py \
  --data-dir ./data \
  --log-dir ./log_ce_optimization \
  --desc ce_joint_finetune \
  --data cifar10

# Category-wise analysis
python adversarial_robustness_pytorch/category_group.py \
  --data-dir ./data \
  --log-dir ./log_ce_optimization \
  --desc ce_joint_finetune \
  --data cifar10
```


## Project Structure

```
CE-optimization-solution/
├── adversarial_robustness_pytorch/
│   ├── train-parx.py              # Main training script
│   ├── par_x/
│   │   ├── train.py              # Ensemble training logic
│   │   └── model.py              # Tree model definition
│   ├── par/
│   │   └── individual_train.py   # Sub-model training
│   ├── core/
│   │   ├── attacks/              # Adversarial attack implementations
│   │   ├── models/               # Model architectures
│   │   └── utils/                # Utility functions
│   └── eval-aa.py               # AutoAttack evaluation
├── data/                        # CIFAR-10 dataset
├── log_ce_optimization/         # Training logs and weights
└── README.md                    # This file
```

## Technical Details

### Confidence Calculation
```python
# Animal model confidence
conf_animal = 1 - F.softmax(animal_logits, dim=1)[:, -1]

# Vehicle model confidence  
conf_vehicle = 1 - F.softmax(vehicle_logits, dim=1)[:, -1]
```

### Fusion Process
```python
# Map sub-model logits to 10-class space
animal_logits[:, animal_classes] = subroot_animal_logits[:, :-1]
vehicle_logits[:, vehicle_classes] = subroot_vehicle_logits[:, :-1]

# Weighted fusion
final_logits = conf_animal * animal_logits + conf_vehicle * vehicle_logits
```

### Label Mapping
```python
# CIFAR-10 class mapping
animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
vehicle_classes = [0, 1, 8, 9]       # airplane, automobile, ship, truck
```

## Results

### Performance Comparison
| Model | Clean Acc | Adv Acc  | 
|-------|-----------|---------------|
| Baseline | 72.20% | 35.55% |
| CE Ensemble | **76.47%** | **41.42%** |



---

**Note**: This project demonstrates how multi-model ensemble approaches with confidence-based fusion can improve both clean accuracy and adversarial robustness, providing a novel solution for robust deep learning systems.



