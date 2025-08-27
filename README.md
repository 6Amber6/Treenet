# tree_model

> Experiments on **hierarchical (HD-CNN)** and **parallel** CNN architectures for **privacy & robustness** under adversarial training.  
> Datasets: CIFAR-10 (clean) + CIFAR-10-C (corruptions) + adversarial evaluation (AutoAttack / robustness benchmarks).

---

## ✨ Goals
- **Compare**: flat baseline CNN vs. hierarchical/parallel variants
- **Train** with TRADES / MART and evaluate robustness
- **Measure** clean accuracy, adversarial accuracy, and per-category behavior

---

## 🧠 Model Variants

### 1) Flat CNN (Baseline)
- `lightresnet20` (≈175k params)
- Standard 10-class classifier.

### 2) Parallel (2-Experts on Coarse Groups: **animal** / **vehicle**)
- **Coarse routing** into two heads:
  - **Animal head** (4 classes)
  - **Vehicle head** (6 classes)
- Fusion options:
  - **Independent** training + **Mixture-of-Experts (MoE)** fusion
  - **Joint** training (final logits supervised) with optional auxiliary losses
  - Weighted or linear-schedule **alpha** for multi-loss

### 3) HD-Tree (Hierarchical)
- **Root (coarse)** → **Subroots/Experts**
- **Soft route**: weighted logits by root probabilities (with small “other -5” encouragement; expand to 10-dim)
- **Hard route**: if-based routing by root predictions
- Supports **pretraining** root/subroots then **finetuning** end-to-end.

> “2 - 4,6 model” refers to **2-way coarse routing** (animal vs vehicle) with **4-class** and **6-class** experts.  
> “10 - 4,6 model” refers to composing experts so that final space is **10 classes** via hierarchical fusion.

---

## 🔧 Environment & Setup

```bash
# (Recommended) Create env
conda create -n tree-model python=3.10 -y
conda activate tree-model

# Install deps
pip install -r requirements.txt

# Prepare data
# data/ should contain CIFAR-10; optional CIFAR-10-C for PRIME augmentations
```
 
## 🚀 Training & Evaluation
### PRIME Augmentation (optional)
From: https://github.com/amodas/PRIME-augmentations.git
```python
python -u train.py \
  --config=config/cifar10_cfg.py \
  --config.model.name=lighttreeresnet20 \
  --config.use_prime=True \
  --config.cc_dir=./data/cifar10c/CIFAR-10-C \
  --config.save_dir=./PRIME/linear_lr/
```
### Adversarial Training (TRADES / MART)
From: https://github.com/imrahulr/adversarial_robustness_pytorch.git
- Train tree model with TRADES
```python
python train-tree.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc trades_tree \
  --data cifar10 \
  --batch-size 1024 \
  --model lighttreeresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6
```
- Train tree model with MART
```python 
python train-tree.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc test \
  --data cifar10 \
  --batch-size 1024 \
  --model lighttreeresnet20 \
  --num-adv-epochs 1 \
  --adv-eval-freq 10 \
  --beta 6 \
  --mart
```
- Train origin (flat) model
```python 
python train.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc trades_origin \
  --data cifar10 \
  --batch-size 1024 \
  --model lightresnet20 \
  --num-adv-epochs 100 \
  --adv-eval-freq 10 \
  --beta 6
```
### Evaluation
- AutoAttack
```python
python eval-aa.py \
  --data-dir ./data \
  --log-dir ./log_test \
  --desc test \
  --data cifar10
```
- Robustness Benchmark (e.g., Linf)
```python 
python eval-rb.py \
  --data-dir ./data \
  --log-dir ./baseline_log/origin \
  --desc origin_10_classifier \
  --data cifar10 \
  --threat Linf
```
- Category Grouping / Analysis
```python 
python category_group.py \
  --data-dir ./data \
  --log-dir ./baseline_log/origin \
  --desc origin_10_classifier \
  --data cifar10
```
## 📊 Results (Aug 15 snapshot)

> Metrics are **Top-1 accuracy** on CIFAR-10 (clean) and under adversarial evaluation.  
> Values are averaged over representative runs; exact results depend on seeds and hyperparameters.

---

### Baseline (Flat CNN)

| Model                   | Params   | Clean Acc | Adv Acc |
|--------------------------|----------|-----------|---------|
| lightresnet20 (origin)  | 175,258  | **72.20%**| **35.55%** |

---

### Parallel Network — **animal + vehicle** (2-Experts)

| Run / Setting | Notes                                                                                   | Clean Acc | Adv Acc |
|---------------|-----------------------------------------------------------------------------------------|-----------|---------|
| `init`        | Independent training; animal head trained on all data; MoE using `conf = 1 - pred[-1]` | 72.71%    | 41.00%  |
| `init` (MoE2) | Independent training; MoE: `conf*animal_logits + conf*vehicle_logits`                   | 73.07%    | 39.18%  |
| `CE`          | Joint training with CE on final logits (MoE fusion)                                     | 76.89%    | 39.06%  |
| `init_nopretrain` | Joint CE(final) training, no pretrain, MoE fusion                                   | 75.62%    | 38.42%  |
| `1-1 linear`  | Joint CE(final) + α·CE_animal + α·CE_vehicle; α linear schedule                         | 75.77%    | 38.37%  |

---

### Parallel Network — **root + animal + vehicle** (3-branch)

| Run / Setting     | Notes                                                                                                               | α Schedule        | Clean Acc | Adv Acc |
|-------------------|---------------------------------------------------------------------------------------------------------------------|-------------------|-----------|---------|
| `inti-par3`       | Independent training; root (coarse) all-data; **soft route** with other-5 encouragement, root-weighted logits       | —                 | 74.32%    | 39.64%  |
| `CE-par3`         | Independent pretrains + joint CE(final logits), soft routing, weighted loss                                         | —                 | 77.10%    | 38.93%  |
| `alpha-par3-static` | Joint CE(final) + α·root + α·animal + α·vehicle (weights 1:0.5:0.3:0.3, fixed)                                    | static            | **78.56%**| 32.87%  |
| `alpha-par3-linear` | Same as above, but α varies linearly during training                                                              | linear schedule   | 76.46%    | 33.96%  |

---

### HD-Tree (Hierarchical CNN)

| Run / Setting | Notes                                                                                                       | Clean Acc | Adv Acc |
|---------------|-------------------------------------------------------------------------------------------------------------|-----------|---------|
| `pretrain coarse` + `pretrain subroots` + `finetune` | Root trained on all data, subroot1 trained on animal-only, subroot2 on vehicle-only; soft/hard routes tested | 74.28%    | 37.37%  |

---

## 🔍 Key Observations
- **Parallel / hierarchical models** often achieve **higher clean accuracy** than baseline (up to 78.56%).  
- **Robust accuracy** gains depend strongly on fusion strategy and loss balancing:
  - Independent → joint training improves clean accuracy.
  - Weighted loss (α schedules) shifts the clean vs. robust tradeoff.
- **HD-Tree** shows promise but requires careful route calibration (soft vs. hard).  

---



