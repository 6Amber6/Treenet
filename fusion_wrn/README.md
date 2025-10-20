# Fusion WRN Training with MART/TRADES

## 核心文件

### 主要训练脚本
- **`train_fusion_wrn_adv_improved.py`** - 改进的训练脚本（支持MART/TRADES切换）
- **`train_fusion_wrn_adv.py`** - 原始训练脚本

### 对比实验
- **`compare_mart_trades.py`** - 完整MART vs TRADES对比
- **`quick_compare.py`** - 快速测试对比
- **`run_comparison.sh`** - 交互式运行脚本

### 其他实现
- **`fusion_trades_robust.py`** - 原始TRADES实现
- **`fusion_trades_robust_mart.py`** - 原始MART实现  
- **`fusion_trades_mart_savesub.py`** - 带子模型保存的MART实现
- **`dp_utils_diffusion.py`** - 扩散模型工具

## 快速使用

```bash
# 快速测试（5分钟）
python quick_compare.py

# 完整对比（2-3小时）
python compare_mart_trades.py

# 交互式运行
./run_comparison.sh
```

## 参数设置
- Batch Size: 128
- Submodel Epochs: 100  
- Fusion Epochs: 120
- Learning Rate: 0.1
- TRADES Beta: 8.0
- Attack: PGD-L∞, eps=8/255
