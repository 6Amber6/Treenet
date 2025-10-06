# DP-SGD Privacy Implementation

基于论文 "A Theory to Instruct Differentially Private Learning via Clipping Bias Reduction" 的DP-SGD实现。

## 文件说明

### 核心文件
- **`dp_models.py`** - 严格按照论文实现的ResNet-20模型，使用GroupNorm
- **`dp_train.py`** - DP-SGD训练脚本，支持4类、6类、10类和融合模型
- **`dp_utils.py`** - 隐私计算、梯度裁剪、数据处理工具
- **`dp_eval.py`** - 模型评估和隐私分析脚本

### 辅助文件
- **`run_example.py`** - 完整示例脚本
- **`requirements.txt`** - 依赖包列表
- **`README.md`** - 本说明文件

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练所有模型
```bash
python dp_train.py --train_all --data_dir ./data --output_dir ./dp_output
```

### 3. 训练单个模型
```bash
# 训练4类模型
python dp_train.py --train_4class --epochs_4class 50 --noise_multiplier 1.0

# 训练6类模型
python dp_train.py --train_6class --epochs_6class 50 --noise_multiplier 1.0

# 训练10类模型
python dp_train.py --train_10class --epochs_10class 50 --noise_multiplier 1.0

# 训练融合模型
python dp_train.py --train_fusion --epochs_fusion 30 --noise_multiplier 1.0
```

### 4. 评估模型
```bash
python dp_eval.py --model_dir ./dp_output --output_dir ./evaluation_output
```

### 5. 运行完整示例
```bash
python run_example.py
```

## 参数说明

### DP-SGD参数
- `--noise_multiplier`: 噪声乘数 (默认: 1.0)
- `--max_grad_norm`: 最大梯度范数 (默认: 1.0)
- `--delta`: 隐私参数δ (默认: 1e-5)

### 训练参数
- `--lr`: 学习率 (默认: 0.01)
- `--batch_size`: 批次大小 (默认: 64)
- `--epochs_4class`: 4类模型训练轮数 (默认: 50)
- `--epochs_6class`: 6类模型训练轮数 (默认: 50)
- `--epochs_10class`: 10类模型训练轮数 (默认: 50)
- `--epochs_fusion`: 融合模型训练轮数 (默认: 30)

## 模型架构

### ResNet-20 (严格按照论文实现)
- 20层残差网络
- GroupNorm替代BatchNorm (groups=8)
- 支持4类、6类、10类分类
- 支持嵌入提取

### 融合模型
- 结合4类和6类模型的嵌入
- 使用DP-SGD训练
- 支持隐私保护

## 隐私保护

- **梯度裁剪**: 限制单个样本的影响
- **噪声添加**: 添加校准噪声保护隐私
- **隐私计算**: 跟踪隐私消耗(ε, δ)
- **RDP组合**: 使用Renyi差分隐私进行组合

## 输出结果

- 训练好的模型 (`.pth`文件)
- 训练历史记录 (包含隐私消耗)
- 评估结果和隐私分析
- 训练曲线和隐私曲线