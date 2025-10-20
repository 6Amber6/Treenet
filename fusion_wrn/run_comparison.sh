#!/bin/bash
# MART vs TRADES 对比运行脚本

echo "=========================================="
echo "MART vs TRADES 对比实验"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
python --version

# 检查必要文件
echo "检查必要文件..."
if [ ! -f "train_fusion_wrn_adv_improved.py" ]; then
    echo "❌ 找不到 train_fusion_wrn_adv_improved.py"
    exit 1
fi

if [ ! -f "compare_mart_trades.py" ]; then
    echo "❌ 找不到 compare_mart_trades.py"
    exit 1
fi

echo "✓ 所有必要文件存在"

# 创建日志目录
mkdir -p logs_comparison
mkdir -p logs_quick

echo ""
echo "选择运行模式："
echo "1. 快速测试 (5分钟，用于验证代码)"
echo "2. 完整对比 (2-3小时，正式实验)"
echo "3. 单独运行TRADES"
echo "4. 单独运行MART"
echo ""

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "🚀 运行快速测试..."
        python quick_compare.py
        ;;
    2)
        echo "🚀 运行完整对比..."
        echo "⚠️  这将需要2-3小时，请确保有足够时间"
        read -p "确认继续? (y/N): " confirm
        if [[ $confirm == [yY] ]]; then
            python compare_mart_trades.py
        else
            echo "取消运行"
        fi
        ;;
    3)
        echo "🚀 运行TRADES训练..."
        python train_fusion_wrn_adv_improved.py \
            --desc "single_trades_$(date +%H%M%S)" \
            --data-dir ./data \
            --log-dir ./logs_single \
            --batch-size 128 \
            --epochs-m 100 \
            --epochs-g 120 \
            --lr 0.1 \
            --lr-m 0.1 \
            --aux_w 0.02 \
            --beta 8.0 \
            --attack linf-pgd \
            --attack-eps 8/255 \
            --attack-step 2/255 \
            --attack-iter 12 \
            --ema-decay 0.999 \
            --train-mode all \
            --seed 42
        ;;
    4)
        echo "🚀 运行MART训练..."
        python train_fusion_wrn_adv_improved.py \
            --desc "single_mart_$(date +%H%M%S)" \
            --data-dir ./data \
            --log-dir ./logs_single \
            --batch-size 128 \
            --epochs-m 100 \
            --epochs-g 120 \
            --lr 0.1 \
            --lr-m 0.1 \
            --aux_w 0.02 \
            --attack linf-pgd \
            --attack-eps 8/255 \
            --attack-step 2/255 \
            --attack-iter 12 \
            --ema-decay 0.999 \
            --train-mode all \
            --use-mart \
            --seed 42
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "📁 日志保存在:"
echo "   - logs_comparison/ (完整对比)"
echo "   - logs_quick/ (快速测试)"
echo "   - logs_single/ (单独训练)"
echo ""
echo "📊 查看结果:"
echo "   - 检查训练日志中的最终准确率"
echo "   - 对比两个方法的性能差异"
echo ""
