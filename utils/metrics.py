# ============================================================
# metrics.py - 模型评估指标
# ============================================================
# 实现:
# 1. 准确率 Accuracy
# 2. 精确率 Precision / 召回率 Recall / F1分数
# 3. 混淆矩阵可视化
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_metrics(y_true, y_pred, average='weighted'):
    """
    计算多种评估指标

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 多分类时如何平均
            'macro' - 各类别平等加权
            'weighted' - 按类别样本数加权

    返回:
        dict 包含 accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_metrics(metrics, prefix=''):
    """格式化输出指标"""
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1-Score:  {metrics['f1']:.4f}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵热力图 (PPT用图!)

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名列表, 如 ['负面', '正面']
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,              # 在格子内显示数字
        fmt='d',                 # 整数格式
        cmap='Blues',            # 颜色方案
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': '样本数'}
    )
    plt.title('混淆矩阵 Confusion Matrix', fontsize=14)
    plt.ylabel('真实标签 True Label', fontsize=12)
    plt.xlabel('预测标签 Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")

    plt.show()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    绘制训练曲线 (PPT用图!)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-o', label='训练损失')
    axes[0].plot(epochs, val_losses, 'r-s', label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练/验证 损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: 准确率曲线
    axes[1].plot(epochs, train_accs, 'b-o', label='训练准确率')
    axes[1].plot(epochs, val_accs, 'r-s', label='验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('训练/验证 准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")

    plt.show()
