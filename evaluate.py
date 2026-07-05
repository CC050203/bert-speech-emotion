# ============================================================
# evaluate.py - 在测试集上评估训练好的模型
# ============================================================
# 功能:
# 1. 加载训练好的 best_model.pt
# 2. 在 1200 条测试集上评估
# 3. 生成混淆矩阵图 (PPT素材!)
# 4. 输出详细分类报告
#
# 使用:
#   python evaluate.py
# ============================================================

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from models.bert_classifier import BertClassifier
from utils.data_loader import load_data
from utils.metrics import compute_metrics, print_metrics, plot_confusion_matrix


# ============================================================
# 配置
# ============================================================
CONFIG = {
    'bert_model': 'bert-base-chinese',
    'num_classes': 2,
    'max_length': 128,
    'batch_size': 16,
    'test_path': 'data/test.csv',
    'model_path': 'checkpoints/best_model.pt',
    'save_dir': 'checkpoints',
}

CLASS_NAMES = ['负面', '正面']  # label 0=负面, 1=正面


@torch.no_grad()
def evaluate_on_test():
    """在测试集上评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ===== 加载 Tokenizer 和数据 =====
    print("\n加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model'])

    print("加载测试数据...")
    test_loader = load_data(
        CONFIG['test_path'], tokenizer,
        batch_size=CONFIG['batch_size'],
        max_length=CONFIG['max_length'],
        shuffle=False
    )
    print(f"测试集大小: {len(test_loader.dataset)}")

    # ===== 加载训练好的模型 =====
    print("\n加载训练好的模型...")
    model = BertClassifier(
        num_classes=CONFIG['num_classes'],
        bert_model_name=CONFIG['bert_model']
    )
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已从 {CONFIG['model_path']} 加载")

    # ===== 在测试集上预测 =====
    print("\n开始在测试集上评估...")
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(test_loader, desc='Testing'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # ===== 计算指标 =====
    avg_loss = total_loss / len(test_loader)
    metrics = compute_metrics(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"Loss: {avg_loss:.4f}")
    print_metrics(metrics, prefix='  ')

    # ===== 详细分类报告 =====
    print("\n" + "=" * 60)
    print("详细分类报告 (Classification Report)")
    print("=" * 60)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    # 保存分类报告到文本文件
    report_path = os.path.join(CONFIG['save_dir'], 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BERT 语音情感识别 - 测试集评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试集大小: {len(all_labels)}\n")
        f.write(f"Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n\n")
        f.write("详细分类报告:\n")
        f.write(report)
    print(f"\n分类报告已保存到: {report_path}")

    # ===== 绘制混淆矩阵 =====
    print("\n生成混淆矩阵图...")
    cm_path = os.path.join(CONFIG['save_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, save_path=cm_path)

    print("\n" + "=" * 60)
    print("✅ 评估完成! 生成的文件:")
    print(f"  - {report_path}  (分类报告)")
    print(f"  - {cm_path}  (混淆矩阵图)")
    print("=" * 60)


if __name__ == '__main__':
    evaluate_on_test()
