# ============================================================
# train.py - BERT情感分类训练主脚本
# ============================================================
# 使用方法:
#   python train.py
#
# 流程:
# 1. 加载数据
# 2. 初始化BERT模型
# 3. 设置优化器和学习率调度
# 4. 训练循环 + 验证
# 5. 保存最佳模型
# 6. 绘制训练曲线
# ============================================================

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

# 导入自定义模块
from models.bert_classifier import BertClassifier
from utils.data_loader import load_data
from utils.metrics import compute_metrics, print_metrics, plot_training_curves


# ============================================================
# 超参数配置区 (可调整, 影响效果)
# ============================================================
CONFIG = {
    'bert_model': 'bert-base-chinese',  # 预训练模型
    'num_classes': 2,                    # 类别数(二分类:正/负)
    'max_length': 128,                   # 文本最大长度
    'batch_size': 16,                    # batch大小, 显存小调到8
    'learning_rate': 2e-5,               # 学习率(BERT推荐2e-5~5e-5)
    'num_epochs': 3,                     # 训练轮数(BERT通常3~5即可)
    'warmup_ratio': 0.1,                 # warmup比例
    'weight_decay': 0.01,                # 权重衰减
    'dropout_rate': 0.3,                 # dropout概率
    'train_path': 'data/train.csv',
    'dev_path': 'data/dev.csv',
    'save_dir': 'checkpoints',
    'seed': 42
}


def set_seed(seed):
    """固定随机种子, 保证实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """训练一个epoch"""
    model.train()  # 切换到训练模式(启用dropout等)
    total_loss = 0
    all_preds, all_labels = [], []

    # tqdm 显示进度条
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # 把数据搬到GPU(如果有)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        # ★ 训练四步走 (PyTorch标准流程)
        optimizer.zero_grad()  # 1. 梯度清零

        # 2. 前向传播
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        # 3. 反向传播
        loss.backward()

        # 梯度裁剪: 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. 更新参数
        optimizer.step()
        scheduler.step()

        # 记录
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # 实时显示loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


@torch.no_grad()  # 验证时不需要计算梯度, 节省显存加速
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()  # 切换到评估模式(禁用dropout)
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
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

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


def main():
    # ========== 准备工作 ==========
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # 自动选择设备: 有GPU用GPU, 否则用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    # ========== 加载Tokenizer和数据 ==========
    print("\n加载Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['bert_model'])

    print("加载训练数据...")
    train_loader = load_data(
        CONFIG['train_path'], tokenizer,
        batch_size=CONFIG['batch_size'],
        max_length=CONFIG['max_length'],
        shuffle=True
    )
    print(f"训练集大小: {len(train_loader.dataset)}")

    print("加载验证数据...")
    dev_loader = load_data(
        CONFIG['dev_path'], tokenizer,
        batch_size=CONFIG['batch_size'],
        max_length=CONFIG['max_length'],
        shuffle=False
    )
    print(f"验证集大小: {len(dev_loader.dataset)}")

    # ========== 初始化模型 ==========
    print("\n初始化BERT模型...")
    model = BertClassifier(
        num_classes=CONFIG['num_classes'],
        bert_model_name=CONFIG['bert_model'],
        dropout_rate=CONFIG['dropout_rate']
    )
    model.to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")

    # ========== 损失函数和优化器 ==========
    # 交叉熵损失: 分类任务标配
    criterion = nn.CrossEntropyLoss()

    # AdamW: BERT训练标配的优化器(改进版Adam, 加了权重衰减)
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # 学习率调度器: 先warmup再线性衰减(BERT训练标准做法)
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ========== 训练循环 ==========
    print("\n开始训练...")
    print("=" * 60)
    best_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 60)

        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"\n训练集结果:")
        print(f"  Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix='  ')

        # 验证
        val_loss, val_metrics = evaluate(model, dev_loader, criterion, device)
        print(f"\n验证集结果:")
        print(f"  Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix='  ')

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"\n★ 新的最佳模型! F1={best_f1:.4f}, 已保存到 {save_path}")

    # ========== 训练结束 ==========
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证F1: {best_f1:.4f}")
    print("=" * 60)

    # 绘制训练曲线
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        save_path=os.path.join(CONFIG['save_dir'], 'training_curves.png')
    )


if __name__ == '__main__':
    main()
