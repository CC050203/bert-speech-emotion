# ============================================================
# data_loader.py - 数据加载与预处理
# ============================================================
# 作用:
# 1. 读取csv数据文件
# 2. 用BERT的Tokenizer对文本编码
# 3. 封装成PyTorch的Dataset和DataLoader
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmotionDataset(Dataset):
    """
    自定义情感数据集类
    继承自 PyTorch 的 Dataset, 必须实现 __len__ 和 __getitem__
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        参数:
            texts: 文本列表 (list of str)
            labels: 标签列表 (list of int)
            tokenizer: BERT的Tokenizer (用于将文本转为模型输入)
            max_length: 最大序列长度 (超过截断, 不足补齐)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        返回第idx个样本, 这是PyTorch加载数据的核心方法
        每条样本会被自动组成batch送入模型
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # ★ 关键步骤: 用BERT的tokenizer对文本编码
        # 输出包含3个张量:
        #   - input_ids: 每个token的ID (词典中的索引)
        #   - attention_mask: 1表示真实token, 0表示padding
        #   - token_type_ids: 区分句子A/B(单句任务全是0)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # 自动加 [CLS] 和 [SEP]
            max_length=self.max_length,
            padding='max_length',          # 不足max_length的补0
            truncation=True,               # 超过max_length的截断
            return_tensors='pt'            # 返回PyTorch张量
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),       # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(csv_path, tokenizer, batch_size=16, max_length=128, shuffle=True):
    """
    从CSV加载数据并返回DataLoader

    参数:
        csv_path: CSV文件路径, 必须包含'text'和'label'两列
        tokenizer: BERT tokenizer
        batch_size: 批大小
        max_length: 文本最大长度
        shuffle: 是否打乱(训练集打乱, 测试集不打乱)

    返回:
        DataLoader对象
    """
    # 读取CSV
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # 构造Dataset
    dataset = EmotionDataset(texts, labels, tokenizer, max_length)

    # 包装成DataLoader
    # num_workers=0 是为了兼容Windows; Linux/Mac可以设为2~4加速
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return loader


# ============================================================
# 测试代码: 直接运行此文件可以测试数据加载是否正常
# ============================================================
if __name__ == '__main__':
    from transformers import BertTokenizer

    print("加载Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    print("测试编码:")
    sample = "今天心情真好,阳光明媚!"
    encoded = tokenizer(sample, return_tensors='pt')
    print(f"原文: {sample}")
    print(f"input_ids: {encoded['input_ids']}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])}")
