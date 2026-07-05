# ============================================================
# bert_classifier.py - 基于BERT的情感分类模型
# ============================================================
# 模型结构:
#
#   输入文本 [CLS] 我 今 天 很 开 心 [SEP]
#        ↓
#   BERT编码器 (12层Transformer Encoder, 1.1亿参数)
#        ↓
#   取 [CLS] 位置的输出向量 (768维)
#        ↓
#   Dropout层 (防止过拟合)
#        ↓
#   全连接层 (768 → 类别数)
#        ↓
#   Softmax → 类别概率
#
# ============================================================

import torch
import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    基于BERT的文本分类模型

    继承 nn.Module 是PyTorch构建模型的标准方式
    需要实现:
        __init__: 定义网络层
        forward:  定义前向传播
    """

    def __init__(self, num_classes=2, bert_model_name='bert-base-chinese', dropout_rate=0.3):
        """
        参数:
            num_classes: 情感类别数 (二分类=2, 多分类按实际)
            bert_model_name: 预训练BERT的名字
                - 'bert-base-chinese': 中文BERT (我们用这个)
                - 'bert-base-uncased': 英文BERT
            dropout_rate: dropout概率, 缓解过拟合
        """
        super(BertClassifier, self).__init__()

        # ★ 加载预训练BERT
        # 第一次会自动从HuggingFace下载约400MB的模型权重
        # 下载后会缓存在 ~/.cache/huggingface/
        self.bert = BertModel.from_pretrained(bert_model_name)

        # BERT输出维度: bert-base是768, bert-large是1024
        bert_hidden_size = self.bert.config.hidden_size  # 768

        # Dropout层: 训练时随机"丢弃"部分神经元, 防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

        # 分类头: 一个全连接层
        # 输入: [CLS]的向量 (768维)
        # 输出: 各类别的logits (num_classes维)
        self.classifier = nn.Linear(bert_hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播

        参数:
            input_ids: [batch_size, seq_len] token的ID
            attention_mask: [batch_size, seq_len] 1表示真实token, 0表示padding
            token_type_ids: [batch_size, seq_len] 句子A/B标记

        返回:
            logits: [batch_size, num_classes] 各类别的得分(未归一化)
        """
        # ★ Step 1: 通过BERT得到上下文表示
        # outputs 是一个对象, 包含:
        #   - last_hidden_state: [batch, seq_len, 768] 每个token的向量
        #   - pooler_output:     [batch, 768] [CLS]的向量(经过一个tanh变换)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # ★ Step 2: 取 [CLS] 的向量代表整句话
        # 方法A: 用 pooler_output (BERT原作者推荐的方式)
        cls_output = outputs.pooler_output  # [batch, 768]

        # 方法B: 直接取 last_hidden_state 的第0个位置 (有时效果更好)
        # cls_output = outputs.last_hidden_state[:, 0, :]

        # ★ Step 3: dropout + 分类
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch, num_classes]

        return logits

    def get_attention_weights(self, input_ids, attention_mask):
        """
        辅助方法: 提取注意力权重 (用于可视化, 是你的加分项!)

        返回:
            attentions: tuple, 每层一个张量, 形状[batch, heads, seq_len, seq_len]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True  # ★ 关键参数: 让BERT返回注意力权重
        )
        return outputs.attentions


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    from transformers import BertTokenizer

    print("初始化模型...")
    model = BertClassifier(num_classes=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 测试输入
    text = "这部电影真的太精彩了!"
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=32)

    print(f"测试文本: {text}")
    print(f"输入shape: {inputs['input_ids'].shape}")

    # 前向传播
    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
    print(f"输出logits: {logits}")
    print(f"输出shape: {logits.shape}")  # 应该是 [1, 2]

    # 转为概率
    probs = torch.softmax(logits, dim=-1)
    print(f"类别概率: {probs}")
