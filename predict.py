# ============================================================
# predict.py - 用训练好的模型做单条预测
# ============================================================
# 使用:
#   from predict import EmotionPredictor
#   predictor = EmotionPredictor('checkpoints/best_model.pt')
#   result = predictor.predict("今天心情真好!")
#   print(result)  # {'label': '正面', 'confidence': 0.98}
# ============================================================

import torch
from transformers import BertTokenizer
from models.bert_classifier import BertClassifier


class EmotionPredictor:
    """情感预测器封装类"""

    LABEL_MAP = {0: '负面', 1: '正面'}  # 二分类
    # 多分类版本(如果是7情感分类):
    # LABEL_MAP = {0: '愤怒', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '中性', 5: '悲伤', 6: '惊讶'}

    def __init__(self, model_path, bert_model_name='bert-base-chinese', num_classes=2):
        """加载训练好的模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.model = BertClassifier(num_classes=num_classes, bert_model_name=bert_model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 评估模式

    @torch.no_grad()
    def predict(self, text, max_length=128):
        """
        预测单条文本

        参数:
            text: 输入文本

        返回:
            dict: {
                'label': 类别名,
                'label_id': 类别ID,
                'confidence': 置信度,
                'all_probs': 所有类别概率
            }
        """
        # 编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        # 预测
        logits = self.model(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

        return {
            'text': text,
            'label': self.LABEL_MAP[pred_id],
            'label_id': pred_id,
            'confidence': confidence,
            'all_probs': {self.LABEL_MAP[i]: probs[0][i].item() for i in range(len(self.LABEL_MAP))}
        }

    def predict_batch(self, texts, max_length=128, batch_size=16):
        """批量预测"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.predict(text, max_length))
        return results


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    predictor = EmotionPredictor('checkpoints/best_model.pt')

    test_texts = [
        "这部电影太精彩了, 我看了好几遍!",
        "服务态度太差, 再也不会来了",
        "今天天气不错,适合出去走走",
        "心情糟透了, 一切都不顺利"
    ]

    print("=" * 60)
    print("情感预测演示")
    print("=" * 60)
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n文本: {result['text']}")
        print(f"预测: {result['label']} (置信度: {result['confidence']:.2%})")
        print(f"详细: {result['all_probs']}")
