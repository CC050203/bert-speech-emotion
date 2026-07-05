# ============================================================
# visualize_attention.py - BERT注意力可视化 (加分项!) V3
# ============================================================
# V3 修复: 新版 transformers 5.x 即使在 from_pretrained 指定 eager
#          也可能不生效。这里在脚本里直接重新加载一个 eager 模式的
#          BERT 来提取注意力, 确保 100% 可用。
#
# 使用:
#   python visualize_attention.py
#
# 生成文件:
#   checkpoints/attention_single.png      单头注意力热力图
#   checkpoints/attention_all_heads.png   某层全部12个头
#   checkpoints/attention_layers.png      不同层的注意力对比
# ============================================================

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import BertTokenizer, BertModel

# matplotlib 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = 'checkpoints/best_model.pt'
SAVE_DIR = 'checkpoints'


def load_attention_model():
    """
    加载一个专门用于注意力可视化的 BERT (eager 模式)

    说明: 我们直接加载原始 bert-base-chinese 的 eager 版本来提取注意力。
    注意力模式是 BERT 的固有结构, 微调只改变了分类头, BERT 主体的注意力
    结构本身是预训练学到的, 用原始 BERT 提取注意力来做原理演示完全合理。
    如果想用微调后的权重, 下面也提供了加载方式。
    """
    print("加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    print("加载 BERT (eager 注意力模式)...")
    # 关键: attn_implementation='eager' 确保能导出注意力
    bert = BertModel.from_pretrained(
        'bert-base-chinese',
        attn_implementation='eager'
    )

    # 尝试把微调后的 BERT 权重加载进来 (可选)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            # 提取出 bert.* 开头的权重 (微调模型里 BERT 部分的权重)
            bert_state = {}
            for k, v in state_dict.items():
                if k.startswith('bert.'):
                    bert_state[k[len('bert.'):]] = v
            if bert_state:
                bert.load_state_dict(bert_state, strict=False)
                print(f"✅ 已加载微调后的 BERT 权重: {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ 加载微调权重失败 ({e}), 使用原始 BERT")
    else:
        print(f"⚠️ 未找到 {MODEL_PATH}, 使用原始 BERT")

    bert.eval()
    return bert, tokenizer


def get_attentions(bert, tokenizer, text, max_length=64):
    """提取注意力权重"""
    inputs = tokenizer(text, return_tensors='pt', padding=True,
                       truncation=True, max_length=max_length)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    actual_len = inputs['attention_mask'][0].sum().item()
    tokens = tokens[:actual_len]

    with torch.no_grad():
        outputs = bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_attentions=True
        )
    # attentions: tuple, 12层, 每层 [batch, heads, seq, seq]
    attentions = outputs.attentions
    return attentions, tokens, actual_len


def visualize_attention_heatmap(bert, tokenizer, text, layer=-1, head=0, save_path=None):
    """可视化单层单头的注意力热力图"""
    attentions, tokens, actual_len = get_attentions(bert, tokenizer, text)
    attn = attentions[layer][0, head, :actual_len, :actual_len].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        cbar_kws={'label': '注意力权重'},
        square=True
    )
    layer_name = '最后一层' if layer == -1 else f'第{layer+1}层'
    plt.title(f'BERT 注意力热力图 ({layer_name}, 注意力头 {head})\n输入文本: "{text}"', fontsize=12)
    plt.xlabel('Key (被关注的词)', fontsize=11)
    plt.ylabel('Query (发起关注的词)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
    plt.close()


def visualize_all_heads(bert, tokenizer, text, layer=-1, save_path=None):
    """可视化某一层全部12个注意力头"""
    attentions, tokens, actual_len = get_attentions(bert, tokenizer, text, max_length=32)
    layer_attn = attentions[layer][0, :, :actual_len, :actual_len].cpu().numpy()
    num_heads = layer_attn.shape[0]

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    layer_name = '最后一层' if layer == -1 else f'第{layer+1}层'
    fig.suptitle(f'BERT {layer_name} - 全部 {num_heads} 个注意力头的注意力分布\n输入文本: "{text}"',
                 fontsize=15)

    for h in range(num_heads):
        ax = axes[h // 4, h % 4]
        sns.heatmap(
            layer_attn[h],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            cbar=False,
            square=True,
            ax=ax
        )
        ax.set_title(f'注意力头 {h}', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
    plt.close()


def visualize_layers_comparison(bert, tokenizer, text, head=0, save_path=None):
    """对比不同层的注意力 (浅层vs深层)"""
    attentions, tokens, actual_len = get_attentions(bert, tokenizer, text, max_length=32)

    layers_to_show = [0, 3, 7, 11]  # 第1,4,8,12层
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f'BERT 不同层的注意力对比 (注意力头 {head})\n输入文本: "{text}"', fontsize=15)

    for idx, layer in enumerate(layers_to_show):
        ax = axes[idx]
        attn = attentions[layer][0, head, :actual_len, :actual_len].cpu().numpy()
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            cbar=False,
            square=True,
            ax=ax
        )
        ax.set_title(f'第 {layer+1} 层', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)

    bert, tokenizer = load_attention_model()

    # 选一个有情感色彩的句子做演示
    text = "这家酒店的服务态度非常好"

    print(f"\n可视化文本: \"{text}\"")
    print("=" * 60)

    print("\n[1/3] 生成单头注意力热力图...")
    visualize_attention_heatmap(
        bert, tokenizer, text, layer=-1, head=0,
        save_path=os.path.join(SAVE_DIR, 'attention_single.png')
    )

    print("\n[2/3] 生成全部12个注意力头的图...")
    visualize_all_heads(
        bert, tokenizer, text, layer=-1,
        save_path=os.path.join(SAVE_DIR, 'attention_all_heads.png')
    )

    print("\n[3/3] 生成不同层的注意力对比图...")
    visualize_layers_comparison(
        bert, tokenizer, text, head=0,
        save_path=os.path.join(SAVE_DIR, 'attention_layers.png')
    )

    print("\n" + "=" * 60)
    print("✅ 注意力可视化完成! 生成的文件:")
    print(f"  - {SAVE_DIR}/attention_single.png      单头注意力热力图")
    print(f"  - {SAVE_DIR}/attention_all_heads.png   全部12个头")
    print(f"  - {SAVE_DIR}/attention_layers.png      不同层对比")
    print("=" * 60)
