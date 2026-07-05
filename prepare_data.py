# ============================================================
# prepare_data.py - 自动下载与准备数据集 (V2 - 修复版)
# ============================================================
# 数据集: lansinuote/ChnSentiCorp (parquet 格式版本)
# - 9600 训练样本 + 1200 验证样本 + 1200 测试样本
# - 二分类: 0=负面, 1=正面
# - 内容: 酒店评论、商品评论等
#
# 使用方法:
#   python prepare_data.py
# ============================================================

import os
import pandas as pd
from datasets import load_dataset


def prepare_chn_senti_corp():
    """下载并准备 ChnSentiCorp 数据集 (Parquet 格式版本)"""
    os.makedirs('data', exist_ok=True)

    print("=" * 60)
    print("下载 ChnSentiCorp 中文情感数据集 (parquet 版本)...")
    print("来源: lansinuote/ChnSentiCorp")
    print("=" * 60)

    # 使用 parquet 格式的 ChnSentiCorp
    dataset = load_dataset('lansinuote/ChnSentiCorp')

    print(f"\n数据集结构: {dataset}")

    splits = {
        'train': 'data/train.csv',
        'validation': 'data/dev.csv',
        'test': 'data/test.csv'
    }

    for split_name, save_path in splits.items():
        if split_name not in dataset:
            print(f"警告: 数据集中没有 {split_name} 拆分")
            continue

        df = pd.DataFrame(dataset[split_name])

        # 统一列名: 把 'review' 改为 'text'
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})

        # 确保只有 text 和 label 两列
        df = df[['text', 'label']]

        # 过滤掉空文本
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']

        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n{split_name}: {len(df)} 条样本, 已保存到 {save_path}")
        print(f"标签分布:\n{df['label'].value_counts()}")
        print(f"前3条样本:")
        print(df.head(3).to_string())

    print("\n" + "=" * 60)
    print("✅ 数据准备完成! 现在可以运行 train.py 训练了")
    print("=" * 60)


def create_demo_data():
    """备用方案: 如果下载失败, 创建演示数据集"""
    os.makedirs('data', exist_ok=True)

    train_data = [
        ("这部电影真的太精彩了!", 1),
        ("服务很周到,房间也干净整洁", 1),
        ("味道好极了,下次还来", 1),
        ("性价比很高,推荐购买", 1),
        ("老师讲课非常生动有趣", 1),
        ("风景美得让人陶醉", 1),
        ("孩子很喜欢这个玩具", 1),
        ("快递速度超快,包装也很好", 1),
        ("效果出乎意料地好", 1),
        ("员工态度很热情专业", 1),
        ("态度极差,完全不推荐", 0),
        ("味道很难吃,再也不来了", 0),
        ("质量太差,刚用就坏了", 0),
        ("等了两个小时还没上菜", 0),
        ("房间又脏又乱,被子有异味", 0),
        ("根本是骗人的,差评!", 0),
        ("效果完全没有宣传的好", 0),
        ("客服一直不回复,失望", 0),
        ("包装破损严重,东西也坏了", 0),
        ("浪费时间和金钱", 0),
    ]

    train_df = pd.DataFrame(train_data * 5, columns=['text', 'label'])
    dev_df = pd.DataFrame(train_data, columns=['text', 'label'])
    test_df = pd.DataFrame(train_data, columns=['text', 'label'])

    train_df.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
    dev_df.to_csv('data/dev.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8-sig')

    print("演示数据已创建!")
    print(f"train: {len(train_df)} 条")
    print(f"dev: {len(dev_df)} 条")
    print(f"test: {len(test_df)} 条")


if __name__ == '__main__':
    try:
        prepare_chn_senti_corp()
    except Exception as e:
        print(f"\n下载失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n切换到本地演示数据...")
        create_demo_data()
