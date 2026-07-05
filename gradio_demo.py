# ============================================================
# gradio_demo.py - 语音情感识别 Web 演示界面 (V3 语音版)
# ============================================================
# 完整演示「语音 → 文字 → 情感」闭环:
#   1. 用户上传音频 或 用麦克风录音
#   2. faster-whisper 把语音转写成文字 (ASR)
#   3. BERT 模型对文字做情感识别
#   4. 输出情感结果
#
# 同时保留纯文本输入功能。
#
# 使用:
#   python gradio_demo.py
#
# 依赖 (需额外安装):
#   pip install faster-whisper
# ============================================================

import gradio as gr
import torch
from transformers import BertTokenizer
from models.bert_classifier import BertClassifier

# ============================================================
# 1. 加载 BERT 情感识别模型
# ============================================================
print("=" * 60)
print("正在加载 BERT 情感识别模型...")
print("=" * 60)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_MAP = {0: '负面 😞', 1: '正面 😊'}

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertClassifier(num_classes=2)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ BERT 情感识别模型加载完成")

# ============================================================
# 2. 加载语音识别模型 (faster-whisper)
# ============================================================
# 说明:
#   - 首次运行会自动下载 whisper 模型 (base模型约140MB)
#   - 模型缓存后,以后无需重新下载
#   - 用 'base' 模型,中文识别效果与速度比较平衡
#   - 如果想更快可改成 'tiny',想更准可改成 'small'
ASR_MODEL = None
ASR_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    print("\n正在加载语音识别模型 (faster-whisper)...")
    print("(首次运行需下载模型, 约140MB, 请耐心等待)")
    ASR_MODEL = WhisperModel('base', device='cpu', compute_type='int8')
    ASR_AVAILABLE = True
    print("✅ 语音识别模型加载完成")
except Exception as e:
    print(f"⚠️ 语音识别模型加载失败: {e}")
    print("   语音输入功能将不可用,但文本输入功能正常。")
    print("   如需语音功能,请运行: pip install faster-whisper")

print("\n正在启动 Web 界面...")


# ============================================================
# 3. 核心功能函数
# ============================================================
@torch.no_grad()
def predict_emotion(text):
    """对文本做情感预测, 返回各类别概率"""
    if not text or not text.strip():
        return {"请输入文本": 1.0}

    encoding = tokenizer(
        text, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    token_type_ids = encoding['token_type_ids'].to(DEVICE)

    logits = model(input_ids, attention_mask, token_type_ids)
    probs = torch.softmax(logits, dim=-1)[0]

    return {
        LABEL_MAP[0]: float(probs[0]),
        LABEL_MAP[1]: float(probs[1])
    }


def transcribe_audio(audio_path):
    """用 faster-whisper 把语音转写成文字 (ASR)"""
    if audio_path is None:
        return "", "⚠️ 请先上传音频或录音"

    if not ASR_AVAILABLE:
        return "", "⚠️ 语音识别模型未加载,请使用文本输入功能"

    try:
        # language='zh' 指定中文; beam_size 影响识别质量
        segments, info = ASR_MODEL.transcribe(
            audio_path, language='zh', beam_size=5
        )
        # 把所有片段拼接成完整文本
        text = ''.join(segment.text for segment in segments).strip()
        if not text:
            return "", "⚠️ 未能识别出语音内容,请重试"
        return text, f"✅ 语音识别完成"
    except Exception as e:
        return "", f"⚠️ 语音识别出错: {e}"


def process_voice(audio_path):
    """
    语音情感识别完整流程:
    语音 → ASR转文字 → BERT情感识别
    """
    # Step 1: 语音转文字
    text, status = transcribe_audio(audio_path)
    if not text:
        return "", status, {"无结果": 1.0}

    # Step 2: 文本情感识别
    result = predict_emotion(text)

    return text, f"{status}  →  情感分析完成", result


# ============================================================
# 4. 构建 Gradio 界面
# ============================================================
with gr.Blocks(title="基于BERT的语音情感识别") as demo:
    gr.Markdown(
        """
        # 🎯 基于BERT的语音情感识别系统

        本系统基于预训练 **BERT** 模型，在中文情感语料上微调，可对文本进行情感分类。
        系统支持**语音输入**和**文本输入**两种方式：

        - **语音输入**：语音 → ASR语音识别转文字 → BERT情感分析（完整的语音情感识别流程）
        - **文本输入**：直接输入文本进行情感分析

        **技术栈**：faster-whisper（ASR）+ BERT-base-chinese（情感识别）+ PyTorch
        **测试集准确率**：94.08%

        ---
        """
    )

    # ========== Tab 1: 语音输入 ==========
    with gr.Tab("🎤 语音输入"):
        gr.Markdown(
            "**完整流程演示**：上传一段语音或用麦克风录音，"
            "系统会先把语音转写成文字，再分析其情感倾向。"
        )
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="🎤 上传音频 或 点击录音",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                voice_btn = gr.Button("🚀 开始语音情感分析", variant="primary")

            with gr.Column():
                voice_status = gr.Textbox(
                    label="📍 处理状态", interactive=False
                )
                voice_text = gr.Textbox(
                    label="📝 语音识别结果（ASR转写的文字）",
                    interactive=False, lines=3
                )
                voice_result = gr.Label(
                    label="📊 情感分析结果", num_top_classes=2
                )

        voice_btn.click(
            fn=process_voice,
            inputs=audio_input,
            outputs=[voice_text, voice_status, voice_result]
        )

    # ========== Tab 2: 文本输入 ==========
    with gr.Tab("📝 文本输入"):
        gr.Markdown("直接输入中文文本，分析其情感倾向。")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="📝 请输入要分析的文本",
                    placeholder="例如：今天心情真好，阳光明媚！",
                    lines=4
                )
                text_btn = gr.Button("🚀 开始分析", variant="primary")

                gr.Examples(
                    examples=[
                        "这部电影真的太精彩了, 我看了好几遍!",
                        "服务态度太差, 再也不会来了",
                        "今天天气不错, 适合出去走走",
                        "心情糟透了, 一切都不顺利",
                        "刚收到录取通知, 激动得睡不着觉!",
                        "排队两小时, 菜难吃, 太失望了",
                    ],
                    inputs=text_input,
                    label="💡 点击试试这些例子"
                )

            with gr.Column():
                text_result = gr.Label(
                    label="📊 情感分析结果", num_top_classes=2
                )

        text_btn.click(
            fn=predict_emotion,
            inputs=text_input,
            outputs=text_result
        )

    gr.Markdown(
        """
        ---
        ### 📌 项目说明
        - **应用方向**：语音情感识别。语音先经 ASR 转写为文字，再由 BERT 进行情感识别
        - **ASR 模块**：faster-whisper（OpenAI Whisper 的高速实现）
        - **情感识别模块**：bert-base-chinese 微调（12层Transformer，1.1亿参数）
        - **数据集**：ChnSentiCorp 中文情感分析数据集（9600训练 / 1200验证 / 1200测试）
        - **课程**：语音信号处理 - 项目作业
        """
    )


# ============================================================
# 5. 启动服务
# ============================================================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
