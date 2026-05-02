import streamlit as st
import random
import torch
from PIL import Image
import numpy as np
import os
import sys
import pickle
from pathlib import Path

import random

# Define the Vocab class that was used to create the pickle files.
# This is necessary for pickle to be able to load the vocab objects.
class Vocab:
    """A simple vocabulary class to map tokens to indices and vice-versa."""
    def __init__(self, itos, unk_token="<unk>"):
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk_idx = self.stoi.get(unk_token)

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        # A simple whitespace tokenizer. This should match the one used during training.
        return text.lower().strip().split()

    def encode(self, text):
        """Converts a text string to a list of token indices."""
        tokens = self.tokenize(text)
        # Lấy index của token <unk> một cách an toàn để tránh lỗi với pickle cũ
        unk_index = getattr(self, 'unk_idx', self.stoi.get("<unk>", 3))
        return [self.stoi.get(token, unk_index) for token in tokens]

    def decode(self, ids):
        """Converts a list of token indices to a text string."""
        tokens = [self.itos[i] for i in ids if self.itos[i] not in ["<bos>", "<pad>"]]
        return " ".join(tokens).split("<eos>")[0].strip()

# ==== CONFIG ====
KAGGLE_OUTPUT = Path(__file__).parent / "kaggle_output"
MODELS_DIR = KAGGLE_OUTPUT / "models"
BLIP_PROC_DIR = MODELS_DIR / "blip_processor"
VOCAB_Q_PATH = KAGGLE_OUTPUT / "q_vocab.pkl"
VOCAB_A_PATH = KAGGLE_OUTPUT / "a_vocab.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOADERS ====
def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_a(model_type, q_vocab, a_vocab):
    from vi_vqa_animal_a_b_model import VQAModelA, PAD
    if model_type == "A1 (LSTM)":
        ckpt = MODELS_DIR / "best_a1.pth"
        decoder_type = "lstm"
    else:
        ckpt = MODELS_DIR / "best_a2.pth"
        decoder_type = "transformer"
    model = VQAModelA(
        q_vocab_size=len(q_vocab.itos),
        a_vocab_size=len(a_vocab.itos),
        image_backbone="resnet50",
        fusion_type="coattention",
        decoder_type=decoder_type,
        feat_dim=512,
    )
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_blip(model_type):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    if model_type == "B1 (BLIP zero-shot)":
        # Use the official pretrained BLIP processor and model
        proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)
    elif model_type == "B2 (BLIP fine-tuned)":
        # Use the local fine-tuned processor and weights
        proc = BlipProcessor.from_pretrained(str(BLIP_PROC_DIR))
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        model.load_state_dict(torch.load(MODELS_DIR / "best_b2.pth", map_location=DEVICE))
        model.to(DEVICE)
    else:
        raise ValueError(f"Unknown BLIP model type: {model_type}")
    model.eval()
    return proc, model

def resolve_image(img_file):
    img = Image.open(img_file).convert("RGB")
    return img

# ==== STREAMLIT UI ====
st.set_page_config(page_title="Vi-VQA Animal Demo", page_icon="🦉", layout="centered", initial_sidebar_state="auto")
st.title("🐾 Vi-VQA Animal Visual Question Answering")
st.markdown("""
<style>
    .stApp {background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);}
    .big-font {font-size: 1.3em; font-weight: 500;}
    .answer-box {background: #f0f4ff; border-radius: 8px; padding: 1em; margin-top: 1em;}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Chọn mô hình:")
model_choice = st.sidebar.selectbox(
    "Model",
    ["A1 (LSTM)", "A2 (Transformer)", "B1 (BLIP zero-shot)", "B2 (BLIP fine-tuned)"]
)

st.sidebar.markdown("""
- **A1**: ResNet50 + BiLSTM + LSTM decoder
- **A2**: ResNet50 + BiLSTM + Transformer decoder
- **B1**: BLIP zero-shot (pretrained)
- **B2**: BLIP fine-tuned (custom)
""")

st.markdown("### 1. Upload ảnh 🖼️")
img_file = st.file_uploader("Chọn ảnh động vật", type=["jpg", "jpeg", "png"])

st.markdown("### 2. Nhập câu hỏi ❓")
question = st.text_input("Câu hỏi (tiếng Việt hoặc tiếng Anh)")


# ====== Lưu và hiển thị lịch sử hỏi đáp ======
import pandas as pd

HISTORY_PATH = Path("vqa_history.csv")
if "vqa_history" not in st.session_state:
    if HISTORY_PATH.exists():
        st.session_state.vqa_history = pd.read_csv(HISTORY_PATH).to_dict("records")
    else:
        st.session_state.vqa_history = []

if img_file:
    img = resolve_image(img_file)
    st.image(img, caption="Ảnh đã upload", use_column_width=True)


# ==== PATTERN GENERATION ====
def get_q_type(question):
    """
    Xác định loại câu hỏi (q_type) dựa trên từ khóa trong câu hỏi.
    """
    q = question.lower()
    if any(x in q for x in ["con gì", "con nào", "loài gì", "loài nào", "animal", "what animal", "which animal", "who", "tên gì", "tên con"]):
        return "Identity"
    if any(x in q for x in ["màu gì", "màu sắc", "color", "what color"]):
        return "Color"
    if any(x in q for x in ["không", "phải không", "đúng không", "có phải", "yes", "no", "is it", "are they", "does it", "do they"]):
        return "YesNooN"
    if any(x in q for x in ["đang làm gì", "làm gì", "doing", "do", "what is", "what are", "hành động", "action"]):
        return "Action"
    if any(x in q for x in ["ở đâu", "môi trường", "sống ở", "environment", "where", "habitat", "đặc điểm", "details", "characteristic"]):
        return "Environment"
    return "Identity"  # fallback

patterns_vi = {
    "Identity": [
        "Đây là {answer}.",
        "Con vật trong hình là {answer}.",
        "Đó là {answer}.",
        "Tên con vật là {answer}."
    ],
    "Color": [
        "Màu sắc của con vật là {answer}.",
        "Con vật có màu {answer}.",
        "{answer} là màu của con vật.",
        "Con vật này mang màu {answer}."
    ],
    "YesNooN": [
        "Câu trả lời là: {answer}.",
        "{answer}.",
        "Đúng vậy, {answer}.",
        "Không, {answer}."
    ],
    "Action": [
        "Con vật đang {answer}.",
        "Hành động của con vật là {answer}.",
        "Nó đang {answer}.",
        "Con vật trong hình đang {answer}."
    ],
    "Environment": [
        "Con vật đang ở {answer}.",
        "Môi trường xung quanh là {answer}.",
        "Đặc điểm môi trường: {answer}.",
        "Con vật sống ở {answer}."
    ]
}
patterns_en = {
    "Identity": [
        "This is a {answer}.",
        "The animal in the picture is a {answer}.",
        "It is a {answer}.",
        "The animal's name is {answer}."
    ],
    "Color": [
        "The animal's color is {answer}.",
        "It is {answer} in color.",
        "{answer} is the color of the animal.",
        "The animal has a {answer} color."
    ],
    "YesNooN": [
        "The answer is: {answer}.",
        "{answer}.",
        "Yes, {answer}.",
        "No, {answer}."
    ],
    "Action": [
        "The animal is {answer}.",
        "Its action is {answer}.",
        "It is {answer}.",
        "The animal in the image is {answer}."
    ],
    "Environment": [
        "The animal is in {answer}.",
        "Its environment is {answer}.",
        "The animal lives in {answer}.",
        "Environment details: {answer}."
    ]
}

def wrap_answer(answer, model_choice, question):
    q_type = get_q_type(question)
    if model_choice in ["A1 (LSTM)", "A2 (Transformer)"]:
        pattern = random.choice(patterns_vi.get(q_type, patterns_vi["Identity"]))
    else:
        pattern = random.choice(patterns_en.get(q_type, patterns_en["Identity"]))
    return pattern.format(answer=answer)

if st.button("Trả lời", type="primary"):
    if not img_file or not question:
        st.warning("Vui lòng upload ảnh và nhập câu hỏi.")
        st.stop()
    img = resolve_image(img_file)
    with st.spinner("Đang sinh câu trả lời..."):
        if model_choice in ["A1 (LSTM)", "A2 (Transformer)"]:
            q_vocab = load_vocab(VOCAB_Q_PATH)
            a_vocab = load_vocab(VOCAB_A_PATH)
            model = load_model_a(model_choice, q_vocab, a_vocab)
            from torchvision import transforms
            tfms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = tfms(img).unsqueeze(0).to(DEVICE)
            q_ids = torch.tensor([q_vocab.encode(question)], dtype=torch.long).to(DEVICE)
            q_lens = torch.tensor([len(q_vocab.encode(question))], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                gen_ids = model.generate(img_tensor, q_ids, q_lens, a_vocab.stoi["<bos>"], a_vocab.stoi["<eos>"], max_len=12)
                answer = a_vocab.decode(gen_ids[0].tolist())
        elif model_choice in ["B1 (BLIP zero-shot)", "B2 (BLIP fine-tuned)"]:
            proc, model_blip = load_blip(model_choice)
            inp = proc(images=img, text=question, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model_blip.generate(**inp, max_new_tokens=20)
            answer = proc.decode(out[0], skip_special_tokens=True)
        else:
            st.error(f"Không nhận diện được lựa chọn mô hình: {model_choice}")
            st.stop()
    answer_sentence = make_full_sentence(answer, model_choice, question)
    st.markdown(f"<div class='answer-box'><span class='big-font'>Đáp án: <b>{answer_sentence}</b></span></div>", unsafe_allow_html=True)
    st.success("Hoàn thành!")
    # Lưu lịch sử
    st.session_state.vqa_history.append({
        "model": model_choice,
        "question": question,
        "answer": answer
    })
    pd.DataFrame(st.session_state.vqa_history).to_csv(HISTORY_PATH, index=False)

# Hiển thị bảng lịch sử hỏi đáp
if st.session_state.vqa_history:
    st.markdown("### 📋 Lịch sử hỏi đáp (có thể tải về)")
    df_hist = pd.DataFrame(st.session_state.vqa_history)
    st.dataframe(df_hist, use_container_width=True)
    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("Tải file CSV lịch sử", data=csv, file_name="vqa_history.csv", mime="text/csv")

st.markdown("""
---
<sub>Vi-VQA Animal Demo | Đồ án Deep Learning 2026</sub>
""")
