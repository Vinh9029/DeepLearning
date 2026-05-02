#!/usr/bin/env python3
"""
Script to build model-b-dl.ipynb from the main notebook.
Extracts needed cells, patches paths, and creates a self-contained B-only notebook.
"""
import json, copy

SRC_NB = 'vi_vqa_animal_a_b_model.ipynb'
DST_NB = 'model-b-dl.ipynb'
eam
KAGGLE_PATH = '/kaggle/input/datasets/trietnminh/datasetdl'

with open(SRC_NB, encoding='utf-8') as f:
    src = json.load(f)

src_cells = src['cells']
code_cells = []
for i, c in enumerate(src_cells):
    if c.get('cell_type') == 'code':
        code_cells.append((i, c))

def get_src(seq):
    """Get source code of code cell by sequence number."""
    return ''.join(code_cells[seq][1].get('source', []))

def make_md(text):
    """Create a markdown cell."""
    lines = text.strip().splitlines(keepends=False)
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def make_code(text):
    """Create a code cell."""
    lines = text.strip().splitlines(keepends=False)
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "code", "metadata": {"trusted": True},
            "source": source, "outputs": [], "execution_count": None}

# ═══════════════════════════════════════════════════════════════
# BUILD CELLS
# ═══════════════════════════════════════════════════════════════
new_cells = []

# ── 0. Title ──
new_cells.append(make_md("""# Vi-VQA Animal — Hướng B (BLIP Zero-shot & Fine-tune)

**Notebook chỉ chạy B1/B2** — dùng checkpoint A1/A2 đã train sẵn.

**Dataset Kaggle:** chứa `best_a1.pth`, `best_a2.pth`, `data_*_en.csv`, `a_vocab.pkl`, `q_vocab.pkl`

**Workflow:**
1. Load data từ Kaggle dataset
2. B1: Zero-shot evaluation (BLIP)
3. B2: Fine-tune BLIP
4. So sánh A1/A2/B1/B2"""))

# ── 1. Install + Imports + DEVICE ──
new_cells.append(make_md("## 1. Cài đặt & Import"))

# Take install cell from source and patch DEVICE
install_src = get_src(0)
# Patch DEVICE line
install_src = install_src.replace(
    'DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Luôn dùng GPU đầu tiên',
    'DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Luôn dùng GPU đầu tiên'
)
if 'cuda:0' not in install_src:
    install_src = install_src.replace(
        'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        'DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'
    )
new_cells.append(make_code(install_src))

# ── 2. Paths — point to Kaggle dataset ──
new_cells.append(make_md("## 2. Đường dẫn dữ liệu"))
new_cells.append(make_code(f'''# ── Đường dẫn Kaggle Dataset (chứa checkpoints + CSV đã dịch) ──
KAGGLE_DS = Path("{KAGGLE_PATH}")

# Dataset ảnh gốc (ViVQA)
DATA_DIR = Path("/kaggle/input/vivqa-data")  # ← chỉnh nếu dataset ảnh tên khác
CSV_PATH = list(DATA_DIR.rglob("*.csv"))[0] if DATA_DIR.exists() else None
IMG_DIR  = DATA_DIR

# Output
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
import os
os.makedirs("outputs", exist_ok=True)

print(f"Kaggle DS: {{KAGGLE_DS}}")
print(f"Data DIR:  {{DATA_DIR}}")
print(f"CSV:       {{CSV_PATH}}")
'''))

# ── 3. Load & preprocess data ──
new_cells.append(make_md("## 3. Load & Tiền xử lý dữ liệu"))

# Preprocessing functions
preprocess_src = get_src(2)
new_cells.append(make_code(preprocess_src))

# Data loading + split (from cell 1 source, but adapted)
data_load_src = get_src(1)
new_cells.append(make_code(data_load_src))

# ── 4. Vocab ──
new_cells.append(make_md("## 4. Vocabulary"))
# Vocab class
vocab_src = get_src(6)
new_cells.append(make_code(vocab_src))

# Load vocab from pickle instead of building
new_cells.append(make_code(f'''# Load vocab từ Kaggle dataset (đã build sẵn)
import pickle

_q_vocab_path = KAGGLE_DS / "q_vocab.pkl"
_a_vocab_path = KAGGLE_DS / "a_vocab.pkl"

if _q_vocab_path.exists() and _a_vocab_path.exists():
    with open(_q_vocab_path, "rb") as f:
        q_vocab = pickle.load(f)
    with open(_a_vocab_path, "rb") as f:
        a_vocab = pickle.load(f)
    print(f"✓ Loaded q_vocab ({{len(q_vocab.itos)}} tokens), a_vocab ({{len(a_vocab.itos)}} tokens)")
else:
    print("⚠ Vocab files not found, building from scratch...")
    q_vocab = Vocab(min_freq=1)
    q_vocab.build(df["question_norm"])
    a_vocab = Vocab(min_freq=1)
    a_vocab.build(df["answer_norm"])
    print(f"Built q_vocab ({{len(q_vocab.itos)}}), a_vocab ({{len(a_vocab.itos)}})")
'''))

# ── 5. Dataset + DataLoader (for A model eval) ──
new_cells.append(make_md("## 5. Dataset & DataLoader (cho eval A models)"))
dataset_src = get_src(7)
new_cells.append(make_code(dataset_src))

# ── 6. Model Architecture A (needed for eval) ──
new_cells.append(make_md("## 6. Kiến trúc Model A (để load checkpoint đánh giá)"))
model_src = get_src(8)
new_cells.append(make_code(model_src))

# ── 7. Training helpers (run_eval_a, show_predictions, etc.) ──
new_cells.append(make_md("## 7. Helper functions"))
helpers_src = get_src(9)
new_cells.append(make_code(helpers_src))

# ── 8. Metrics ──
new_cells.append(make_md("## 8. Metrics (EM, BLEU, ROUGE, METEOR, BERTScore)"))
metrics_src = get_src(11)
new_cells.append(make_code(metrics_src))

# ── 9. Copy A1/A2 checkpoints from Kaggle DS ──
new_cells.append(make_md("## 9. Copy checkpoints A1/A2 từ Kaggle Dataset"))
new_cells.append(make_code(f'''import shutil

for fname in ["best_a1.pth", "best_a2.pth"]:
    src_path = KAGGLE_DS / fname
    dst_path = OUTPUT_DIR / fname
    if src_path.exists() and not dst_path.exists():
        shutil.copy2(str(src_path), str(dst_path))
        print(f"✓ Copied {{fname}} → {{dst_path}}")
    elif dst_path.exists():
        print(f"✓ {{fname}} already exists")
    else:
        print(f"⚠ {{fname}} not found in dataset")
'''))

# ── 10. Translator ──
new_cells.append(make_md("## 10. Translator VI↔EN (MarianMT)"))
translator_src = get_src(12)
new_cells.append(make_code(translator_src))

# ── 11. Load pre-translated CSV ──
new_cells.append(make_md("## 11. Load CSV đã dịch sẵn"))
new_cells.append(make_code(f'''# Load CSV đã dịch từ Kaggle Dataset (không cần dịch lại)
TRANSLATED_TRAIN = KAGGLE_DS / "data_train_en.csv"
TRANSLATED_VAL   = KAGGLE_DS / "data_val_en.csv"
TRANSLATED_TEST  = KAGGLE_DS / "data_test_en.csv"

if TRANSLATED_TRAIN.exists():
    train_df_en = pd.read_csv(TRANSLATED_TRAIN)
    val_df_en   = pd.read_csv(TRANSLATED_VAL)
    test_df_en  = pd.read_csv(TRANSLATED_TEST)
    print(f"✓ Load CSV đã dịch: train={{len(train_df_en)}}, val={{len(val_df_en)}}, test={{len(test_df_en)}}")
else:
    raise FileNotFoundError(f"Không tìm thấy CSV đã dịch tại {{KAGGLE_DS}}")
'''))

# ── 12. B1 Zero-shot ──
new_cells.append(make_md("## 12. B1 — BLIP Zero-shot Evaluation"))
b1_src = get_src(14)
new_cells.append(make_code(b1_src))

# ── 13. BlipFineTuneDataset + blip_collate ──
new_cells.append(make_md("## 13. Dataset & Collate cho BLIP Fine-tune"))
collate_src = get_src(15)
new_cells.append(make_code(collate_src))

# ── 14. B2 Fine-tune ──
new_cells.append(make_md("## 14. B2 — Fine-tune BLIP"))
b2_src = get_src(17)
new_cells.append(make_code(b2_src))

# ── 15. B2 Evaluation ──
new_cells.append(make_md("## 15. B2 — Evaluation"))
b2_eval_src = get_src(18)
new_cells.append(make_code(b2_eval_src))

# ── 16. RL Skeleton ──
new_cells.append(make_md("## 16. RL / Preference Optimization (Skeleton)"))
rl_src = get_src(19)
new_cells.append(make_code(rl_src))

# ── 17. Section 15 — Comparison ──
new_cells.append(make_md("## 17. So sánh & Đánh giá tổng hợp (A1/A2/B1/B2)"))
comparison_src = get_src(20)
new_cells.append(make_code(comparison_src))

# ═══════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════
out_nb = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": new_cells
}

# Clear all outputs
for c in out_nb['cells']:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open(DST_NB, 'w', encoding='utf-8') as f:
    json.dump(out_nb, f, ensure_ascii=False, indent=1)

print(f"✅ Created {DST_NB} with {len(new_cells)} cells")
print(f"   Code cells: {sum(1 for c in new_cells if c['cell_type'] == 'code')}")
print(f"   Markdown cells: {sum(1 for c in new_cells if c['cell_type'] == 'markdown')}")
