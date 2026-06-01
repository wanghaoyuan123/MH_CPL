# ==========================================
# lime_all_depressed_from_oof_M5_pure.py
# 对【全部纯抑郁样本】跑 LIME
# 使用【M5_pure 每个 fold 对应的 best_model】做解释（K-fold 严谨版）
#
# 说明：
# 1) 基于 text_only_kfold_M5_pure
# 2) 优先直接使用 M5 OOF 中保存的 text
# 3) 如果 OOF 没有 text，则回 M5 数据文件中取 text_M5
# 4) 输出单独放到 lime_outputs_all_depressed_M5_pure
# ==========================================

import os
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from lime.lime_text import LimeTextExplainer


# ===============================
# 0) 路径：以脚本所在目录为基准
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OOF_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "text_only_kfold_M5_pure",
    "oof_predictions_all.csv"
)

# M5 数据文件路径：
# 先找脚本同目录下的文件；
# 如果没有，再找 OUTPUT 文件夹里的文件。
DATA_PATH_1 = os.path.join(
    BASE_DIR,
    "ML_dataset_balanced_5fold_text_M5_pure_depression.xlsx"
)

DATA_PATH_2 = os.path.join(
    BASE_DIR,
    "OUTPUT",
    "ML_dataset_balanced_5fold_text_M5_pure_depression.xlsx"
)

if os.path.exists(DATA_PATH_1):
    DATA_PATH = DATA_PATH_1
elif os.path.exists(DATA_PATH_2):
    DATA_PATH = DATA_PATH_2
else:
    DATA_PATH = DATA_PATH_1

EXP_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "text_only_kfold_M5_pure"
)

FOLDS_DIR = os.path.join(EXP_DIR, "folds")

OUT_DIR = os.path.join(
    BASE_DIR,
    "lime_outputs_all_depressed_M5_pure"
)

TEXT_COL = "text_M5"
ID_COL = "ID"
LABEL_COL = "label"

BASE_NUM_SAMPLES = 200
MIN_CHARS = 20
MAX_LEN = 512

# 只保留对“抑郁类 label=1”有正向贡献的 token
# 如果想更严格，可以改成 0.02 或 0.05
WEIGHT_MIN = 0.0

SEED = 42


# ===============================
# 1) 复现性
# ===============================
def set_all_seeds(seed=42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ===============================
# 2) 安全读取 CSV
# ===============================
def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


# ===============================
# 3) 检查路径
# ===============================
print("BASE_DIR:", BASE_DIR)
print("OOF_PATH:", OOF_PATH)
print("OOF exists:", os.path.exists(OOF_PATH))
print("DATA_PATH:", DATA_PATH)
print("DATA exists:", os.path.exists(DATA_PATH))
print("EXP_DIR:", EXP_DIR)
print("EXP exists:", os.path.exists(EXP_DIR))
print("FOLDS_DIR:", FOLDS_DIR)
print("FOLDS exists:", os.path.exists(FOLDS_DIR))
print("OUT_DIR:", OUT_DIR)

if not os.path.exists(OOF_PATH):
    raise FileNotFoundError(f"OOF file not found: {OOF_PATH}")

if not os.path.exists(FOLDS_DIR):
    raise FileNotFoundError(f"FOLDS_DIR not found: {FOLDS_DIR}")


# ===============================
# 4) 读取 M5 OOF 结果
# ===============================
oof = read_csv_safely(OOF_PATH)

required_oof = {"id", "y_true", "p_depression", "fold"}
missing = required_oof - set(oof.columns)

if missing:
    raise ValueError(
        f"OOF missing columns: {missing}. OOF columns={list(oof.columns)}"
    )

oof["id"] = oof["id"].astype(str)
oof["y_true"] = oof["y_true"].astype(int)
oof["p_depression"] = oof["p_depression"].astype(float)
oof["fold"] = oof["fold"].astype(int)

# 只保留真实抑郁样本
dep = oof[oof["y_true"] == 1].copy().reset_index(drop=True)
n_dep = len(dep)

print("\nOOF depressed samples, y_true=1:", n_dep)

if n_dep == 0:
    raise ValueError("No depressed samples found in OOF.")


# ===============================
# 5) 保存全部抑郁样本 ID 清单
# ===============================
sel = dep.copy()

sel_path = os.path.join(
    OUT_DIR,
    "lime_all_depressed_ids_M5_pure.csv"
)

sel.to_csv(sel_path, index=False, encoding="utf-8-sig")

print("Saved selected IDs:", sel_path)
print("Total selected depressed samples:", len(sel))


# ===============================
# 6) 优先直接使用 OOF 中保存的 text
#    如果 OOF 没有 text，则回 DATA_PATH 中取 text_M5
# ===============================
if "text" in sel.columns:
    print("\nUsing text directly from M5 OOF predictions.")

    df_sel = sel[["id", "fold", "p_depression", "text", "y_true"]].copy()

    df_sel.rename(
        columns={
            "id": ID_COL,
            "text": TEXT_COL,
            "y_true": LABEL_COL
        },
        inplace=True
    )

    df_sel[ID_COL] = df_sel[ID_COL].astype(str)
    df_sel[TEXT_COL] = df_sel[TEXT_COL].fillna("").astype(str)
    df_sel[LABEL_COL] = df_sel[LABEL_COL].astype(int)
    df_sel["fold"] = df_sel["fold"].astype(int)

else:
    print("\nOOF does not contain text column. Loading text from DATA_PATH...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"DATA file not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    required_data = {ID_COL, TEXT_COL, LABEL_COL}
    missing2 = required_data - set(df.columns)

    if missing2:
        raise ValueError(
            f"DATA missing columns: {missing2}. DATA columns={list(df.columns)}"
        )

    df[ID_COL] = df[ID_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    df_sel = df[df[ID_COL].isin(set(sel["id"]))].copy()

    df_sel = df_sel.merge(
        sel[["id", "fold", "p_depression"]],
        left_on=ID_COL,
        right_on="id",
        how="left"
    )

    print("Matched selected samples in DATA:", len(df_sel))
    print("df_sel columns after merge:")
    print(df_sel.columns.tolist())

    if "fold" not in df_sel.columns:
        if "fold_y" in df_sel.columns:
            df_sel["fold"] = df_sel["fold_y"]
        elif "fold_x" in df_sel.columns:
            df_sel["fold"] = df_sel["fold_x"]

    if "fold" not in df_sel.columns:
        raise ValueError(
            f"After merge, missing fold column. Existing columns={list(df_sel.columns)}"
        )

    if df_sel["fold"].isna().any():
        bad = df_sel[df_sel["fold"].isna()][[ID_COL, "fold"]].head(20)
        raise ValueError(
            f"[ID mismatch] Some selected IDs not found in DATA or merge failed. Example:\n{bad}"
        )

# 再次强制只保留 label=1
df_sel = df_sel[df_sel[LABEL_COL] == 1].copy()

print("After filtering label=1:", len(df_sel))

if len(df_sel) == 0:
    raise ValueError("No depressed samples left after filtering label=1.")


# ===============================
# 7) 简单检查文本长度
# ===============================
df_sel["text_len"] = df_sel[TEXT_COL].astype(str).str.len()

print("\nM5 text length summary:")
print(df_sel["text_len"].describe())

short_n = (df_sel["text_len"] < MIN_CHARS).sum()
print(f"Samples shorter than MIN_CHARS={MIN_CHARS}: {short_n}")


# ===============================
# 8) 按 fold 缓存模型 / tokenizer / predictor
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

tokenizer_cache = {}
predictor_cache = {}


def get_model_dir_for_fold(fold: int) -> str:
    return os.path.join(
        FOLDS_DIR,
        f"fold_{fold}",
        "best_model"
    )


def get_predictor_for_fold(fold: int):
    if fold in predictor_cache:
        return predictor_cache[fold]

    model_dir = get_model_dir_for_fold(fold)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"best_model not found: {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    mdl.eval()

    tokenizer_cache[fold] = tok

    def predictor(texts):
        inputs = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = mdl(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        return probs.detach().cpu().numpy()

    predictor_cache[fold] = predictor

    return predictor


# ===============================
# 9) LIME explainer
# ===============================
class_names = ["Non-Depression", "Depression"]

lime_explainer = LimeTextExplainer(
    class_names=class_names
)


# ===============================
# 10) 批量跑 LIME
# ===============================
count_all = Counter()
weight_sum_all = defaultdict(float)
weight_max_all = defaultdict(float)

detail_rows = []

skipped_short = 0
skipped_error = 0
processed = 0

records = df_sel[[ID_COL, TEXT_COL, "fold", "p_depression"]].to_dict("records")
N = len(records)

print(f"\nRunning LIME on ALL depressed M5 samples: {N}")

for i, row in enumerate(records, 1):
    sample_id = row[ID_COL]
    text = row[TEXT_COL]
    fold = int(row["fold"])
    p_dep = float(row["p_depression"])

    if not isinstance(text, str):
        skipped_short += 1
        continue

    text_s = text.strip()

    if len(text_s) < MIN_CHARS:
        skipped_short += 1
        continue

    predictor = get_predictor_for_fold(fold)
    tok = tokenizer_cache[fold]

    num_tokens = len(tok.tokenize(text_s))
    num_samples = int(min(BASE_NUM_SAMPLES, max(50, num_tokens * 10)))

    try:
        exp = lime_explainer.explain_instance(
            text_s,
            predictor,
            num_samples=num_samples
        )

        # 只保留对“抑郁类 label=1”有正向贡献的 token
        for token, w in exp.as_list(label=1):
            if w > WEIGHT_MIN:
                count_all[token] += 1
                weight_sum_all[token] += float(w)

                if float(w) > weight_max_all[token]:
                    weight_max_all[token] = float(w)

                detail_rows.append({
                    "id": sample_id,
                    "fold": fold,
                    "p_depression": p_dep,
                    "token": token,
                    "weight": float(w),
                    "text_len": len(text_s)
                })

        processed += 1

    except Exception as e:
        skipped_error += 1

        if skipped_error <= 10:
            print(
                f"[WARN] LIME error at i={i}, "
                f"id={sample_id}, fold={fold}, p={p_dep:.4f}: {e}"
            )

        continue

    if i % 50 == 0:
        print(
            f"progress {i}/{N} | "
            f"processed={processed} | "
            f"skipped_short={skipped_short} | "
            f"skipped_error={skipped_error}"
        )

print("\n===== LIME batch summary =====")
print("Processed:", processed)
print("Skipped short:", skipped_short)
print("Skipped error:", skipped_error)
print("Total attempted:", N)


# ===============================
# 11) 保存统计表
# ===============================
def save_tables(prefix: str, counter: Counter, w_sum: dict, w_max: dict):
    # 按出现次数排序
    df_count = pd.DataFrame(
        counter.most_common(),
        columns=["token", "count"]
    )

    count_path = os.path.join(
        OUT_DIR,
        f"{prefix}_count.csv"
    )

    df_count.to_csv(
        count_path,
        index=False,
        encoding="utf-8-sig"
    )

    rows = []

    for tok, cnt in counter.items():
        rows.append({
            "token": tok,
            "count": int(cnt),
            "weight_sum": float(w_sum[tok]),
            "weight_mean": float(w_sum[tok] / cnt) if cnt > 0 else 0.0,
            "weight_max": float(w_max[tok]),
        })

    df_w = pd.DataFrame(rows)

    if not df_w.empty:
        df_w = df_w.sort_values(
            ["count", "weight_sum"],
            ascending=False
        ).reset_index(drop=True)

    weight_path = os.path.join(
        OUT_DIR,
        f"{prefix}_weight.csv"
    )

    df_w.to_csv(
        weight_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", count_path)
    print("Saved:", weight_path)


save_tables(
    "lime_all_M5_pure",
    count_all,
    weight_sum_all,
    weight_max_all
)


# ===============================
# 12) 保存逐样本明细
# ===============================
if len(detail_rows) > 0:
    df_detail = pd.DataFrame(detail_rows)

    detail_path = os.path.join(
        OUT_DIR,
        "lime_all_detail_M5_pure.csv"
    )

    df_detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", detail_path)

else:
    print("[WARN] No LIME detail rows generated.")


# ===============================
# 13) 保存运行摘要
# ===============================
summary = pd.DataFrame([{
    "model": "M5_pure",
    "oof_path": OOF_PATH,
    "data_path": DATA_PATH,
    "exp_dir": EXP_DIR,
    "folds_dir": FOLDS_DIR,
    "out_dir": OUT_DIR,
    "n_depressed_oof": int(n_dep),
    "n_after_label_filter": int(len(df_sel)),
    "processed": int(processed),
    "skipped_short": int(skipped_short),
    "skipped_error": int(skipped_error),
    "weight_min": float(WEIGHT_MIN),
    "base_num_samples": int(BASE_NUM_SAMPLES),
    "min_chars": int(MIN_CHARS),
    "max_len": int(MAX_LEN),
    "seed": int(SEED)
}])

summary_path = os.path.join(
    OUT_DIR,
    "lime_run_summary_M5_pure.csv"
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", summary_path)

print("\nSaved all outputs to:", OUT_DIR)