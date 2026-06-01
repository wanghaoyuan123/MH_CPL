# ==========================================
# lime_all_depressed_from_oof_M2_pure.py
# 对【全部纯抑郁样本】跑 LIME
# 使用【M2_pure 每个fold对应的best_model】做解释（K-fold严谨版）
#
# 说明：
# 1) 基于 text_only_kfold_M2_pure
# 2) 优先直接使用 OOF 中保存的 text
# 3) 如果 OOF 没有 text，则回纯抑郁版总数据里取
# 4) 输出单独放到纯抑郁版目录，避免覆盖旧结果
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
# 0) 路径（纯抑郁版）
# ===============================
OOF_PATH  = r"experiments/text_only_kfold_M2_pure/oof_predictions_all.csv"
DATA_PATH = r"ML_dataset_balanced_5fold_pure_depression.xlsx"

# 训练实验根目录（里面有 folds/fold_k/best_model）
EXP_DIR   = r"experiments/text_only_kfold_M2_pure"
FOLDS_DIR = os.path.join(EXP_DIR, "folds")

# 新输出目录，避免覆盖旧版
OUT_DIR   = r"lime_outputs_all_depressed_M2_pure"

TEXT_COL  = "history_text"
ID_COL    = "ID"
LABEL_COL = "label"

BASE_NUM_SAMPLES = 200
MIN_CHARS = 20
MAX_LEN = 512

WEIGHT_MIN = 0.0   # 只保留正向贡献；更严格可改 0.02 / 0.05
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
# 2) 安全读取CSV（兼容 utf-8 / utf-8-sig）
# ===============================
def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


# ===============================
# 3) 读取 OOF 结果
# ===============================
oof = read_csv_safely(OOF_PATH)

required_oof = {"id", "y_true", "p_depression", "fold"}
missing = required_oof - set(oof.columns)
if missing:
    raise ValueError(f"OOF missing columns: {missing}. OOF columns={list(oof.columns)}")

oof["id"] = oof["id"].astype(str)
oof["y_true"] = oof["y_true"].astype(int)
oof["p_depression"] = oof["p_depression"].astype(float)
oof["fold"] = oof["fold"].astype(int)

# 只保留 OOF 中真实抑郁样本（纯抑郁建模数据中的 label=1）
dep = oof[oof["y_true"] == 1].copy().reset_index(drop=True)
n_dep = len(dep)
print("OOF depressed (y_true=1):", n_dep)

if n_dep == 0:
    raise ValueError("No depressed samples found in OOF.")


# ===============================
# 4) 保存全部抑郁样本 ID 清单
# ===============================
sel = dep.copy()
sel_path = os.path.join(OUT_DIR, "lime_all_depressed_ids_pure.csv")
sel.to_csv(sel_path, index=False, encoding="utf-8-sig")
print("Saved selected IDs:", sel_path)
print("Total selected depressed samples:", len(sel))


# ===============================
# 5) 优先直接使用 OOF 里的 text
#    如果 OOF 没有 text，则回 DATA 里取
# ===============================
if "text" in sel.columns:
    print("Using text directly from OOF predictions.")
    df_sel = sel[["id", "fold", "p_depression", "text", "y_true"]].copy()
    df_sel.rename(columns={"id": ID_COL, "text": TEXT_COL, "y_true": LABEL_COL}, inplace=True)

    df_sel[ID_COL] = df_sel[ID_COL].astype(str)
    df_sel[TEXT_COL] = df_sel[TEXT_COL].fillna("").astype(str)
    df_sel[LABEL_COL] = df_sel[LABEL_COL].astype(int)
    df_sel["fold"] = df_sel["fold"].astype(int)

else:
    print("OOF does not contain text column. Loading text from DATA_PATH ...")

    df = pd.read_excel(DATA_PATH)

    required_data = {ID_COL, TEXT_COL, LABEL_COL}
    missing2 = required_data - set(df.columns)
    if missing2:
        raise ValueError(f"DATA missing columns: {missing2}. DATA columns={list(df.columns)}")

    df[ID_COL] = df[ID_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    df_sel = df[df[ID_COL].isin(set(sel["id"]))].copy()
    df_sel = df_sel.merge(
        sel[["id", "fold", "p_depression"]],
        left_on=ID_COL, right_on="id", how="left"
    )

    print("Matched selected samples in DATA:", len(df_sel))
    print("df_sel columns after merge:\n", df_sel.columns.tolist())

    # 修复 fold 列名冲突（fold_x / fold_y）
    if "fold" not in df_sel.columns:
        if "fold_y" in df_sel.columns:
            df_sel["fold"] = df_sel["fold_y"]
        elif "fold_x" in df_sel.columns:
            df_sel["fold"] = df_sel["fold_x"]

    # 检查 merge 是否成功
    need_cols = ["fold"]
    missing_need = [c for c in need_cols if c not in df_sel.columns]
    if missing_need:
        raise ValueError(f"After merge, missing columns: {missing_need}. Existing columns={list(df_sel.columns)}")

    if df_sel["fold"].isna().any():
        bad = df_sel[df_sel["fold"].isna()][[ID_COL, "fold"]].head(20)
        raise ValueError(f"[ID mismatch] Some selected IDs not found in DATA or merge failed. Example:\n{bad}")

# 再次强制只保留 label=1
df_sel = df_sel[df_sel[LABEL_COL] == 1].copy()
print("After filtering label=1:", len(df_sel))

if len(df_sel) == 0:
    raise ValueError("No depressed samples left after filtering label=1.")


# ===============================
# 6) 按 fold 缓存模型/分词器/预测函数
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer_cache = {}
predictor_cache = {}

def get_model_dir_for_fold(fold: int) -> str:
    return os.path.join(FOLDS_DIR, f"fold_{fold}", "best_model")

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
# 7) LIME explainer
# ===============================
class_names = ["Non-Depression", "Depression"]
lime_explainer = LimeTextExplainer(class_names=class_names)


# ===============================
# 8) 批量跑 LIME：总体统计
# ===============================
count_all = Counter()
weight_sum_all = defaultdict(float)
weight_max_all = defaultdict(float)

# 保存逐样本明细，便于后续回溯
detail_rows = []

skipped_short = 0
skipped_error = 0
processed = 0

records = df_sel[[ID_COL, TEXT_COL, "fold", "p_depression"]].to_dict("records")
N = len(records)
print(f"Running LIME on ALL depressed samples: {N}")

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

        # 只保留对“抑郁类(label=1)”有正向贡献的词
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
                    "weight": float(w)
                })

        processed += 1

    except Exception as e:
        skipped_error += 1
        if skipped_error <= 10:
            print(f"[WARN] LIME error at i={i}, id={sample_id}, fold={fold}, p={p_dep:.4f}: {e}")
        continue

    if i % 50 == 0:
        print(f"progress {i}/{N} | processed={processed} | skipped_short={skipped_short} | skipped_error={skipped_error}")

print("\n===== LIME batch summary =====")
print("Processed:", processed)
print("Skipped short:", skipped_short)
print("Skipped error:", skipped_error)
print("Total attempted:", N)


# ===============================
# 9) 输出表格（count + weight）
# ===============================
def save_tables(prefix: str, counter: Counter, w_sum: dict, w_max: dict):
    # 按出现次数排序
    df_count = pd.DataFrame(counter.most_common(), columns=["token", "count"])
    df_count.to_csv(os.path.join(OUT_DIR, f"{prefix}_count.csv"), index=False, encoding="utf-8-sig")

    # 带权重信息
    rows = []
    for tok, cnt in counter.items():
        rows.append({
            "token": tok,
            "count": int(cnt),
            "weight_sum": float(w_sum[tok]),
            "weight_mean": float(w_sum[tok] / cnt) if cnt > 0 else 0.0,
            "weight_max": float(w_max[tok]),
        })

    df_w = pd.DataFrame(rows).sort_values(["count", "weight_sum"], ascending=False).reset_index(drop=True)
    df_w.to_csv(os.path.join(OUT_DIR, f"{prefix}_weight.csv"), index=False, encoding="utf-8-sig")

save_tables("lime_all_pure", count_all, weight_sum_all, weight_max_all)

# 保存逐样本明细
if len(detail_rows) > 0:
    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(os.path.join(OUT_DIR, "lime_all_detail_pure.csv"), index=False, encoding="utf-8-sig")

print("Saved outputs to:", OUT_DIR)