# ===============================
# train_kfold_text_only_M2_pure.py
# M2 纯抑郁版：
# 1) 使用纯抑郁数据：ML_dataset_balanced_5fold_pure_depression.xlsx
# 2) best checkpoint 按 eval_auc 选择
# 3) 使用 DataCollatorWithPadding 做动态 padding
# 4) 加入 EarlyStopping
# 5) Trainer 写成版本兼容形式：processing_class / tokenizer 二选一
# 6) 保留 outer 5-fold + inner val + val选阈值 + test评估 的整体结构
# 7) 本模型作为“统一split基线”，供 M1 / M3 / M4 复用
# ===============================

import os
import json
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)


# -----------------------
# 0) Reproducibility
# -----------------------
def set_gen_seed(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def safe_auc(y_true, p1):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, p1))
    except Exception:
        return float("nan")


def extract_logits(predictions):
    if isinstance(predictions, tuple):
        return predictions[0]
    return predictions


# -----------------------
# 1) Threshold selection on VAL
# -----------------------
def select_threshold_by_f1_pos1(y_true, p1, grid=None):
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5, None

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        y_hat = (p1 >= t).astype(int)
        f1p = f1_score(y_true, y_hat, average="binary", pos_label=1, zero_division=0)
        if f1p > best_f1:
            best_f1 = float(f1p)
            best_t = float(t)

    return best_t, float(best_f1)


def select_threshold_by_recall_min(y_true, p1, recall_min=0.90, grid=None):
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5, None

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    candidates = []
    for t in grid:
        y_hat = (p1 >= t).astype(int)
        rec = recall_score(y_true, y_hat, pos_label=1, zero_division=0)
        prec = precision_score(y_true, y_hat, pos_label=1, zero_division=0)
        f1p = f1_score(y_true, y_hat, average="binary", pos_label=1, zero_division=0)
        if rec >= recall_min:
            candidates.append((t, prec, f1p, rec))

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        t, prec, f1p, rec = candidates[0]
        return float(t), float(f1p)

    return select_threshold_by_f1_pos1(y_true, p1, grid=grid)


# -----------------------
# 2) TrainingArguments builder
# -----------------------
def build_training_args(
    output_dir,
    num_train_epochs,
    batch_size,
    seed,
    learning_rate=2e-5,
    warmup_ratio=0.1
):
    common = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        seed=seed,
        data_seed=seed
    )

    try:
        return TrainingArguments(evaluation_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **common)


# -----------------------
# 3) Main
# -----------------------
def main(
    data_file="ML_dataset_balanced_5fold_pure_depression.xlsx",
    exp_root="experiments",
    exp_name="text_only_kfold_M2_pure",
    model_ckpt="bert-base-chinese",
    seed=42,
    inner_val_ratio=0.2,
    num_train_epochs=6,
    batch_size=8,
    max_length=512,
    threshold_mode="f1",
    recall_min=0.90,
    save_text_in_predictions=True,
    early_stopping_patience=2
):
    set_gen_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load data
    df = pd.read_excel(data_file)
    required_cols = ["ID", "history_text", "label", "fold"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_cols].copy()
    df.columns = ["id", "text", "label", "fold"]
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df = df[df["text"].str.strip() != ""].copy()

    folds = sorted(df["fold"].unique().tolist())

    print("\n===== FULL DATASET INFO =====")
    print("Total samples:", len(df))
    print("Label distribution:\n", df["label"].value_counts())
    print("Fold distribution:\n", df["fold"].value_counts().sort_index())
    print("\nPer-fold label counts:")
    print(df.groupby("fold")["label"].value_counts().unstack(fill_value=0).sort_index())
    print("\nFolds:", folds)

    # ---- dirs
    exp_dir = os.path.join(exp_root, exp_name)
    folds_dir = os.path.join(exp_dir, "folds")
    os.makedirs(folds_dir, exist_ok=True)

    # ---- save config
    cfg = dict(
        data_file=data_file,
        model_ckpt=model_ckpt,
        seed=seed,
        inner_val_ratio=inner_val_ratio,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        max_length=max_length,
        threshold_mode=threshold_mode,
        recall_min=recall_min,
        folds=folds,
        save_text_in_predictions=save_text_in_predictions,
        early_stopping_patience=early_stopping_patience
    )
    with open(os.path.join(exp_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ---- tokenizer + dynamic padding
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    def compute_metrics(pred):
        y_true = pred.label_ids.astype(int)
        logits = extract_logits(pred.predictions)
        probs = softmax_np(logits, axis=1)
        p1 = probs[:, 1]
        y_hat = (p1 >= 0.5).astype(int)

        return {
            "accuracy": float(accuracy_score(y_true, y_hat)),
            "f1_weighted": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
            "f1_pos1": float(f1_score(y_true, y_hat, average="binary", pos_label=1, zero_division=0)),
            "precision_pos1": float(precision_score(y_true, y_hat, pos_label=1, zero_division=0)),
            "recall_pos1": float(recall_score(y_true, y_hat, pos_label=1, zero_division=0)),
            "pred_pos_rate": float(y_hat.mean()),
            "auc": float(safe_auc(y_true, p1)),
        }

    cv_rows = []
    oof_rows = []
    diag_rows = []

    # -----------------------
    # Outer K-Fold
    # -----------------------
    for fold in folds:
        print("\n" + "=" * 70)
        print(f"OUTER FOLD = {fold} (this fold is TEST)")

        fold_dir = os.path.join(folds_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "best_model"), exist_ok=True)

        df_test = df[df["fold"] == fold].copy()
        df_trainval = df[df["fold"] != fold].copy()

        df_train, df_val = train_test_split(
            df_trainval,
            test_size=inner_val_ratio,
            random_state=seed + int(fold),
            stratify=df_trainval["label"]
        )

        print("Train/Val/Test sizes:", len(df_train), len(df_val), len(df_test))
        print("Val label counts:\n", df_val["label"].value_counts())
        print("Test label counts:\n", df_test["label"].value_counts())

        split_df = pd.concat([
            df_train[["id"]].assign(split="train"),
            df_val[["id"]].assign(split="val"),
            df_test[["id"]].assign(split="test"),
        ], ignore_index=True)
        split_df.to_csv(os.path.join(fold_dir, "splits.csv"), index=False, encoding="utf-8-sig")

        train_ds = Dataset.from_pandas(df_train[["id", "text", "label"]], preserve_index=False)
        val_ds = Dataset.from_pandas(df_val[["id", "text", "label"]], preserve_index=False)
        test_ds = Dataset.from_pandas(df_test[["id", "text", "label"]], preserve_index=False)

        train_enc = train_ds.map(tokenize, batched=True)
        val_enc = val_ds.map(tokenize, batched=True)
        test_enc = test_ds.map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            num_labels=2
        ).to(device)

        ckpt_dir = os.path.join(fold_dir, "checkpoints")
        training_args = build_training_args(
            output_dir=ckpt_dir,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            seed=seed + int(fold),
            learning_rate=2e-5,
            warmup_ratio=0.1
        )

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_enc,
            eval_dataset=val_enc,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )

        try:
            trainer = Trainer(
                processing_class=tokenizer,
                **trainer_kwargs
            )
        except TypeError:
            trainer = Trainer(
                tokenizer=tokenizer,
                **trainer_kwargs
            )

        trainer.train()
        trainer.model.eval()

        print("Best checkpoint:", trainer.state.best_model_checkpoint)
        print("Best metric (eval_auc):", trainer.state.best_metric)

        rows = []
        current_train_loss = None
        best_eval_auc = -1.0
        best_epoch = None

        for entry in trainer.state.log_history:
            if "loss" in entry:
                current_train_loss = entry["loss"]

            if "eval_loss" in entry:
                ep = entry.get("epoch")
                eval_loss = entry.get("eval_loss")
                eval_auc = entry.get("eval_auc")

                is_best = False
                if eval_auc is not None and eval_auc > best_eval_auc:
                    best_eval_auc = eval_auc
                    best_epoch = ep
                    is_best = True

                rows.append({
                    "Epoch": ep,
                    "Training Loss": current_train_loss,
                    "Validation Loss": eval_loss,
                    "Val Accuracy@0.5": entry.get("eval_accuracy"),
                    "Val F1_pos1@0.5": entry.get("eval_f1_pos1"),
                    "Val Precision_pos1@0.5": entry.get("eval_precision_pos1"),
                    "Val Recall_pos1@0.5": entry.get("eval_recall_pos1"),
                    "Val PredPosRate@0.5": entry.get("eval_pred_pos_rate"),
                    "Val AUC": entry.get("eval_auc"),
                    "Is Best": is_best
                })

        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(
            os.path.join(fold_dir, "training_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        if not df_summary.empty:
            plt.figure()
            plt.plot(df_summary["Epoch"], df_summary["Training Loss"], label="Training Loss")
            plt.plot(df_summary["Epoch"], df_summary["Validation Loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Fold {fold} - Training/Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(fold_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            if "Val F1_pos1@0.5" in df_summary.columns:
                plt.plot(df_summary["Epoch"], df_summary["Val F1_pos1@0.5"], label="Val F1_pos1@0.5")
            if "Val Recall_pos1@0.5" in df_summary.columns:
                plt.plot(df_summary["Epoch"], df_summary["Val Recall_pos1@0.5"], label="Val Recall_pos1@0.5")
            if "Val Precision_pos1@0.5" in df_summary.columns:
                plt.plot(df_summary["Epoch"], df_summary["Val Precision_pos1@0.5"], label="Val Precision_pos1@0.5")
            if "Val AUC" in df_summary.columns:
                plt.plot(df_summary["Epoch"], df_summary["Val AUC"], label="Val AUC")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"Fold {fold} - Val Metrics (@0.5)")
            plt.legend()
            plt.savefig(os.path.join(fold_dir, "val_metrics_curve.png"), dpi=300, bbox_inches="tight")
            plt.close()

        val_out = trainer.predict(val_enc)
        y_val = val_out.label_ids.astype(int)
        val_logits = extract_logits(val_out.predictions)
        val_probs = softmax_np(val_logits, axis=1)
        p1_val = val_probs[:, 1]

        if threshold_mode == "recall":
            best_t, best_t_f1 = select_threshold_by_recall_min(
                y_val, p1_val, recall_min=recall_min
            )
        else:
            best_t, best_t_f1 = select_threshold_by_f1_pos1(y_val, p1_val)

        val_auc = safe_auc(y_val, p1_val)
        val_pred_pos_rate_05 = float((p1_val >= 0.5).mean())
        val_pred_pos_rate_bt = float((p1_val >= best_t).mean())

        print(f"[Fold {fold}] VAL AUC={val_auc:.3f}  pred_pos_rate@0.5={val_pred_pos_rate_05:.3f}")
        print(
            f"[Fold {fold}] Best threshold from VAL = {best_t:.2f} "
            f"(val_f1_pos1={best_t_f1}) pred_pos_rate@best_t={val_pred_pos_rate_bt:.3f}"
        )

        test_out = trainer.predict(test_enc)
        y_true = test_out.label_ids.astype(int)
        test_logits = extract_logits(test_out.predictions)
        test_probs = softmax_np(test_logits, axis=1)
        p1_test = test_probs[:, 1]
        y_pred = (p1_test >= best_t).astype(int)

        test_auc = safe_auc(y_true, p1_test)
        pred_pos_rate_05 = float((p1_test >= 0.5).mean())
        pred_pos_rate_bt = float(y_pred.mean())

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        test_metrics = {
            "fold": int(fold),
            "n_test": int(len(df_test)),
            "best_checkpoint": str(trainer.state.best_model_checkpoint),
            "best_metric_eval_auc": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
            "best_epoch_by_eval_auc": float(best_epoch) if best_epoch is not None else None,
            "threshold_mode": threshold_mode,
            "threshold_from_val": float(best_t),
            "val_f1_pos1_at_best_t": float(best_t_f1) if best_t_f1 is not None else None,
            "val_auc": float(val_auc),
            "val_pred_pos_rate@0.5": float(val_pred_pos_rate_05),
            "val_pred_pos_rate@best_t": float(val_pred_pos_rate_bt),
            "test_auc": float(test_auc),
            "test_pred_pos_rate@0.5": float(pred_pos_rate_05),
            "test_pred_pos_rate@best_t": float(pred_pos_rate_bt),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "test_f1_pos1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
            "test_precision_pos1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "test_recall_pos1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        pd.DataFrame([test_metrics]).to_csv(
            os.path.join(fold_dir, "test_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        cm_df = pd.DataFrame(cm, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
        cm_df.to_csv(os.path.join(fold_dir, "confusion_matrix.csv"), encoding="utf-8-sig")

        out_dict = {
            "id": df_test["id"].astype(str).tolist(),
            "fold": [int(fold)] * len(df_test),
            "y_true": y_true,
            "p_depression": p1_test,
            "y_pred@best_t": y_pred,
            "y_pred@0.5": (p1_test >= 0.5).astype(int),
        }
        if save_text_in_predictions:
            out_dict["text"] = df_test["text"].astype(str).tolist()

        pred_df = pd.DataFrame(out_dict)
        pred_df.to_csv(
            os.path.join(fold_dir, "test_predictions.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        best_dir = os.path.join(fold_dir, "best_model")
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

        cv_rows.append(test_metrics)
        oof_rows.append(pred_df)

        def prob_summary(y, p1):
            y = np.asarray(y).astype(int)
            p1 = np.asarray(p1).astype(float)
            return {
                "p1_mean": float(np.mean(p1)),
                "p1_std": float(np.std(p1)),
                "p1_min": float(np.min(p1)),
                "p1_p25": float(np.quantile(p1, 0.25)),
                "p1_median": float(np.median(p1)),
                "p1_p75": float(np.quantile(p1, 0.75)),
                "p1_max": float(np.max(p1)),
                "p1_mean_pos": float(np.mean(p1[y == 1])) if np.any(y == 1) else float("nan"),
                "p1_mean_neg": float(np.mean(p1[y == 0])) if np.any(y == 0) else float("nan"),
            }

        d = {"fold": int(fold)}
        d.update({f"val_{k}": v for k, v in prob_summary(y_val, p1_val).items()})
        d.update({f"test_{k}": v for k, v in prob_summary(y_true, p1_test).items()})
        diag_rows.append(d)

        print(f"[Fold {fold}] TEST done. Saved to {fold_dir}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(exp_dir, "cv_metrics_by_fold.csv"), index=False, encoding="utf-8-sig")

    metric_cols = [
        "val_auc", "test_auc",
        "test_accuracy", "test_f1_weighted", "test_f1_pos1",
        "test_precision_pos1", "test_recall_pos1",
        "test_pred_pos_rate@0.5", "test_pred_pos_rate@best_t"
    ]
    summary_rows = []
    for col in metric_cols:
        vals = pd.to_numeric(cv_df[col], errors="coerce").to_numpy()
        summary_rows.append({
            "metric": col,
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(exp_dir, "cv_summary_mean_std.csv"), index=False, encoding="utf-8-sig")

    oof_all = pd.concat(oof_rows, ignore_index=True)
    oof_all.to_csv(os.path.join(exp_dir, "oof_predictions_all.csv"), index=False, encoding="utf-8-sig")

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(os.path.join(exp_dir, "diagnostics_prob_summary.csv"), index=False, encoding="utf-8-sig")

    if oof_all["id"].nunique() != df["id"].nunique():
        print(
            "[WARN] OOF unique id mismatch:",
            oof_all["id"].nunique(),
            "vs",
            df["id"].nunique(),
            " -> check duplicate IDs or fold assignment."
        )

    print("\n===== DONE =====")
    print("Saved all outputs to:", exp_dir)
    print(summary_df)


if __name__ == "__main__":
    main(
        data_file="ML_dataset_balanced_5fold_pure_depression.xlsx",
        exp_root="experiments",
        exp_name="text_only_kfold_M2_pure",
        model_ckpt="bert-base-chinese",
        seed=42,
        inner_val_ratio=0.2,
        num_train_epochs=6,
        batch_size=8,
        max_length=512,
        threshold_mode="f1",
        recall_min=0.90,
        save_text_in_predictions=True,
        early_stopping_patience=2
    )