# ===============================
# train_kfold_scale_only_M0_pure.py
# M0 pure: 仅用 4 个量表总分 的基线模型
#
# 统一版本：
# 1) 使用纯抑郁数据：ML_dataset_balanced_5fold_pure_depression.xlsx
# 2) 不重新切分 train/val/test
# 3) 直接复用 M2_pure 已保存的 splits.csv
# 4) 每个 outer fold 内，仅用 train 集对量表总分做标准化
# 5) 模型：Logistic Regression
# 6) threshold 在 validation 上按 positive-class F1 选择
# ===============================

import os
import json
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)


# -----------------------
# 路径基准：脚本所在目录
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------
# 0) Reproducibility
# -----------------------
def set_gen_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def safe_auc(y_true, p1):
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, p1))
    except Exception:
        return float("nan")


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
# 2) Main
# -----------------------
def main(
    data_file=os.path.join(BASE_DIR, "ML_dataset_balanced_5fold_pure_depression.xlsx"),
    exp_root=os.path.join(BASE_DIR, "experiments"),
    exp_name="scale_only_kfold_M0_pure",
    seed=42,
    threshold_mode="f1",   # "f1" or "recall"
    recall_min=0.90,
    split_ref_root=os.path.join(BASE_DIR, "experiments", "text_only_kfold_M2_pure", "folds")
):
    set_gen_seed(seed)

    scale_cols = ["total_DSRSC", "total_PHQ", "total_CDI", "total_DASS"]

    # ---- load data
    df = pd.read_excel(data_file)
    required_cols = ["ID", "label", "fold"] + scale_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_cols].copy()
    df.columns = ["id", "label", "fold"] + scale_cols

    df["id"] = df["id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["fold"] = df["fold"].astype(int)

    for c in scale_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 不允许静默丢样本：发现缺失直接报错
    bad_scale = df[scale_cols].isna().any(axis=1)
    if bad_scale.any():
        bad_ids = df.loc[bad_scale, "id"].astype(str).tolist()[:10]
        raise ValueError(f"Missing scale totals found in base file. Example IDs: {bad_ids}")

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
        split_ref_root=split_ref_root,
        seed=seed,
        threshold_mode=threshold_mode,
        recall_min=recall_min,
        folds=folds,
        scale_cols=scale_cols,
        model="LogisticRegression"
    )
    with open(os.path.join(exp_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

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

        # ---- 直接复用 M2_pure 已保存的 split
        split_file = os.path.join(split_ref_root, f"fold_{fold}", "splits.csv")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

        split_df = pd.read_csv(split_file)
        if "id" not in split_df.columns or "split" not in split_df.columns:
            raise ValueError(f"Invalid split file format: {split_file}")

        split_df["id"] = split_df["id"].astype(str)

        train_ids = set(split_df.loc[split_df["split"] == "train", "id"])
        val_ids = set(split_df.loc[split_df["split"] == "val", "id"])
        test_ids = set(split_df.loc[split_df["split"] == "test", "id"])

        df_train = df[df["id"].isin(train_ids)].copy()
        df_val = df[df["id"].isin(val_ids)].copy()
        df_test = df[df["id"].isin(test_ids)].copy()

        # ---- 基本检查：确保 split 完全匹配
        if len(df_train) != len(train_ids):
            missing_ids = sorted(list(train_ids - set(df_train["id"])))
            raise ValueError(f"[Fold {fold}] Train IDs mismatch. Missing examples: {missing_ids[:10]}")
        if len(df_val) != len(val_ids):
            missing_ids = sorted(list(val_ids - set(df_val["id"])))
            raise ValueError(f"[Fold {fold}] Val IDs mismatch. Missing examples: {missing_ids[:10]}")
        if len(df_test) != len(test_ids):
            missing_ids = sorted(list(test_ids - set(df_test["id"])))
            raise ValueError(f"[Fold {fold}] Test IDs mismatch. Missing examples: {missing_ids[:10]}")

        # ---- scale standardization: fit on train only
        train_mean = df_train[scale_cols].mean()
        train_std = df_train[scale_cols].std(ddof=0).replace(0, 1.0)

        for subset in [df_train, df_val, df_test]:
            subset.loc[:, scale_cols] = (subset[scale_cols] - train_mean) / train_std

        print("Train/Val/Test sizes:", len(df_train), len(df_val), len(df_test))
        print("Val label counts:\n", df_val["label"].value_counts())
        print("Test label counts:\n", df_test["label"].value_counts())

        # ---- 保存实际使用的 split
        split_df.to_csv(os.path.join(fold_dir, "splits.csv"), index=False, encoding="utf-8-sig")

        # ---- 保存标准化参数
        scaler_df = pd.DataFrame({
            "feature": scale_cols,
            "train_mean": [train_mean[c] for c in scale_cols],
            "train_std": [train_std[c] for c in scale_cols],
        })
        scaler_df.to_csv(os.path.join(fold_dir, "scale_standardization.csv"), index=False, encoding="utf-8-sig")

        # ---- 准备数据
        X_train = df_train[scale_cols].to_numpy(dtype=float)
        y_train = df_train["label"].to_numpy(dtype=int)

        X_val = df_val[scale_cols].to_numpy(dtype=float)
        y_val = df_val["label"].to_numpy(dtype=int)

        X_test = df_test[scale_cols].to_numpy(dtype=float)
        y_test = df_test["label"].to_numpy(dtype=int)

        # ---- 模型
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=seed + int(fold)
        )
        model.fit(X_train, y_train)

        # ---- 保存模型
        joblib.dump(model, os.path.join(fold_dir, "best_model.joblib"))

        # ---- 保存系数
        coef_df = pd.DataFrame({
            "feature": scale_cols,
            "coef": model.coef_[0]
        }).sort_values("coef", ascending=False)
        coef_df.to_csv(os.path.join(fold_dir, "model_coefficients.csv"), index=False, encoding="utf-8-sig")

        # -----------------------
        # Select threshold on VAL
        # -----------------------
        p1_val = model.predict_proba(X_val)[:, 1]

        if threshold_mode == "recall":
            best_t, best_t_f1 = select_threshold_by_recall_min(
                y_val, p1_val, recall_min=recall_min
            )
        else:
            best_t, best_t_f1 = select_threshold_by_f1_pos1(y_val, p1_val)

        val_auc = safe_auc(y_val, p1_val)
        val_pred_05 = (p1_val >= 0.5).astype(int)
        val_pred_bt = (p1_val >= best_t).astype(int)

        val_metrics = {
            "fold": int(fold),
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "n_test": int(len(df_test)),
            "threshold_mode": threshold_mode,
            "threshold_from_val": float(best_t),
            "val_f1_pos1_at_best_t": float(best_t_f1) if best_t_f1 is not None else None,
            "val_auc": float(val_auc),
            "val_accuracy@0.5": float(accuracy_score(y_val, val_pred_05)),
            "val_f1_pos1@0.5": float(f1_score(y_val, val_pred_05, average="binary", pos_label=1, zero_division=0)),
            "val_precision_pos1@0.5": float(precision_score(y_val, val_pred_05, pos_label=1, zero_division=0)),
            "val_recall_pos1@0.5": float(recall_score(y_val, val_pred_05, pos_label=1, zero_division=0)),
            "val_pred_pos_rate@0.5": float(val_pred_05.mean()),
            "val_pred_pos_rate@best_t": float(val_pred_bt.mean()),
        }

        pd.DataFrame([val_metrics]).to_csv(
            os.path.join(fold_dir, "validation_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        print(f"[Fold {fold}] VAL AUC={val_auc:.3f}  pred_pos_rate@0.5={float(val_pred_05.mean()):.3f}")
        print(
            f"[Fold {fold}] Best threshold from VAL = {best_t:.2f} "
            f"(val_f1_pos1={best_t_f1}) pred_pos_rate@best_t={float(val_pred_bt.mean()):.3f}"
        )

        # -----------------------
        # Test evaluation using best_t
        # -----------------------
        p1_test = model.predict_proba(X_test)[:, 1]
        y_pred = (p1_test >= best_t).astype(int)

        test_auc = safe_auc(y_test, p1_test)
        pred_pos_rate_05 = float((p1_test >= 0.5).mean())
        pred_pos_rate_bt = float(y_pred.mean())

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        test_metrics = {
            "fold": int(fold),
            "n_test": int(len(df_test)),
            "model": "LogisticRegression",
            "threshold_mode": threshold_mode,
            "threshold_from_val": float(best_t),
            "val_f1_pos1_at_best_t": float(best_t_f1) if best_t_f1 is not None else None,
            "val_auc": float(val_auc),
            "val_pred_pos_rate@0.5": float((p1_val >= 0.5).mean()),
            "val_pred_pos_rate@best_t": float((p1_val >= best_t).mean()),
            "test_auc": float(test_auc),
            "test_pred_pos_rate@0.5": float(pred_pos_rate_05),
            "test_pred_pos_rate@best_t": float(pred_pos_rate_bt),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_f1_pos1": float(f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)),
            "test_precision_pos1": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            "test_recall_pos1": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
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

        # ---- per-sample predictions
        out_dict = {
            "id": df_test["id"].astype(str).tolist(),
            "fold": [int(fold)] * len(df_test),
            "y_true": y_test,
            "p_depression": p1_test,
            "y_pred@best_t": y_pred,
            "y_pred@0.5": (p1_test >= 0.5).astype(int),
        }

        for c in scale_cols:
            out_dict[c] = df_test[c].tolist()

        pred_df = pd.DataFrame(out_dict)
        pred_df.to_csv(
            os.path.join(fold_dir, "test_predictions.csv"),
            index=False,
            encoding="utf-8-sig"
        )

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
        d.update({f"test_{k}": v for k, v in prob_summary(y_test, p1_test).items()})
        diag_rows.append(d)

        print(f"[Fold {fold}] TEST done. Saved to {fold_dir}")

    # -----------------------
    # CV summary + OOF
    # -----------------------
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
            " -> check duplicate IDs or split reference."
        )

    print("\n===== DONE =====")
    print("Saved all outputs to:", exp_dir)
    print(summary_df)


if __name__ == "__main__":
    main()
