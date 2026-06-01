# ==========================================================
# organize_deepseek_candidates_by_symptom.py
#
# 目的：
# 把 DeepSeek 逐条病历抽取结果整理成“按症状审核”的表。
#
# 输入：
#   1. deepseek_symptom_expression_outputs/deepseek_extraction_detail.csv
#   2. symptom_expression_reference.xlsx
#
# 输出：
#   deepseek_symptom_expression_outputs/deepseek_candidates_review_by_symptom.xlsx
#
# 输出内容：
#   Sheet1: 按症状整理_人工审核
#   Sheet2: 每个症状候选汇总
#   Sheet3: 未匹配症状_需人工看
#   Sheet4: 原始有效明细
# ==========================================================

import os
import re
from collections import Counter

import pandas as pd


# ==========================================================
# 0. 路径设置
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(BASE_DIR, "deepseek_symptom_expression_outputs")

DETAIL_FILE = os.path.join(OUT_DIR, "deepseek_extraction_detail.csv")

REFERENCE_FILE = os.path.join(BASE_DIR, "symptom_expression_reference.xlsx")

REVIEW_OUT = os.path.join(OUT_DIR, "deepseek_candidates_review_by_symptom.xlsx")


# 如果你的参考表第一行是表头，改成 True
REFERENCE_HAS_HEADER = False


# ==========================================================
# 1. 清理函数
# ==========================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\n", "")
    x = x.replace("\r", "")
    x = x.replace("\t", "")
    x = x.strip()
    return x


def normalize_text(x):
    """
    用于判断短语是否重复。
    """
    x = clean_text(x)
    x = re.sub(r"\s+", "", x)
    x = x.replace("，", "")
    x = x.replace("。", "")
    x = x.replace("；", "")
    x = x.replace("、", "")
    x = x.replace(",", "")
    x = x.replace(".", "")
    x = x.replace(";", "")
    return x


def mode_value(series):
    """
    取最常见值。
    """
    vals = [clean_text(x) for x in series.tolist() if clean_text(x) != ""]
    if len(vals) == 0:
        return ""
    return Counter(vals).most_common(1)[0][0]


def join_unique(series, max_n=8):
    """
    把若干值合并成一个字符串。
    """
    vals = []
    for x in series.tolist():
        x = clean_text(x)
        if x != "" and x not in vals:
            vals.append(x)
    return "；".join(vals[:max_n])


# ==========================================================
# 2. 读取参考表：A列症状 + E列以后已有表述
# ==========================================================
def read_reference_table(path):
    ref = pd.read_excel(path, header=None)

    if REFERENCE_HAS_HEADER:
        ref = ref.iloc[1:, :].copy()

    rows = []
    symptom_order = {}
    expression_to_symptom = {}

    for i in range(len(ref)):
        symptom = clean_text(ref.iloc[i, 0])

        expressions = []
        for j in range(4, ref.shape[1]):
            value = clean_text(ref.iloc[i, j])
            if value != "":
                expressions.append(value)

        if symptom == "" and len(expressions) == 0:
            continue

        if symptom != "" and symptom not in symptom_order:
            symptom_order[symptom] = len(symptom_order) + 1

        for expr in expressions:
            expr_norm = normalize_text(expr)
            if expr_norm != "" and symptom != "":
                expression_to_symptom[expr_norm] = symptom

        rows.append({
            "symptom": symptom,
            "expressions": expressions
        })

    return rows, symptom_order, expression_to_symptom


# ==========================================================
# 3. 主程序
# ==========================================================
def main():
    print("DETAIL_FILE:", DETAIL_FILE)
    print("REFERENCE_FILE:", REFERENCE_FILE)
    print("REVIEW_OUT:", REVIEW_OUT)

    if not os.path.exists(DETAIL_FILE):
        raise FileNotFoundError("找不到 detail 文件：" + DETAIL_FILE)

    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError("找不到参考表文件：" + REFERENCE_FILE)

    reference_rows, symptom_order, expression_to_symptom = read_reference_table(REFERENCE_FILE)

    detail = pd.read_csv(DETAIL_FILE, encoding="utf-8-sig")

    if len(detail) == 0:
        raise ValueError("detail 文件是空的。")

    # ======================================================
    # 3.1 基础过滤
    # ======================================================
    detail["phrase"] = detail["phrase"].apply(clean_text)
    detail["matched_symptom"] = detail["matched_symptom"].apply(clean_text)
    detail["matched_existing_expression"] = detail["matched_existing_expression"].apply(clean_text)
    detail["relation_to_reference"] = detail["relation_to_reference"].apply(clean_text)
    detail["confidence"] = detail["confidence"].apply(clean_text)
    detail["reason"] = detail["reason"].apply(clean_text)

    if "phrase_in_text" in detail.columns:
        detail = detail[detail["phrase_in_text"] == True].copy()

    detail = detail[detail["phrase"] != ""].copy()

    # 只保留 variant / new_candidate
    detail = detail[
        detail["relation_to_reference"].isin(["variant", "new_candidate"])
    ].copy()

    # ======================================================
    # 3.2 给每条结果确定一个“整理用症状”
    # ======================================================
    def decide_review_symptom(row):
        matched_symptom = clean_text(row.get("matched_symptom", ""))
        matched_expr = clean_text(row.get("matched_existing_expression", ""))

        # 第一优先：模型已经匹配到 A列症状
        if matched_symptom != "":
            return matched_symptom

        # 第二优先：根据 matched_existing_expression 反查它属于哪个 A列症状
        expr_norm = normalize_text(matched_expr)
        if expr_norm in expression_to_symptom:
            return expression_to_symptom[expr_norm]

        # 第三优先：如果 phrase 本身正好是已有表述，也反查症状
        phrase_norm = normalize_text(row.get("phrase", ""))
        if phrase_norm in expression_to_symptom:
            return expression_to_symptom[phrase_norm]

        # 都找不到，就放到未匹配
        return "未匹配症状_需人工判断"

    detail["review_symptom"] = detail.apply(decide_review_symptom, axis=1)
    detail["phrase_norm"] = detail["phrase"].apply(normalize_text)

    # ======================================================
    # 3.3 按 症状 + phrase 聚合
    # ======================================================
    group_cols = ["review_symptom", "phrase_norm"]

    rows = []

    for (review_symptom, phrase_norm), g in detail.groupby(group_cols):
        phrase_display = mode_value(g["phrase"])

        relation_main = mode_value(g["relation_to_reference"])
        confidence_main = mode_value(g["confidence"])
        matched_symptom_main = mode_value(g["matched_symptom"])
        matched_expr_main = mode_value(g["matched_existing_expression"])

        n_samples = g["ID"].astype(str).nunique() if "ID" in g.columns else len(g)

        dep_hits = int((g["label"] == 1).sum()) if "label" in g.columns else ""
        non_hits = int((g["label"] == 0).sum()) if "label" in g.columns else ""

        example_ids = join_unique(g["ID"].astype(str), max_n=5) if "ID" in g.columns else ""
        example_reasons = join_unique(g["reason"], max_n=3)

        if confidence_main == "high":
            priority = "优先看"
        elif confidence_main == "medium":
            priority = "可看"
        else:
            priority = "低优先"

        rows.append({
            "症状_用于整理": review_symptom,
            "候选表述_phrase": phrase_display,
            "出现样本数": n_samples,
            "抑郁组命中数": dep_hits,
            "非抑郁组命中数": non_hits,
            "模型关系": relation_main,
            "模型置信度": confidence_main,
            "人工审核优先级": priority,
            "模型匹配A列症状": matched_symptom_main,
            "最接近已有表述": matched_expr_main,
            "示例ID": example_ids,
            "模型理由示例": example_reasons,
            "人工决定_纳入不纳入待定": "",
            "最终归入症状": "",
            "最终标准表述": "",
            "人工备注": ""
        })

    review_df = pd.DataFrame(rows)

    if len(review_df) == 0:
        raise ValueError("过滤后没有可整理的候选。")

    # 症状顺序：优先按参考表 A 列原始顺序
    def symptom_sort_key(x):
        return symptom_order.get(x, 999999)

    review_df["症状排序"] = review_df["症状_用于整理"].apply(symptom_sort_key)

    review_df = review_df.sort_values(
        ["症状排序", "人工审核优先级", "出现样本数"],
        ascending=[True, True, False]
    ).drop(columns=["症状排序"])

    # ======================================================
    # 3.4 每个症状压缩成一行，方便快速看总体
    # ======================================================
    summary_rows = []

    for symptom, g in review_df.groupby("症状_用于整理", sort=False):
        candidates = []

        for _, r in g.iterrows():
            item = (
                str(r["候选表述_phrase"])
                + f"(n={r['出现样本数']}, {r['模型关系']}, {r['模型置信度']})"
            )
            candidates.append(item)

        summary_rows.append({
            "症状": symptom,
            "候选表述数量": len(g),
            "候选表述汇总": "；".join(candidates),
            "高置信候选数": int((g["模型置信度"] == "high").sum()),
            "中置信候选数": int((g["模型置信度"] == "medium").sum()),
            "低置信候选数": int((g["模型置信度"] == "low").sum())
        })

    symptom_summary_df = pd.DataFrame(summary_rows)

    unmatched_df = review_df[
        review_df["症状_用于整理"] == "未匹配症状_需人工判断"
    ].copy()

    # ======================================================
    # 3.5 写入 Excel
    # ======================================================
    with pd.ExcelWriter(REVIEW_OUT, engine="openpyxl") as writer:
        review_df.to_excel(writer, sheet_name="按症状整理_人工审核", index=False)
        symptom_summary_df.to_excel(writer, sheet_name="每个症状候选汇总", index=False)
        unmatched_df.to_excel(writer, sheet_name="未匹配症状_需人工看", index=False)
        detail.to_excel(writer, sheet_name="原始有效明细", index=False)

    print("\n整理完成。输出文件：")
    print(REVIEW_OUT)

    print("\n建议优先看：")
    print("1. Sheet：每个症状候选汇总")
    print("2. Sheet：按症状整理_人工审核")
    print("3. Sheet：未匹配症状_需人工看")


if __name__ == "__main__":
    main()