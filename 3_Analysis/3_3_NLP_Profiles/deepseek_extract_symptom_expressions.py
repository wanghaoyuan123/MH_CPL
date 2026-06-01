# ==========================================================
# deepseek_extract_symptom_expressions_fast.py
#
# 全量快速版：
# 1. 读取 ML_dataset_balanced_5fold_pure_depression.xlsx
# 2. 读取 symptom_expression_reference.xlsx
#    只用 A列 = 症状体系
#    只用 E列以后 = 已有病历表述
# 3. 调用 DeepSeek API 批量扫描 history_text
# 4. 只抽取 variant / new_candidate
# 5. 不输出 exact
# 6. 关闭 thinking mode
# 7. DeepSeek 返回空内容时直接记录 error，不反复等待
#
# 输出文件夹：
# deepseek_symptom_expression_outputs
#
# 主要输出：
# 1. deepseek_extraction_detail.csv
# 2. deepseek_extraction_summary.csv
# 3. deepseek_extraction_new_or_variant.csv
# 4. deepseek_extraction_progress.csv
# 5. deepseek_extraction_errors.csv
# ==========================================================

import os
import json
import traceback
from collections import Counter

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ==========================================================
# 0. 路径设置
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "ML_dataset_balanced_5fold_pure_depression.xlsx"
)

REFERENCE_FILE = os.path.join(
    BASE_DIR,
    "symptom_expression_reference.xlsx"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "deepseek_symptom_expression_outputs"
)

os.makedirs(OUT_DIR, exist_ok=True)

DETAIL_OUT = os.path.join(OUT_DIR, "deepseek_extraction_detail.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "deepseek_extraction_summary.csv")
NEW_VARIANT_OUT = os.path.join(OUT_DIR, "deepseek_extraction_new_or_variant.csv")
PROGRESS_OUT = os.path.join(OUT_DIR, "deepseek_extraction_progress.csv")
ERROR_OUT = os.path.join(OUT_DIR, "deepseek_extraction_errors.csv")


# ==========================================================
# 1. DeepSeek 设置
# ==========================================================
MODEL_NAME = "deepseek-v4-flash"

TEMPERATURE = 0
MAX_TOKENS = 1000

# 全量运行
MAX_ROWS = None

# 每处理多少条保存一次
SAVE_EVERY = 10

# 如果 symptom_expression_reference.xlsx 第一行是表头，改成 True。
# 如果第一行就是正式数据，保持 False。
REFERENCE_HAS_HEADER = False


# ==========================================================
# 2. 清理函数
# ==========================================================
def clean_cell(x):
    if pd.isna(x):
        return ""

    x = str(x)
    x = x.replace("\n", "")
    x = x.replace("\r", "")
    x = x.replace("\t", "")
    x = x.strip()

    return x


def clean_phrase(x):
    if x is None:
        return ""

    x = str(x)
    x = x.replace("\n", "")
    x = x.replace("\r", "")
    x = x.replace("\t", "")
    x = x.strip()

    return x


def normalize_for_contains(x):
    """
    用于判断 phrase 是否真的在原文中。
    去掉空格、换行、制表符，避免因为格式问题误判。
    """
    if x is None:
        return ""

    x = str(x)
    x = x.replace("\n", "")
    x = x.replace("\r", "")
    x = x.replace("\t", "")
    x = x.replace(" ", "")
    x = x.strip()

    return x


def is_phrase_in_text(phrase, text):
    phrase_norm = normalize_for_contains(phrase)
    text_norm = normalize_for_contains(text)

    if phrase_norm == "" or text_norm == "":
        return False

    return phrase_norm in text_norm


# ==========================================================
# 3. 读取参考表：只读 A列 + E列以后
# ==========================================================
def read_reference_table(path):
    ref = pd.read_excel(path, header=None)

    if REFERENCE_HAS_HEADER:
        ref = ref.iloc[1:, :].copy()

    rows = []

    for i in range(len(ref)):
        # A列：症状体系
        symptom = clean_cell(ref.iloc[i, 0])

        # E列以后：已有病历表述
        expressions = []

        for j in range(4, ref.shape[1]):
            value = clean_cell(ref.iloc[i, j])

            if value != "":
                expressions.append(value)

        if symptom == "" and len(expressions) == 0:
            continue

        rows.append({
            "symptom": symptom,
            "expressions": expressions
        })

    return rows


def make_reference_block(reference_rows):
    lines = []

    for idx, row in enumerate(reference_rows, 1):
        symptom = row["symptom"]
        expressions = row["expressions"]

        if len(expressions) == 0:
            expr_text = "无"
        else:
            expr_text = "；".join(expressions)

        one_block = (
            f"{idx}. 症状：{symptom}\n"
            f"   已有表述：{expr_text}"
        )

        lines.append(one_block)

    return "\n\n".join(lines)


# ==========================================================
# 4. 构造提示词：只找 variant / new_candidate
# ==========================================================
def build_prompt(history_text, reference_block):
    prompt = f"""
你是一名精神科临床文本标注助手。

我正在整理儿童青少年门诊病历中的症状表述库。

下面给你两部分内容：
1. 一张“症状体系 + 已有病历表述”的参考表；
2. 一段原始病历文本。

参考表说明：
- “症状”来自我人工整理的症状体系；
- “已有表述”是我已经整理出的病历原文表述；
- 你的任务不是重复找已有表述，而是帮助发现可能遗漏的表述。

你的任务：
只从原始病历文本中抽取以下两类内容：

A. variant：
和参考表中已有表述意思相同或相近，但原文写法不同的连续片段。

B. new_candidate：
参考表中可能没有覆盖，但明显属于某个症状体系的新的原文连续片段。

严格不要输出：
1. 和已有表述完全一样的 exact phrase；
2. 诊断名，例如“抑郁状态”“焦虑状态”“精神障碍”“注意缺陷多动障碍”；
3. 病历结构词，例如“今日来诊”“精神检查”“个人史”“既往史”“病史资料”“查体”；
4. 纯就诊流程词，例如“来诊”“复诊”“门诊”“建议治疗”“随诊”“完善检查”；
5. 单纯时间、年份、日期、年级，例如“2022”“2023”“初一”“目前初二”“近半年”；
6. 单独的家庭成员或场景词，例如“母亲”“父亲”“学校”“老师”“同学”；
7. 单独的程度副词或连接词，例如“有时”“目前”“一直”“明显”“较前”；
8. 医生模板化的正常精神检查描述，例如“意识清晰”“接触可”“对答切题”“定向力完整”。

非常重要的规则：
1. phrase 必须是原始病历文本中连续出现的原文，不能改写，不能总结，不能创造。
2. 如果一个长短语已经完整表达症状，不要再重复输出其中的短词。
3. 如果原文中只有已有表述的完全匹配，没有变体或新候选，返回空数组。
4. 如果没有可抽取内容，返回空数组。
5. 对于 new_candidate，也要尽量填写它最接近的参考表症状 matched_symptom。
6. 如果无法匹配任何参考表症状，但确实像症状或功能受损，可以保留为 new_candidate，matched_symptom 写空字符串。

输出必须是 JSON。
不要输出任何 JSON 以外的解释文字。

JSON 格式如下：
{{
  "items": [
    {{
      "phrase": "原始病历文本中连续出现的症状或功能受损表述",
      "matched_symptom": "匹配到的参考表症状；如果没有明确匹配，写空字符串",
      "matched_existing_expression": "最接近的已有表述；如果没有，写空字符串",
      "relation_to_reference": "variant / new_candidate",
      "confidence": "high / medium / low",
      "reason": "简短说明为什么它是遗漏表述或已有表述的变体"
    }}
  ]
}}

没有可抽取内容时输出：
{{"items": []}}

症状体系与已有表述参考表：
{reference_block}

原始病历文本：
{history_text}
"""
    return prompt.strip()


# ==========================================================
# 5. 调用 DeepSeek：关闭 thinking，空内容直接记 error
# ==========================================================
def call_deepseek(client, history_text, reference_block):
    prompt = build_prompt(history_text, reference_block)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严谨的精神科临床文本标注助手。"
                    "你必须只输出合法 JSON。"
                    "所有 phrase 必须来自原文中的连续片段。"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    if content is None or content.strip() == "":
        raise ValueError(f"DeepSeek 返回空内容，finish_reason={finish_reason}")

    content = content.strip()

    # 防止模型偶尔输出 markdown 代码块
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    data = json.loads(content)

    if "items" not in data:
        data = {"items": []}

    if not isinstance(data["items"], list):
        data = {"items": []}

    response_id = getattr(response, "id", "")

    return data, response_id


# ==========================================================
# 6. 保存输出
# ==========================================================
def save_outputs(detail_rows, progress_rows, error_rows, df_all):
    detail_df = pd.DataFrame(detail_rows)

    if len(detail_df) == 0:
        detail_df = pd.DataFrame(columns=[
            "ID",
            "label",
            "fold",
            "phrase",
            "matched_symptom",
            "matched_existing_expression",
            "relation_to_reference",
            "confidence",
            "reason",
            "phrase_in_text",
            "response_id"
        ])

    detail_df.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")

    progress_df = pd.DataFrame(progress_rows)

    if len(progress_df) == 0:
        progress_df = pd.DataFrame(columns=[
            "ID",
            "status",
            "n_items",
            "error"
        ])

    progress_df.to_csv(PROGRESS_OUT, index=False, encoding="utf-8-sig")

    if len(error_rows) > 0:
        error_df = pd.DataFrame(error_rows)
        error_df.to_csv(ERROR_OUT, index=False, encoding="utf-8-sig")

    # 只汇总模型输出中确实存在于原文的 phrase
    if "phrase_in_text" not in detail_df.columns:
        pd.DataFrame().to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(NEW_VARIANT_OUT, index=False, encoding="utf-8-sig")
        return

    valid = detail_df[detail_df["phrase_in_text"] == True].copy()

    if len(valid) == 0:
        pd.DataFrame().to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(NEW_VARIANT_OUT, index=False, encoding="utf-8-sig")
        return

    # 同一个样本内同一个 phrase 只算一次
    valid = valid.drop_duplicates(subset=["ID", "phrase"])

    n_dep = int((df_all["label"] == 1).sum())
    n_non = int((df_all["label"] == 0).sum())

    rows = []

    for phrase, g in valid.groupby("phrase"):
        dep_hits = int((g["label"] == 1).sum())
        non_hits = int((g["label"] == 0).sum())

        dep_prop = dep_hits / n_dep if n_dep > 0 else 0
        non_prop = non_hits / n_non if n_non > 0 else 0

        symptom_counter = Counter(g["matched_symptom"].fillna("").astype(str).tolist())
        expression_counter = Counter(g["matched_existing_expression"].fillna("").astype(str).tolist())
        relation_counter = Counter(g["relation_to_reference"].fillna("").astype(str).tolist())
        confidence_counter = Counter(g["confidence"].fillna("").astype(str).tolist())

        rows.append({
            "phrase": phrase,
            "n_samples": int(g["ID"].nunique()),
            "dep_hits": dep_hits,
            "non_hits": non_hits,
            "dep_prop": dep_prop,
            "non_prop": non_prop,
            "diff_prop": dep_prop - non_prop,
            "main_matched_symptom": symptom_counter.most_common(1)[0][0] if len(symptom_counter) > 0 else "",
            "main_matched_existing_expression": expression_counter.most_common(1)[0][0] if len(expression_counter) > 0 else "",
            "main_relation_to_reference": relation_counter.most_common(1)[0][0] if len(relation_counter) > 0 else "",
            "main_confidence": confidence_counter.most_common(1)[0][0] if len(confidence_counter) > 0 else ""
        })

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        ["n_samples", "diff_prop"],
        ascending=[False, False]
    )

    summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    new_variant_df = summary_df[
        summary_df["main_relation_to_reference"].isin(["variant", "new_candidate"])
    ].copy()

    new_variant_df = new_variant_df.sort_values(
        ["n_samples", "diff_prop"],
        ascending=[False, False]
    )

    new_variant_df.to_csv(NEW_VARIANT_OUT, index=False, encoding="utf-8-sig")


# ==========================================================
# 7. 主程序
# ==========================================================
def main():
    print("BASE_DIR:", BASE_DIR)
    print("DATA_FILE:", DATA_FILE)
    print("REFERENCE_FILE:", REFERENCE_FILE)
    print("OUT_DIR:", OUT_DIR)
    print("MODEL_NAME:", MODEL_NAME)
    print("MAX_ROWS:", MAX_ROWS)

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if api_key is None or api_key.strip() == "":
        raise ValueError(
            "没有检测到 DEEPSEEK_API_KEY。请先设置环境变量，或者在 PyCharm 的 Environment variables 里添加。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("找不到数据文件：" + DATA_FILE)

    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError("找不到参考表文件：" + REFERENCE_FILE)

    # --------------------------
    # 7.1 读取原始建模数据
    # --------------------------
    df = pd.read_excel(DATA_FILE)

    required_cols = ["ID", "history_text", "label"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"数据缺少必要列：{col}。当前列名为：{list(df.columns)}"
            )

    if "fold" not in df.columns:
        df["fold"] = ""

    df["ID"] = df["ID"].astype(str)
    df["history_text"] = df["history_text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["history_text"].str.strip() != ""].copy()

    # --------------------------
    # 7.2 读取参考表
    # --------------------------
    reference_rows = read_reference_table(REFERENCE_FILE)
    reference_block = make_reference_block(reference_rows)

    print("\n数据样本数：", len(df))
    print("参考表症状行数：", len(reference_rows))
    print("label 分布：")
    print(df["label"].value_counts())

    # --------------------------
    # 7.3 全量运行
    # --------------------------
    if MAX_ROWS is not None:
        df_run = df.head(MAX_ROWS).copy()
        print("\n当前为试跑/分批模式，只检查前", MAX_ROWS, "条。")
    else:
        df_run = df.copy()
        print("\n当前为全量模式，将处理全部样本。")

    # --------------------------
    # 7.4 断点续跑
    # 如果输出文件夹已删除，这里会自动从头开始。
    # --------------------------
    done_ids = set()
    detail_rows = []
    progress_rows = []
    error_rows = []

    if os.path.exists(PROGRESS_OUT):
        try:
            old_progress = pd.read_csv(PROGRESS_OUT, encoding="utf-8-sig")

            if "ID" in old_progress.columns and "status" in old_progress.columns:
                done_ids = set(
                    old_progress.loc[
                        old_progress["status"] == "done",
                        "ID"
                    ].astype(str).tolist()
                )

                progress_rows = old_progress.to_dict("records")

            print("检测到历史进度，已完成 ID 数：", len(done_ids))

        except Exception:
            print("历史 progress 读取失败，将重新开始记录 progress。")
            done_ids = set()
            progress_rows = []

    if os.path.exists(DETAIL_OUT):
        try:
            old_detail = pd.read_csv(DETAIL_OUT, encoding="utf-8-sig")
            detail_rows = old_detail.to_dict("records")
            print("检测到历史 detail，已有行数：", len(detail_rows))

        except Exception:
            print("历史 detail 读取失败，将重新写入。")
            detail_rows = []

    if os.path.exists(ERROR_OUT):
        try:
            old_error = pd.read_csv(ERROR_OUT, encoding="utf-8-sig")
            error_rows = old_error.to_dict("records")

        except Exception:
            error_rows = []

    # --------------------------
    # 7.5 逐条调用 DeepSeek
    # --------------------------
    processed_since_save = 0

    for _, row in tqdm(df_run.iterrows(), total=len(df_run)):
        sample_id = str(row["ID"])

        if sample_id in done_ids:
            continue

        history_text = str(row["history_text"])
        label = int(row["label"])
        fold = row["fold"]

        try:
            result, response_id = call_deepseek(
                client=client,
                history_text=history_text,
                reference_block=reference_block
            )

            items = result.get("items", [])

            kept_items = 0

            for item in items:
                phrase = clean_phrase(item.get("phrase", ""))

                if phrase == "":
                    continue

                relation = clean_cell(item.get("relation_to_reference", ""))

                # 只保留 variant / new_candidate
                if relation not in ["variant", "new_candidate"]:
                    continue

                phrase_in_text = is_phrase_in_text(phrase, history_text)

                detail_rows.append({
                    "ID": sample_id,
                    "label": label,
                    "fold": fold,
                    "phrase": phrase,
                    "matched_symptom": clean_cell(item.get("matched_symptom", "")),
                    "matched_existing_expression": clean_cell(item.get("matched_existing_expression", "")),
                    "relation_to_reference": relation,
                    "confidence": clean_cell(item.get("confidence", "")),
                    "reason": clean_cell(item.get("reason", "")),
                    "phrase_in_text": phrase_in_text,
                    "response_id": response_id
                })

                kept_items += 1

            progress_rows.append({
                "ID": sample_id,
                "status": "done",
                "n_items": kept_items,
                "error": ""
            })

            done_ids.add(sample_id)
            processed_since_save += 1

        except Exception as e:
            error_message = str(e)

            print("\n[ERROR] ID =", sample_id)
            print(error_message)

            error_rows.append({
                "ID": sample_id,
                "status": "error",
                "error": error_message,
                "traceback": traceback.format_exc()
            })

            progress_rows.append({
                "ID": sample_id,
                "status": "error",
                "n_items": 0,
                "error": error_message
            })

            processed_since_save += 1

        if processed_since_save >= SAVE_EVERY:
            save_outputs(
                detail_rows=detail_rows,
                progress_rows=progress_rows,
                error_rows=error_rows,
                df_all=df
            )

            print("\n已中途保存。")
            processed_since_save = 0

    # --------------------------
    # 7.6 最终保存
    # --------------------------
    save_outputs(
        detail_rows=detail_rows,
        progress_rows=progress_rows,
        error_rows=error_rows,
        df_all=df
    )

    print("\n全部完成。输出文件：")
    print("1.", DETAIL_OUT)
    print("2.", SUMMARY_OUT)
    print("3.", NEW_VARIANT_OUT)
    print("4.", PROGRESS_OUT)

    if len(error_rows) > 0:
        print("5.", ERROR_OUT)


if __name__ == "__main__":
    main()