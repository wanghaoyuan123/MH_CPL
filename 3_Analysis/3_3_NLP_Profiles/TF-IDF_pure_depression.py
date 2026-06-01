# ==========================
# TF-IDF 症状文本分析（纯抑郁版）
# 说明：
# 1) 请把本脚本放在“纯抑郁”文件夹里
# 2) 同文件夹中放入：
#    - Complains_output_depression_pure.xlsx
#    - hit_stopwords.txt
#    - symptom_keywords.txt
#    - bigram_stopwords.txt
# 3) 所有输出也会保存到这个文件夹
# ==========================

import re
import ast
from pathlib import Path
from collections import defaultdict

import jieba
import pandas as pd
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# ==========================
# 0. 基础路径与文件名
# ==========================
BASE_DIR = Path(__file__).resolve().parent

stopwords_file = BASE_DIR / "hit_stopwords.txt"
symptom_file = BASE_DIR / "symptom_keywords.txt"
bigram_stopwords_file = BASE_DIR / "bigram_stopwords.txt"
data_file = BASE_DIR / "Complains_output_depression_pure.xlsx"

processed_file = BASE_DIR / "processed_result_症状文本分析_pure.xlsx"
tfidf_unigram_file = BASE_DIR / "TFIDF_top100_候选症状_pure.xlsx"
tfidf_bigram_file = BASE_DIR / "TFIDF_bigram_候选症状_filtered_pure.xlsx"
tfidf_bigram_cov_file = BASE_DIR / "TFIDF_bigram_with_coverage_pure.xlsx"
coverage_png = BASE_DIR / "tfidf_bigram_cumulative_coverage_pure.png"
marginal_png = BASE_DIR / "tfidf_bigram_marginal_gain_pure.png"

print(f"当前脚本目录：{BASE_DIR}")
print(f"输入病历文件：{data_file.name}")

# ==========================
# 1. 加载停用词 & 症状词
# ==========================
def load_wordlist(file_path):
    words = set()
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words

stop_words = load_wordlist(stopwords_file)
symptom_keywords = load_wordlist(symptom_file)

print(f"✅ 停用词数量：{len(stop_words)}")
print(f"✅ 症状词数量：{len(symptom_keywords)}")

# 强制 jieba 把“症状短语”当成一个整体
for kw in symptom_keywords:
    jieba.add_word(kw)
    jieba.suggest_freq(kw, True)

# ==========================
# 2. 读取病历数据
# ==========================
df = pd.read_excel(data_file)
print(f"✅ 读取病历 {len(df)} 条")

if "现病史" not in df.columns:
    raise ValueError(f"输入文件中缺少“现病史”列。现有列：{list(df.columns)}")

# ==========================
# 3. 文本预处理函数
# ==========================
def preprocess_text(text):
    """
    功能：
    1）清洗符号
    2）jieba 分词（症状短语不被拆）
    3）症状词优先，其余再去停用词
    """
    if pd.isna(text):
        return []

    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", str(text))
    raw_tokens = jieba.lcut(text)

    tokens = []
    for t in raw_tokens:
        if t in symptom_keywords:
            tokens.append(t)
        elif t not in stop_words and len(t) > 1:
            tokens.append(t)

    return tokens

# ==========================
# 4. 主处理流程（生成 tokens + bigrams）
# ==========================
result = defaultdict(dict)

for idx, row in df.iterrows():
    raw_text = row["现病史"]

    tokens = preprocess_text(raw_text)
    bigrams = list(ngrams(tokens, 2))

    # 症状命中：直接在原文中查短语
    hits = [kw for kw in symptom_keywords if kw in str(raw_text)]

    result[idx] = {
        "原始文本": raw_text,
        "processed_tokens": tokens,
        "bigrams": bigrams,
        "症状命中": ",".join(hits)
    }

print("✅ 文本处理完成")

# ==========================
# 5. 保存处理结果
# ==========================
result_df = pd.DataFrame.from_dict(result, orient="index")
result_df.to_excel(processed_file, index=False)
print(f"✅ 结果已保存：{processed_file.name}")

# ==========================
# 6. TF-IDF（unigram）
# ==========================
def tokens_to_text(x):
    if isinstance(x, str):
        try:
            tokens = ast.literal_eval(x)
            return " ".join(tokens)
        except Exception:
            return ""
    elif isinstance(x, list):
        return " ".join(x)
    else:
        return ""

documents = result_df["processed_tokens"].apply(tokens_to_text).tolist()
print(f"✅ TF-IDF 输入文档数：{len(documents)}")

vectorizer = TfidfVectorizer(
    min_df=3,
    max_df=0.6,
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

avg_tfidf = tfidf_matrix.mean(axis=0).A1
tfidf_scores = list(zip(feature_names, avg_tfidf))
tfidf_sorted = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

TOP_N = 100
top_words = tfidf_sorted[:TOP_N]
output_df = pd.DataFrame(top_words, columns=["候选词", "平均TF-IDF"])
output_df.to_excel(tfidf_unigram_file, index=False)

print(f"🔥 TF-IDF 完成，已导出：{tfidf_unigram_file.name}")

# ==========================
# 7. 加载 Bigram 停用词
# ==========================
def load_bigram_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

bigram_stopwords = load_bigram_stopwords(bigram_stopwords_file)
print(f"✅ Bigram 停用词数量：{len(bigram_stopwords)}")

# ==========================
# 8. Bigram TF-IDF（加入 bigram 停用词过滤）
# ==========================
documents = result_df["processed_tokens"].apply(tokens_to_text).tolist()
print(f"✅ Bigram TF-IDF 输入文档数：{len(documents)}")

vectorizer = TfidfVectorizer(
    ngram_range=(2, 2),
    min_df=3,
    max_df=0.6,
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

avg_tfidf = tfidf_matrix.mean(axis=0).A1
tfidf_scores = list(zip(feature_names, avg_tfidf))

tfidf_filtered = [
    (term, score)
    for term, score in tfidf_scores
    if term not in bigram_stopwords
]

print(f"🧹 过滤前 bigram 数量：{len(tfidf_scores)}")
print(f"🧹 过滤后 bigram 数量：{len(tfidf_filtered)}")

tfidf_sorted = sorted(tfidf_filtered, key=lambda x: x[1], reverse=True)

TOP_N = 100
output_df = pd.DataFrame(
    tfidf_sorted[:TOP_N],
    columns=["二元短语", "平均TF-IDF"]
)
output_df.to_excel(tfidf_bigram_file, index=False)

print(f"🔥 Bigram TF-IDF（已过滤）完成，已导出：{tfidf_bigram_file.name}")

# ==========================
# 9. 边际收益分析
# ==========================
df_bigram = pd.read_excel(tfidf_bigram_file)

term_col = df_bigram.columns[0]
score_col = df_bigram.columns[1]

df_bigram = df_bigram.sort_values(score_col, ascending=False).reset_index(drop=True)

scores = df_bigram[score_col].values
total_score = scores.sum()

df_bigram["cumulative_score"] = scores.cumsum()
df_bigram["coverage"] = df_bigram["cumulative_score"] / total_score
df_bigram["marginal_gain"] = df_bigram["coverage"].diff().fillna(df_bigram["coverage"])

def find_k_for_coverage(target):
    return int((df_bigram["coverage"] >= target).idxmax() + 1)

k_80 = find_k_for_coverage(0.8)
k_90 = find_k_for_coverage(0.9)

threshold = 0.01
low_gain = df_bigram["marginal_gain"] < threshold

stop_k = None
count = 0
for i, v in enumerate(low_gain):
    if v:
        count += 1
        if count >= 3:
            stop_k = i + 1
            break
    else:
        count = 0

# 图1：累计覆盖率
plt.figure()
plt.plot(df_bigram.index + 1, df_bigram["coverage"])
plt.axhline(0.8, linestyle="--")
plt.axhline(0.9, linestyle="--")
plt.xlabel("Top-K Bigram")
plt.ylabel("Cumulative TF-IDF Coverage")
plt.title("TF-IDF Bigram Cumulative Coverage (Pure Depression)")
plt.tight_layout()
plt.savefig(coverage_png, dpi=300)
plt.close()

# 图2：边际增益
plt.figure()
plt.plot(df_bigram.index + 1, df_bigram["marginal_gain"])
plt.axhline(threshold, linestyle="--")
plt.xlabel("Top-K Bigram")
plt.ylabel("Marginal Gain (ΔCoverage)")
plt.title("TF-IDF Bigram Marginal Gain (Pure Depression)")
plt.tight_layout()
plt.savefig(marginal_png, dpi=300)
plt.close()

# 保存带分析的表
df_bigram.to_excel(tfidf_bigram_cov_file, index=False)

print("===== TF-IDF Bigram Coverage Summary =====")
print(f"K for 80% coverage: {k_80}")
print(f"K for 90% coverage: {k_90}")
print(f"Suggested stop_k (Δcoverage<0.01 for 3 consecutive): {stop_k}")

print(f"Saved: {tfidf_bigram_cov_file.name}")
print(f"Saved: {coverage_png.name}")
print(f"Saved: {marginal_png.name}")