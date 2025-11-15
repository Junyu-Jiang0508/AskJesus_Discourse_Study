import pandas as pd
from sklearn.utils import resample
import math

# ==== 参数 ====
target_n = 200  # 总样本数
cluster_col = "cluster"
text_col = "sentence"

# ==== 读取数据 ====
df_csv = pd.read_csv("output/fewshot_candidate_5002.csv")
df_excel = pd.read_excel("output/前500筛选.excel.xlsx")

# ==== 清洗：保留非空文本与 cluster ====
df_csv = df_csv.dropna(subset=[text_col, cluster_col])
df_excel = df_excel.dropna(subset=[text_col, cluster_col])

df_csv[cluster_col] = df_csv[cluster_col].astype(int)
df_excel[cluster_col] = df_excel[cluster_col].astype(int)

# ==== 分层抽样函数（修复错误） ====
def stratified_sample(df, cluster_col, n_total):
    n_cluster = df[cluster_col].nunique()
    n_per_cluster = math.ceil(n_total / n_cluster)
    samples = []
    for cid, group in df.groupby(cluster_col):
        n_sample = min(n_per_cluster, len(group))
        sample = resample(group, n_samples=n_sample, random_state=42, replace=False)
        samples.append(sample)
    combined = pd.concat(samples)
    # 再次确认是否总数大于目标值（才做 sample 截断）
    if len(combined) > n_total:
        combined = combined.sample(n=n_total, random_state=42)
    return combined

# ==== 抽样执行 ====
sample_csv = stratified_sample(df_csv, cluster_col, target_n)
sample_excel = stratified_sample(df_excel, cluster_col, target_n)

# ==== 输出保存 ====
sample_csv.to_csv("output/stratified_sample_csv_200.csv", index=False)
sample_excel.to_csv("output/stratified_sample_excel_200.csv", index=False)

print("✅ 成功完成分层抽样并保存两份 200 条样本")
