import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import torch.nn.functional as F

# ==== 参数配置 ====
MODEL_PATH = "model/bge-base-zh"  # 替换为你的本地路径
CSV_FILE = "output/split_txt_sentences.csv"  # 替换为你的文件名
COLUMN_NAME = "sentence"
NUM_CLUSTERS = 100
SAMPLES_PER_CLUSTER = 10
OUTPUT_FILE = "output/fewshot_candidate_1000.csv"

# ==== 加载数据 ====
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=[COLUMN_NAME])
sentences = df[COLUMN_NAME].tolist()

# ==== 加载模型 ====
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

# ==== 嵌入函数 ====
def embed(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model(**encoded)
        cls_embedding = output.last_hidden_state[:, 0]
        normalized = F.normalize(cls_embedding, p=2, dim=1)
        all_embeddings.append(normalized.cpu().numpy())
    return np.vstack(all_embeddings)

# ==== 编码句子 ====
embeddings = embed(sentences)

# ==== 聚类 ====
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init="auto")
labels = kmeans.fit_predict(embeddings)

# ==== 每簇抽样 ====
samples = []
for cluster_id in range(NUM_CLUSTERS):
    indices = np.where(labels == cluster_id)[0]
    chosen = np.random.choice(indices, size=min(SAMPLES_PER_CLUSTER, len(indices)), replace=False)
    for idx in chosen:
        samples.append({
            "cluster": cluster_id,
            "sentence": sentences[idx]
        })

# ==== 保存为文件 ====
sample_df = pd.DataFrame(samples)
sample_df.to_csv(OUTPUT_FILE, index=False)
print(f" 已保存 1000 条候选样本至 {OUTPUT_FILE}")
