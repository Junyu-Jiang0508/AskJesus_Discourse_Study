import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

# 1. 读取标签文件
primary_labels = pd.read_csv("data/labeled_set/primary_label.csv", encoding='mbcs')
secondary_labels = pd.read_csv("data/labeled_set/secondary_label.csv", encoding='mbcs')

# 2. 读取主数据集
df = pd.read_csv("data/labeled_set/set.csv", encoding='mbcs')

# 处理缺失值
df['primary_label'] = df['primary_label'].fillna('Unknown').astype(str).str.strip()
df['secondary_label'] = df['secondary_label'].fillna('Unknown').astype(str).str.strip()

# 3. 构建组合标签并编码为整数索引
df['combo'] = df['primary_label'] + "_" + df['secondary_label']
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['combo'])
all_labels = sorted(df['combo'].unique().tolist())

# 打印标签分布
print(f"总样本数: {len(df)}")
print(f"标签类别数: {len(all_labels)}")
print(df['combo'].value_counts())

# 4. 划分训练/验证集
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. 加载本地模型和tokenizer
model_path = "model/bge-base-zh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(all_labels)  # 多类分类
)

# 6. 计算类别权重（解决不平衡问题）
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"类别权重: {class_weights}")

# 7. 转换为HuggingFace Dataset
text_column = 'sentence' if 'sentence' in df.columns else 'text'
print(f"使用文本列: '{text_column}'")


def tokenize_batch(batch):
    return tokenizer(batch[text_column], padding="max_length", truncation=True, max_length=256)


dataset_train = Dataset.from_pandas(train_df)
dataset_eval = Dataset.from_pandas(eval_df)
dataset_train = dataset_train.map(tokenize_batch, batched=True)
dataset_eval = dataset_eval.map(tokenize_batch, batched=True)


# 8. 添加整数标签
def add_label_tensor(example):
    example["labels"] = torch.tensor(example['label'], dtype=torch.long)
    return example


dataset_train = dataset_train.map(add_label_tensor)
dataset_eval = dataset_eval.map(add_label_tensor)

# 9. 保留必要列
keep_cols = ["input_ids", "attention_mask", "labels"]
dataset_train = dataset_train.remove_columns([c for c in dataset_train.column_names if c not in keep_cols])
dataset_eval = dataset_eval.remove_columns([c for c in dataset_eval.column_names if c not in keep_cols])
dataset_train.set_format("torch")
dataset_eval.set_format("torch")


# 10. 评估函数（多类分类）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# 11. 修正加权损失训练器（关键修复）
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        解决 'num_items_in_batch' 参数兼容性问题
        并处理可能出现的维度错误
        """
        # 安全获取标签
        labels = inputs.pop("labels") if "labels" in inputs else None

        # 模型前向传播
        outputs = model(**inputs)

        # 检查logits维度
        logits = outputs.logits
        if len(logits.shape) == 1:  # 单样本时维度可能减少
            logits = logits.unsqueeze(0)

        # 检查labels维度
        if labels.dim() == 0:  # 单样本时维度可能减少
            labels = labels.unsqueeze(0)

        # 计算加权交叉熵损失
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# 12. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # 进一步减小以支持小内存
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # 减少轮次以快速测试
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=5,
    seed=42,
    report_to="none",  # 禁用wandb等报告
    fp16=False,  # 确保不使用混合精度
)

# 13. 创建训练器
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    compute_metrics=compute_metrics,
)

# 14. 训练与评估
print("开始训练...")
trainer.train()

print("最终评估...")
eval_results = trainer.evaluate()
print("评估结果:", eval_results)

# 15. 保存最佳模型
trainer.save_model("./best_model")