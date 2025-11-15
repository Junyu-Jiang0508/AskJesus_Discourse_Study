import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

# 设置数据路径
DATA_DIR = "data/labeled_set"  # 根据您的实际路径调整
DATA_FILE = "set.csv"


def load_and_preprocess_data():
    """从set.csv加载并预处理标注数据"""
    # 构建完整文件路径
    data_path = os.path.join(DATA_DIR, DATA_FILE)

    try:
        # 加载数据
        df = pd.read_csv(data_path)
        print(f"成功加载数据: {df.shape[0]}条记录")

        # 检查必需列
        required_columns = ['sentence', 'primary_label', 'secondary_label']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            raise ValueError(f"CSV文件中缺少必需列: {missing_cols}")

        # 数据清洗：移除空值行
        initial_count = df.shape[0]
        df = df.dropna(subset=['primary_label', 'secondary_label'])
        cleaned_count = df.shape[0]

        if initial_count > cleaned_count:
            print(f"已移除 {initial_count - cleaned_count} 条包含空标签的记录")

        # 验证标签质量
        invalid_primary = df[df['primary_label'].str.strip() == '']
        invalid_secondary = df[df['secondary_label'].str.strip() == '']

        if not invalid_primary.empty or not invalid_secondary.empty:
            print("警告：发现空字符串标签")
            df = df[(df['primary_label'].str.strip() != '') &
                    (df['secondary_label'].str.strip() != '')]

        # 创建标签编码器
        primary_encoder = LabelEncoder()
        secondary_encoder = LabelEncoder()

        # 转换标签列
        df['primary_encoded'] = primary_encoder.fit_transform(df['primary_label'])
        df['secondary_encoded'] = secondary_encoder.fit_transform(df['secondary_label'])

        # 保存标签映射信息
        label_mappings = {
            'primary': dict(enumerate(primary_encoder.classes_)),
            'secondary': dict(enumerate(secondary_encoder.classes_))
        }

        print(f"主标签类别: {list(label_mappings['primary'].values())}")
        print(f"次标签类别: {list(label_mappings['secondary'].values())}")

        return df, label_mappings

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        print(f"请检查文件路径: {data_path}")
        print("确保文件存在且格式正确")
        raise


def prepare_datasets(df):
    """准备训练和验证数据集"""
    # 划分数据集 (80%训练, 20%验证)
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[['primary_label', 'secondary_label']]
        )

        print(f"训练集: {train_df.shape[0]}条")
        print(f"验证集: {val_df.shape[0]}条")

        # 转换为Hugging Face Dataset格式
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        return train_dataset, val_dataset

    except ValueError as ve:
        print(f"数据集划分错误: {str(ve)}")
        print("尝试不使用分层抽样")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        print(f"训练集: {train_df.shape[0]}条")
        print(f"验证集: {val_df.shape[0]}条")

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        return train_dataset, val_dataset


def tokenize_datasets(datasets, tokenizer, max_length=256):
    """标记化处理数据集"""

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_train = datasets[0].map(tokenize_function, batched=True)
    tokenized_val = datasets[1].map(tokenize_function, batched=True)

    # 设置标签列
    tokenized_train = tokenized_train.rename_column("primary_encoded", "labels")
    tokenized_val = tokenized_val.rename_column("primary_encoded", "labels")

    return tokenized_train, tokenized_val


def train_bert_classifier(train_dataset, val_dataset, num_labels):
    """训练BERT分类模型"""
    try:
        # 模型初始化
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels
        )

        # 训练参数配置
        training_args = TrainingArguments(
            output_dir="./model_output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            load_best_model_at_end=True
        )

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        # 开始训练
        print("开始模型训练...")
        trainer.train()

        return trainer

    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        raise


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # 计算准确率
    accuracy = (predictions == labels).mean()

    return {"accuracy": accuracy}


def save_model_and_labels(trainer, tokenizer, label_mappings):
    """保存模型和标签信息"""
    try:
        # 创建保存目录
        os.makedirs("./saved_model", exist_ok=True)

        # 保存模型和tokenizer
        trainer.save_model("./saved_model")
        tokenizer.save_pretrained("./saved_model")

        # 保存标签映射
        pd.DataFrame({
            'id': list(label_mappings['primary'].keys()),
            'primary_label': list(label_mappings['primary'].values())
        }).to_csv("./saved_model/primary_labels.csv", index=False)

        pd.DataFrame({
            'id': list(label_mappings['secondary'].keys()),
            'secondary_label': list(label_mappings['secondary'].values())
        }).to_csv("./saved_model/secondary_labels.csv", index=False)

        print("模型和标签信息已保存至./saved_model目录")

    except Exception as e:
        print(f"保存模型时出错: {str(e)}")


# 主执行流程
if __name__ == "__main__":
    # 设置CPU环境
    torch.set_num_threads(4)  # 限制CPU线程数

    try:
        print("=" * 50)
        print("开始Ask_Jesus文本分类模型训练")
        print("=" * 50)

        # 1. 加载和预处理数据
        print("\n步骤1: 加载和预处理数据...")
        df, label_mappings = load_and_preprocess_data()

        # 2. 准备数据集
        print("\n步骤2: 准备数据集...")
        train_dataset, val_dataset = prepare_datasets(df)

        # 3. 初始化tokenizer
        print("\n步骤3: 初始化tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 4. 标记化处理
        print("\n步骤4: 标记化处理数据集...")
        tokenized_train, tokenized_val = tokenize_datasets(
            (train_dataset, val_dataset),
            tokenizer
        )

        # 5. 训练模型（主标签分类）
        print("\n步骤5: 训练BERT分类模型...")
        trainer = train_bert_classifier(
            tokenized_train,
            tokenized_val,
            num_labels=len(label_mappings['primary'])
        )

        # 6. 保存模型和标签信息
        print("\n步骤6: 保存模型和标签信息...")
        save_model_and_labels(trainer, tokenizer, label_mappings)

        # 7. 最终评估
        print("\n最终验证结果:")
        eval_results = trainer.evaluate()
        print(f"验证准确率: {eval_results['eval_accuracy']:.4f}")
        print(f"验证损失: {eval_results['eval_loss']:.4f}")

        print("\n" + "=" * 50)
        print("训练成功完成!")
        print("=" * 50)

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"训练失败: {str(e)}")
        print("=" * 50)