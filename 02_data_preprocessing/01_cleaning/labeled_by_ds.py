#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格的双重标签自动标注系统
适用于宗教文本分析场景
依赖：pandas openai tqdm
"""

import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import datetime

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ===== 系统配置 =====
API_KEY = "sk-9b631abec1a6411d88579b58bfbf9599"  # 替换为您的API密钥
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 15  # 并发工作线程数
MAX_RETRIES = 3  # API调用重试次数
RETRY_INTERVAL = 2.0  # 重试间隔(秒)
MIN_WORD_COUNT = 5  # 最小分析字数

# 路径配置
DATA_FOLDER = Path("data")
OUTPUT_FOLDER = Path("output/labeled")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 标签定义文件
PRIMARY_LABEL_CSV = DATA_FOLDER / "labeled_set" / "primary_label.csv"
SECONDARY_LABEL_CSV = DATA_FOLDER / "labeled_set" / "secondary_label.csv"
LABELED_SET_CSV = DATA_FOLDER / "labeled_set" / "set.csv"

# 待分析文件列表
AUDIENCE_FILE = DATA_FOLDER / "by_perspective_audience.csv"
JESUS_FILE = DATA_FOLDER / "by_perspective_jesus.csv"

# 初始化API客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ===== 辅助函数 =====
def read_csv_auto(path: Path) -> pd.DataFrame:
    """自动检测文件编码并读取CSV"""
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "cp1252", "latin1")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # 最终尝试，带错误替换
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def get_valid_labels(df: pd.DataFrame, column: str) -> set:
    """获取有效标签集合"""
    return set(df[column].unique())


# ===== 核心功能 =====
def build_prompt(text: str,
                 label1_desc: pd.DataFrame,
                 label2_desc: pd.DataFrame,
                 few_shot: str) -> str:
    """
    构建严格约束的提示词
    确保模型遵循双层标签分类规则
    """
    # 提取标签描述
    label1_guide = "\n".join(
        f"- {row['primary_label']}：{row['description']}"
        for _, row in label1_desc.iterrows()
    )
    label2_guide = "\n".join(
        f"- {row['secondary_label']}：{row['description']}"
        for _, row in label2_desc.iterrows()
    )

    return f"""
你是一位专业的宗教文本分析专家。请严格遵循以下规则对文本进行双重标签分类：

【分层标签规则】
1. 主标签必须从【主框架维度】列表中选择且仅选一项
2. 副标签必须从【风格与语用维度】列表中选择且仅选一项
3. 禁止创建新标签或标签混合（如"标签A/标签B"形式）
4. 当文本明显不属于任何类别时，才使用"无法识别"

【主框架维度】- 严格单选：
{label1_guide}

【风格与语用维度】- 严格单选：
{label2_guide}

【参考示例】- 注意标签必须严格匹配：
{few_shot}

【待分析文本】：
\"\"\"{text}\"\"\"

请仅返回以下格式（严格匹配）：
标签1: [主框架维度标签]
标签2: [风格与语用维度标签]
"""


# 标签解析正则表达式
LABEL1_RE = re.compile(r"标签1[:：]\s*([^\s\n]+)")
LABEL2_RE = re.compile(r"标签2[:：]\s*([^\s\n]+)")


def parse_labels(resp: str) -> dict:
    """解析模型响应中的标签"""
    l1 = LABEL1_RE.search(resp)
    l2 = LABEL2_RE.search(resp)
    return {
        "primary_label": l1.group(1).strip() if l1 else "格式错误",
        "secondary_label": l2.group(1).strip() if l2 else "格式错误",
    }


def validate_labels(result: dict,
                    valid_primary: set,
                    valid_secondary: set) -> dict:
    """
    严格验证标签有效性
    确保标签格式正确且在预定义集合中
    """
    primary = result["primary_label"]
    secondary = result["secondary_label"]

    # 检测标签混合
    if any(sep in primary for sep in ("/", "&", "or", "+")):
        result["primary_label"] = "标签混合错误"
    if any(sep in secondary for sep in ("/", "&", "or", "+")):
        result["secondary_label"] = "标签混合错误"

    # 主标签有效性验证
    if primary not in valid_primary:
        if primary not in ("无法识别", "格式错误", "标注失败"):
            result["primary_label"] = "无效主标签"

    # 副标签有效性验证
    if secondary not in valid_secondary:
        if secondary not in ("无法识别", "格式错误", "标注失败"):
            result["secondary_label"] = "无效副标签"

    return result


def process_text(sentence: str,
                 label1_desc: pd.DataFrame,
                 label2_desc: pd.DataFrame,
                 few_shot: str,
                 valid_primary: set,
                 valid_secondary: set) -> dict:
    """
    处理单条文本的标签分类
    包含严格验证和错误处理
    """
    # 跳过过短的文本
    if len(sentence.split()) < MIN_WORD_COUNT:
        return {
            "sentence": sentence,
            "primary_label": "忽略:文本过短",
            "secondary_label": "忽略:文本过短"
        }

    prompt = build_prompt(sentence, label1_desc, label2_desc, few_shot)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                timeout=60,
            )
            content = resp.choices[0].message.content.strip()
            labels = parse_labels(content)
            labels = validate_labels(labels, valid_primary, valid_secondary)
            return {"sentence": sentence, **labels}

        except Exception as e:
            err_type = "API错误" if "openai" in str(e).lower() else "系统错误"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_INTERVAL * (attempt + 1))
            else:
                return {
                    "sentence": sentence,
                    "primary_label": f"标注失败:{err_type}",
                    "secondary_label": f"标注失败:{err_type}"
                }


def process_file(input_file: Path,
                 output_prefix: str,
                 label1_desc: pd.DataFrame,
                 label2_desc: pd.DataFrame,
                 few_shot: str):
    """
    处理整个CSV文件的标注任务
    使用多线程提高效率
    """
    print(f"\n开始处理文件: {input_file.name}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        df = read_csv_auto(input_file)
    except Exception as e:
        print(f"文件读取失败: {e}")
        return

    # 确定文本列
    text_column = "sentence" if "sentence" in df.columns else "text"
    if text_column not in df.columns:
        print(f"错误: 文件 {input_file.name} 缺少文本列")
        return

    # 准备标签验证集合
    valid_primary = get_valid_labels(label1_desc, "primary_label")
    valid_secondary = get_valid_labels(label2_desc, "secondary_label")

    # 提取文本并过滤
    sentences = df[text_column].dropna().astype(str).tolist()
    total = len(sentences)

    if total == 0:
        print("无可标注文本")
        return

    print(f"开始标注 {total} 条文本（最小字数: {MIN_WORD_COUNT}）")
    start_time = time.time()
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_text,
                s,
                label1_desc,
                label2_desc,
                few_shot,
                valid_primary,
                valid_secondary
            ): s for s in sentences
        }

        with tqdm(total=total, desc="标注进度") as pbar:
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    # 分离错误和成功结果
                    if "错误" in result["primary_label"] or "失败" in result["primary_label"]:
                        errors.append(result)
                    else:
                        results.append(result)
                except Exception as e:
                    errors.append({
                        "sentence": futures[fut],
                        "primary_label": f"处理异常: {type(e).__name__}",
                        "secondary_label": f"处理异常: {type(e).__name__}"
                    })
                pbar.update(1)

    # 保存结果
    if results:
        result_df = pd.DataFrame(results)
        output_file = OUTPUT_FOLDER / f"{output_prefix}_{timestamp}_labeled.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✓ 标注成功: {len(results)} 条，保存至: {output_file}")

    # 保存错误信息
    if errors:
        error_df = pd.DataFrame(errors)
        error_file = OUTPUT_FOLDER / f"{output_prefix}_{timestamp}_errors.csv"
        error_df.to_csv(error_file, index=False, encoding="utf-8-sig")
        print(f"⚠ 标注失败: {len(errors)} 条，错误详情: {error_file}")

    elapsed = time.time() - start_time
    avg_time = elapsed / total if total > 0 else 0
    print(f"处理完成: 总计 {total} 条，平均 {avg_time:.2f} 秒/条")
    print(f"成功率: {len(results) / total:.1%}")


# ===== 主函数 =====
def main():
    """主处理函数"""
    print("=" * 50)
    print("宗教文本双重标签自动标注系统")
    print("=" * 50)

    # 读取标签定义
    try:
        label1_desc = read_csv_auto(PRIMARY_LABEL_CSV)
        label2_desc = read_csv_auto(SECONDARY_LABEL_CSV)
        labeled_df = read_csv_auto(LABELED_SET_CSV)

        # 筛选有效的主副标签
        valid_primary = get_valid_labels(label1_desc, "primary_label")
        valid_secondary = get_valid_labels(label2_desc, "secondary_label")

        # 筛选高质量示例（排除无效标签）
        filtered_df = labeled_df[
            (labeled_df["primary_label"].isin(valid_primary)) &
            (labeled_df["secondary_label"].isin(valid_secondary))
            ]

        # 获取示例（至少3个，最多5个）
        sample_size = min(5, max(3, len(filtered_df)))
        good_examples = filtered_df.sample(sample_size, random_state=42)

        # 构建强约束示例
        few_shot_examples = "\n".join(
            f'文本: "{row.sentence}"\n标签1: {row.primary_label}\n标签2: {row.secondary_label}'
            for _, row in good_examples.iterrows()
        )

        print(f"系统初始化完成，使用 {sample_size} 个高质量示例")
    except Exception as e:
        print(f"标签定义读取失败: {e}")
        return

    # 处理目标文件
    if AUDIENCE_FILE.exists():
        process_file(
            AUDIENCE_FILE,
            "audience",
            label1_desc,
            label2_desc,
            few_shot_examples
        )
    else:
        print(f"文件未找到: {AUDIENCE_FILE}")

    if JESUS_FILE.exists():
        process_file(
            JESUS_FILE,
            "jesus",
            label1_desc,
            label2_desc,
            few_shot_examples
        )
    else:
        print(f"文件未找到: {JESUS_FILE}")

    print("\n所有文件处理完成")


if __name__ == "__main__":
    main()
