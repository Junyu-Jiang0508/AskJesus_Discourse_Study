#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格的单标签自动标注系统（使用你原始prompt）
适用于宗教文本分析场景
依赖：pandas openai tqdm
"""

import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ===== 系统配置 =====
API_KEY = "sk-9b631abec1a6411d88579b58bfbf9599"  # 替换为您的API密钥
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 15
MAX_RETRIES = 3
RETRY_INTERVAL = 2.0
MIN_WORD_COUNT = 5

# 路径配置
DATA_FOLDER = Path("output/labeled")
OUTPUT_FOLDER = Path("output/labeled")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 待分析文件列表
JESUS_FILE = DATA_FOLDER / "by_perspective_jesus_labeled.csv"

# 初始化API客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 硬编码标签集
VALID_PRIMARY = {"绝对化断言", "权威暗示", "神圣隐喻", "无法识别"}

# ===== 辅助函数 =====
def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "cp1252", "latin1")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")

# ===== 核心功能 =====
def build_prompt(text: str) -> str:
    # 这是你原来完整的prompt，没改动
    prompt = f"""
你是一位具备语言学与神学素养的人工智能分析专家，任务是对宗教文本中的语言使用方式进行严格的风格判断。请依据下列定义，在三个语言风格中严格单选一个标签（禁止混合、禁止自创、禁止解释性扩展）：

【唯一合法的标签选项】：
- 绝对化断言：使用“必须、永远、从不”等绝对性语言
- 权威暗示：通过引用权威来建立正当性
- 神圣隐喻：使用比喻、象征等表达神圣概念

【规则】：
1. 每段文本只能分配一个标签；
2. 禁止标签组合（如“权威暗示/绝对化断言”）；
3. 不允许创造新标签；
4. 无法判断的文本请直接标记为“无法识别”；
5. 返回必须严格按照以下格式：

文本：\"\"\"{text}\"\"\"
标签: [绝对化断言 / 权威暗示 / 神圣隐喻 / 无法识别]

请开始标注，注意标签必须精准选择，严禁自由解释。
"""
    return prompt

LABEL_RE = re.compile(r"标签[:：]\s*([^\s\n]+)")

def parse_labels(resp: str) -> dict:
    m = LABEL_RE.search(resp)
    label = m.group(1).strip() if m else "格式错误"
    return {"primary_label": label}

def validate_labels(result: dict) -> dict:
    primary = result["primary_label"]
    if any(sep in primary for sep in ("/", "&", "or", "+")):
        result["primary_label"] = "标签混合错误"
    if primary not in VALID_PRIMARY and primary not in ("格式错误", "标注失败"):
        result["primary_label"] = "无效主标签"
    return result

def process_text(sentence: str) -> dict:
    if len(sentence.split()) < MIN_WORD_COUNT:
        return {"sentence": sentence, "primary_label": "忽略:文本过短"}
    prompt = build_prompt(sentence)
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
            labels = validate_labels(labels)
            return {"sentence": sentence, **labels}
        except Exception as e:
            err_type = "API错误" if "openai" in str(e).lower() else "系统错误"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_INTERVAL * (attempt + 1))
            else:
                return {"sentence": sentence, "primary_label": f"标注失败:{err_type}"}

def process_file(input_file: Path, output_prefix: str):
    print(f"\n开始处理文件: {input_file.name}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        df = read_csv_auto(input_file)
    except Exception as e:
        print(f"文件读取失败: {e}")
        return

    text_column = "sentence" if "sentence" in df.columns else "text"
    if text_column not in df.columns:
        print(f"错误: 文件 {input_file.name} 缺少文本列")
        return

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
        futures = {executor.submit(process_text, s): s for s in sentences}

        with tqdm(total=total, desc="标注进度") as pbar:
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if "错误" in result["primary_label"] or "失败" in result["primary_label"]:
                        errors.append(result)
                    else:
                        results.append(result)
                except Exception as e:
                    errors.append({
                        "sentence": futures[fut],
                        "primary_label": f"处理异常: {type(e).__name__}"
                    })
                pbar.update(1)

    if results:
        result_df = pd.DataFrame(results)
        output_file = OUTPUT_FOLDER / f"{output_prefix}_{timestamp}_labeled.csv"
        result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✓ 标注成功: {len(results)} 条，保存至: {output_file}")

    if errors:
        error_df = pd.DataFrame(errors)
        error_file = OUTPUT_FOLDER / f"{output_prefix}_{timestamp}_errors.csv"
        error_df.to_csv(error_file, index=False, encoding="utf-8-sig")
        print(f"⚠ 标注失败: {len(errors)} 条，错误详情: {error_file}")

    elapsed = time.time() - start_time
    avg_time = elapsed / total if total > 0 else 0
    print(f"处理完成: 总计 {total} 条，平均 {avg_time:.2f} 秒/条")
    print(f"成功率: {len(results) / total:.1%}")

def main():
    print("=" * 50)
    print("宗教文本单标签自动标注系统")
    print("=" * 50)

    if JESUS_FILE.exists():
        process_file(
            JESUS_FILE,
            "jesus"
        )
    else:
        print(f"文件未找到: {JESUS_FILE}")

    print("\n所有文件处理完成")

if __name__ == "__main__":
    main()
