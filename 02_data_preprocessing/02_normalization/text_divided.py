import pandas as pd
import re
import os
from pathlib import Path


def process_csv_file(input_file, output_folder):
    """
    处理CSV文件：按英文格式切割句子并保留文件名

    参数:
        input_file (str): 输入CSV文件路径
        output_folder (str): 输出文件夹路径
    """
    # 读取CSV文件
    try:
        # 尝试自动检测文本列名
        df = pd.read_csv(input_file)
        text_columns = ["text", "sentence", "content"]
        text_col = None

        # 自动检测文本列名
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            # 如果找不到标准列名，使用第一列文本类型的列
            text_col = next((col for col in df.columns if pd.api.types.is_string_dtype(df[col])), None)
            if text_col is None:
                raise ValueError(f"在文件 {input_file} 中找不到文本列")

    except Exception as e:
        print(f"读取文件 {input_file} 失败: {e}")
        return None

    # 提取文件名（不含扩展名）
    file_name = Path(input_file).stem

    # 清理文本：确保为字符串类型并填充空值
    df['temp_text'] = df[text_col].fillna('').astype(str)

    # 句子分割函数（按英文标点拆分）
    def split_sentences(text):
        # 使用英文标点分割句子，保留标点
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # 处理特殊情况（如缩略词）
        merged = []
        for s in sentences:
            # 合并可能的错误分割（如"I think so.So do I"）
            if s.endswith('.') and len(s) > 2 and s[-3] not in ' .' and not s[-2].isupper():
                # 处理连续句子但被错误分割的情况
                if merged and merged[-1].endswith('.'):
                    merged[-1] += " " + s
                else:
                    merged.append(s)
            else:
                merged.append(s)

        return [s.strip() for s in merged if s.strip()]

    # 处理后的数据存储
    processed_data = []
    for _, row in df.iterrows():
        text = row['temp_text']
        sentences = split_sentences(text)
        for sentence in sentences:
            # 根据第二张图片的格式，只保留文件名部分
            processed_data.append([file_name, sentence])

    # 转换为DataFrame
    processed_df = pd.DataFrame(processed_data, columns=['filename', 'sentence'])

    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存为CSV文件，使用文件名作为分割标识
    output_file = output_path / f"{file_name}_processed.csv"
    processed_df.to_csv(output_file, index=False)

    return {
        "input_file": input_file,
        "output_file": str(output_file),
        "total_sentences": len(processed_df),
        "sample_text": processed_df['sentence'].head(5).tolist() if len(processed_df) > 0 else []
    }


# 主处理函数
def main():
    # 根据第一张图片设置输入文件路径
    data_folder = Path("data")
    input_files = [
        data_folder / "by_perspective_audience.csv",
        data_folder / "by_perspective_jesus.csv"
    ]

    # 根据第二张图片设置输出文件夹路径
    output_folder = "output/processed_text"

    # 处理每个文件
    for input_file in input_files:
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在 - {input_file}")
            continue

        result = process_csv_file(input_file, output_folder)
        if result is None:
            continue

        print(f"处理完成: {result['input_file']}")
        print(f"输出文件: {result['output_file']}")
        print(f"总句子数: {result['total_sentences']}")

        # 打印前5个句子作为样本
        if result['sample_text']:
            print("前5个句子示例:")
            for i, text in enumerate(result['sample_text'], 1):
                print(f" {i}. {text[:60]}{'...' if len(text) > 60 else ''}")

        print("-" * 80)


if __name__ == "__main__":
    main()