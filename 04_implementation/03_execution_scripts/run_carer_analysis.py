#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行CARER情绪分析的脚本

这个脚本将处理现有的Ask_Jesus数据，进行情绪分析，并生成可视化结果。
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carer_emotion_analysis import CAREREmotionAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ask_jesus_data():
    """加载Ask_Jesus平台的数据"""
    logger.info("开始加载Ask_Jesus数据...")
    
    data_files = {
        'jesus_responses': 'data/by_perspective_jesus.csv',
        'audience_comments': 'data/by_perspective_audience.csv',
        'labeled_data': 'output/labeled/by_perspective_jesus_labeled.csv'
    }
    
    loaded_data = {}
    
    for name, file_path in data_files.items():
        try:
            if Path(file_path).exists():
                logger.info(f"加载 {name}: {file_path}")
                data = pd.read_csv(file_path, encoding='utf-8')
                loaded_data[name] = data
                logger.info(f"成功加载 {name}: {len(data)} 条记录")
            else:
                logger.warning(f"文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"加载 {name} 失败: {e}")
    
    return loaded_data


def prepare_texts_for_analysis(data_dict):
    """准备用于情绪分析的文本数据"""
    logger.info("准备文本数据用于情绪分析...")
    
    all_texts = []
    text_sources = []
    
    # 处理耶稣回应数据
    if 'jesus_responses' in data_dict:
        jesus_data = data_dict['jesus_responses']
        
        # 查找包含文本内容的列
        text_columns = [col for col in jesus_data.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'content', 'message', 'response', 'answer'])]
        
        if text_columns:
            logger.info(f"在耶稣回应数据中找到文本列: {text_columns}")
            
            for col in text_columns:
                texts = jesus_data[col].dropna().astype(str).tolist()
                all_texts.extend(texts)
                text_sources.extend([f"jesus_{col}"] * len(texts))
        
        # 如果没有找到明确的文本列，尝试使用第一列
        elif len(jesus_data.columns) > 0:
            first_col = jesus_data.columns[0]
            texts = jesus_data[first_col].dropna().astype(str).tolist()
            all_texts.extend(texts)
            text_sources.extend([f"jesus_{first_col}"] * len(texts))
    
    # 处理观众评论数据
    if 'audience_comments' in data_dict:
        audience_data = data_dict['audience_comments']
        
        # 查找包含文本内容的列
        text_columns = [col for col in audience_data.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'content', 'message', 'comment', 'chat'])]
        
        if text_columns:
            logger.info(f"在观众评论数据中找到文本列: {text_columns}")
            
            for col in text_columns:
                texts = audience_data[col].dropna().astype(str).tolist()
                all_texts.extend(texts)
                text_sources.extend([f"audience_{col}"] * len(texts))
    
    # 处理标注数据
    if 'labeled_data' in data_dict:
        labeled_data = data_dict['labeled_data']
        
        # 查找包含文本内容的列
        text_columns = [col for col in labeled_data.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'content', 'message', 'sentence'])]
        
        if text_columns:
            logger.info(f"在标注数据中找到文本列: {text_columns}")
            
            for col in text_columns:
                texts = labeled_data[col].dropna().astype(str).tolist()
                all_texts.extend(texts)
                text_sources.extend([f"labeled_{col}"] * len(texts))
    
    # 过滤和清理文本
    cleaned_texts = []
    cleaned_sources = []
    
    for text, source in zip(all_texts, text_sources):
        # 过滤太短的文本
        if len(text.strip()) > 10:
            cleaned_texts.append(text.strip())
            cleaned_sources.append(source)
    
    logger.info(f"准备完成: {len(cleaned_texts)} 条有效文本")
    
    return cleaned_texts, cleaned_sources


def create_sample_data_for_testing():
    """创建测试用的样本数据"""
    logger.info("创建测试用的样本数据...")
    
    sample_texts = [
        "Jesus loves you unconditionally and will always be there for you.",
        "I'm feeling sad and lost, can you help me find hope?",
        "The Bible teaches us to be kind and compassionate to everyone.",
        "I'm angry about the injustice in the world today.",
        "God's grace is sufficient for all our needs.",
        "I feel blessed to have such wonderful friends in my life.",
        "Sometimes I'm afraid of what the future holds.",
        "I'm surprised by how much joy this community brings me.",
        "The teachings of Jesus give me comfort in difficult times.",
        "I'm disgusted by the hatred I see in the world.",
        "Let us pray together for peace and understanding.",
        "The Lord is my shepherd, I shall not want.",
        "I'm grateful for all the blessings in my life.",
        "Sometimes I feel overwhelmed by life's challenges.",
        "The love of Christ fills my heart with joy.",
        "I'm concerned about the state of our world today.",
        "God's mercy is new every morning.",
        "I find strength in reading the scriptures.",
        "The fellowship of believers gives me hope.",
        "I'm thankful for the guidance of the Holy Spirit."
    ]
    
    return sample_texts


def run_emotion_analysis(texts, sources=None):
    """运行情绪分析"""
    logger.info("开始运行CARER情绪分析...")
    
    try:
        # 初始化情绪分析器
        analyzer = CAREREmotionAnalyzer()
        
        # 分析情绪
        logger.info("分析文本情绪...")
        emotion_results = analyzer.analyze_emotions(texts)
        
        # 添加来源信息
        if sources and len(sources) == len(emotion_results):
            emotion_results['text_source'] = sources
        
        # 分析情绪分布
        logger.info("分析情绪分布...")
        distribution_results = analyzer.analyze_emotion_distribution()
        
        # 评估权威建构效果
        logger.info("评估权威建构效果...")
        authority_results = analyzer.evaluate_authority_construction(texts)
        
        # 创建可视化
        logger.info("创建可视化图表...")
        analyzer.create_visualizations("carer_analysis_results")
        
        # 生成报告
        logger.info("生成分析报告...")
        analyzer.generate_report("carer_analysis_report.txt")
        
        logger.info("CARER情绪分析完成！")
        
        return analyzer, emotion_results, distribution_results, authority_results
        
    except Exception as e:
        logger.error(f"情绪分析失败: {e}")
        raise


def analyze_frame_emotion_correlation(analyzer, frame_file_path):
    """分析框架与情绪的关联"""
    logger.info("分析框架与情绪的关联...")
    
    try:
        # 加载框架数据
        analyzer.load_frame_data(frame_file_path)
        
        # 合并情绪和框架数据
        merged_data = analyzer.merge_emotion_frame_data()
        
        # 分析关联
        correlation_results = analyzer.analyze_frame_emotion_correlation(merged_data)
        
        logger.info("框架-情绪关联分析完成")
        return correlation_results
        
    except Exception as e:
        logger.error(f"框架-情绪关联分析失败: {e}")
        return None


def print_analysis_summary(emotion_results, distribution_results, authority_results):
    """打印分析摘要"""
    logger.info("=" * 60)
    logger.info("CARER情绪分析摘要")
    logger.info("=" * 60)
    
    if emotion_results is not None:
        logger.info(f"总文本数量: {len(emotion_results)}")
        
        # 情绪分布
        if 'primary_emotion' in emotion_results.columns:
            emotion_counts = emotion_results['primary_emotion'].value_counts()
            logger.info("\n情绪分布:")
            for emotion, count in emotion_counts.items():
                percentage = (count / len(emotion_results)) * 100
                logger.info(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # 置信度统计
        if 'emotion_confidence' in emotion_results.columns:
            avg_confidence = emotion_results['emotion_confidence'].mean()
            logger.info(f"\n平均情绪置信度: {avg_confidence:.3f}")
    
    # 权威建构指标
    if authority_results and 'authority_indicators' in authority_results:
        logger.info("\n权威建构指标:")
        for indicator_type, data in authority_results['authority_indicators'].items():
            logger.info(f"  {indicator_type}:")
            logger.info(f"    总出现次数: {data['total_occurrences']}")
            logger.info(f"    平均每文本: {data['average_per_text']:.2f}")


def main():
    """主函数"""
    logger.info("开始Ask_Jesus CARER情绪分析...")
    
    try:
        # 1. 加载数据
        data_dict = load_ask_jesus_data()
        
        if not data_dict:
            logger.warning("没有找到有效的数据文件，使用样本数据进行测试...")
            texts = create_sample_data_for_testing()
            sources = ["sample"] * len(texts)
        else:
            # 2. 准备文本数据
            texts, sources = prepare_texts_for_analysis(data_dict)
            
            if not texts:
                logger.warning("没有找到有效的文本数据，使用样本数据进行测试...")
                texts = create_sample_data_for_testing()
                sources = ["sample"] * len(texts)
        
        # 3. 运行情绪分析
        analyzer, emotion_results, distribution_results, authority_results = run_emotion_analysis(texts, sources)
        
        # 4. 打印分析摘要
        print_analysis_summary(emotion_results, distribution_results, authority_results)
        
        # 5. 尝试分析框架-情绪关联（如果有框架数据）
        frame_files = [
            'output/labeled/by_perspective_jesus_labeled.csv',
            'data/labeled_set/primary_label.csv',
            'data/labeled_set/secondary_label.csv'
        ]
        
        for frame_file in frame_files:
            if Path(frame_file).exists():
                logger.info(f"尝试分析框架-情绪关联: {frame_file}")
                correlation_results = analyze_frame_emotion_correlation(analyzer, frame_file)
                if correlation_results:
                    logger.info("框架-情绪关联分析成功")
                    break
        
        logger.info("=" * 60)
        logger.info("分析完成！结果保存在以下位置:")
        logger.info("- 可视化图表: carer_analysis_results/")
        logger.info("- 分析报告: carer_analysis_report.txt")
        logger.info("- 情绪数据: 在内存中 (emotion_results)")
        logger.info("=" * 60)
        
        return analyzer, emotion_results, distribution_results, authority_results
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    try:
        analyzer, emotion_results, distribution_results, authority_results = main()
        
        # 保存情绪结果到CSV文件
        if emotion_results is not None:
            output_file = "carer_emotion_results.csv"
            emotion_results.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"情绪分析结果已保存到: {output_file}")
            
            # 显示前几行结果
            print("\n情绪分析结果预览:")
            print(emotion_results[['primary_emotion', 'emotion_confidence']].head(10))
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)
