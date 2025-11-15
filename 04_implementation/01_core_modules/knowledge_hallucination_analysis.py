#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识幻觉分析模块

这个模块专门用于分析AI生成的宗教文本中的知识幻觉现象，
包括圣经原文比对、新术语检测和绝对化断言识别。
"""

import logging
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from difflib import SequenceMatcher
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 知识幻觉检测模式
HALLUCINATION_PATTERNS = {
    'absolute_assertions': [
        r'\b(always|never|everyone|nobody|everything|nothing)\b',
        r'\b(must|should|have to|need to)\b',
        r'\b(perfect|complete|absolute|total|ultimate)\b',
        r'\b(proven|certain|definite|guaranteed)\b',
        r'\b(impossible|inevitable|unavoidable)\b',
        r'\b(only|exclusively|solely|uniquely)\b'
    ],
    'unverifiable_claims': [
        r'\b(studies show|research proves|scientists say)\b',
        r'\b(it is known|it is established|it is proven)\b',
        r'\b(experts agree|scholars confirm|authorities state)\b',
        r'\b(historically|traditionally|classically)\b'
    ],
    'religious_authority_abuse': [
        r'\b(God told me|Jesus revealed|the Lord showed me)\b',
        r'\b(divine inspiration|holy revelation|sacred vision)\b',
        r'\b(prophetic word|apostolic authority|biblical truth)\b'
    ]
}

# 圣经关键概念和术语
BIBLICAL_CONCEPTS = {
    'salvation_terms': [
        'salvation', 'redemption', 'atonement', 'justification', 'sanctification',
        'grace', 'mercy', 'forgiveness', 'repentance', 'faith'
    ],
    'divine_attributes': [
        'omnipotent', 'omniscient', 'omnipresent', 'eternal', 'immutable',
        'holy', 'righteous', 'loving', 'just', 'merciful'
    ],
    'christological_terms': [
        'incarnation', 'resurrection', 'ascension', 'second coming', 'messiah',
        'son of god', 'lamb of god', 'good shepherd', 'way truth life'
    ],
    'ecclesiological_terms': [
        'church', 'body of christ', 'bride of christ', 'kingdom of god',
        'fellowship', 'communion', 'baptism', 'eucharist', 'worship'
    ]
}


class KnowledgeHallucinationAnalyzer:
    """
    知识幻觉分析器
    
    专门用于检测和分析AI生成的宗教文本中的知识幻觉现象
    """
    
    def __init__(self, bible_reference_file: Optional[str] = None):
        """
        初始化知识幻觉分析器
        
        Args:
            bible_reference_file: 圣经参考文件路径（可选）
        """
        self.bible_reference_file = bible_reference_file
        self.bible_references = {}
        self.hallucination_results = {}
        
        # 加载圣经参考数据
        if bible_reference_file:
            self._load_bible_references()
        
        logger.info("知识幻觉分析器初始化完成")
    
    def _load_bible_references(self):
        """加载圣经参考数据"""
        try:
            if Path(self.bible_reference_file).exists():
                # 根据文件类型加载数据
                if self.bible_reference_file.endswith('.json'):
                    with open(self.bible_reference_file, 'r', encoding='utf-8') as f:
                        self.bible_references = json.load(f)
                elif self.bible_reference_file.endswith('.csv'):
                    self.bible_references = pd.read_csv(self.bible_reference_file, encoding='utf-8')
                else:
                    logger.warning(f"不支持的圣经参考文件格式: {self.bible_reference_file}")
                
                logger.info(f"成功加载圣经参考数据: {len(self.bible_references)} 条记录")
            else:
                logger.warning(f"圣经参考文件不存在: {self.bible_reference_file}")
        except Exception as e:
            logger.error(f"加载圣经参考数据失败: {e}")
    
    def detect_absolute_assertions(self, texts: List[str]) -> pd.DataFrame:
        """
        检测文本中的绝对化断言
        
        Args:
            texts: 要分析的文本列表
            
        Returns:
            包含检测结果的DataFrame
        """
        logger.info(f"开始检测 {len(texts)} 个文本中的绝对化断言")
        
        results = []
        
        for i, text in enumerate(texts):
            text_result = {
                'text_id': i,
                'text': text,
                'total_absolute_assertions': 0,
                'absolute_assertion_types': [],
                'absolute_assertion_examples': [],
                'hallucination_risk_score': 0.0
            }
            
            # 检测各种类型的绝对化断言
            for assertion_type, patterns in HALLUCINATION_PATTERNS.items():
                type_count = 0
                examples = []
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    type_count += len(matches)
                    examples.extend(matches)
                
                if type_count > 0:
                    text_result['absolute_assertion_types'].append(assertion_type)
                    text_result['absolute_assertion_examples'].extend(examples)
                    text_result['total_absolute_assertions'] += type_count
            
            # 计算幻觉风险分数
            text_result['hallucination_risk_score'] = self._calculate_hallucination_risk(
                text_result['total_absolute_assertions'], 
                len(text)
            )
            
            results.append(text_result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(texts)} 个文本")
        
        # 创建DataFrame
        results_df = pd.DataFrame(results)
        
        # 按幻觉风险分数排序
        results_df = results_df.sort_values('hallucination_risk_score', ascending=False)
        
        logger.info(f"绝对化断言检测完成，发现 {results_df['total_absolute_assertions'].sum()} 个断言")
        
        return results_df
    
    def _calculate_hallucination_risk(self, assertion_count: int, text_length: int) -> float:
        """
        计算幻觉风险分数
        
        Args:
            assertion_count: 绝对化断言数量
            text_length: 文本长度
            
        Returns:
            风险分数 (0-1)
        """
        # 基于断言密度和文本长度的风险计算
        density = assertion_count / max(text_length, 1)
        
        # 使用sigmoid函数计算风险分数
        risk_score = 1 / (1 + np.exp(-10 * (density - 0.01)))
        
        return min(risk_score, 1.0)
    
    def detect_new_terminology(self, texts: List[str]) -> Dict[str, Any]:
        """
        检测新术语的出现
        
        Args:
            texts: 要分析的文本列表
            
        Returns:
            新术语检测结果
        """
        logger.info("开始检测新术语")
        
        # 收集所有文本中的词汇
        all_words = set()
        word_frequency = {}
        
        for text in texts:
            # 提取词汇（简单的分词）
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            for word in words:
                if len(word) > 3:  # 过滤短词
                    all_words.add(word)
                    word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # 识别可能的新术语
        new_terminology = []
        established_terms = set()
        
        # 从圣经概念中提取已建立的术语
        for category, terms in BIBLICAL_CONCEPTS.items():
            established_terms.update(terms)
        
        # 检测新术语
        for word, freq in word_frequency.items():
            if word not in established_terms and freq >= 2:  # 出现至少2次
                # 计算与已建立术语的相似度
                max_similarity = max([
                    SequenceMatcher(None, word, term).ratio() 
                    for term in established_terms
                ]) if established_terms else 0
                
                if max_similarity < 0.8:  # 相似度阈值
                    new_terminology.append({
                        'term': word,
                        'frequency': freq,
                        'max_similarity': max_similarity,
                        'novelty_score': 1 - max_similarity
                    })
        
        # 按新颖性分数排序
        new_terminology.sort(key=lambda x: x['novelty_score'], reverse=True)
        
        results = {
            'total_words': len(all_words),
            'established_terms': len(established_terms),
            'new_terminology_count': len(new_terminology),
            'new_terminology': new_terminology[:50],  # 返回前50个
            'word_frequency': dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:100])
        }
        
        logger.info(f"新术语检测完成，发现 {len(new_terminology)} 个可能的新术语")
        
        return results
    
    def analyze_emotional_manipulation(self, texts: List[str], emotion_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        分析情绪操纵策略
        
        Args:
            texts: 要分析的文本列表
            emotion_data: 情绪分析数据（可选）
            
        Returns:
            情绪操纵分析结果
        """
        logger.info("开始分析情绪操纵策略")
        
        results = {
            'emotional_manipulation_indicators': {},
            'emotion_authority_correlation': {},
            'manipulation_effectiveness': {}
        }
        
        # 检测情绪操纵指标
        for manipulation_type, patterns in HALLUCINATION_PATTERNS.items():
            if 'emotional' in manipulation_type or 'authority' in manipulation_type:
                indicator_counts = []
                
                for text in texts:
                    count = 0
                    for pattern in patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        count += len(matches)
                    indicator_counts.append(count)
                
                results['emotional_manipulation_indicators'][manipulation_type] = {
                    'total_occurrences': sum(indicator_counts),
                    'average_per_text': np.mean(indicator_counts),
                    'distribution': indicator_counts
                }
        
        # 如果有情绪数据，分析情绪与权威指标的相关性
        if emotion_data is not None and len(emotion_data) == len(texts):
            # 计算每个文本的权威指标分数
            authority_scores = []
            for text in texts:
                score = 0
                for patterns in HALLUCINATION_PATTERNS.values():
                    for pattern in patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        score += len(matches)
                authority_scores.append(score)
            
            # 分析权威分数与情绪的关系
            if 'primary_emotion' in emotion_data.columns:
                emotion_authority_corr = {}
                for emotion in emotion_data['primary_emotion'].unique():
                    emotion_mask = emotion_data['primary_emotion'] == emotion
                    emotion_authority_scores = [authority_scores[i] for i in range(len(texts)) if emotion_mask.iloc[i]]
                    
                    if emotion_authority_scores:
                        emotion_authority_corr[emotion] = {
                            'count': len(emotion_authority_scores),
                            'avg_authority_score': np.mean(emotion_authority_scores),
                            'max_authority_score': max(emotion_authority_scores)
                        }
                
                results['emotion_authority_correlation'] = emotion_authority_corr
            
            # 分析操纵效果
            if 'emotion_confidence' in emotion_data.columns:
                # 计算权威分数与情绪置信度的相关性
                correlation = np.corrcoef(authority_scores, emotion_data['emotion_confidence'])[0, 1]
                results['manipulation_effectiveness']['authority_emotion_correlation'] = correlation
                
                # 分析高权威文本的情绪特征
                high_authority_indices = [i for i, score in enumerate(authority_scores) if score > np.mean(authority_scores) + np.std(authority_scores)]
                if high_authority_indices:
                    high_authority_emotions = emotion_data.iloc[high_authority_indices]['primary_emotion'].value_counts()
                    results['manipulation_effectiveness']['high_authority_emotions'] = high_authority_emotions.to_dict()
        
        logger.info("情绪操纵策略分析完成")
        
        return results
    
    def generate_hallucination_report(self, 
                                    absolute_assertions_df: pd.DataFrame,
                                    new_terminology_results: Dict[str, Any],
                                    emotional_manipulation_results: Dict[str, Any],
                                    output_file: str = "knowledge_hallucination_report.txt") -> str:
        """
        生成知识幻觉分析报告
        
        Args:
            absolute_assertions_df: 绝对化断言检测结果
            new_terminology_results: 新术语检测结果
            emotional_manipulation_results: 情绪操纵分析结果
            output_file: 输出文件路径
            
        Returns:
            报告文件路径
        """
        logger.info(f"生成知识幻觉分析报告: {output_file}")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("知识幻觉分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 执行摘要
        report_lines.append("1. 执行摘要")
        report_lines.append("-" * 40)
        
        total_texts = len(absolute_assertions_df)
        total_assertions = absolute_assertions_df['total_absolute_assertions'].sum()
        avg_risk_score = absolute_assertions_df['hallucination_risk_score'].mean()
        
        report_lines.append(f"分析文本总数: {total_texts}")
        report_lines.append(f"检测到的绝对化断言总数: {total_assertions}")
        report_lines.append(f"平均幻觉风险分数: {avg_risk_score:.3f}")
        report_lines.append(f"发现的新术语数量: {new_terminology_results['new_terminology_count']}")
        report_lines.append("")
        
        # 2. 绝对化断言分析
        report_lines.append("2. 绝对化断言分析")
        report_lines.append("-" * 40)
        
        # 高风险文本
        high_risk_texts = absolute_assertions_df[absolute_assertions_df['hallucination_risk_score'] > 0.7]
        report_lines.append(f"高风险文本数量 (风险分数 > 0.7): {len(high_risk_texts)}")
        
        if len(high_risk_texts) > 0:
            report_lines.append("高风险文本示例:")
            for _, row in high_risk_texts.head(5).iterrows():
                report_lines.append(f"  文本ID {row['text_id']}: 风险分数 {row['hallucination_risk_score']:.3f}")
                report_lines.append(f"    断言数量: {row['total_absolute_assertions']}")
                if row['absolute_assertion_examples']:
                    examples = row['absolute_assertion_examples'][:3]  # 显示前3个例子
                    report_lines.append(f"    断言例子: {', '.join(examples)}")
                report_lines.append("")
        
        # 3. 新术语分析
        report_lines.append("3. 新术语分析")
        report_lines.append("-" * 40)
        
        if new_terminology_results['new_terminology']:
            report_lines.append("主要新术语 (按新颖性排序):")
            for term_info in new_terminology_results['new_terminology'][:10]:
                report_lines.append(f"  {term_info['term']}: 频率={term_info['frequency']}, 新颖性={term_info['novelty_score']:.3f}")
        else:
            report_lines.append("未发现明显的新术语")
        
        report_lines.append("")
        
        # 4. 情绪操纵分析
        report_lines.append("4. 情绪操纵分析")
        report_lines.append("-" * 40)
        
        if emotional_manipulation_results['emotional_manipulation_indicators']:
            report_lines.append("情绪操纵指标统计:")
            for indicator_type, data in emotional_manipulation_results['emotional_manipulation_indicators'].items():
                report_lines.append(f"  {indicator_type}:")
                report_lines.append(f"    总出现次数: {data['total_occurrences']}")
                report_lines.append(f"    平均每文本: {data['average_per_text']:.2f}")
        
        if emotional_manipulation_results['emotion_authority_correlation']:
            report_lines.append("\n情绪与权威指标关联:")
            for emotion, corr_data in emotional_manipulation_results['emotion_authority_correlation'].items():
                report_lines.append(f"  {emotion}: 平均权威分数={corr_data['avg_authority_score']:.2f}")
        
        report_lines.append("")
        
        # 5. 风险评估与建议
        report_lines.append("5. 风险评估与建议")
        report_lines.append("-" * 40)
        
        # 计算整体风险等级
        if avg_risk_score > 0.7:
            risk_level = "高"
            risk_description = "存在显著的知识幻觉风险"
        elif avg_risk_score > 0.4:
            risk_level = "中"
            risk_description = "存在中等程度的知识幻觉风险"
        else:
            risk_level = "低"
            risk_description = "知识幻觉风险较低"
        
        report_lines.append(f"整体风险等级: {risk_level}")
        report_lines.append(f"风险描述: {risk_description}")
        report_lines.append("")
        
        report_lines.append("建议措施:")
        if risk_level in ["中", "高"]:
            report_lines.append("  - 加强对AI生成内容的审核和验证")
            report_lines.append("  - 建立更严格的宗教文本生成标准")
            report_lines.append("  - 增加人工审核环节")
            report_lines.append("  - 改进模型训练以减少幻觉生成")
        else:
            report_lines.append("  - 继续保持当前的内容质量标准")
            report_lines.append("  - 定期监控和评估")
        
        # 写入报告文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"知识幻觉分析报告已保存到: {output_file}")
        
        return output_file
    
    def create_hallucination_visualizations(self, 
                                          absolute_assertions_df: pd.DataFrame,
                                          new_terminology_results: Dict[str, Any],
                                          output_dir: str = "hallucination_visualizations"):
        """
        创建知识幻觉分析的可视化图表
        
        Args:
            absolute_assertions_df: 绝对化断言检测结果
            new_terminology_results: 新术语检测结果
            output_dir: 输出目录
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 创建输出目录
            Path(output_dir).mkdir(exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 幻觉风险分数分布
            plt.figure(figsize=(10, 6))
            plt.hist(absolute_assertions_df['hallucination_risk_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('幻觉风险分数')
            plt.ylabel('文本数量')
            plt.title('知识幻觉风险分数分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/hallucination_risk_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 绝对化断言类型分布
            if 'absolute_assertion_types' in absolute_assertions_df.columns:
                # 统计各种类型的断言
                type_counts = {}
                for types in absolute_assertions_df['absolute_assertion_types']:
                    for type_name in types:
                        type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                if type_counts:
                    plt.figure(figsize=(10, 6))
                    types = list(type_counts.keys())
                    counts = list(type_counts.values())
                    
                    bars = plt.bar(range(len(types)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(types))))
                    plt.xlabel('断言类型')
                    plt.ylabel('出现次数')
                    plt.title('绝对化断言类型分布')
                    plt.xticks(range(len(types)), types, rotation=45)
                    
                    # 添加数值标签
                    for bar, count in zip(bars, counts):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                str(count), ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/assertion_type_distribution.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 3. 新术语新颖性分布
            if new_terminology_results['new_terminology']:
                novelty_scores = [term['novelty_score'] for term in new_terminology_results['new_terminology']]
                
                plt.figure(figsize=(10, 6))
                plt.hist(novelty_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.xlabel('新颖性分数')
                plt.ylabel('术语数量')
                plt.title('新术语新颖性分布')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/terminology_novelty_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. 风险分数与断言数量的散点图
            plt.figure(figsize=(10, 6))
            plt.scatter(absolute_assertions_df['total_absolute_assertions'], 
                       absolute_assertions_df['hallucination_risk_score'], 
                       alpha=0.6, color='red')
            plt.xlabel('绝对化断言数量')
            plt.ylabel('幻觉风险分数')
            plt.title('断言数量与幻觉风险的关系')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/assertions_vs_risk_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"知识幻觉可视化图表已保存到: {output_dir}")
            
        except ImportError as e:
            logger.warning(f"无法创建可视化图表，缺少必要的库: {e}")
        except Exception as e:
            logger.error(f"创建可视化图表时发生错误: {e}")


def main():
    """主函数 - 演示知识幻觉分析功能"""
    
    # 创建分析器实例
    analyzer = KnowledgeHallucinationAnalyzer()
    
    # 示例文本
    sample_texts = [
        "Jesus always loves everyone and will never abandon you.",
        "God has proven that salvation is guaranteed for all believers.",
        "The Bible clearly states that everyone must follow these rules.",
        "Research shows that prayer has been scientifically proven to work.",
        "God told me personally that this is the absolute truth.",
        "The Lord revealed to me that everyone will be saved.",
        "Studies confirm that faith is the only way to heaven.",
        "Jesus commanded that we must never question authority.",
        "The scriptures prove that this interpretation is correct.",
        "God's word guarantees that believers will never suffer."
    ]
    
    # 1. 检测绝对化断言
    print("检测绝对化断言...")
    assertions_df = analyzer.detect_absolute_assertions(sample_texts)
    print(f"检测到 {assertions_df['total_absolute_assertions'].sum()} 个绝对化断言")
    
    # 2. 检测新术语
    print("\n检测新术语...")
    terminology_results = analyzer.detect_new_terminology(sample_texts)
    print(f"发现 {terminology_results['new_terminology_count']} 个可能的新术语")
    
    # 3. 分析情绪操纵
    print("\n分析情绪操纵策略...")
    manipulation_results = analyzer.analyze_emotional_manipulation(sample_texts)
    print("情绪操纵分析完成")
    
    # 4. 生成报告
    print("\n生成分析报告...")
    report_file = analyzer.generate_hallucination_report(
        assertions_df, terminology_results, manipulation_results
    )
    print(f"报告已保存到: {report_file}")
    
    # 5. 创建可视化
    print("\n创建可视化图表...")
    analyzer.create_hallucination_visualizations(assertions_df, terminology_results)
    print("可视化图表创建完成")
    
    print("\n知识幻觉分析演示完成！")


if __name__ == "__main__":
    main()
