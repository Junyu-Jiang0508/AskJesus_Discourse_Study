#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Ask_Jesus Analysis Script

This script integrates CARER emotion analysis and knowledge hallucination analysis,
providing comprehensive data analysis support for your research paper.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import List

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carer_emotion_analysis import CAREREmotionAnalyzer
from knowledge_hallucination_analysis import KnowledgeHallucinationAnalyzer

# 配置日志
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'complete_analysis_{timestamp}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CompleteAskJesusAnalyzer:
    """
    Complete Ask_Jesus Analyzer
    
    Integrates emotion analysis, frame analysis, knowledge hallucination detection and other functions
    """
    
    def __init__(self):
        """Initialize analyzer"""
        self.emotion_analyzer = None
        self.hallucination_analyzer = None
        self.analysis_results = {}
        
        logger.info("Complete analyzer initialization completed")
    
    def run_emotion_analysis(self, texts: List[str], sources: List[str] = None):
        """
        Run CARER emotion analysis
        
        Args:
            texts: List of texts
            sources: List of text sources
        """
        logger.info("Starting CARER emotion analysis...")
        
        try:
            # 初始化情绪分析器
            self.emotion_analyzer = CAREREmotionAnalyzer()
            
            # Analyze emotions
            emotion_results = self.emotion_analyzer.analyze_emotions(texts)
            
            # Add source information
            if sources and len(sources) == len(emotion_results):
                emotion_results['text_source'] = sources
            
            # Analyze emotion distribution
            distribution_results = self.emotion_analyzer.analyze_emotion_distribution()
            
            # Evaluate authority construction effectiveness
            authority_results = self.emotion_analyzer.evaluate_authority_construction(texts)
            
            # Store results
            self.analysis_results['emotion_analysis'] = {
                'emotion_results': emotion_results,
                'distribution_results': distribution_results,
                'authority_results': authority_results
            }
            
            logger.info("CARER emotion analysis completed")
            return emotion_results, distribution_results, authority_results
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            raise
    
    def run_hallucination_analysis(self, texts: List[str]):
        """
        Run knowledge hallucination analysis
        
        Args:
            texts: List of texts
        """
        logger.info("Starting knowledge hallucination analysis...")
        
        try:
            # 初始化知识幻觉分析器
            self.hallucination_analyzer = KnowledgeHallucinationAnalyzer()
            
            # Detect absolute assertions
            assertions_df = self.hallucination_analyzer.detect_absolute_assertions(texts)
            
            # Detect new terminology
            terminology_results = self.hallucination_analyzer.detect_new_terminology(texts)
            
            # Analyze emotional manipulation (if emotion data available)
            emotion_data = None
            if 'emotion_analysis' in self.analysis_results:
                emotion_data = self.analysis_results['emotion_analysis']['emotion_results']
            
            manipulation_results = self.hallucination_analyzer.analyze_emotional_manipulation(
                texts, emotion_data
            )
            
            # Store results
            self.analysis_results['hallucination_analysis'] = {
                'assertions_df': assertions_df,
                'terminology_results': terminology_results,
                'manipulation_results': manipulation_results
            }
            
            logger.info("Knowledge hallucination analysis completed")
            return assertions_df, terminology_results, manipulation_results
            
        except Exception as e:
            logger.error(f"Knowledge hallucination analysis failed: {e}")
            raise
    
    def analyze_frame_emotion_correlation(self, frame_file_path: str):
        """
        Analyze correlation between frames and emotions
        
        Args:
            frame_file_path: Path to frame data file
        """
        if not self.emotion_analyzer:
            logger.error("Please run emotion analysis first")
            return None
        
        logger.info("Analyzing correlation between frames and emotions...")
        
        try:
            # Load frame data
            self.emotion_analyzer.load_frame_data(frame_file_path)
            
            # Merge emotion and frame data
            merged_data = self.emotion_analyzer.merge_emotion_frame_data()
            
            # Analyze correlation
            correlation_results = self.emotion_analyzer.analyze_frame_emotion_correlation(merged_data)
            
            # Store results
            self.analysis_results['frame_emotion_correlation'] = correlation_results
            
            logger.info("Frame-emotion correlation analysis completed")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Frame-emotion correlation analysis failed: {e}")
            return None
    
    def create_all_visualizations(self, output_dir: str = "complete_analysis_results"):
        """
        Create all visualization charts
        
        Args:
            output_dir: Output directory
        """
        logger.info(f"Creating all visualization charts: {output_dir}")
        
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Create emotion analysis visualizations
            if self.emotion_analyzer:
                emotion_dir = f"{output_dir}/emotion_analysis"
                self.emotion_analyzer.create_visualizations(emotion_dir)
                logger.info(f"Emotion analysis visualizations saved to: {emotion_dir}")
            
            # Create knowledge hallucination visualizations
            if self.hallucination_analyzer and 'hallucination_analysis' in self.analysis_results:
                hallucination_dir = f"{output_dir}/hallucination_analysis"
                assertions_df = self.analysis_results['hallucination_analysis']['assertions_df']
                terminology_results = self.analysis_results['hallucination_analysis']['terminology_results']
                
                self.hallucination_analyzer.create_hallucination_visualizations(
                    assertions_df, terminology_results, hallucination_dir
                )
                logger.info(f"Knowledge hallucination visualizations saved to: {hallucination_dir}")
            
            # Create comprehensive analysis charts
            self._create_comprehensive_visualizations(output_dir)
            
            logger.info("All visualization charts created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create visualization charts: {e}")
    
    def _create_comprehensive_visualizations(self, output_dir: str):
        """Create comprehensive analysis charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set font for Chinese characters
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. Emotion vs Hallucination Risk Relationship Chart
            if ('emotion_analysis' in self.analysis_results and 
                'hallucination_analysis' in self.analysis_results):
                
                emotion_data = self.analysis_results['emotion_analysis']['emotion_results']
                assertions_df = self.analysis_results['hallucination_analysis']['assertions_df']
                
                if len(emotion_data) == len(assertions_df):
                    plt.figure(figsize=(12, 8))
                    
                    # Group by emotion type, calculate average hallucination risk
                    emotion_risk_data = []
                    for emotion in emotion_data['primary_emotion'].unique():
                        emotion_mask = emotion_data['primary_emotion'] == emotion
                        emotion_indices = emotion_mask[emotion_mask].index
                        avg_risk = assertions_df.iloc[emotion_indices]['hallucination_risk_score'].mean()
                        emotion_risk_data.append((emotion, avg_risk))
                    
                    emotions, risks = zip(*emotion_risk_data)
                    
                    bars = plt.bar(range(len(emotions)), risks, color=plt.cm.Set3(np.linspace(0, 1, len(emotions))))
                    plt.xlabel('Emotion Category')
                    plt.ylabel('Average Hallucination Risk Score')
                    plt.title('Average Hallucination Risk by Emotion Category')
                    plt.xticks(range(len(emotions)), emotions, rotation=45)
                    
                    # Add value labels
                    for bar, risk in zip(bars, risks):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{risk:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/emotion_hallucination_correlation.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 2. Authority Construction Effectiveness Comprehensive Assessment Chart
            if 'emotion_analysis' in self.analysis_results:
                authority_results = self.analysis_results['emotion_analysis']['authority_results']
                
                if 'authority_indicators' in authority_results:
                    plt.figure(figsize=(10, 6))
                    
                    indicator_types = list(authority_results['authority_indicators'].keys())
                    avg_counts = [authority_results['authority_indicators'][t]['average_per_text'] 
                                for t in indicator_types]
                    
                    bars = plt.bar(range(len(indicator_types)), avg_counts, color='lightcoral')
                    plt.xlabel('Authority Construction Indicator Type')
                    plt.ylabel('Average Occurrences per Text')
                    plt.title('AI Authority Construction Indicator Usage Frequency')
                    plt.xticks(range(len(indicator_types)), indicator_types, rotation=45)
                    
                    # Add value labels
                    for bar, count in zip(bars, avg_counts):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                f'{count:.2f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/authority_construction_indicators.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info("Comprehensive analysis charts created successfully")
            
        except ImportError as e:
            logger.warning(f"Cannot create comprehensive analysis charts, missing required libraries: {e}")
        except Exception as e:
            logger.error(f"Error occurred while creating comprehensive analysis charts: {e}")
    
    def generate_comprehensive_report(self, output_file: str = None):
        """
        Generate comprehensive analysis report
        
        Args:
            output_file: Output file path
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"complete_analysis_report_{timestamp}.txt"
        
        logger.info(f"Generating comprehensive analysis report: {output_file}")
        
        # Generate emotion analysis report
        if self.emotion_analyzer:
            self.emotion_analyzer.generate_report(f"emotion_analysis_report_{timestamp}.txt")
        
        # Generate knowledge hallucination report
        if (self.hallucination_analyzer and 
            'hallucination_analysis' in self.analysis_results):
            
            assertions_df = self.analysis_results['hallucination_analysis']['assertions_df']
            terminology_results = self.analysis_results['hallucination_analysis']['terminology_results']
            manipulation_results = self.analysis_results['hallucination_analysis']['manipulation_results']
            
            self.hallucination_analyzer.generate_hallucination_report(
                assertions_df, terminology_results, manipulation_results,
                f"hallucination_analysis_report_{timestamp}.txt"
            )
        
        # Generate comprehensive report
        self._generate_comprehensive_report(output_file)
        
        logger.info(f"Comprehensive analysis report saved to: {output_file}")
        return output_file
    
    def _generate_comprehensive_report(self, output_file: str):
        """Generate comprehensive report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Ask_Jesus Platform Comprehensive Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. Executive Summary
        report_lines.append("1. Executive Summary")
        report_lines.append("-" * 40)
        
        if 'emotion_analysis' in self.analysis_results:
            emotion_data = self.analysis_results['emotion_analysis']['emotion_results']
            report_lines.append(f"Number of texts analyzed for emotion: {len(emotion_data)}")
            
            if 'primary_emotion' in emotion_data.columns:
                dominant_emotion = emotion_data['primary_emotion'].mode()[0]
                report_lines.append(f"Dominant emotion: {dominant_emotion}")
        
        if 'hallucination_analysis' in self.analysis_results:
            assertions_df = self.analysis_results['hallucination_analysis']['assertions_df']
            total_assertions = assertions_df['total_absolute_assertions'].sum()
            avg_risk = assertions_df['hallucination_risk_score'].mean()
            report_lines.append(f"Total absolute assertions detected: {total_assertions}")
            report_lines.append(f"Average hallucination risk score: {avg_risk:.3f}")
        
        report_lines.append("")
        
        # 2. Key Findings
        report_lines.append("2. Key Findings")
        report_lines.append("-" * 40)
        
        # Emotion distribution findings
        if 'emotion_analysis' in self.analysis_results:
            emotion_data = self.analysis_results['emotion_analysis']['emotion_results']
            if 'primary_emotion' in emotion_data.columns:
                emotion_counts = emotion_data['primary_emotion'].value_counts()
                report_lines.append("Emotion distribution:")
                for emotion, count in emotion_counts.items():
                    percentage = (count / len(emotion_data)) * 100
                    report_lines.append(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Hallucination risk findings
        if 'hallucination_analysis' in self.analysis_results:
            assertions_df = self.analysis_results['hallucination_analysis']['assertions_df']
            high_risk_count = len(assertions_df[assertions_df['hallucination_risk_score'] > 0.7])
            report_lines.append(f"\nHigh-risk text count (risk score > 0.7): {high_risk_count}")
        
        report_lines.append("")
        
        # 3. Research Significance
        report_lines.append("3. Research Significance")
        report_lines.append("-" * 40)
        report_lines.append("This study reveals through CARER emotion analysis and knowledge hallucination detection:")
        report_lines.append("  - Characteristics of AI-generated religious texts in emotion construction")
        report_lines.append("  - Knowledge hallucinations that may arise in AI authority construction")
        report_lines.append("  - Correlation patterns between emotions and authority construction")
        report_lines.append("  - New mechanisms for sacred construction in religious texts in the digital age")
        report_lines.append("")
        
        # 4. Recommendations
        report_lines.append("4. Recommendations")
        report_lines.append("-" * 40)
        report_lines.append("Based on analysis results, we recommend:")
        report_lines.append("  - Establish ethical guidelines for AI religious text generation")
        report_lines.append("  - Strengthen content review and verification mechanisms")
        report_lines.append("  - Develop more accurate hallucination detection tools")
        report_lines.append("  - Promote positive interaction between technology and religion")
        report_lines.append("")
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


def load_and_prepare_data():
    """Load and prepare data"""
    logger.info("Starting to load Ask_Jesus data...")
    
    data_files = {
        'jesus_responses': 'data/by_perspective_jesus.csv',
        'audience_comments': 'data/by_perspective_audience.csv',
        'labeled_data': 'output/labeled/by_perspective_jesus_labeled.csv'
    }
    
    loaded_data = {}
    
    for name, file_path in data_files.items():
        try:
            if Path(file_path).exists():
                logger.info(f"Loading {name}: {file_path}")
                data = pd.read_csv(file_path, encoding='utf-8')
                loaded_data[name] = data
                logger.info(f"Successfully loaded {name}: {len(data)} records")
            else:
                logger.warning(f"File does not exist: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    
    return loaded_data


def prepare_texts_for_analysis(data_dict):
    """Prepare text data for analysis"""
    logger.info("Preparing text data for analysis...")
    
    all_texts = []
    text_sources = []
    
    # Process Jesus response data
    if 'jesus_responses' in data_dict:
        jesus_data = data_dict['jesus_responses']
        
        # Find columns containing text content
        text_columns = [col for col in jesus_data.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'content', 'message', 'response', 'answer'])]
        
        if text_columns:
            logger.info(f"Found text columns in Jesus response data: {text_columns}")
            
            for col in text_columns:
                texts = jesus_data[col].dropna().astype(str).tolist()
                all_texts.extend(texts)
                text_sources.extend([f"jesus_{col}"] * len(texts))
        
        # If no clear text column found, try using the first column
        elif len(jesus_data.columns) > 0:
            first_col = jesus_data.columns[0]
            texts = jesus_data[first_col].dropna().astype(str).tolist()
            all_texts.extend(texts)
            text_sources.extend([f"jesus_{first_col}"] * len(texts))
    
    # Process audience comment data
    if 'audience_comments' in data_dict:
        audience_data = data_dict['audience_comments']
        
        # Find columns containing text content
        text_columns = [col for col in audience_data.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'content', 'message', 'comment', 'chat'])]
        
        if text_columns:
            logger.info(f"Found text columns in audience comment data: {text_columns}")
            
            for col in text_columns:
                texts = audience_data[col].dropna().astype(str).tolist()
                all_texts.extend(texts)
                text_sources.extend([f"audience_{col}"] * len(texts))
    
    # Filter and clean texts
    cleaned_texts = []
    cleaned_sources = []
    
    for text, source in zip(all_texts, text_sources):
        # Filter texts that are too short
        if len(text.strip()) > 10:
            cleaned_texts.append(text.strip())
            cleaned_sources.append(source)
    
    logger.info(f"Preparation completed: {len(cleaned_texts)} valid texts")
    
    return cleaned_texts, cleaned_sources


def create_sample_data():
    """Create sample data for testing"""
    logger.info("Creating sample data for testing...")
    
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


def main():
    """Main function"""
    logger.info("Starting Ask_Jesus comprehensive analysis...")
    
    try:
        # 1. Load data
        data_dict = load_and_prepare_data()
        
        if not data_dict:
            logger.warning("No valid data files found, using sample data for testing...")
            texts = create_sample_data()
            sources = ["sample"] * len(texts)
        else:
            # 2. Prepare text data
            texts, sources = prepare_texts_for_analysis(data_dict)
            
            if not texts:
                logger.warning("No valid text data found, using sample data for testing...")
                texts = create_sample_data()
                sources = ["sample"] * len(texts)
        
        # 3. Initialize complete analyzer
        analyzer = CompleteAskJesusAnalyzer()
        
        # 4. Run emotion analysis
        logger.info("=" * 60)
        logger.info("Phase 1: CARER Emotion Analysis")
        logger.info("=" * 60)
        
        emotion_results, distribution_results, authority_results = analyzer.run_emotion_analysis(texts, sources)
        
        # 5. Run knowledge hallucination analysis
        logger.info("=" * 60)
        logger.info("Phase 2: Knowledge Hallucination Analysis")
        logger.info("=" * 60)
        
        assertions_df, terminology_results, manipulation_results = analyzer.run_hallucination_analysis(texts)
        
        # 6. Try to analyze frame-emotion correlation
        logger.info("=" * 60)
        logger.info("Phase 3: Frame-Emotion Correlation Analysis")
        logger.info("=" * 60)
        
        frame_files = [
            'output/labeled/by_perspective_jesus_labeled.csv',
            'data/labeled_set/primary_label.csv',
            'data/labeled_set/secondary_label.csv'
        ]
        
        for frame_file in frame_files:
            if Path(frame_file).exists():
                logger.info(f"Attempting to analyze frame-emotion correlation: {frame_file}")
                correlation_results = analyzer.analyze_frame_emotion_correlation(frame_file)
                if correlation_results:
                    logger.info("Frame-emotion correlation analysis successful")
                    break
        
        # 7. Create all visualizations
        logger.info("=" * 60)
        logger.info("Phase 4: Creating Visualization Charts")
        logger.info("=" * 60)
        
        analyzer.create_all_visualizations()
        
        # 8. Generate comprehensive report
        logger.info("=" * 60)
        logger.info("Phase 5: Generating Analysis Report")
        logger.info("=" * 60)
        
        report_file = analyzer.generate_comprehensive_report()
        
        # 9. Save result data
        logger.info("=" * 60)
        logger.info("Phase 6: Saving Analysis Results")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save emotion analysis results
        if emotion_results is not None:
            emotion_output_file = f"emotion_analysis_results_{timestamp}.csv"
            emotion_results.to_csv(emotion_output_file, index=False, encoding='utf-8')
            logger.info(f"Emotion analysis results saved to: {emotion_output_file}")
        
        # Save knowledge hallucination analysis results
        if assertions_df is not None:
            hallucination_output_file = f"hallucination_analysis_results_{timestamp}.csv"
            assertions_df.to_csv(hallucination_output_file, index=False, encoding='utf-8')
            logger.info(f"Knowledge hallucination analysis results saved to: {hallucination_output_file}")
        
        logger.info("=" * 60)
        logger.info("Analysis completed! Results saved in the following locations:")
        logger.info(f"- Comprehensive report: {report_file}")
        logger.info("- Visualization charts: complete_analysis_results/")
        logger.info("- Emotion analysis report: emotion_analysis_report_*.txt")
        logger.info("- Knowledge hallucination report: hallucination_analysis_report_*.txt")
        logger.info("- Data results: *_results_*.csv")
        logger.info("=" * 60)
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Error occurred during analysis: {e}")
        raise


if __name__ == "__main__":
    try:
        analyzer = main()
        print("\nComplete analysis executed successfully!")
        print("Please check the generated analysis reports and visualization charts.")
        
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        print(f"\nProgram execution failed: {e}")
        print("Please check the error log and try again.")
        sys.exit(1)
