#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARER Emotion Analysis System for Ask_Jesus Religious Text Analysis

This module implements comprehensive emotion analysis for religious AI-generated texts,
including emotion distribution analysis, frame-emotion correlation, and authority 
construction effectiveness evaluation. Enhanced with 8emos emotion features and 
religious text-specific patterns.

Author: [Your Name]
Date: [Current Date]
Version: 2.0.0
License: MIT

Dependencies:
    - pandas >= 1.3.0
    - numpy >= 1.20.0
    - matplotlib >= 3.3.0
    - seaborn >= 0.11.0
    - scipy >= 1.7.0
    - scikit-learn >= 1.0.0
    - transformers >= 4.20.0
    - torch >= 1.12.0
    - plotly >= 5.0.0
    - wordcloud >= 1.9.0
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import json
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# Configure warnings and logging
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('carer_emotion_analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import transformers with proper error handling
try:
    import transformers
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    logger.error(f"Transformers library not found: {e}")
    logger.info("Installing transformers...")
    os.system("pip install transformers torch")
    try:
        import transformers
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        logger.info("Transformers library installed successfully")
    except ImportError:
        logger.error("Failed to install transformers library")
        sys.exit(1)

# Configuration constants
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_RANDOM_STATE = 42
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (12, 8)

# Matplotlib configuration for Chinese support
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Enhanced CARER Emotion Categories with 8emos integration
CARER_EMOTIONS = {
    'joy': 'Joy',
    'sadness': 'Sadness', 
    'anger': 'Anger',
    'fear': 'Fear',
    'surprise': 'Surprise',
    'disgust': 'Disgust',
    'neutral': 'Neutral',
    'anticipation': 'Anticipation',  # Added from 8emos
    'trust': 'Trust'                 # Added from 8emos
}

# 8emos Emotion Feature Weights (PFIEF patterns)
EMOTION_FEATURE_WEIGHTS = {
    'joy': {
        'hashtag_patterns': 23.51,
        'exclamation_patterns': 18.94,
        'positive_words': 16.95,
        'personal_pronouns': 15.64
    },
    'sadness': {
        'negative_words': 24.05,
        'sad_emoticons': 11.43,
        'melancholy_patterns': 17.43,
        'isolation_words': 15.07
    },
    'anger': {
        'aggressive_patterns': 20.64,
        'intensity_words': 18.24,
        'confrontation_patterns': 16.58,
        'forceful_language': 15.69
    },
    'fear': {
        'threat_patterns': 22.63,
        'uncertainty_words': 19.78,
        'anxiety_patterns': 17.96,
        'protective_language': 16.24
    },
    'surprise': {
        'exclamation_patterns': 25.63,
        'question_patterns': 21.51,
        'unexpected_patterns': 19.63,
        'reaction_words': 18.24
    },
    'disgust': {
        'negative_patterns': 23.45,
        'rejection_words': 20.12,
        'aversion_patterns': 18.67,
        'contempt_language': 16.89
    },
    'anticipation': {
        'future_words': 24.78,
        'hope_patterns': 21.34,
        'expectation_words': 19.56,
        'preparation_patterns': 17.89
    },
    'trust': {
        'confidence_words': 25.12,
        'belief_patterns': 22.45,
        'faith_words': 20.78,
        'reliability_patterns': 18.92
    }
}

# Religious Authority Indicators with enhanced patterns
AUTHORITY_INDICATORS = {
    'absolute_assertions': [
        r'\b(always|never|everyone|nobody|everything|nothing)\b',
        r'\b(must|should|have to|need to)\b',
        r'\b(perfect|complete|absolute|total|ultimate)\b',
        r'\b(proven|certain|definite|guaranteed)\b',
        r'\b(without doubt|undoubtedly|certainly|absolutely)\b'
    ],
    'divine_references': [
        r'\b(God|Jesus|Christ|Lord|Heaven|divine|holy|sacred)\b',
        r'\b(scripture|Bible|gospel|revelation|prophecy)\b',
        r'\b(salvation|redemption|eternal|immortal)\b',
        r'\b(almighty|omnipotent|omniscient|omnipresent)\b'
    ],
    'emotional_manipulation': [
        r'\b(trust me|believe me|I know|I understand)\b',
        r'\b(you must|you should|you need)\b',
        r'\b(if you don\'t|unless you|without this)\b',
        r'\b(only through|the only way|the truth is)\b'
    ],
    'religious_authority': [
        r'\b(as the scripture says|the Bible teaches|God reveals)\b',
        r'\b(divine wisdom|holy guidance|spiritual truth)\b',
        r'\b(eternal principle|divine law|sacred teaching)\b'
    ]
}

# Religious Text Emotion Patterns
RELIGIOUS_EMOTION_PATTERNS = {
    'joy': [
        r'\b(rejoice|blessed|glad|happy|joyful|celebrate)\b',
        r'\b(praise|worship|thankful|grateful|blessing)\b',
        r'\b(victory|triumph|overcome|conquer|prevail)\b'
    ],
    'sadness': [
        r'\b(sorrow|grief|mourn|weep|lament|suffer)\b',
        r'\b(pain|anguish|despair|hopeless|broken)\b',
        r'\b(lost|abandoned|rejected|forsaken|alone)\b'
    ],
    'anger': [
        r'\b(wrath|fury|rage|indignation|outrage)\b',
        r'\b(judgment|condemn|punish|retribution|vengeance)\b',
        r'\b(offended|betrayed|deceived|lied to|cheated)\b'
    ],
    'fear': [
        r'\b(afraid|terrified|dread|horror|panic)\b',
        r'\b(tremble|quake|shudder|cower|hide)\b',
        r'\b(danger|threat|peril|doom|destruction)\b'
    ],
    'surprise': [
        r'\b(amazed|astonished|stunned|shocked|bewildered)\b',
        r'\b(miracle|wonder|marvel|extraordinary|unexpected)\b',
        r'\b(revelation|discovery|unveiled|revealed|manifested)\b'
    ],
    'disgust': [
        r'\b(abhor|detest|loathe|despise|abominate)\b',
        r'\b(sin|evil|wicked|corrupt|depraved)\b',
        r'\b(filth|impurity|defilement|pollution|contamination)\b'
    ],
    'anticipation': [
        r'\b(expect|hope|wait|long for|yearn)\b',
        r'\b(promise|prophecy|foretell|predict|foresee)\b',
        r'\b(coming|approaching|near|soon|at hand)\b'
    ],
    'trust': [
        r'\b(faith|believe|trust|confidence|assurance)\b',
        r'\b(rely|depend|lean on|count on|rest in)\b',
        r'\b(faithful|reliable|steadfast|unwavering|constant)\b'
    ]
}


class CAREREmotionAnalyzer:
    """
    Comprehensive emotion analysis system for religious AI-generated texts.
    
    This class implements emotion recognition, frame-emotion correlation analysis,
    and authority construction effectiveness evaluation.
    """
    
    def __init__(self, 
                 emotion_model: str = DEFAULT_EMOTION_MODEL,
                 max_length: int = DEFAULT_MAX_LENGTH,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 random_state: int = DEFAULT_RANDOM_STATE):
        """
        Initialize the CARER emotion analyzer.
        
        Args:
            emotion_model: Pre-trained emotion classification model
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
            random_state: Random state for reproducibility
        """
        self.emotion_model = emotion_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Initialize emotion classifier
        self.emotion_classifier = None
        self.tokenizer = None
        self.model = None
        
        # Data storage
        self.emotion_data = None
        self.frame_data = None
        self.correlation_data = None
        
        # Initialize the model
        self._initialize_model()
        
        logger.info("CARER Emotion Analyzer initialized successfully")
    
    def _initialize_model(self):
        """Initialize the emotion classification model."""
        try:
            logger.info(f"Loading emotion model: {self.emotion_model}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.emotion_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.emotion_model)
            
            # Create emotion classifier pipeline
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                batch_size=self.batch_size,
                return_all_scores=True
            )
            
            logger.info("Emotion model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise
    
    def analyze_emotions(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze emotions in a list of texts using enhanced CARER + 8emos approach.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            DataFrame with emotion analysis results
        """
        logger.info(f"Analyzing emotions for {len(texts)} texts using enhanced CARER + 8emos")
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                # Clean and truncate text
                cleaned_text = self._preprocess_text(text)
                
                if not cleaned_text.strip():
                    continue
                
                # Get base emotion predictions from transformer model
                predictions = self.emotion_classifier(cleaned_text)[0]
                
                # Extract base emotion scores
                base_emotion_scores = {}
                for pred in predictions:
                    label = pred['label'].lower()
                    score = pred['score']
                    base_emotion_scores[label] = score
                
                # Apply 8emos feature enhancement
                enhanced_scores = self._apply_8emos_enhancement(cleaned_text, base_emotion_scores)
                
                # Apply religious text pattern enhancement
                religious_enhanced_scores = self._apply_religious_pattern_enhancement(cleaned_text, enhanced_scores)
                
                # Get primary emotion from enhanced scores
                primary_emotion = max(religious_enhanced_scores.items(), key=lambda x: x[1])
                
                # Store results
                result = {
                    'text_id': i,
                    'original_text': text,
                    'cleaned_text': cleaned_text,
                    'primary_emotion': primary_emotion[0],
                    'emotion_confidence': primary_emotion[1],
                    'base_confidence': base_emotion_scores.get(primary_emotion[0], 0.0),
                    'enhancement_factor': primary_emotion[1] / base_emotion_scores.get(primary_emotion[0], 0.1)
                }
                
                # Add individual emotion scores (both base and enhanced)
                for emotion in CARER_EMOTIONS.keys():
                    if emotion in base_emotion_scores:
                        result[f'{emotion}_base_score'] = base_emotion_scores[emotion]
                    if emotion in religious_enhanced_scores:
                        result[f'{emotion}_enhanced_score'] = religious_enhanced_scores[emotion]
                
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                continue
        
        # Create DataFrame
        self.emotion_data = pd.DataFrame(results)
        logger.info(f"Enhanced emotion analysis completed. Processed {len(self.emotion_data)} texts")
        
        return self.emotion_data
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for emotion analysis.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def load_frame_data(self, frame_file: str) -> pd.DataFrame:
        """
        Load frame classification data.
        
        Args:
            frame_file: Path to frame classification CSV file
            
        Returns:
            DataFrame with frame data
        """
        try:
            logger.info(f"Loading frame data from: {frame_file}")
            self.frame_data = pd.read_csv(frame_file, encoding='utf-8')
            logger.info(f"Loaded {len(self.frame_data)} frame records")
            return self.frame_data
            
        except Exception as e:
            logger.error(f"Failed to load frame data: {e}")
            raise
    
    def _apply_8emos_enhancement(self, text: str, base_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply 8emos emotion feature enhancement using PFIEF patterns.
        
        Args:
            text: Preprocessed text
            base_scores: Base emotion scores from transformer model
            
        Returns:
            Enhanced emotion scores
        """
        enhanced_scores = base_scores.copy()
        
        # Initialize missing emotions with base scores
        for emotion in CARER_EMOTIONS.keys():
            if emotion not in enhanced_scores:
                enhanced_scores[emotion] = 0.1  # Small base score
        
        # Apply 8emos feature weights
        for emotion, features in EMOTION_FEATURE_WEIGHTS.items():
            enhancement_score = 0.0
            
            # Check for hashtag patterns
            if 'hashtag_patterns' in features:
                hashtag_count = len(re.findall(r'#\w+', text))
                if hashtag_count > 0:
                    enhancement_score += features['hashtag_patterns'] * 0.01 * hashtag_count
            
            # Check for exclamation patterns
            if 'exclamation_patterns' in features:
                exclamation_count = len(re.findall(r'!+', text))
                if exclamation_count > 0:
                    enhancement_score += features['exclamation_patterns'] * 0.01 * exclamation_count
            
            # Check for question patterns
            if 'question_patterns' in features:
                question_count = len(re.findall(r'\?+', text))
                if question_count > 0:
                    enhancement_score += features['question_patterns'] * 0.01 * question_count
            
            # Check for personal pronouns
            if 'personal_pronouns' in features:
                pronoun_count = len(re.findall(r'\b(I|you|he|she|we|they|me|him|her|us|them)\b', text, re.IGNORECASE))
                if pronoun_count > 0:
                    enhancement_score += features['personal_pronouns'] * 0.005 * pronoun_count
            
            # Apply enhancement
            if enhancement_score > 0:
                enhanced_scores[emotion] = min(1.0, enhanced_scores[emotion] + enhancement_score)
        
        return enhanced_scores
    
    def _apply_religious_pattern_enhancement(self, text: str, enhanced_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply religious text-specific emotion pattern enhancement.
        
        Args:
            text: Preprocessed text
            enhanced_scores: Emotion scores after 8emos enhancement
            
        Returns:
            Further enhanced emotion scores
        """
        religious_enhanced_scores = enhanced_scores.copy()
        
        # Apply religious emotion patterns
        for emotion, patterns in RELIGIOUS_EMOTION_PATTERNS.items():
            pattern_score = 0.0
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    pattern_score += len(matches) * 0.1  # Boost for each match
            
            # Apply religious enhancement
            if pattern_score > 0:
                religious_enhanced_scores[emotion] = min(1.0, religious_enhanced_scores[emotion] + pattern_score)
        
        # Special handling for religious authority indicators
        authority_score = 0.0
        for indicator_type, patterns in AUTHORITY_INDICATORS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    authority_score += len(matches) * 0.05
        
        # Boost trust emotion for religious authority
        if authority_score > 0:
            religious_enhanced_scores['trust'] = min(1.0, religious_enhanced_scores['trust'] + authority_score)
        
        return religious_enhanced_scores
    
    def merge_emotion_frame_data(self) -> pd.DataFrame:
        """
        Merge emotion and frame data for correlation analysis.
        
        Returns:
            Merged DataFrame
        """
        if self.emotion_data is None or self.frame_data is None:
            raise ValueError("Both emotion and frame data must be loaded first")
        
        logger.info("Merging emotion and frame data")
        
        # Merge on text_id
        merged_data = pd.merge(
            self.emotion_data, 
            self.frame_data, 
            on='text_id', 
            how='inner'
        )
        
        logger.info(f"Merged data contains {len(merged_data)} records")
        return merged_data
    
    def analyze_emotion_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of emotions in the dataset.
        
        Returns:
            Dictionary with emotion distribution statistics
        """
        if self.emotion_data is None:
            raise ValueError("Emotion data must be loaded first")
        
        logger.info("Analyzing emotion distribution")
        
        # Basic emotion counts
        emotion_counts = self.emotion_data['primary_emotion'].value_counts()
        
        # Emotion confidence statistics
        confidence_stats = self.emotion_data['emotion_confidence'].describe()
        
        # Emotion score distributions (both base and enhanced)
        emotion_scores = {}
        for emotion in CARER_EMOTIONS.keys():
            base_score_col = f'{emotion}_base_score'
            enhanced_score_col = f'{emotion}_enhanced_score'
            
            if base_score_col in self.emotion_data.columns:
                emotion_scores[f'{emotion}_base'] = self.emotion_data[base_score_col].describe()
            if enhanced_score_col in self.emotion_data.columns:
                emotion_scores[f'{emotion}_enhanced'] = self.emotion_data[enhanced_score_col].describe()
        
        results = {
            'emotion_counts': emotion_counts,
            'confidence_stats': confidence_stats,
            'emotion_scores': emotion_scores,
            'total_texts': len(self.emotion_data)
        }
        
        logger.info("Emotion distribution analysis completed")
        return results
    
    def analyze_frame_emotion_correlation(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlation between frames and emotions.
        
        Args:
            merged_data: DataFrame with both emotion and frame data
            
        Returns:
            Dictionary with correlation analysis results
        """
        logger.info("Analyzing frame-emotion correlations")
        
        results = {}
        
        # Analyze primary frame vs emotion correlation
        if 'primary_label' in merged_data.columns:
            frame_emotion_crosstab = pd.crosstab(
                merged_data['primary_label'], 
                merged_data['primary_emotion']
            )
            
            # Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(frame_emotion_crosstab)
            
            results['primary_frame_emotion_crosstab'] = frame_emotion_crosstab
            results['primary_frame_chi2'] = {'chi2': chi2, 'p_value': p_value, 'dof': dof}
        
        # Analyze secondary frame vs emotion correlation
        if 'secondary_label' in merged_data.columns:
            sec_frame_emotion_crosstab = pd.crosstab(
                merged_data['secondary_label'], 
                merged_data['primary_emotion']
            )
            
            # Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(sec_frame_emotion_crosstab)
            
            results['secondary_frame_emotion_crosstab'] = sec_frame_emotion_crosstab
            results['secondary_frame_chi2'] = {'chi2': chi2, 'p_value': p_value, 'dof': dof}
        
        # Emotion score correlations with frame features
        emotion_score_cols = [col for col in merged_data.columns if col.endswith('_score')]
        if emotion_score_cols:
            # Create correlation matrix
            correlation_matrix = merged_data[emotion_score_cols].corr()
            results['emotion_score_correlations'] = correlation_matrix
        
        logger.info("Frame-emotion correlation analysis completed")
        return results
    
    def evaluate_authority_construction(self, texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of AI authority construction.
        
        Args:
            texts: List of AI-generated texts to analyze
            
        Returns:
            Dictionary with authority construction analysis results
        """
        logger.info("Evaluating AI authority construction effectiveness")
        
        results = {
            'authority_indicators': {},
            'emotional_manipulation': {},
            'trust_signals': {},
            'overall_effectiveness': {}
        }
        
        # Analyze authority indicators
        for indicator_type, patterns in AUTHORITY_INDICATORS.items():
            indicator_counts = []
            for text in texts:
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    count += len(matches)
                indicator_counts.append(count)
            
            results['authority_indicators'][indicator_type] = {
                'total_occurrences': sum(indicator_counts),
                'average_per_text': np.mean(indicator_counts),
                'distribution': indicator_counts
            }
        
        # Analyze emotional manipulation effectiveness
        if self.emotion_data is not None:
            # Check if high authority indicators correlate with specific emotions
            authority_scores = []
            for text in texts:
                score = 0
                for patterns in AUTHORITY_INDICATORS.values():
                    for pattern in patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        score += len(matches)
                authority_scores.append(score)
            
            # Correlate with emotion confidence
            if len(authority_scores) == len(self.emotion_data):
                correlation = np.corrcoef(authority_scores, self.emotion_data['emotion_confidence'])[0, 1]
                results['emotional_manipulation']['authority_emotion_correlation'] = correlation
        
        logger.info("Authority construction evaluation completed")
        return results
    
    def create_visualizations(self, output_dir: str = "carer_visualizations"):
        """
        Create comprehensive visualizations for the analysis.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if self.emotion_data is None:
            raise ValueError("Emotion data must be loaded first")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info(f"Creating visualizations in: {output_dir}")
        
        # 1. Emotion Distribution
        self._create_emotion_distribution_plot(output_dir)
        
        # 2. Frame-Emotion Correlation Heatmap
        if self.frame_data is not None:
            self._create_frame_emotion_heatmap(output_dir)
        
        # 3. Emotion Confidence Distribution
        self._create_emotion_confidence_plot(output_dir)
        
        # 4. Emotion Score Radar Chart
        self._create_emotion_radar_chart(output_dir)
        
        # 5. Interactive Emotion Timeline (if temporal data available)
        self._create_emotion_timeline(output_dir)
        
        # 6. 8emos Feature Analysis Visualizations
        self.create_8emos_visualizations(output_dir)
        
        logger.info("All visualizations created successfully")
    
    def _create_emotion_distribution_plot(self, output_dir: str):
        """Create emotion distribution bar plot."""
        plt.figure(figsize=DEFAULT_FIGSIZE)
        
        emotion_counts = self.emotion_data['primary_emotion'].value_counts()
        
        # Create bar plot
        bars = plt.bar(range(len(emotion_counts)), emotion_counts.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(emotion_counts))))
        
        plt.xlabel('情绪类别')
        plt.ylabel('文本数量')
        plt.title('CARER情绪分布分析')
        plt.xticks(range(len(emotion_counts)), 
                  [CARER_EMOTIONS.get(emotion, emotion) for emotion in emotion_counts.index], 
                  rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, emotion_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/emotion_distribution.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()
    
    def _create_frame_emotion_heatmap(self, output_dir: str):
        """Create frame-emotion correlation heatmap."""
        if self.frame_data is None:
            return
        
        # Merge data for heatmap
        merged_data = self.merge_emotion_frame_data()
        
        if 'primary_label' in merged_data.columns:
            # Create crosstab
            crosstab = pd.crosstab(merged_data['primary_label'], merged_data['primary_emotion'])
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', 
                       cbar_kws={'label': '文本数量'})
            
            plt.xlabel('情绪类别')
            plt.ylabel('主框架类别')
            plt.title('框架-情绪关联热力图')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/frame_emotion_heatmap.png", dpi=DEFAULT_DPI, bbox_inches='tight')
            plt.close()
    
    def _create_emotion_confidence_plot(self, output_dir: str):
        """Create emotion confidence distribution plot."""
        plt.figure(figsize=DEFAULT_FIGSIZE)
        
        # Create violin plot for emotion confidence by emotion type
        emotion_data = []
        emotion_labels = []
        
        for emotion in CARER_EMOTIONS.keys():
            mask = self.emotion_data['primary_emotion'] == emotion
            if mask.any():
                emotion_data.append(self.emotion_data[mask]['emotion_confidence'].values)
                emotion_labels.append(CARER_EMOTIONS.get(emotion, emotion))
        
        if emotion_data:
            plt.violinplot(emotion_data, positions=range(len(emotion_data)))
            plt.xlabel('情绪类别')
            plt.ylabel('置信度分数')
            plt.title('各情绪类别的置信度分布')
            plt.xticks(range(len(emotion_labels)), emotion_labels, rotation=45)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/emotion_confidence_distribution.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()
    
    def _create_emotion_radar_chart(self, output_dir: str):
        """Create emotion score radar chart."""
        # Calculate average emotion scores
        emotion_scores = {}
        for emotion in CARER_EMOTIONS.keys():
            score_col = f'{emotion}_score'
            if score_col in self.emotion_data.columns:
                emotion_scores[emotion] = self.emotion_data[score_col].mean()
        
        if not emotion_scores:
            return
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(emotion_scores), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        values = list(emotion_scores.values())
        values += values[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, label='平均情绪分数')
        ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([CARER_EMOTIONS.get(emotion, emotion) for emotion in emotion_scores.keys()])
        ax.set_ylim(0, 1)
        ax.set_title('情绪分数雷达图', size=16, y=1.08)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/emotion_radar_chart.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()
    
    def _create_emotion_timeline(self, output_dir: str):
        """Create interactive emotion timeline plot."""
        if 'timestamp' not in self.emotion_data.columns:
            return
        
        # Create interactive timeline using plotly
        fig = px.scatter(
            self.emotion_data, 
            x='timestamp', 
            y='emotion_confidence',
            color='primary_emotion',
            title='情绪变化时间线',
            labels={'timestamp': '时间', 'emotion_confidence': '情绪置信度', 'primary_emotion': '主要情绪'}
        )
        
        fig.write_html(f"{output_dir}/emotion_timeline.html")
    
    def generate_report(self, output_file: str = "carer_emotion_analysis_report.txt"):
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Path to save the report
        """
        logger.info(f"Generating analysis report: {output_file}")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CARER情绪分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 数据概览
        if self.emotion_data is not None:
            report_lines.append("1. 数据概览")
            report_lines.append("-" * 40)
            report_lines.append(f"总文本数量: {len(self.emotion_data)}")
            report_lines.append(f"情绪类别数量: {len(CARER_EMOTIONS)}")
            report_lines.append("")
        
        # 2. 情绪分布分析
        if self.emotion_data is not None:
            report_lines.append("2. 情绪分布分析")
            report_lines.append("-" * 40)
            
            emotion_counts = self.emotion_data['primary_emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                percentage = (count / len(self.emotion_data)) * 100
                emotion_name = CARER_EMOTIONS.get(emotion, emotion)
                report_lines.append(f"{emotion_name}: {count} ({percentage:.1f}%)")
            
            report_lines.append("")
            
            # Confidence statistics
            avg_confidence = self.emotion_data['emotion_confidence'].mean()
            report_lines.append(f"平均情绪置信度: {avg_confidence:.3f}")
            report_lines.append("")
        
        # 3. 框架-情绪关联分析
        if self.frame_data is not None:
            report_lines.append("3. 框架-情绪关联分析")
            report_lines.append("-" * 40)
            
            try:
                merged_data = self.merge_emotion_frame_data()
                correlation_results = self.analyze_frame_emotion_correlation(merged_data)
                
                if 'primary_frame_chi2' in correlation_results:
                    chi2_result = correlation_results['primary_frame_chi2']
                    report_lines.append(f"主框架-情绪卡方检验:")
                    report_lines.append(f"  卡方值: {chi2_result['chi2']:.3f}")
                    report_lines.append(f"  p值: {chi2_result['p_value']:.3f}")
                    report_lines.append(f"  自由度: {chi2_result['dof']}")
                    report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"框架-情绪关联分析失败: {e}")
                report_lines.append("")
        
        # 4. 权威建构效果评估
        if self.emotion_data is not None:
            report_lines.append("4. 权威建构效果评估")
            report_lines.append("-" * 40)
            
            try:
                texts = self.emotion_data['original_text'].tolist()
                authority_results = self.evaluate_authority_construction(texts)
                
                for indicator_type, data in authority_results['authority_indicators'].items():
                    report_lines.append(f"{indicator_type}:")
                    report_lines.append(f"  总出现次数: {data['total_occurrences']}")
                    report_lines.append(f"  平均每文本: {data['average_per_text']:.2f}")
                    report_lines.append("")
                
            except Exception as e:
                report_lines.append(f"权威建构效果评估失败: {e}")
                report_lines.append("")
        
        # 5. 结论与建议
        report_lines.append("5. 结论与建议")
        report_lines.append("-" * 40)
        
        if self.emotion_data is not None:
            # Find dominant emotion
            dominant_emotion = self.emotion_data['primary_emotion'].mode()[0]
            dominant_emotion_name = CARER_EMOTIONS.get(dominant_emotion, dominant_emotion)
            
            report_lines.append(f"主要发现:")
            report_lines.append(f"  - 主导情绪: {dominant_emotion_name}")
            report_lines.append(f"  - 情绪识别置信度: {avg_confidence:.3f}")
            
            if avg_confidence > 0.8:
                report_lines.append("  - 情绪识别质量: 优秀")
            elif avg_confidence > 0.6:
                report_lines.append("  - 情绪识别质量: 良好")
            else:
                report_lines.append("  - 情绪识别质量: 需要改进")
        
        report_lines.append("")
        report_lines.append("建议:")
        report_lines.append("  - 进一步分析高置信度情绪样本的特征")
        report_lines.append("  - 探索情绪与特定宗教框架的关联模式")
        report_lines.append("  - 评估AI权威建构对用户情绪的实际影响")
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Analysis report saved to: {output_file}")
    
    def analyze_8emos_features(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of 8emos feature enhancement.
        
        Returns:
            Dictionary with 8emos feature analysis results
        """
        if self.emotion_data is None:
            raise ValueError("Emotion data must be loaded first")
        
        logger.info("Analyzing 8emos feature enhancement effectiveness")
        
        results = {
            'enhancement_stats': {},
            'feature_impact': {},
            'religious_pattern_analysis': {}
        }
        
        # Analyze enhancement factors
        if 'enhancement_factor' in self.emotion_data.columns:
            enhancement_stats = self.emotion_data['enhancement_factor'].describe()
            results['enhancement_stats'] = enhancement_stats.to_dict()
            
            # Find texts with significant enhancement
            high_enhancement = self.emotion_data[self.emotion_data['enhancement_factor'] > 1.5]
            results['high_enhancement_count'] = len(high_enhancement)
            results['high_enhancement_examples'] = high_enhancement[['text_id', 'enhancement_factor']].head(10).to_dict('records')
        
        # Analyze feature impact by emotion
        for emotion in CARER_EMOTIONS.keys():
            base_col = f'{emotion}_base_score'
            enhanced_col = f'{emotion}_enhanced_score'
            
            if base_col in self.emotion_data.columns and enhanced_col in self.emotion_data.columns:
                base_scores = self.emotion_data[base_col].dropna()
                enhanced_scores = self.emotion_data[enhanced_col].dropna()
                
                if len(base_scores) > 0 and len(enhanced_scores) > 0:
                    improvement = enhanced_scores.mean() - base_scores.mean()
                    results['feature_impact'][emotion] = {
                        'base_mean': base_scores.mean(),
                        'enhanced_mean': enhanced_scores.mean(),
                        'improvement': improvement,
                        'improvement_percentage': (improvement / base_scores.mean()) * 100 if base_scores.mean() > 0 else 0
                    }
        
        # Analyze religious pattern effectiveness
        religious_patterns = {}
        for emotion, patterns in RELIGIOUS_EMOTION_PATTERNS.items():
            pattern_matches = 0
            for text in self.emotion_data['cleaned_text']:
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        pattern_matches += 1
                        break
            
            religious_patterns[emotion] = {
                'patterns': patterns,
                'total_matches': pattern_matches,
                'match_percentage': (pattern_matches / len(self.emotion_data)) * 100
            }
        
        results['religious_pattern_analysis'] = religious_patterns
        
        logger.info("8emos feature analysis completed")
        return results
    
    def create_8emos_visualizations(self, output_dir: str):
        """Create visualizations for 8emos feature analysis."""
        if self.emotion_data is None:
            return
        
        logger.info("Creating 8emos feature visualizations")
        
        # 1. Enhancement Factor Distribution
        if 'enhancement_factor' in self.emotion_data.columns:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(self.emotion_data['enhancement_factor'], bins=30, alpha=0.7, color='skyblue')
            plt.xlabel('Enhancement Factor')
            plt.ylabel('Frequency')
            plt.title('8emos Enhancement Factor Distribution')
            plt.axvline(x=1.0, color='red', linestyle='--', label='No Enhancement')
            plt.legend()
            
            # 2. Base vs Enhanced Score Comparison
            plt.subplot(2, 2, 2)
            emotions_to_plot = ['joy', 'sadness', 'anger', 'fear', 'trust']
            base_means = []
            enhanced_means = []
            valid_emotions = []
            
            for emotion in emotions_to_plot:
                base_col = f'{emotion}_base_score'
                enhanced_col = f'{emotion}_enhanced_score'
                
                if base_col in self.emotion_data.columns and enhanced_col in self.emotion_data.columns:
                    base_mean = self.emotion_data[base_col].mean()
                    enhanced_mean = self.emotion_data[enhanced_col].mean()
                    
                    if not np.isnan(base_mean) and not np.isnan(enhanced_mean):
                        base_means.append(base_mean)
                        enhanced_means.append(enhanced_mean)
                        valid_emotions.append(emotion)
            
            if base_means and enhanced_means and len(base_means) == len(enhanced_means):
                x = np.arange(len(valid_emotions))
                width = 0.35
                
                plt.bar(x - width/2, base_means, width, label='Base Scores', alpha=0.8)
                plt.bar(x + width/2, enhanced_means, width, label='Enhanced Scores', alpha=0.8)
                
                plt.xlabel('Emotions')
                plt.ylabel('Average Score')
                plt.title('Base vs Enhanced Emotion Scores')
                plt.xticks(x, valid_emotions)
                plt.legend()
                plt.xticks(rotation=45)
            
            # 3. Religious Pattern Effectiveness
            plt.subplot(2, 2, 3)
            pattern_analysis = self.analyze_8emos_features()
            religious_data = pattern_analysis['religious_pattern_analysis']
            
            if religious_data:
                emotions = list(religious_data.keys())
                match_percentages = [religious_data[emotion]['match_percentage'] for emotion in emotions]
                
                plt.bar(emotions, match_percentages, color='lightcoral')
                plt.xlabel('Emotions')
                plt.ylabel('Pattern Match Percentage (%)')
                plt.title('Religious Pattern Effectiveness')
                plt.xticks(rotation=45)
            
            # 4. Enhancement by Text Length
            plt.subplot(2, 2, 4)
            if 'enhancement_factor' in self.emotion_data.columns:
                text_lengths = [len(text) for text in self.emotion_data['cleaned_text']]
                enhancement_factors = self.emotion_data['enhancement_factor']
                
                plt.scatter(text_lengths, enhancement_factors, alpha=0.6, color='green')
                plt.xlabel('Text Length')
                plt.ylabel('Enhancement Factor')
                plt.title('Enhancement Factor vs Text Length')
                
                # Add trend line
                z = np.polyfit(text_lengths, enhancement_factors, 1)
                p = np.poly1d(z)
                plt.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/8emos_feature_analysis.png", dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info("8emos feature visualizations created successfully")


def main():
    """Main function to demonstrate the CARER emotion analysis system."""
    
    # Initialize analyzer
    analyzer = CAREREmotionAnalyzer()
    
    # Example usage
    logger.info("CARER Emotion Analysis System Demo")
    
    # Sample texts for demonstration
    sample_texts = [
        "Jesus loves you unconditionally and will always be there for you.",
        "I'm feeling sad and lost, can you help me find hope?",
        "The Bible teaches us to be kind and compassionate to everyone.",
        "I'm angry about the injustice in the world today.",
        "God's grace is sufficient for all our needs."
    ]
    
    # Analyze emotions with 8emos enhancement
    emotion_results = analyzer.analyze_emotions(sample_texts)
    print("Enhanced emotion analysis results:")
    print(emotion_results[['primary_emotion', 'emotion_confidence', 'enhancement_factor']].head())
    
    # Analyze emotion distribution
    distribution_results = analyzer.analyze_emotion_distribution()
    print(f"\nEmotion distribution: {distribution_results['emotion_counts'].to_dict()}")
    
    # Analyze 8emos feature effectiveness
    print("\n8emos Feature Analysis:")
    eightemos_results = analyzer.analyze_8emos_features()
    print(f"High enhancement texts: {eightemos_results.get('high_enhancement_count', 0)}")
    
    # Evaluate authority construction
    authority_results = analyzer.evaluate_authority_construction(sample_texts)
    print(f"\nAuthority construction indicators: {authority_results['authority_indicators'].keys()}")
    
    # Create visualizations including 8emos
    analyzer.create_visualizations()
    
    # Generate report
    analyzer.generate_report()
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
