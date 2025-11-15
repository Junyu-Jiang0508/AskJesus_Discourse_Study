#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARER Configuration File for Ask_Jesus Emotion Analysis

This file contains all configuration parameters for the CARER emotion analysis system,
including 8emos feature weights, religious text patterns, and model settings.

Author: [Your Name]
Date: [Current Date]
Version: 2.0.0
License: MIT
"""

import os
from pathlib import Path

# Base Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Model Configuration
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_RANDOM_STATE = 42

# Visualization Configuration
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (12, 8)
FIGURE_FORMAT = 'png'

# 8emos Feature Enhancement Configuration
EMOTION_ENHANCEMENT_CONFIG = {
    'hashtag_weight': 0.01,
    'exclamation_weight': 0.01,
    'question_weight': 0.01,
    'pronoun_weight': 0.005,
    'religious_pattern_weight': 0.1,
    'authority_indicator_weight': 0.05,
    'min_base_score': 0.1,
    'max_enhancement': 1.0
}

# Religious Text Pattern Weights
RELIGIOUS_PATTERN_WEIGHTS = {
    'joy': 1.2,
    'sadness': 1.1,
    'anger': 1.15,
    'fear': 1.1,
    'surprise': 1.25,
    'disgust': 1.1,
    'anticipation': 1.3,
    'trust': 1.4  # Highest weight for religious authority
}

# Authority Construction Thresholds
AUTHORITY_THRESHOLDS = {
    'high_risk': 0.8,
    'medium_risk': 0.6,
    'low_risk': 0.4,
    'min_confidence': 0.3
}

# Emotion Classification Thresholds
EMOTION_THRESHOLDS = {
    'min_confidence': 0.2,
    'high_confidence': 0.8,
    'enhancement_threshold': 1.5,
    'pattern_match_threshold': 0.1
}

# File Paths
DATA_PATHS = {
    'frame_data': BASE_DIR / "data" / "labeled_set",
    'output_results': OUTPUT_DIR / "carer_analysis",
    'visualizations': OUTPUT_DIR / "carer_analysis" / "visualizations",
    'reports': OUTPUT_DIR / "carer_analysis" / "reports"
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': {
        'filename': LOG_DIR / 'carer_analysis.log',
        'encoding': 'utf-8',
        'maxBytes': 10485760,  # 10MB
        'backupCount': 5
    }
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'enable_8emos_enhancement': True,
    'enable_religious_patterns': True,
    'enable_authority_analysis': True,
    'enable_frame_correlation': True,
    'enable_visualization': True,
    'enable_report_generation': True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'use_gpu': False,
    'max_workers': 4,
    'chunk_size': 100,
    'memory_limit': '4GB'
}

# Validation Configuration
VALIDATION_CONFIG = {
    'min_text_length': 10,
    'max_text_length': 1000,
    'required_columns': ['text_id', 'original_text', 'cleaned_text'],
    'emotion_validation': {
        'min_score': 0.0,
        'max_score': 1.0,
        'required_emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
    }
}

def get_config():
    """Get the complete configuration dictionary."""
    return {
        'base': {
            'base_dir': str(BASE_DIR),
            'output_dir': str(OUTPUT_DIR),
            'log_dir': str(LOG_DIR),
            'model_dir': str(MODEL_DIR)
        },
        'model': {
            'emotion_model': DEFAULT_EMOTION_MODEL,
            'max_length': DEFAULT_MAX_LENGTH,
            'batch_size': DEFAULT_BATCH_SIZE,
            'random_state': DEFAULT_RANDOM_STATE
        },
        'visualization': {
            'dpi': DEFAULT_DPI,
            'figsize': DEFAULT_FIGSIZE,
            'format': FIGURE_FORMAT
        },
        '8emos': EMOTION_ENHANCEMENT_CONFIG,
        'religious_patterns': RELIGIOUS_PATTERN_WEIGHTS,
        'authority': AUTHORITY_THRESHOLDS,
        'emotion': EMOTION_THRESHOLDS,
        'paths': {k: str(v) for k, v in DATA_PATHS.items()},
        'logging': LOGGING_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'validation': VALIDATION_CONFIG
    }

def validate_config():
    """Validate the configuration settings."""
    errors = []
    
    # Check if required directories exist
    for path_name, path in DATA_PATHS.items():
        if not path.exists():
            errors.append(f"Required path does not exist: {path_name} -> {path}")
    
    # Check if model path is accessible
    if not MODEL_DIR.exists():
        errors.append(f"Model directory does not exist: {MODEL_DIR}")
    
    # Validate thresholds
    for threshold_name, threshold_value in EMOTION_THRESHOLDS.items():
        if not isinstance(threshold_value, (int, float)) or threshold_value < 0:
            errors.append(f"Invalid threshold value: {threshold_name} = {threshold_value}")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

def update_config(updates: dict):
    """Update configuration with new values."""
    config = get_config()
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    updated_config = deep_update(config, updates)
    return updated_config

if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print("Configuration validation passed!")
        
        config = get_config()
        print(f"Base directory: {config['base']['base_dir']}")
        print(f"Output directory: {config['base']['output_dir']}")
        print(f"8emos enhancement enabled: {config['analysis']['enable_8emos_enhancement']}")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
