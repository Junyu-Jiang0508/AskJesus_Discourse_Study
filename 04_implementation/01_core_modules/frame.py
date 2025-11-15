#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Association Analysis Framework for Text Classification Data

This module provides comprehensive statistical analysis of labeled text data,
including joint frequency analysis, conditional probability calculations,
chi-squared tests, correspondence analysis, and mutual information analysis.

Author: [Your Name]
Date: [Current Date]
Version: 1.0.0
License: MIT

Dependencies:
    - pandas >= 1.3.0
    - numpy >= 1.20.0
    - matplotlib >= 3.3.0
    - seaborn >= 0.11.0
    - scipy >= 1.7.0
    - scikit-learn >= 1.0.0
    - prince >= 0.6.0

References:
    - Agresti, A. (2013). Categorical Data Analysis. Wiley.
    - Greenacre, M. (2017). Correspondence Analysis in Practice. CRC Press.
    - Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory. Wiley.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# Configure warnings and logging
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import prince with proper error handling
try:
    import prince
except ImportError as e:
    logger.error(f"Prince library not found: {e}")
    logger.info("Installing prince...")
    os.system("pip install prince")
    try:
        import prince
        logger.info("Prince library installed successfully")
    except ImportError:
        logger.error("Failed to install prince library")
        sys.exit(1)

# Configuration constants
DEFAULT_FREQ_THRESHOLD = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_CA_COMPONENTS = 2
DEFAULT_CA_ITERATIONS = 10
DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (12, 10)
DEFAULT_CA_FIGSIZE = (10, 8)

# Matplotlib configuration
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


class LabelAssociationAnalyzer:
    """
    A comprehensive analyzer for label association patterns in text classification data.
    
    This class implements various statistical methods to analyze the relationships
    between primary and secondary labels in labeled text datasets.
    
    Attributes:
        freq_threshold (int): Minimum frequency threshold for high-frequency pairs
        random_state (int): Random state for reproducible results
        ca_components (int): Number of components for correspondence analysis
        ca_iterations (int): Number of iterations for correspondence analysis
    """
    
    def __init__(self, 
                 freq_threshold: int = DEFAULT_FREQ_THRESHOLD,
                 random_state: int = DEFAULT_RANDOM_STATE,
                 ca_components: int = DEFAULT_CA_COMPONENTS,
                 ca_iterations: int = DEFAULT_CA_ITERATIONS):
        """
        Initialize the LabelAssociationAnalyzer.
        
        Args:
            freq_threshold: Minimum frequency for high-frequency pairs
            random_state: Random state for reproducibility
            ca_components: Number of components for CA
            ca_iterations: Number of iterations for CA
        """
        self.freq_threshold = freq_threshold
        self.random_state = random_state
        self.ca_components = ca_components
        self.ca_iterations = ca_iterations
        
        # Validate parameters
        if freq_threshold < 1:
            raise ValueError("freq_threshold must be at least 1")
        if ca_components < 2:
            raise ValueError("ca_components must be at least 2")
        if ca_iterations < 1:
            raise ValueError("ca_iterations must be at least 1")
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data format and content.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        required_columns = ['primary_label', 'secondary_label', 'sentence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if df['primary_label'].isna().any() or df['secondary_label'].isna().any():
            logger.warning("Found missing values in labels")
        
        logger.info(f"Data validation passed. Shape: {df.shape}")
    
    def _create_output_directory(self, output_dir: Union[str, Path]) -> Path:
        """
        Create output directory if it doesn't exist.
        
        Args:
            output_dir: Path to output directory
            
        Returns:
            Path object of the created directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_path}")
        return output_path
    
    def analyze_joint_frequency(self, df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
        """
        Analyze joint frequency distribution of primary and secondary labels.
        
        Args:
            df: Input DataFrame
            output_path: Output directory path
            
        Returns:
            Joint frequency contingency table
        """
        logger.info("Analyzing joint frequency distribution...")
        
        joint_freq = pd.crosstab(df['primary_label'], df['secondary_label'])
        joint_freq.to_csv(output_path / 'joint_frequency.csv')
        
        logger.info(f"Joint frequency table shape: {joint_freq.shape}")
        return joint_freq
    
    def analyze_conditional_probability(self, joint_freq: pd.DataFrame, output_path: Path) -> pd.DataFrame:
        """
        Calculate conditional probability P(secondary | primary).
        
        Args:
            joint_freq: Joint frequency table
            output_path: Output directory path
            
        Returns:
            Conditional probability matrix
        """
        logger.info("Calculating conditional probabilities...")
        
        cond_prob = joint_freq.div(joint_freq.sum(axis=1), axis=0)
        cond_prob.to_csv(output_path / 'conditional_probability_secondary_given_primary.csv')
        
        return cond_prob
    
    def analyze_label_distributions(self, df: pd.DataFrame, output_path: Path) -> Tuple[pd.Series, pd.Series]:
        """
        Analyze distribution of primary and secondary labels.
        
        Args:
            df: Input DataFrame
            output_path: Output directory path
            
        Returns:
            Tuple of primary and secondary label distributions
        """
        logger.info("Analyzing label distributions...")
        
        primary_dist = df['primary_label'].value_counts(normalize=True)
        secondary_dist = df['secondary_label'].value_counts(normalize=True)
        
        primary_dist.to_csv(output_path / 'primary_label_distribution.csv')
        secondary_dist.to_csv(output_path / 'secondary_label_distribution.csv')
        
        return primary_dist, secondary_dist
    
    def analyze_high_frequency_pairs(self, joint_freq: pd.DataFrame, df: pd.DataFrame, 
                                   output_path: Path) -> pd.DataFrame:
        """
        Identify and analyze high-frequency label pairs.
        
        Args:
            joint_freq: Joint frequency table
            df: Original DataFrame
            output_path: Output directory path
            
        Returns:
            DataFrame with high-frequency pairs and examples
        """
        logger.info(f"Analyzing high-frequency pairs (threshold: {self.freq_threshold})...")
        
        high_freq_pairs = joint_freq.stack().loc[lambda x: x >= self.freq_threshold].sort_values(ascending=False)
        
        examples = []
        for (primary, secondary), count in high_freq_pairs.items():
            sample_df = df[(df['primary_label'] == primary) & (df['secondary_label'] == secondary)]
            if not sample_df.empty:
                sample_text = sample_df['sentence'].sample(1, random_state=self.random_state).values[0]
            else:
                sample_text = "No sample text available."
            examples.append((primary, secondary, count, sample_text))
        
        examples_df = pd.DataFrame(examples, 
                                 columns=['primary_label', 'secondary_label', 'count', 'sample_sentence'])
        examples_df.to_csv(output_path / 'high_frequency_pairs_examples.csv', index=False)
        
        logger.info(f"Found {len(examples)} high-frequency pairs")
        return examples_df
    
    def perform_chi_squared_analysis(self, joint_freq: pd.DataFrame, output_path: Path) -> Dict[str, float]:
        """
        Perform chi-squared test and calculate Cramér's V.
        
        Args:
            joint_freq: Joint frequency table
            output_path: Output directory path
            
        Returns:
            Dictionary with test statistics
        """
        logger.info("Performing chi-squared analysis...")
        
        chi2, p, dof, expected = stats.chi2_contingency(joint_freq)
        n = joint_freq.values.sum()
        phi2 = chi2 / n
        r, k = joint_freq.shape
        cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))
        
        results = {
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'cramers_v': cramers_v
        }
        
        # Save results
        with open(output_path / 'association_stats.txt', 'w', encoding='utf-8') as f:
            f.write(f"Chi-squared statistic: {chi2:.6f}\n")
            f.write(f"p-value: {p:.6f}\n")
            f.write(f"Degrees of freedom: {dof}\n")
            f.write(f"Cramér's V: {cramers_v:.6f}\n")
            f.write(f"Sample size: {n}\n")
            f.write(f"Effect size interpretation:\n")
            if cramers_v < 0.1:
                f.write("  - Negligible association\n")
            elif cramers_v < 0.3:
                f.write("  - Weak association\n")
            elif cramers_v < 0.5:
                f.write("  - Moderate association\n")
            else:
                f.write("  - Strong association\n")
        
        logger.info(f"Chi-squared test: Chi2({dof}) = {chi2:.4f}, p = {p:.6f}")
        logger.info(f"Cramer's V = {cramers_v:.4f}")
        
        return results
    
    def analyze_standardized_residuals(self, joint_freq: pd.DataFrame, expected: np.ndarray, 
                                     output_path: Path) -> pd.DataFrame:
        """
        Calculate and visualize standardized residuals.
        
        Args:
            joint_freq: Observed frequencies
            expected: Expected frequencies
            output_path: Output directory path
            
        Returns:
            Standardized residuals DataFrame
        """
        logger.info("Analyzing standardized residuals...")
        
        std_residuals = (joint_freq - expected) / np.sqrt(expected)
        std_residuals.to_csv(output_path / 'standardized_residuals.csv')
        
        # Create heatmap
        plt.figure(figsize=DEFAULT_FIGSIZE)
        sns.heatmap(std_residuals, cmap="coolwarm", annot=True, fmt=".2f", center=0,
                   cbar_kws={'label': 'Standardized Residuals'})
        plt.title("Standardized Residuals Heatmap\n(Primary vs Secondary Labels)")
        plt.xlabel("Secondary Label")
        plt.ylabel("Primary Label")
        plt.tight_layout()
        plt.savefig(output_path / 'standardized_residuals_heatmap.png', dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        
        return std_residuals
    
    def analyze_mutual_information(self, df: pd.DataFrame, output_path: Path) -> Dict[str, float]:
        """
        Calculate mutual information and normalized mutual information.
        
        Args:
            df: Input DataFrame
            output_path: Output directory path
            
        Returns:
            Dictionary with MI and NMI values
        """
        logger.info("Calculating mutual information...")
        
        mi = mutual_info_score(df['primary_label'], df['secondary_label'])
        nmi = normalized_mutual_info_score(df['primary_label'], df['secondary_label'])
        
        results = {'mi': mi, 'nmi': nmi}
        
        with open(output_path / 'mutual_information.txt', 'w', encoding='utf-8') as f:
            f.write(f"Mutual Information (MI): {mi:.6f}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi:.6f}\n")
            f.write(f"Interpretation:\n")
            f.write(f"  - MI ranges from 0 (independent) to ∞\n")
            f.write(f"  - NMI ranges from 0 (independent) to 1 (perfect association)\n")
        
        logger.info(f"Mutual Information: {mi:.4f}")
        logger.info(f"Normalized Mutual Information: {nmi:.4f}")
        
        return results
    
    def perform_correspondence_analysis(self, joint_freq: pd.DataFrame, output_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Perform correspondence analysis and generate visualizations.
        
        Args:
            joint_freq: Joint frequency table
            output_path: Output directory path
            
        Returns:
            Dictionary with CA results
        """
        logger.info("Performing correspondence analysis...")
        
        ca = prince.CA(n_components=self.ca_components, 
                      n_iter=self.ca_iterations, 
                      engine='sklearn', 
                      random_state=self.random_state)
        ca = ca.fit(joint_freq)
        
        row_coords = ca.row_coordinates(joint_freq)
        col_coords = ca.column_coordinates(joint_freq)
        
        # Save coordinates
        row_coords.to_csv(output_path / 'ca_primary_coordinates.csv')
        col_coords.to_csv(output_path / 'ca_secondary_coordinates.csv')
        
        # Create visualization
        fig, ax = plt.subplots(figsize=DEFAULT_CA_FIGSIZE)
        
        # Plot primary labels
        ax.scatter(row_coords[0], row_coords[1], c='blue', s=100, alpha=0.7, 
                  label='Primary Labels', edgecolors='black', linewidth=0.5)
        for i, txt in enumerate(row_coords.index):
            ax.annotate(txt, (row_coords.iloc[i, 0], row_coords.iloc[i, 1]), 
                       fontsize=10, ha='center', va='bottom')
        
        # Plot secondary labels
        ax.scatter(col_coords[0], col_coords[1], c='red', marker='s', s=100, alpha=0.7,
                  label='Secondary Labels', edgecolors='black', linewidth=0.5)
        for i, txt in enumerate(col_coords.index):
            ax.annotate(txt, (col_coords.iloc[i, 0], col_coords.iloc[i, 1]), 
                       fontsize=10, ha='center', va='top')
        
        # Add reference lines
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
        ax.axvline(0, color='gray', lw=0.8, alpha=0.5)
        
        # Labels and title
        ax.set_xlabel(f'Component 1 ({ca.percentage_of_variance_[0]:.1f}%)')
        ax.set_ylabel(f'Component 2 ({ca.percentage_of_variance_[1]:.1f}%)')
        ax.set_title("Correspondence Analysis (CA) Plot\nLabel Associations")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'correspondence_analysis_plot.png', dpi=DEFAULT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"CA explained variance: {ca.percentage_of_variance_[:2]}")
        
        return {
            'row_coordinates': row_coords,
            'column_coordinates': col_coords,
            'explained_variance': ca.percentage_of_variance_
        }
    
    def run_complete_analysis(self, csv_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            csv_path: Path to input CSV file
            output_dir: Path to output directory
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting analysis for: {csv_path}")
        
        try:
            # Load and validate data
            df = pd.read_csv(csv_path)
            self._validate_data(df)
            
            # Create output directory
            output_path = self._create_output_directory(output_dir)
            
            # Run analyses
            joint_freq = self.analyze_joint_frequency(df, output_path)
            cond_prob = self.analyze_conditional_probability(joint_freq, output_path)
            primary_dist, secondary_dist = self.analyze_label_distributions(df, output_path)
            high_freq_pairs = self.analyze_high_frequency_pairs(joint_freq, df, output_path)
            
            # Statistical tests
            chi2_results = self.perform_chi_squared_analysis(joint_freq, output_path)
            _, _, _, expected = stats.chi2_contingency(joint_freq)
            std_residuals = self.analyze_standardized_residuals(joint_freq, expected, output_path)
            mi_results = self.analyze_mutual_information(df, output_path)
            ca_results = self.perform_correspondence_analysis(joint_freq, output_path)
            
            # Generate summary report
            self._generate_summary_report(df, output_path, {
                'joint_freq': joint_freq,
                'chi2_results': chi2_results,
                'mi_results': mi_results,
                'ca_results': ca_results,
                'high_frequency_pairs': high_freq_pairs
            })
            
            logger.info(f"Analysis completed successfully for: {csv_path}")
            
            return {
                'joint_frequency': joint_freq,
                'conditional_probability': cond_prob,
                'primary_distribution': primary_dist,
                'secondary_distribution': secondary_dist,
                'high_frequency_pairs': high_freq_pairs,
                'chi2_results': chi2_results,
                'standardized_residuals': std_residuals,
                'mutual_information': mi_results,
                'correspondence_analysis': ca_results
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {csv_path}: {str(e)}")
            raise
    
    def _generate_summary_report(self, df: pd.DataFrame, output_path: Path, results: Dict) -> None:
        """
        Generate comprehensive summary report.
        
        Args:
            df: Input DataFrame
            output_path: Output directory path
            results: Analysis results
        """
        logger.info("Generating summary report...")
        
        with open(output_path / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LABEL ASSOCIATION ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples: {len(df):,}\n")
            f.write(f"Primary labels: {df['primary_label'].nunique()}\n")
            f.write(f"Secondary labels: {df['secondary_label'].nunique()}\n")
            f.write(f"Unique label combinations: {df.groupby(['primary_label', 'secondary_label']).size().shape[0]}\n\n")
            
            f.write("STATISTICAL TESTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Chi-squared test: Chi2({results['chi2_results']['dof']}) = {results['chi2_results']['chi2']:.4f}\n")
            f.write(f"p-value: {results['chi2_results']['p_value']:.6f}\n")
            f.write(f"Cramér's V: {results['chi2_results']['cramers_v']:.4f}\n")
            f.write(f"Mutual Information: {results['mi_results']['mi']:.4f}\n")
            f.write(f"Normalized MI: {results['mi_results']['nmi']:.4f}\n\n")
            
            f.write("CORRESPONDENCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Explained inertia (Component 1): {results['ca_results']['explained_variance'][0]:.2f}%\n")
            f.write(f"Explained inertia (Component 2): {results['ca_results']['explained_variance'][1]:.2f}%\n")
            f.write(f"Total explained inertia: {sum(results['ca_results']['explained_variance'][:2]):.2f}%\n\n")
            
            f.write("HIGH-FREQUENCY PAIRS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Pairs above threshold ({self.freq_threshold}): {len(results['high_frequency_pairs'])}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Analysis completed successfully.\n")
            f.write("=" * 80 + "\n")


def main():
    """Main execution function."""
    # Configuration
    files = {
        "by_perspective_jesus": "output/labeled/by_perspective_jesus_labeled.csv",
        "by_perspective_audience": "output/labeled/by_perspective_audience_filtered_labeled.csv"
    }
    
    # Initialize analyzer
    analyzer = LabelAssociationAnalyzer(
        freq_threshold=DEFAULT_FREQ_THRESHOLD,
        random_state=DEFAULT_RANDOM_STATE,
        ca_components=DEFAULT_CA_COMPONENTS,
        ca_iterations=DEFAULT_CA_ITERATIONS
    )
    
    # Run analysis for each dataset
    for name, path in files.items():
        try:
            logger.info(f"Starting analysis for {name}...")
            out_dir = f"output_stats/{name}"
            
            results = analyzer.run_complete_analysis(path, out_dir)
            
            logger.info(f"Analysis completed for {name}")
            logger.info(f"Results saved to: {out_dir}")
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {path}")
        except Exception as e:
            logger.error(f"Analysis failed for {name}: {str(e)}")
            continue
    
    logger.info("All analyses completed.")


if __name__ == "__main__":
    main()
