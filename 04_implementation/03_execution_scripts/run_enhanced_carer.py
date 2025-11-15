#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced CARER + 8emos Emotion Analysis Runner

This script demonstrates the enhanced CARER emotion analysis system with 8emos feature
enhancement and religious text-specific patterns for Ask_Jesus religious text analysis.

Author: [Your Name]
Date: [Current Date]
Version: 2.0.0
License: MIT
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import the enhanced CARER analyzer
from carer_emotion_analysis import CAREREmotionAnalyzer
from carer_config import get_config, validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'enhanced_carer_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_sample_texts():
    """Load comprehensive sample religious texts for analysis."""
    sample_texts = [
        # Joy examples (15 texts)
        "Rejoice in the Lord always! I will say it again: Rejoice! God's love brings us eternal joy and happiness.",
        "Blessed are those who trust in the Lord. His grace is sufficient for all our needs and brings us peace.",
        "Praise God from whom all blessings flow! We celebrate His goodness and mercy every day.",
        "My heart is filled with joy as I witness the miracles of God in my life. Every day is a gift from above.",
        "The Lord has turned my mourning into dancing! I am overwhelmed with gratitude for His faithfulness.",
        "What a wonderful day to serve the Lord! His love fills my heart with unspeakable joy and contentment.",
        "I am blessed beyond measure by God's abundant grace. His mercies are new every morning.",
        "The joy of the Lord is my strength! I find my happiness in serving Him and others.",
        "God's love has transformed my life completely. I am filled with joy and hope for the future.",
        "Every blessing I receive comes from above. I am grateful for God's constant provision and care.",
        "The Lord has answered my prayers in ways I never expected. My heart is overflowing with joy!",
        "I find my greatest happiness in walking with God and following His ways. His joy is my reward.",
        "God's faithfulness throughout my life fills me with joy and confidence. He never fails!",
        "I am blessed to be a child of God. His love brings me endless joy and peace.",
        "The Lord has given me a heart of gratitude. Every day I find new reasons to rejoice in Him.",
        
        # Sadness examples (15 texts)
        "My soul is overwhelmed with sorrow to the point of death. I feel lost and abandoned in this world.",
        "I mourn for the brokenness of humanity. The pain and suffering seem endless and hopeless.",
        "My heart is heavy with grief. I weep for those who have lost their way and need comfort.",
        "I am drowning in sorrow and despair. The weight of my sins feels too heavy to bear.",
        "My soul cries out in anguish. I feel so alone and forgotten by everyone, including God.",
        "The pain in my heart is unbearable. I don't know how to go on without the one I loved.",
        "I am consumed by sadness and regret. The mistakes of my past haunt me every day.",
        "My spirit is crushed and broken. I have lost all hope and faith in everything.",
        "I weep bitterly for the suffering I see around me. The world is filled with so much pain.",
        "My heart aches with loneliness and isolation. I feel like no one understands my pain.",
        "I am overwhelmed by grief and loss. Everything I once held dear has been taken from me.",
        "My soul is in deep distress. I cannot find peace or comfort in anything I do.",
        "I am broken and shattered by life's cruel blows. My faith is hanging by a thread.",
        "The darkness of depression surrounds me. I cannot see any light or hope ahead.",
        "I am drowning in a sea of sorrow. Every day feels like a struggle just to survive.",
        
        # Anger examples (15 texts)
        "I am filled with righteous anger at the injustice in this world. The wicked must be held accountable.",
        "How dare they mock the sacred teachings! This blasphemy cannot go unpunished.",
        "I am outraged by the corruption and evil that surrounds us. Justice must prevail!",
        "My blood boils when I see the innocent suffering while the guilty go free.",
        "I am furious at the way people twist God's word for their own selfish purposes.",
        "The hypocrisy of those who claim to be righteous fills me with righteous indignation.",
        "I am enraged by the systematic oppression of the poor and vulnerable in our society.",
        "How can anyone justify such cruelty and heartlessness? It makes me sick with anger.",
        "I am burning with anger at the lies and deception being spread in God's name.",
        "The injustice I witness every day fills me with a fury that cannot be contained.",
        "I am outraged by the way people use religion to justify hatred and violence.",
        "My anger burns hot against those who exploit the weak and defenseless.",
        "I am furious at the corruption that has infiltrated every level of our institutions.",
        "The way people mock and ridicule faith makes my blood boil with righteous anger.",
        "I am consumed by anger at the evil that seems to be winning in this world.",
        
        # Fear examples (15 texts)
        "I tremble with fear at the thought of God's judgment. The day of reckoning approaches.",
        "My heart is filled with dread and anxiety. I fear what the future may bring.",
        "I am terrified by the darkness that threatens to consume us all. We need divine protection.",
        "The uncertainty of tomorrow fills me with paralyzing fear and anxiety.",
        "I am afraid that I am not good enough for God's love and forgiveness.",
        "The thought of losing my faith terrifies me beyond words. What if I'm wrong?",
        "I am gripped by fear of the unknown. The future seems so dark and frightening.",
        "My heart races with fear when I think about the consequences of my actions.",
        "I am terrified of being alone and abandoned in this cruel world.",
        "The fear of death and what comes after haunts my every waking moment.",
        "I am paralyzed by fear of failure and rejection from those I love.",
        "The darkness of evil in this world fills me with terror and dread.",
        "I am afraid that my prayers are not being heard or answered by God.",
        "The fear of losing everything I hold dear keeps me awake at night.",
        "I am consumed by fear of the judgment and condemnation of others.",
        
        # Surprise examples (15 texts)
        "I am amazed and astonished by this miraculous revelation! God has shown us something extraordinary.",
        "What a wonderful surprise! The Lord has answered our prayers in ways we never expected.",
        "I am stunned by this divine manifestation. The heavens have opened to reveal God's glory.",
        "I never expected God to work in such a miraculous way! This is beyond my wildest dreams.",
        "What an incredible surprise! The Lord has turned my greatest trial into my greatest blessing.",
        "I am astonished by the way God has orchestrated events in my life. It's truly miraculous!",
        "I am shocked and amazed by the sudden answer to my prayers. God is so good!",
        "What a wonderful surprise! God has opened doors I never thought possible.",
        "I am stunned by the unexpected blessing that has come into my life. Thank you, Lord!",
        "I never saw this coming! God has surprised me with His amazing grace and mercy.",
        "What an incredible revelation! The Lord has shown me things I never understood before.",
        "I am amazed by the miraculous way God has provided for my needs. He is faithful!",
        "I am astonished by the sudden change in circumstances. God works in mysterious ways!",
        "What a wonderful surprise! The Lord has given me exactly what I needed, not what I wanted.",
        "I am stunned by the unexpected joy that has come from my greatest sorrow. God is amazing!",
        
        # Disgust examples (15 texts)
        "I abhor the sin and corruption that plagues our society. This filth must be cleansed.",
        "The wickedness and depravity I see fills me with disgust and revulsion.",
        "I detest the impurity and defilement that surrounds us. We must remain pure and holy.",
        "I am sickened by the moral decay and degradation I witness every day.",
        "The corruption and evil in this world fills me with utter disgust and contempt.",
        "I abhor the way people twist and pervert God's holy word for their own gain.",
        "The depravity and wickedness I see makes me physically ill with disgust.",
        "I am repulsed by the hypocrisy and false piety of those who claim to be righteous.",
        "The moral filth and corruption in our society fills me with righteous disgust.",
        "I am sickened by the way people use religion to justify their evil actions.",
        "The impurity and defilement I witness fills me with revulsion and contempt.",
        "I abhor the way people mock and ridicule what is sacred and holy.",
        "The corruption and decay in our institutions fills me with disgust and anger.",
        "I am repulsed by the moral bankruptcy and spiritual emptiness I see around me.",
        "The evil and wickedness in this world fills me with righteous disgust and outrage.",
        
        # Anticipation examples (15 texts)
        "I eagerly await the coming of the Lord. His return is near and we must be prepared.",
        "I hope and long for the day when all things will be made new. The promise is coming soon.",
        "I expect great things from God. He has promised to fulfill His word and I trust in Him.",
        "I am filled with anticipation for the wonderful plans God has for my future.",
        "I eagerly look forward to the day when God will wipe away every tear from our eyes.",
        "I am excited about the amazing things God is going to do in my life and ministry.",
        "I anticipate with joy the fulfillment of God's promises in my life.",
        "I am looking forward to the day when all things will be restored and made whole.",
        "I eagerly await the manifestation of God's glory and power in our midst.",
        "I am filled with hope and anticipation for the great things ahead.",
        "I look forward with excitement to the day when God's kingdom will come in full.",
        "I am anticipating the wonderful ways God will use me for His glory.",
        "I eagerly await the day when all suffering and pain will come to an end.",
        "I am excited about the future God has planned for me and my family.",
        "I look forward with great anticipation to the day when faith will become sight.",
        
        # Trust examples (15 texts)
        "I have complete faith and trust in God's plan. He is faithful and reliable in all things.",
        "I rely on the Lord with unwavering confidence. His promises are steadfast and true.",
        "I depend on God's wisdom and guidance. He alone is worthy of our complete trust.",
        "I trust God completely with my life and future. He has never failed me.",
        "I have unwavering faith in God's goodness and love. He is always trustworthy.",
        "I rely on God's strength and power. He is my rock and my fortress.",
        "I trust in God's perfect timing and plan. He knows what is best for me.",
        "I have complete confidence in God's word and promises. He is faithful to fulfill them.",
        "I rely on God's mercy and grace. He is always ready to forgive and restore.",
        "I trust God with my deepest fears and concerns. He cares for me deeply.",
        "I have faith in God's ability to work all things for good. He is sovereign.",
        "I rely on God's protection and provision. He is my shield and my provider.",
        "I trust in God's justice and righteousness. He will make all things right.",
        "I have confidence in God's love and faithfulness. He will never abandon me.",
        "I rely on God's wisdom and understanding. He knows the end from the beginning.",
        
        # Neutral examples (15 texts)
        "The Bible teaches us important principles about living a righteous life.",
        "Scripture provides guidance for making wise decisions in difficult situations.",
        "Religious texts offer valuable insights into human nature and behavior.",
        "The teachings of Jesus provide a framework for ethical decision-making.",
        "Biblical wisdom can be applied to many aspects of modern life.",
        "Religious principles offer a foundation for building strong relationships.",
        "The word of God contains timeless truths that apply to every generation.",
        "Scriptural teachings provide guidance for navigating life's challenges.",
        "Biblical principles offer a moral compass for personal conduct.",
        "Religious wisdom can help us understand the deeper meaning of life.",
        "The teachings of scripture provide a basis for ethical behavior.",
        "Biblical guidance can help us make wise choices in complex situations.",
        "Religious principles offer a framework for understanding human relationships.",
        "Scriptural wisdom provides insights into the nature of good and evil.",
        "The word of God offers practical advice for daily living and decision-making."
    ]
    
    return sample_texts

def run_enhanced_analysis():
    """Run the enhanced CARER + 8emos analysis."""
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        validate_config()
        config = get_config()
        logger.info("Configuration validation passed!")
        
        # Initialize enhanced analyzer
        logger.info("Initializing Enhanced CARER + 8emos Analyzer...")
        analyzer = CAREREmotionAnalyzer(
            emotion_model=config['model']['emotion_model'],
            max_length=config['model']['max_length'],
            batch_size=config['model']['batch_size'],
            random_state=config['model']['random_state']
        )
        
        # Load sample texts
        logger.info("Loading sample religious texts...")
        sample_texts = load_sample_texts()
        logger.info(f"Loaded {len(sample_texts)} sample texts")
        
        # Run enhanced emotion analysis
        logger.info("Running enhanced emotion analysis with 8emos features...")
        emotion_results = analyzer.analyze_emotions(sample_texts)
        
        # Display results summary
        logger.info("=== Enhanced Emotion Analysis Results ===")
        logger.info(f"Total texts analyzed: {len(emotion_results)}")
        
        # Show emotion distribution
        emotion_counts = emotion_results['primary_emotion'].value_counts()
        logger.info("\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(emotion_results)) * 100
            logger.info(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Show enhancement statistics
        if 'enhancement_factor' in emotion_results.columns:
            avg_enhancement = emotion_results['enhancement_factor'].mean()
            max_enhancement = emotion_results['enhancement_factor'].max()
            min_enhancement = emotion_results['enhancement_factor'].min()
            
            logger.info(f"\nEnhancement Statistics:")
            logger.info(f"  Average enhancement factor: {avg_enhancement:.3f}")
            logger.info(f"  Maximum enhancement factor: {max_enhancement:.3f}")
            logger.info(f"  Minimum enhancement factor: {min_enhancement:.3f}")
            
            # Show high enhancement examples
            high_enhancement = emotion_results[emotion_results['enhancement_factor'] > 1.5]
            if len(high_enhancement) > 0:
                logger.info(f"\nHigh Enhancement Examples (factor > 1.5): {len(high_enhancement)} texts")
                for _, row in high_enhancement.head(3).iterrows():
                    logger.info(f"  Text {row['text_id']}: {row['enhancement_factor']:.3f} - {row['primary_emotion']}")
        
        # Analyze 8emos features
        logger.info("\nAnalyzing 8emos feature effectiveness...")
        eightemos_results = analyzer.analyze_8emos_features()
        
        # Show feature impact
        if 'feature_impact' in eightemos_results:
            logger.info("\n8emos Feature Impact by Emotion:")
            for emotion, impact in eightemos_results['feature_impact'].items():
                improvement = impact.get('improvement_percentage', 0)
                logger.info(f"  {emotion}: {improvement:+.1f}% improvement")
        
        # Show religious pattern analysis
        if 'religious_pattern_analysis' in eightemos_results:
            logger.info("\nReligious Pattern Effectiveness:")
            for emotion, analysis in eightemos_results['religious_pattern_analysis'].items():
                match_pct = analysis.get('match_percentage', 0)
                total_matches = analysis.get('total_matches', 0)
                logger.info(f"  {emotion}: {match_pct:.1f}% ({total_matches} matches)")
        
        # Create output directory
        output_dir = Path(config['paths']['output_results'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        logger.info("\nCreating enhanced visualizations...")
        analyzer.create_visualizations(str(output_dir))
        
        # Generate comprehensive report
        logger.info("Generating comprehensive analysis report...")
        analyzer.generate_report(str(output_dir / "enhanced_carer_report.txt"))
        
        # Save detailed results
        results_file = output_dir / "enhanced_emotion_results.csv"
        emotion_results.to_csv(results_file, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save 8emos analysis results
        import json
        eightemos_file = output_dir / "8emos_analysis_results.json"
        with open(eightemos_file, 'w', encoding='utf-8') as f:
            json.dump(eightemos_results, f, indent=2, ensure_ascii=False)
        logger.info(f"8emos analysis results saved to: {eightemos_file}")
        
        logger.info("\n=== Enhanced CARER + 8emos Analysis Completed Successfully ===")
        logger.info(f"Output directory: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function."""
    logger.info("Starting Enhanced CARER + 8emos Emotion Analysis System")
    logger.info("=" * 60)
    
    # Check if required packages are available
    try:
        import transformers
        import torch
        import plotly
        import wordcloud
        logger.info("All required packages are available")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.info("Please install required packages: pip install transformers torch plotly wordcloud")
        return False
    
    # Run the enhanced analysis
    success = run_enhanced_analysis()
    
    if success:
        logger.info("\n🎉 Enhanced CARER + 8emos analysis completed successfully!")
        logger.info("📊 Check the output directory for results and visualizations")
        logger.info("📈 The system now includes:")
        logger.info("   - 8emos emotion feature enhancement")
        logger.info("   - Religious text pattern recognition")
        logger.info("   - Enhanced authority construction analysis")
        logger.info("   - Comprehensive 8emos feature analysis")
    else:
        logger.error("\n❌ Enhanced analysis failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    main()
