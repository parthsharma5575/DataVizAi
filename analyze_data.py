#!/usr/bin/env python3
"""
Direct analysis script for data exfiltration CSV file
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from data_processor import DataProcessor
from gemini_service import GeminiService
from report_generator import ReportGenerator

def main():
    # Initialize processors
    data_processor = DataProcessor()
    gemini_service = GeminiService()
    report_generator = ReportGenerator()
    
    # Load data
    print("Loading data exfiltration CSV file...")
    filepath = "data_exfiltration.csv"
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return
    
    try:
        # Load the data
        df = data_processor.load_data(filepath)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Analyze columns
        print("\nAnalyzing column structure...")
        columns_info = data_processor.analyze_columns(df)
        
        for col_name, info in columns_info.items():
            print(f"  {col_name}: {info['dtype']} ({info['null_count']} missing, {info['unique_count']} unique)")
        
        # Data cleaning configuration
        cleaning_config = {
            'imputation_method': 'median',
            'outlier_method': 'iqr',
            'validation_rules': ['no_negative_values', 'data_consistency'],
            'apply_weights': False,
            'weight_column': None
        }
        
        print(f"\nApplying data cleaning with config: {cleaning_config}")
        
        # Handle missing values
        if cleaning_config['imputation_method'] != 'none':
            print("Handling missing values...")
            df = data_processor.handle_missing_values(df, method=cleaning_config['imputation_method'])
        
        # Detect outliers
        if cleaning_config['outlier_method'] != 'none':
            print("Detecting and handling outliers...")
            df = data_processor.detect_and_handle_outliers(df, method=cleaning_config['outlier_method'])
        
        # Apply validation rules
        print("Applying validation rules...")
        validation_results = data_processor.apply_validation_rules(df, cleaning_config['validation_rules'])
        
        # Calculate summary statistics
        print("Calculating summary statistics...")
        summary_stats = data_processor.calculate_summary_statistics(df)
        
        # Get AI insights
        print("Generating AI insights...")
        try:
            ai_insights = gemini_service.analyze_survey_data(df, summary_stats)
            print("AI insights generated successfully!")
        except Exception as ai_error:
            print(f"AI insights generation failed: {ai_error}")
            ai_insights = {
                'summary': 'AI analysis unavailable due to API limitations',
                'key_findings': ['Data contains network traffic patterns', 'Multiple file types detected'],
                'recommendations': ['Review data sources', 'Consider security implications']
            }
        
        # Prepare results
        results_data = {
            'summary_stats': summary_stats,
            'validation_results': validation_results,
            'ai_insights': ai_insights,
            'cleaning_config': cleaning_config,
            'original_shape': df.shape,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        output_file = "analysis_results.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_safe_results = {}
            for key, value in results_data.items():
                if key == 'summary_stats':
                    json_safe_results[key] = {
                        col: {
                            stat_name: float(stat_value) if hasattr(stat_value, 'dtype') else stat_value
                            for stat_name, stat_value in stats.items()
                        }
                        for col, stats in value.items()
                    }
                else:
                    json_safe_results[key] = value
            
            json.dump(json_safe_results, f, indent=2, default=str)
        
        print(f"\nAnalysis results saved to: {output_file}")
        
        # Generate PDF report
        print("Generating PDF report...")
        try:
            # Create a mock session object for PDF generation
            class MockSession:
                def __init__(self):
                    self.original_filename = 'data_exfiltration.csv'
                    self.total_rows = df.shape[0]
                    self.total_columns = df.shape[1]
                    self.file_size = os.path.getsize(filepath)
                    self.status = 'completed'
                    self.missing_values_count = int(df.isnull().sum().sum())
                    self.outliers_detected = 0
            
            report_data = {
                'session': MockSession(),
                'results': results_data,
                'processing_logs': [{
                    'step_name': 'data_processing',
                    'step_status': 'completed',
                    'start_time': datetime.now(),
                    'end_time': datetime.now(),
                    'log_message': 'Data processing completed successfully'
                }]
            }
            
            pdf_filename = report_generator.generate_pdf_report(report_data, session_id=1)
            print(f"PDF report generated: {pdf_filename}")
            
        except Exception as pdf_error:
            print(f"PDF generation failed: {pdf_error}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"File: data_exfiltration.csv")
        print(f"Records: {df.shape[0]:,}")
        print(f"Variables: {df.shape[1]}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        print(f"\nNumeric columns:")
        for col in df.select_dtypes(include=['number']).columns:
            print(f"  - {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
        
        print(f"\nCategorical columns:")
        for col in df.select_dtypes(include=['object']).columns:
            print(f"  - {col}: {df[col].nunique()} unique values")
        
        if validation_results:
            print(f"\nValidation Results:")
            print(f"  - Passed: {len(validation_results.get('passed', []))}")
            print(f"  - Warnings: {len(validation_results.get('warnings', []))}")
            print(f"  - Failed: {len(validation_results.get('failed', []))}")
        
        print(f"\nFiles generated:")
        print(f"  - analysis_results.json (detailed results)")
        if 'pdf_filename' in locals():
            print(f"  - {pdf_filename} (PDF report)")
        
        print(f"\n{'='*60}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()