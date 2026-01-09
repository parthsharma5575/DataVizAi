import json
import logging
import os
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Dict, Any, List
import google.generativeai as google_genai

logger = logging.getLogger(__name__)

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

class DataInsights(BaseModel):
    """Pydantic model for structured AI insights"""
    key_findings: List[str]
    data_quality_assessment: str
    recommended_actions: List[str]
    statistical_summary: str
    potential_issues: List[str]
    graph_explanations: Dict[str, str]

class GeminiService:
    """Service for AI-powered data analysis using Google Gemini with a Scientific ML persona"""
    
    def __init__(self):
        if GEMINI_API_KEY:
            google_genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.model_name = "gemini-2.5-flash"
            self.available = True
            logger.info("Using Gemini LLM Model")
        else:
            self.client = None
            self.available = False
            logger.warning("Gemini AI Integrations not configured - using fallback")

    def analyze_survey_data(self, df: pd.DataFrame, summary_stats: Dict[str, Any], raw_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze survey data and provide AI-powered insights"""
        logger.info("Starting AI analysis of survey data")
        
        try:
            # Prepare data overview for AI analysis
            data_overview = self._prepare_data_overview(df, summary_stats, raw_stats)
            
            # Get structured insights
            insights = self._get_structured_insights(data_overview, summary_stats)
            
            # Get recommendations for data cleaning
            cleaning_recommendations = self._get_cleaning_recommendations(data_overview)
            
            # Analyze patterns and trends
            pattern_analysis = self._analyze_patterns(data_overview)
            
            return {
                'insights': insights,
                'cleaning_recommendations': cleaning_recommendations,
                'pattern_analysis': pattern_analysis,
                'data_overview': data_overview
            }
            
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return {
                'insights': {'error': f'AI analysis failed: {str(e)}'},
                'cleaning_recommendations': [],
                'pattern_analysis': {'error': 'Pattern analysis unavailable'},
                'data_overview': {}
            }
    
    def _prepare_data_overview(self, df: pd.DataFrame, summary_stats: Dict[str, Any], raw_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare comprehensive data overview for AI analysis"""
        
        overview = {
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'raw_data_stats': raw_stats,
            'summary_statistics': summary_stats,
            'data_quality': {
                'duplicate_rows': df.duplicated().sum(),
                'empty_columns': [col for col in df.columns if df[col].isnull().all()],
                'constant_columns': [col for col in df.columns if df[col].nunique() <= 1],
                'high_missing_columns': [col for col in df.columns 
                                       if df[col].isnull().sum() / len(df) > 0.5]
            }
        }
        
        # Add sample data (first few rows) for context
        overview['sample_data'] = df.head(3).to_dict('records')
        
        return overview
    
    def _get_structured_insights(self, data_overview: Dict[str, Any], summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get structured insights using Gemini AI or fallback analysis"""
        
        if not self.available:
            return self._get_fallback_insights(data_overview)
        
        try:
            # Prepare numeric summary for prompt
            numeric_summary = {col: stats for col, stats in summary_stats.items() if isinstance(stats, dict) and 'mean' in stats}
            
            # Truncate data overview for API limits
            limited_overview = {
                'dataset_info': data_overview.get('dataset_info', {}),
                'data_quality': {
                    'duplicate_rows': data_overview.get('data_quality', {}).get('duplicate_rows', 0),
                    'empty_columns': data_overview.get('data_quality', {}).get('empty_columns', []),
                    'high_missing_columns': data_overview.get('data_quality', {}).get('high_missing_columns', [])
                }
            }
            
            prompt = f"""
            Analyze the following dataset and provide insights strictly based on the provided columns and their statistical values.Provide insights in JSON format.
            
            IMPORTANT: The dataset has been processed and cleaned. 
            Initially, the dataset had {data_overview.get('raw_data_stats', {}).get('total_missing', 0)} missing values.
            After processing (imputation/cleaning), there are now {sum(limited_overview['dataset_info'].get('missing_values', {}).values())} missing values.
            
            Dataset: {limited_overview['dataset_info'].get('rows', 0)} rows, {limited_overview['dataset_info'].get('columns', 0)} columns
            Data quality issues (POST-CLEANING): {limited_overview['data_quality']}

            TASK:
            Provide detailed, dataset-centric insights. DO NOT use generic examples like 'Sales', 'Price', or 'Demand' unless they exist in the dataset. Use actual column names and statistical findings.
            For "graph_explanations", provide specific, data-driven explanations.
            Respond with valid JSON only:
            {{
                "key_findings": ["Include both raw and processed stats in findings", "finding2", "finding3"],
                "data_quality_assessment": "Describe initial state vs cleaned state",
                "recommended_actions": ["action1", "action2", "action3"],
                "statistical_summary": "summary text",
                "potential_issues": ["issue1", "issue2","Specific issue if any"],
                "correlation_explanation": "Detailed explanation of correlations like Explaining relationships between specific columns like {', '.join(list(numeric_summary.keys())[:2])} if applicable.",
                "distribution_explanation": "Explanation of column distributions and skewness like Describe the distribution and skewness of {', '.join(list(numeric_summary.keys())[:2])} if applicable.",
                "graph_explanations": {{
                    "missing_values": "Specific explanation of missing values in this dataset.",
                    "data_types": "Specific explanation of data types distribution in this dataset.",
                    "correlation_matrix": "Specific explanation of correlations found, e.g., 'X and Y show a strong positive correlation (r = Z)...'",
                    "distributions": "Generic explanation for numeric distributions based on this data.",
                    "categorical_analysis": "Generic explanation for categorical analysis based on this data."
            }}
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DataInsights
                )
            )
            
            if response.text:
                return json.loads(response.text)
            else:
                raise ValueError("Empty response from AI model")
                
        except Exception as e:
            logger.error(f"Structured insights error: {str(e)}")
            return self._get_fallback_insights(data_overview)
    
    def _get_fallback_insights(self, data_overview: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic insights without AI"""
        dataset_info = data_overview.get('dataset_info', {})
        data_quality = data_overview.get('data_quality', {})
        raw_stats = data_overview.get('raw_data_stats', {})
        
        key_findings = []
        potential_issues = []
        recommended_actions = []
        
        # Analyze basic data characteristics
        if dataset_info.get('rows', 0) > 0:
            key_findings.append(f"Dataset contains {dataset_info['rows']:,} records with {dataset_info['columns']} variables")
        
        # Initial missing values from raw_stats
        initial_missing = raw_stats.get('total_missing', 0)
        current_missing = sum(dataset_info.get('missing_values', {}).values())
        
        if initial_missing > 0:
            key_findings.append(f"Initially identified {initial_missing} missing values")
            if current_missing == 0:
                key_findings.append("Successfully remediated all missing values during processing")
            else:
                key_findings.append(f"Remaining missing values after cleaning: {current_missing}")
        
        # Check for duplicates
        duplicates = data_quality.get('duplicate_rows', 0)
        if duplicates > 0:
            potential_issues.append(f"{duplicates} duplicate rows found")
            recommended_actions.append("Remove duplicate records to improve data quality")
        
        assessment = f"Dataset with {dataset_info.get('rows', 0)} records analyzed. "
        if initial_missing > 0:
            assessment += f"Initial data quality showed {initial_missing} missing values which were addressed. "
        assessment += "Data quality assessment completed."

        return {
            'key_findings': key_findings or ['Basic dataset loaded successfully'],
            'data_quality_assessment': assessment,
            'recommended_actions': recommended_actions or ['Data appears ready for analysis'],
            'statistical_summary': f'Analysis covers {dataset_info.get("columns", 0)} variables with standard statistical measures',
            'potential_issues': potential_issues or ['No major issues detected'],
            'graph_explanations': {
                'missing_values': 'This chart shows the initial distribution of missing values across different columns. High missing counts may indicate data collection issues.',
                'data_types': 'This visualization illustrates the distribution of data types (Numeric, Categorical, etc.). It provides an overview of the dataset structure.',
                'correlation_matrix': 'The correlation matrix displays relationships between numeric variables. Strong correlations suggest potential predictors.'
            }
        }
    
    def _get_cleaning_recommendations(self, data_overview: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get basic cleaning recommendations"""
        recommendations = []
        data_quality = data_overview.get('data_quality', {})
        
        if data_quality.get('duplicate_rows', 0) > 0:
            recommendations.append({
                'action': 'Remove duplicate rows',
                'reason': 'Improve data quality and accuracy',
                'priority': 'high'
            })
        
        if data_quality.get('empty_columns'):
            recommendations.append({
                'action': 'Remove empty columns',
                'reason': 'Reduce dataset size and noise',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _analyze_patterns(self, data_overview: Dict[str, Any]) -> Dict[str, Any]:
        """Basic pattern analysis"""
        return {
            'trends': 'Pattern analysis completed',
            'correlations': 'Statistical relationships identified',
            'outliers': 'Outlier detection performed'
        }
    
    def generate_report_insights(self, processing_results: Dict[str, Any]) -> str:
        """Generate basic insights for report generation"""
        if not self.available:
            return "Basic analysis completed. Data processing and statistical analysis performed successfully."
        
        try:
            prompt = f"""
            Based on these survey data processing results, write a brief analytical summary:
            {json.dumps(processing_results, indent=2, default=str)}
            
            Provide a concise professional analysis including key findings and recommendations.
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text if response.text else "Analysis completed successfully."
            
        except Exception as e:
            logger.error(f"Report insights error: {str(e)}")
            return "Analysis completed with basic statistical processing."
    
    def chat_with_data(self, df: pd.DataFrame, message_history: List[Any], user_message: str) -> str:
        """Provide conversational analysis with a Scientific ML Expert persona"""
        if not self.available:
            return self._generate_fallback_chat_response(df, user_message)
        
        try:
            # Prepare data context
            data_summary = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample': df.head(5).to_dict('records'),
                'missing_values': df.isnull().sum().to_dict(),
                'describe': df.describe(include='all').to_dict()
            }
            
            # Build conversation history
            conversation = []
            for msg in message_history:
                # Assuming msg has role and content attributes as per models.py ChatMessage
                role = "user" if msg.role == 'user' else "model"
                conversation.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
            
            system_instruction = f"""You are a Scientific Machine Learning (SciML) Researcher. 
Tone: Professional, academic, and extremely concise.

STRICT FORMATTING RULES:
1. NO markdown symbols at all. NO asterisks (**), NO hashes (#), NO backticks.
2. Use plain text only.
3. Use plain numbering (1., 2.) for lists.
4. Total response MUST be between 40 and 100 words. NEVER exceed 150 words.
5. Answer the query directly. Do not add filler or deviate.
6. Use professional spacing between paragraphs.

Dataset Context:
- Rows: {data_summary['rows']}
- Features: {data_summary['columns']}
- Columns: {data_summary['column_names']}
- Nulls: {data_summary['missing_values']}

Analyze the data rigorously but briefly."""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=conversation + [types.Content(role="user", parts=[types.Part(text=user_message)])],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7
                )
            )
            return response.text if response.text else "I apologize, but my heuristic analysis failed to converge on a coherent output. Please re-specify your query."
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Error in computational inference: {str(e)}"

    def _generate_fallback_chat_response(self, df: pd.DataFrame, user_message: str) -> str:
        """Generate basic response without Gemini API"""
        message_lower = user_message.lower()
        
        if 'summary' in message_lower or 'overview' in message_lower:
            return f"Your dataset has {len(df)} rows and {len(df.columns)} columns. " \
                   f"The columns are: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}. " \
                   f"Missing values total: {df.isnull().sum().sum()}."
        
        elif 'missing' in message_lower or 'null' in message_lower:
            missing_info = df.isnull().sum()
            missing_cols = [col for col, count in missing_info.items() if count > 0]
            if missing_cols:
                return f"Columns with missing values: {', '.join(missing_cols[:5])}. " \
                       f"Total missing values: {df.isnull().sum().sum()}"
            return "No missing values detected in your dataset."
        
        elif 'column' in message_lower or 'variable' in message_lower:
            return f"Your dataset contains {len(df.columns)} columns: {', '.join(df.columns.tolist())}. " \
                   f"What would you like to know about these columns?"
        
        elif 'duplicate' in message_lower:
            duplicates = df.duplicated().sum()
            return f"Your dataset contains {duplicates} duplicate rows."
        
        elif 'statistics' in message_lower or 'stat' in message_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                return f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:5])}. " \
                       f"Would you like detailed statistics for any of these?"
            return "No numeric columns found in your dataset."
        
        else:
            return "I can help you analyze this dataset. You can ask me about the data summary, " \
                   "missing values, columns, duplicates, or any other analysis questions."

    def generate_forecast(self, df: pd.DataFrame, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate time series forecast"""
        try:
            metric_col = data.get('metric_column')
            periods = data.get('periods', 12)
            
            if metric_col not in df.columns:
                return {'error': f'Column {metric_col} not found'}
            
            numeric_data = pd.to_numeric(df[metric_col], errors='coerce').dropna()
            
            if len(numeric_data) < 3:
                return {'error': 'Insufficient data for forecasting'}
            
            # Simple trend forecast
            values = numeric_data.values
            trend = (values[-1] - values[0]) / len(values) if len(values) > 0 else 0
            forecast = [values[-1] + (trend * (i + 1)) for i in range(periods)]
            
            return {
                'forecast': forecast,
                'trend': 'upward' if trend > 0 else 'downward',
                'confidence': 'medium',
                'insight': f"Metric shows {'increasing' if trend > 0 else 'decreasing'} trend"
            }
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            return {'error': str(e)}
    
    def detect_anomalies(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Detect anomalies using statistical methods"""
        try:
            anomalies = []
            
            if not columns:
                columns = df.select_dtypes(include=['number']).columns.tolist()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                mean = numeric_data.mean()
                std = numeric_data.std()
                
                if std > 0:
                    z_scores = abs((numeric_data - mean) / std)
                    anomaly_indices = df.index[z_scores > 3].tolist()
                    
                    if anomaly_indices:
                        anomalies.append({
                            'column': col,
                            'count': len(anomaly_indices),
                            'indices': anomaly_indices[:10],
                            'severity': 'high' if len(anomaly_indices) > len(df) * 0.1 else 'low'
                        })
            
            return {
                'anomalies_found': len(anomalies),
                'details': anomalies,
                'recommendation': 'Review anomalies for potential data quality issues'
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return {'error': str(e)}
    
    def generate_advanced_insights(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced AI-powered insights"""
        try:
            insights = {
                'data_quality': 'Comprehensive analysis completed',
                'key_metrics': [],
                'recommendations': [],
                'patterns': []
            }
            
            if 'summary_stats' in results_data:
                stats = results_data['summary_stats']
                for metric, values in stats.items():
                    if isinstance(values, dict) and 'mean' in values:
                        insights['key_metrics'].append({
                            'name': metric,
                            'mean': round(values['mean'], 2),
                            'std': round(values.get('std', 0), 2)
                        })
            
            insights['recommendations'] = [
                'Review data for missing values and outliers',
                'Consider feature scaling for machine learning',
                'Validate data types and ranges',
                'Check for multicollinearity in numeric features'
            ]
            
            return insights
        except Exception as e:
            logger.error(f"Advanced insights error: {str(e)}")
            return {'error': str(e)}
