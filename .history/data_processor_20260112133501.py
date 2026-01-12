import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import logging
from json_utils import safe_json_serialize
from outlier_detector import OutlierDetector

logger = logging.getLogger(__name__)

class DataProcessor:
    """Main class for data processing and cleaning operations"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def load_data(self, filepath):
        """Load data from CSV or Excel file"""
        try:
            file_extension = filepath.lower().split('.')[-1]
            
            if file_extension == 'csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(f"Successfully loaded CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any supported encoding")
            
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(filepath)
                logger.info("Successfully loaded Excel file")
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_columns(self, df):
        """Analyze column types and characteristics"""
        column_info = {}
        
        for col in df.columns:
            col_data = df[col]
            info = {
                'name': col,
                'dtype': str(col_data.dtype),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(df)) * 100,
                'unique_count': col_data.nunique(),
                'is_numeric': pd.api.types.is_numeric_dtype(col_data),
                'is_categorical': pd.api.types.is_categorical_dtype(col_data) or col_data.nunique() < 20,
                'sample_values': col_data.dropna().head(5).tolist()
            }
            
            if info['is_numeric']:
                info.update({
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                })
            
            column_info[col] = safe_json_serialize(info)
        
        return column_info
    
    def handle_missing_values(self, df, method='mean'):
        """Handle missing values using specified method"""
        logger.info(f"Handling missing values using method: {method}")
        
        # Create a copy to avoid modifying original
        df_cleaned = df.copy()
        
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    if method == 'mean':
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    elif method == 'median':
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    elif method == 'mode':
                        mode_val = df_cleaned[col].mode()
                        if len(mode_val) > 0:
                            df_cleaned[col].fillna(mode_val[0], inplace=True)
                    elif method == 'knn':
                        # Use KNN imputation for numeric columns
                        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            imputer = KNNImputer(n_neighbors=5)
                            df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
                else:
                    # For categorical columns, use mode
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col].fillna(mode_val[0], inplace=True)
        
        logger.info(f"Missing values handled. Remaining nulls: {df_cleaned.isnull().sum().sum()}")
        return df_cleaned
    
    def detect_and_handle_outliers(self, df, method='iqr', action='cap'):
        """Detect and handle outliers using specified method"""
        logger.info(f"Detecting outliers using method: {method}")
        
        df_cleaned = df.copy()
        outlier_info = {}
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        if method in ['isolation_forest', 'local_outlier_factor', 'dbscan']:
            # Use OutlierDetector for ML methods
            detector = OutlierDetector()
            outlier_mask = detector.detection_methods[method](df_cleaned[numeric_columns])
            outlier_count = np.sum(outlier_mask)
            if outlier_count > 0:
                outlier_info['ml_method'] = {
                    'count': int(outlier_count),
                    'percentage': (outlier_count / len(df_cleaned)) * 100,
                    'method': method
                }
                if action == 'remove':
                    df_cleaned = df_cleaned[~outlier_mask]
                # For cap, do nothing as ML methods don't have bounds
        else:
            # Per-column detection for statistical methods
            for col in numeric_columns:
                col_data = df_cleaned[col].dropna()
                outliers = []
                lower_bound = None
                upper_bound = None
                
                if method == 'iqr':
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(col_data))
                    threshold = 3.0
                    outliers = col_data[z_scores > threshold]
                    lower_bound = col_data.mean() - threshold * col_data.std()
                    upper_bound = col_data.mean() + threshold * col_data.std()
                    
                elif method == 'modified_z_score':
                    from scipy.stats import median_abs_deviation
                    mad = median_abs_deviation(col_data, nan_policy='omit')
                    median = col_data.median()
                    modified_z = 0.6745 * (col_data - median) / mad
                    threshold = 3.5
                    outliers = col_data[np.abs(modified_z) > threshold]
                    lower_bound = median - (threshold / 0.6745) * mad
                    upper_bound = median + (threshold / 0.6745) * mad
                    
                elif method == 'percentile':
                    lower_percentile = 1
                    upper_percentile = 99
                    lower_bound = col_data.quantile(lower_percentile / 100)
                    upper_bound = col_data.quantile(upper_percentile / 100)
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                elif method == 'winsorization':
                    lower_bound = col_data.quantile(0.05)
                    upper_bound = col_data.quantile(0.95)
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_info[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(col_data)) * 100,
                        'values': outliers.tolist()[:10]  # Store first 10 outliers
                    }
                    
                    if action == 'cap' and lower_bound is not None and upper_bound is not None:
                        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                    elif action == 'remove':
                        outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                        df_cleaned.loc[outlier_mask, col] = np.nan
        
        logger.info(f"Outlier detection completed. Found outliers in {len(outlier_info)} columns")
        return df_cleaned, outlier_info
    
    def apply_validation_rules(self, df, rules):
        """Apply custom validation rules to the data"""
        logger.info("Applying validation rules")
        
        validation_results = {
            'passed': 0,
            'failed': 0,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Basic consistency checks
            if 'consistency' in rules:
                # Check for duplicate rows
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
                
                # Check for data type consistency
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        non_numeric = pd.to_numeric(df[col], errors='coerce').isnull().sum() - df[col].isnull().sum()
                        if non_numeric > 0:
                            validation_results['warnings'].append(f"Column '{col}' has {non_numeric} non-numeric values")
            
            # Skip pattern validation
            if 'skip_patterns' in rules:
                # Check for logical skip patterns (basic implementation)
                # This would need customization based on survey structure
                validation_results['warnings'].append("Skip pattern validation not implemented for this dataset")
            
            # Range validation for numeric columns
            if 'range_validation' in rules:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Check for reasonable ranges (this is a basic implementation)
                        extreme_values = col_data[(col_data < col_data.quantile(0.01)) | 
                                                (col_data > col_data.quantile(0.99))]
                        if len(extreme_values) > 0:
                            validation_results['warnings'].append(
                                f"Column '{col}' has {len(extreme_values)} extreme values"
                            )
            
            validation_results['passed'] = len(validation_results['warnings']) + len(validation_results['errors'])
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Validation error: {str(e)}")
        
        return validation_results
    
    def calculate_weighted_summary(self, df, weight_column):
        """Calculate weighted summary statistics"""
        logger.info(f"Calculating weighted summary using column: {weight_column}")
        
        if weight_column not in df.columns:
            raise ValueError(f"Weight column '{weight_column}' not found in data")
        
        weights = df[weight_column].dropna()
        summary = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != weight_column]
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            col_weights = df.loc[col_data.index, weight_column]
            
            if len(col_data) > 0 and len(col_weights) > 0:
                weighted_mean = np.average(col_data, weights=col_weights)
                
                # Calculate weighted variance
                weighted_var = np.average((col_data - weighted_mean) ** 2, weights=col_weights)
                weighted_std = np.sqrt(weighted_var)
                
                # Calculate margin of error (assuming 95% confidence level)
                n_effective = (col_weights.sum() ** 2) / (col_weights ** 2).sum()  # Effective sample size
                margin_of_error = 1.96 * (weighted_std / np.sqrt(n_effective))
                
                summary[col] = {
                    'mean': float(weighted_mean),  # Add template-compatible keys
                    'median': float(np.median(col_data)) if not col_data.empty else 0,
                    'std': float(weighted_std),
                    'min': float(col_data.min()) if not col_data.empty else 0,
                    'max': float(col_data.max()) if not col_data.empty else 0,
                    'count': int(len(col_data)),
                    'weighted_mean': float(weighted_mean),
                    'weighted_std': float(weighted_std),
                    'margin_of_error': float(margin_of_error),
                    'effective_n': float(n_effective)
                }
        
        return safe_json_serialize(summary)
    
    def calculate_summary_statistics(self, df):
        """Calculate basic summary statistics"""
        logger.info("Calculating summary statistics")
        
        summary = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if pd.api.types.is_numeric_dtype(col_data) and len(col_data) > 0:
                summary[col] = {
                    'count': len(col_data),
                    'mean': float(col_data.mean()) if not col_data.empty else 0,
                    'median': float(col_data.quantile(0.50)) if not col_data.empty else 0,
                    'std': float(col_data.std()) if not col_data.empty else 0,
                    'min': float(col_data.min()) if not col_data.empty else 0,
                    'max': float(col_data.max()) if not col_data.empty else 0,
                    'q25': float(col_data.quantile(0.25)) if not col_data.empty else 0,
                    'q50': float(col_data.quantile(0.50)) if not col_data.empty else 0,
                    'q75': float(col_data.quantile(0.75)) if not col_data.empty else 0,
                    'missing': int(df[col].isnull().sum()),
                    'missing_pct': float((df[col].isnull().sum() / len(df)) * 100)
                }
            else:
                summary[col] = {
                    'count': len(col_data),
                    'unique': col_data.nunique(),
                    'top': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else None,
                    'freq': col_data.value_counts().iloc[0] if len(col_data) > 0 else 0,
                    'missing': df[col].isnull().sum(),
                    'missing_pct': (df[col].isnull().sum() / len(df)) * 100
                }
        
        return safe_json_serialize(summary)
    
    def get_data_quality_score(self, df):
        """Calculate overall data quality score"""
        logger.info("Calculating data quality score")
        
        scores = {
            'completeness': 0,
            'consistency': 0,
            'validity': 0,
            'overall': 0
        }
        
        try:
            # Completeness score (based on missing values)
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            scores['completeness'] = ((total_cells - missing_cells) / total_cells) * 100
            
            # Consistency score (based on duplicates and data types)
            duplicate_rows = df.duplicated().sum()
            consistency_score = 100
            if duplicate_rows > 0:
                consistency_score -= (duplicate_rows / len(df)) * 50
            scores['consistency'] = max(consistency_score, 0)
            
            # Validity score (basic implementation)
            scores['validity'] = 85  # Placeholder - would need domain-specific rules
            
            # Overall score
            scores['overall'] = (scores['completeness'] + scores['consistency'] + scores['validity']) / 3
            
        except Exception as e:
            logger.error(f"Error calculating data quality score: {str(e)}")
            scores = {'completeness': 0, 'consistency': 0, 'validity': 0, 'overall': 0}
        
        return scores
