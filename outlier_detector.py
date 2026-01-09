"""
Enhanced Outlier Detection System with Accuracy Estimation
Supports multiple detection methods with performance evaluation
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
from typing import Dict, List, Tuple, Any
from json_utils import safe_json_serialize

logger = logging.getLogger(__name__)

class OutlierDetector:
    """
    Advanced outlier detection system with multiple methods and accuracy estimation
    """
    
    def __init__(self):
        self.detection_methods = {
            'z_score': self._detect_z_score,
            'iqr': self._detect_iqr,
            'isolation_forest': self._detect_isolation_forest,
            'local_outlier_factor': self._detect_local_outlier_factor,
            'modified_z_score': self._detect_modified_z_score,
            'dbscan': self._detect_dbscan,
            'percentile': self._detect_percentile
        }
        
        self.method_descriptions = {
            'z_score': 'Z-Score (Standard Deviation)',
            'iqr': 'Interquartile Range (IQR)',
            'isolation_forest': 'Isolation Forest (ML)',
            'local_outlier_factor': 'Local Outlier Factor (LOF)',
            'modified_z_score': 'Modified Z-Score (MAD)',
            'dbscan': 'DBSCAN Clustering',
            'percentile': 'Percentile-based Detection'
        }
    
    def analyze_sample_and_estimate_accuracy(self, df: pd.DataFrame, sample_size: int = 10) -> Dict[str, Any]:
        """
        Analyze sample data and estimate outlier detection accuracy for each method
        
        Args:
            df: DataFrame to analyze
            sample_size: Number of rows to sample for analysis
            
        Returns:
            Dictionary with method estimates and sample analysis
        """
        logger.info(f"Analyzing sample of {sample_size} rows for outlier detection estimation")
        
        # Get sample data (first N rows and some random rows for better representation)
        header_sample = df.head(sample_size // 2)
        random_sample = df.sample(n=min(sample_size // 2, len(df) - sample_size // 2), random_state=42)
        sample_df = pd.concat([header_sample, random_sample]).reset_index(drop=True)
        
        # Extract numeric columns for analysis
        numeric_columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            logger.warning("No numeric columns found for outlier analysis")
            return {
                'error': 'No numeric columns found for outlier analysis',
                'sample_info': self._get_sample_info(sample_df),
                'method_estimates': {}
            }
        
        sample_numeric = sample_df[numeric_columns]
        
        # Analyze each detection method
        method_estimates = {}
        overall_sample_stats = self._calculate_sample_statistics(sample_numeric)
        
        for method_name, method_func in self.detection_methods.items():
            try:
                estimate = self._estimate_method_accuracy(
                    sample_numeric, method_name, method_func, overall_sample_stats
                )
                
                # Extrapolate to full dataset
                full_dataset_estimate = self._extrapolate_to_full_dataset(
                    estimate, len(sample_df), len(df), len(numeric_columns), len(df.columns)
                )
                
                method_estimates[method_name] = {
                    'display_name': self.method_descriptions[method_name],
                    'sample_accuracy': estimate['accuracy_score'],
                    'sample_outlier_percentage': estimate['outlier_percentage'],
                    'estimated_full_accuracy': full_dataset_estimate['estimated_accuracy'],
                    'estimated_outliers_count': full_dataset_estimate['estimated_outliers'],
                    'confidence_level': full_dataset_estimate['confidence'],
                    'method_suitability': estimate['suitability_score'],
                    'processing_time_estimate': estimate['processing_time'],
                    'sample_details': estimate
                }
                
            except Exception as e:
                logger.warning(f"Failed to estimate accuracy for {method_name}: {str(e)}")
                method_estimates[method_name] = {
                    'display_name': self.method_descriptions[method_name],
                    'error': f"Estimation failed: {str(e)}",
                    'sample_accuracy': 0,
                    'estimated_full_accuracy': 0,
                    'confidence_level': 'low'
                }
        
        return {
            'sample_info': self._get_sample_info(sample_df),
            'dataset_overview': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_columns),
                'sample_size': len(sample_df),
                'numeric_column_names': numeric_columns
            },
            'method_estimates': method_estimates,
            'recommended_method': self._recommend_best_method(method_estimates),
            'overall_data_quality': self._assess_data_quality(sample_df, overall_sample_stats)
        }
    
    def _estimate_method_accuracy(self, sample_data: pd.DataFrame, method_name: str, 
                                method_func: callable, sample_stats: Dict) -> Dict[str, Any]:
        """Estimate accuracy of a specific outlier detection method"""
        
        start_time = pd.Timestamp.now()
        
        # Apply the detection method
        outlier_mask = method_func(sample_data)
        outlier_count = np.sum(outlier_mask)
        outlier_percentage = (outlier_count / len(sample_data)) * 100
        
        processing_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Calculate accuracy metrics using statistical heuristics
        accuracy_score = self._calculate_heuristic_accuracy(
            sample_data, outlier_mask, method_name, sample_stats
        )
        
        # Assess method suitability for this dataset
        suitability_score = self._assess_method_suitability(
            sample_data, method_name, sample_stats
        )
        
        return {
            'outlier_count': int(outlier_count),
            'outlier_percentage': round(outlier_percentage, 2),
            'accuracy_score': round(accuracy_score, 2),
            'suitability_score': round(suitability_score, 2),
            'processing_time': round(processing_time, 4),
            'outlier_indices': outlier_mask.tolist(),
            'method_specific_metrics': self._get_method_specific_metrics(
                sample_data, outlier_mask, method_name
            )
        }
    
    def _calculate_heuristic_accuracy(self, data: pd.DataFrame, outlier_mask: np.ndarray, 
                                    method_name: str, sample_stats: Dict) -> float:
        """
        Calculate heuristic accuracy score based on statistical properties
        This uses domain knowledge and statistical heuristics as ground truth proxy
        """
        
        if np.sum(outlier_mask) == 0:
            return 85.0  # No outliers detected - could be good or bad depending on data
        
        outlier_data = data[outlier_mask]
        normal_data = data[~outlier_mask]
        
        # Initialize accuracy score
        accuracy_components = []
        
        # Component 1: Statistical separation quality
        for col in data.select_dtypes(include=[np.number]).columns:
            if len(outlier_data) > 0 and len(normal_data) > 0:
                outlier_mean = outlier_data[col].mean()
                normal_mean = normal_data[col].mean()
                overall_std = data[col].std()
                
                if overall_std > 0:
                    separation = abs(outlier_mean - normal_mean) / overall_std
                    accuracy_components.append(min(separation * 20, 95))  # Cap at 95%
        
        # Component 2: Outlier percentage reasonableness (typically 1-10% for real data)
        outlier_pct = (np.sum(outlier_mask) / len(data)) * 100
        if 1 <= outlier_pct <= 10:
            pct_score = 90
        elif 0.1 <= outlier_pct <= 20:
            pct_score = max(70 - abs(outlier_pct - 5) * 3, 40)
        else:
            pct_score = max(30 - abs(outlier_pct - 5), 10)
        
        accuracy_components.append(pct_score)
        
        # Component 3: Method-specific heuristics
        method_specific_score = self._get_method_specific_accuracy(
            data, outlier_mask, method_name, sample_stats
        )
        accuracy_components.append(method_specific_score)
        
        # Component 4: Data distribution fit
        distribution_score = self._assess_distribution_fit(data, outlier_mask)
        accuracy_components.append(distribution_score)
        
        # Weighted average of components
        weights = [0.3, 0.25, 0.25, 0.2]
        if len(accuracy_components) == len(weights):
            final_accuracy = np.average(accuracy_components, weights=weights)
        else:
            # Fallback to simple average if components mismatch
            final_accuracy = np.mean(accuracy_components) if accuracy_components else 60
        
        return max(min(final_accuracy, 95), 15)  # Clamp between 15% and 95%
    
    def _get_method_specific_accuracy(self, data: pd.DataFrame, outlier_mask: np.ndarray, 
                                    method_name: str, sample_stats: Dict) -> float:
        """Get method-specific accuracy heuristics based on actual data characteristics"""
        
        # Calculate actual performance metrics for each method
        num_outliers = np.sum(outlier_mask)
        outlier_percentage = (num_outliers / len(data)) * 100 if len(data) > 0 else 0
        
        # Calculate data-specific characteristics
        data_variance = self._calculate_data_variance(data)
        data_skewness = self._calculate_data_skewness(data)
        data_normality = sample_stats.get('avg_normality_score', 0.5)
        
        if method_name == 'z_score':
            # Z-score accuracy depends on normality and variance consistency
            base_score = 45
            normality_bonus = data_normality * 35
            variance_penalty = min(data_variance * 5, 15)  # High variance reduces accuracy
            return max(base_score + normality_bonus - variance_penalty, 25)
        
        elif method_name == 'iqr':
            # IQR is robust but less effective with normal distributions
            base_score = 55
            robustness_bonus = (1 - data_normality) * 25  # Better for non-normal data
            outlier_rate_bonus = self._calculate_outlier_rate_bonus(outlier_percentage, optimal_range=(2, 8))
            return max(base_score + robustness_bonus + outlier_rate_bonus, 25)
        
        elif method_name == 'isolation_forest':
            # Isolation Forest works well with high-dimensional and complex patterns
            base_score = 50
            dimensionality_bonus = min(data.shape[1] * 8, 30)  # Better with more features
            complexity_bonus = abs(data_skewness) * 10  # Better with complex distributions
            outlier_rate_bonus = self._calculate_outlier_rate_bonus(outlier_percentage, optimal_range=(5, 15))
            return max(base_score + dimensionality_bonus + complexity_bonus + outlier_rate_bonus, 30)
        
        elif method_name == 'local_outlier_factor':
            # LOF excels with local density variations
            base_score = 40
            density_bonus = self._calculate_density_variation_bonus(data) * 25
            size_penalty = max(0, (len(data) - 100) * 0.1)  # Performance degrades with very large datasets
            outlier_rate_bonus = self._calculate_outlier_rate_bonus(outlier_percentage, optimal_range=(3, 12))
            return max(base_score + density_bonus - size_penalty + outlier_rate_bonus, 25)
        
        elif method_name == 'modified_z_score':
            # Modified Z-score (MAD-based) is more robust than standard Z-score
            base_score = 50
            robustness_bonus = (1 - data_normality) * 20  # Better for non-normal data
            consistency_bonus = self._calculate_consistency_bonus(data, outlier_mask) * 15
            return max(base_score + robustness_bonus + consistency_bonus, 30)
        
        elif method_name == 'dbscan':
            # DBSCAN is effective for cluster-based outliers
            base_score = 35
            cluster_bonus = self._calculate_cluster_quality(data) * 30
            parameter_sensitivity = 10  # DBSCAN is sensitive to parameters
            outlier_rate_bonus = self._calculate_outlier_rate_bonus(outlier_percentage, optimal_range=(1, 10))
            return max(base_score + cluster_bonus - parameter_sensitivity + outlier_rate_bonus, 20)
        
        elif method_name == 'percentile':
            # Percentile-based is simple but effective baseline
            base_score = 45
            distribution_bonus = (1 - abs(data_skewness - 0.5)) * 15  # Better with moderate skewness
            simplicity_bonus = 15  # Reliable and interpretable
            return max(base_score + distribution_bonus + simplicity_bonus, 35)
        
        return 50  # Default score
    
    def _calculate_data_variance(self, data: pd.DataFrame) -> float:
        """Calculate normalized variance across numeric columns"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 0.5
            
            variances = []
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 1:
                    # Normalize variance by mean to get coefficient of variation
                    mean_val = col_data.mean()
                    if mean_val != 0:
                        cv = col_data.std() / abs(mean_val)
                        variances.append(min(cv, 2.0))  # Cap at 2.0
            
            return np.mean(variances) if variances else 0.5
        except Exception:
            return 0.5
    
    def _calculate_data_skewness(self, data: pd.DataFrame) -> float:
        """Calculate average skewness across numeric columns"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return 0.0
            
            skewness_values = []
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 3:
                    skew = stats.skew(col_data)
                    if not np.isnan(skew):
                        skewness_values.append(abs(skew))
            
            return np.mean(skewness_values) if skewness_values else 0.0
        except Exception:
            return 0.0
    
    def _calculate_outlier_rate_bonus(self, outlier_percentage: float, optimal_range: tuple) -> float:
        """Calculate bonus based on how close the outlier rate is to optimal range"""
        min_optimal, max_optimal = optimal_range
        
        if min_optimal <= outlier_percentage <= max_optimal:
            return 15  # Optimal range bonus
        elif outlier_percentage < min_optimal:
            distance = min_optimal - outlier_percentage
            return max(15 - distance * 2, -10)  # Penalty for too few outliers
        else:
            distance = outlier_percentage - max_optimal
            return max(15 - distance * 1.5, -15)  # Penalty for too many outliers
    
    def _calculate_density_variation_bonus(self, data: pd.DataFrame) -> float:
        """Calculate density variation for LOF effectiveness"""
        try:
            numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())
            if len(numeric_data) < 5 or numeric_data.shape[1] == 0:
                return 0.3
            
            # Use standard deviation of distances to measure density variation
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(numeric_data[:min(20, len(numeric_data))], metric='euclidean')
            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            
            if distance_mean > 0:
                density_variation = distance_std / distance_mean
                return min(density_variation, 1.0)  # Cap at 1.0
            
            return 0.3
        except Exception:
            return 0.3
    
    def _calculate_consistency_bonus(self, data: pd.DataFrame, outlier_mask: np.ndarray) -> float:
        """Calculate consistency bonus for Modified Z-score"""
        try:
            if np.sum(outlier_mask) == 0:
                return 0.5
            
            numeric_data = data.select_dtypes(include=[np.number])
            consistency_scores = []
            
            for col in numeric_data.columns:
                col_data = numeric_data[col].dropna()
                if len(col_data) > 3:
                    # Calculate MAD-based consistency
                    median_val = col_data.median()
                    mad = np.median(np.abs(col_data - median_val))
                    
                    if mad > 0:
                        outlier_values = col_data[outlier_mask[:len(col_data)]]
                        if len(outlier_values) > 0:
                            # Check if outliers are consistently extreme
                            outlier_deviations = np.abs(outlier_values - median_val) / mad
                            consistency = np.std(outlier_deviations) / np.mean(outlier_deviations) if np.mean(outlier_deviations) > 0 else 1
                            consistency_scores.append(1 - min(consistency, 1))
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
        except Exception:
            return 0.5
    
    def _calculate_cluster_quality(self, data: pd.DataFrame) -> float:
        """Calculate cluster quality for DBSCAN effectiveness"""
        try:
            numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())
            if len(numeric_data) < 6 or numeric_data.shape[1] == 0:
                return 0.2
            
            # Use silhouette analysis concept - simplified
            sample_size = min(20, len(numeric_data))
            sample_data = numeric_data.sample(n=sample_size, random_state=42)
            
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(sample_data, metric='euclidean')
            
            # Calculate cluster tendency using Hopkins statistic approximation
            random_sample = np.random.random((sample_size, numeric_data.shape[1]))
            random_distances = pairwise_distances(random_sample, metric='euclidean')
            
            data_nearest = np.min(distances + np.eye(len(distances)) * 1e6, axis=1)
            random_nearest = np.min(random_distances + np.eye(len(random_distances)) * 1e6, axis=1)
            
            hopkins = np.sum(random_nearest) / (np.sum(data_nearest) + np.sum(random_nearest))
            
            # Hopkins value close to 0.5 indicates no clustering, close to 1 indicates strong clustering
            cluster_quality = abs(hopkins - 0.5) * 2  # Convert to 0-1 scale
            return min(cluster_quality, 1.0)
            
        except Exception:
            return 0.2
    
    def _assess_distribution_fit(self, data: pd.DataFrame, outlier_mask: np.ndarray) -> float:
        """Assess how well the outlier detection fits the data distribution"""
        
        scores = []
        for col in data.select_dtypes(include=[np.number]).columns:
            try:
                # Test normality of non-outlier data
                normal_data = data.loc[~outlier_mask, col].dropna()
                if len(normal_data) > 8:  # Minimum for statistical tests
                    stat, p_value = stats.shapiro(normal_data[:50])  # Limit sample for performance
                    
                    # Higher p-value means more normal distribution after outlier removal
                    normality_improvement = min(p_value * 100, 85)
                    scores.append(normality_improvement)
                    
            except Exception:
                scores.append(60)  # Default if test fails
        
        return np.mean(scores) if scores else 60
    
    def _assess_method_suitability(self, data: pd.DataFrame, method_name: str, 
                                 sample_stats: Dict) -> float:
        """Assess how suitable a method is for the given dataset characteristics"""
        
        n_samples, n_features = data.shape
        
        # Base suitability factors
        size_factor = min(n_samples / 50, 1.0)  # Preference for larger samples
        complexity_factor = min(n_features / 5, 1.0)  # Some methods better with more features
        
        method_factors = {
            'z_score': {
                'min_normality': sample_stats.get('avg_normality_score', 0.5),
                'size_preference': 0.8,
                'complexity_preference': 0.6
            },
            'iqr': {
                'min_normality': 0.3,  # Works with any distribution
                'size_preference': 0.9,
                'complexity_preference': 0.8
            },
            'isolation_forest': {
                'min_normality': 0.1,  # Distribution-agnostic
                'size_preference': 0.7,  # Needs reasonable sample size
                'complexity_preference': 1.0  # Excellent with high dimensions
            },
            'local_outlier_factor': {
                'min_normality': 0.2,
                'size_preference': 0.6,  # Computationally intensive
                'complexity_preference': 0.8
            },
            'modified_z_score': {
                'min_normality': 0.4,
                'size_preference': 0.8,
                'complexity_preference': 0.7
            },
            'dbscan': {
                'min_normality': 0.1,
                'size_preference': 0.5,  # Can be sensitive to parameters
                'complexity_preference': 0.6
            },
            'percentile': {
                'min_normality': 0.1,  # Works with any distribution
                'size_preference': 1.0,
                'complexity_preference': 0.5
            }
        }
        
        factors = method_factors.get(method_name, {
            'min_normality': 0.5,
            'size_preference': 0.7,
            'complexity_preference': 0.7
        })
        
        # Calculate weighted suitability score
        suitability = (
            factors['min_normality'] * 30 +
            factors['size_preference'] * size_factor * 35 +
            factors['complexity_preference'] * complexity_factor * 35
        )
        
        return max(min(suitability, 95), 25)
    
    def _extrapolate_to_full_dataset(self, sample_estimate: Dict, sample_size: int, 
                                   full_size: int, numeric_cols: int, total_cols: int) -> Dict[str, Any]:
        """Extrapolate sample results to full dataset with confidence estimation"""
        
        # Scale factor for dataset size
        size_ratio = full_size / sample_size
        
        # Confidence decreases as extrapolation increases
        if size_ratio <= 2:
            confidence = 'high'
            accuracy_adjustment = 0
        elif size_ratio <= 10:
            confidence = 'medium'
            accuracy_adjustment = -5
        elif size_ratio <= 100:
            confidence = 'medium-low'
            accuracy_adjustment = -10
        else:
            confidence = 'low'
            accuracy_adjustment = -15
        
        # Adjust accuracy based on data complexity
        complexity_factor = total_cols / max(numeric_cols, 1)
        if complexity_factor > 3:
            accuracy_adjustment -= 5
        
        estimated_accuracy = max(
            sample_estimate['accuracy_score'] + accuracy_adjustment, 
            20
        )
        
        # Estimate outlier count in full dataset
        sample_outlier_rate = sample_estimate['outlier_percentage'] / 100
        estimated_outliers = int(full_size * sample_outlier_rate)
        
        return {
            'estimated_accuracy': round(estimated_accuracy, 1),
            'estimated_outliers': estimated_outliers,
            'confidence': confidence,
            'extrapolation_factor': round(size_ratio, 1),
            'accuracy_adjustment': accuracy_adjustment
        }
    
    def _calculate_sample_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the sample data"""
        
        stats_dict = {
            'mean_values': data.mean().to_dict(),
            'std_values': data.std().to_dict(),
            'skewness': data.skew().to_dict(),
            'kurtosis': data.kurtosis().to_dict()
        }
        
        # Calculate normality scores
        normality_scores = []
        for col in data.columns:
            try:
                if len(data[col].dropna()) > 8:
                    stat, p_value = stats.shapiro(data[col].dropna()[:50])
                    normality_scores.append(p_value)
            except:
                normality_scores.append(0.5)
        
        stats_dict['avg_normality_score'] = np.mean(normality_scores) if normality_scores else 0.5
        stats_dict['density_variation'] = np.std([data[col].std() for col in data.columns])
        stats_dict['cluster_tendency'] = self._estimate_cluster_tendency(data)
        
        return stats_dict
    
    def _estimate_cluster_tendency(self, data: pd.DataFrame) -> float:
        """Estimate the tendency of data to form clusters"""
        try:
            # Simple clustering tendency using variance ratios
            if len(data) < 4:
                return 0.5
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.fillna(data.mean()))
            
            # Calculate within-cluster vs between-cluster variance proxy
            total_variance = np.var(scaled_data, axis=0).mean()
            
            # Simple heuristic based on data spread
            cluster_score = min(total_variance / 2, 1.0)
            return cluster_score
            
        except Exception:
            return 0.5
    
    def _get_sample_info(self, sample_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic information about the sample"""
        return {
            'sample_size': len(sample_df),
            'columns': list(sample_df.columns),
            'numeric_columns': sample_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': sample_df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'missing_values': sample_df.isnull().sum().to_dict(),
            'data_types': sample_df.dtypes.astype(str).to_dict()
        }
    
    def _recommend_best_method(self, method_estimates: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend the best outlier detection method based on estimates"""
        
        if not method_estimates:
            return {'method': 'iqr', 'reason': 'Default - no estimates available'}
        
        # Score each method based on multiple criteria
        method_scores = {}
        for method, estimate in method_estimates.items():
            if 'error' in estimate:
                continue
                
            # Weighted scoring
            score = (
                estimate.get('estimated_full_accuracy', 0) * 0.4 +
                estimate.get('method_suitability', 0) * 0.3 +
                (100 - estimate.get('processing_time_estimate', 1) * 1000) * 0.2 +  # Prefer faster methods
                {'high': 100, 'medium': 80, 'medium-low': 60, 'low': 40}.get(estimate.get('confidence_level', 'low'), 40) * 0.1
            )
            
            method_scores[method] = score
        
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            
            return {
                'method': best_method,
                'display_name': self.method_descriptions.get(best_method, best_method),
                'score': round(best_score, 1),
                'reason': f"Best overall performance (score: {best_score:.1f})",
                'all_scores': {k: round(v, 1) for k, v in method_scores.items()}
            }
        
        return {'method': 'iqr', 'reason': 'Default fallback method'}
    
    def _assess_data_quality(self, sample_df: pd.DataFrame, sample_stats: Dict) -> Dict[str, Any]:
        """Assess overall data quality of the sample"""
        
        # Missing value assessment
        missing_pct = (sample_df.isnull().sum().sum() / (len(sample_df) * len(sample_df.columns))) * 100
        
        if missing_pct < 5:
            missing_quality = 'excellent'
        elif missing_pct < 15:
            missing_quality = 'good'
        elif missing_pct < 30:
            missing_quality = 'fair'
        else:
            missing_quality = 'poor'
        
        # Distribution quality
        avg_normality = sample_stats.get('avg_normality_score', 0.5)
        if avg_normality > 0.7:
            distribution_quality = 'excellent'
        elif avg_normality > 0.5:
            distribution_quality = 'good'
        elif avg_normality > 0.3:
            distribution_quality = 'fair'
        else:
            distribution_quality = 'poor'
        
        # Overall assessment
        quality_scores = {
            'excellent': 4,
            'good': 3,
            'fair': 2,
            'poor': 1
        }
        
        overall_score = np.mean([
            quality_scores[missing_quality],
            quality_scores[distribution_quality]
        ])
        
        if overall_score >= 3.5:
            overall_quality = 'excellent'
        elif overall_score >= 2.5:
            overall_quality = 'good'
        elif overall_score >= 1.5:
            overall_quality = 'fair'
        else:
            overall_quality = 'poor'
        
        return {
            'overall': overall_quality,
            'missing_values': missing_quality,
            'distributions': distribution_quality,
            'missing_percentage': round(missing_pct, 2),
            'normality_score': round(avg_normality, 3),
            'recommendations': self._get_quality_recommendations(missing_quality, distribution_quality)
        }
    
    def _get_quality_recommendations(self, missing_quality: str, distribution_quality: str) -> List[str]:
        """Get data quality improvement recommendations"""
        recommendations = []
        
        if missing_quality in ['fair', 'poor']:
            recommendations.append("Consider data imputation techniques for missing values")
        
        if distribution_quality in ['fair', 'poor']:
            recommendations.append("Data may benefit from transformation (log, square root, etc.)")
            recommendations.append("Consider robust outlier detection methods (IQR, MAD-based)")
        
        if not recommendations:
            recommendations.append("Data quality appears good for outlier detection")
        
        return recommendations
    
    def _get_method_specific_metrics(self, data: pd.DataFrame, outlier_mask: np.ndarray, 
                                   method_name: str) -> Dict[str, Any]:
        """Get additional metrics specific to each detection method"""
        
        metrics = {}
        
        if method_name == 'z_score':
            metrics['threshold_used'] = 3.0
            if np.sum(outlier_mask) > 0:
                outlier_data = data[outlier_mask]
                metrics['max_z_score'] = float(np.max(np.abs(stats.zscore(outlier_data, nan_policy='omit'))))
        
        elif method_name == 'iqr':
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                metrics[f'{col}_iqr'] = float(IQR)
        
        elif method_name in ['isolation_forest', 'local_outlier_factor']:
            metrics['algorithm_type'] = 'machine_learning'
            metrics['features_used'] = data.select_dtypes(include=[np.number]).columns.tolist()
        
        return metrics
    
    # Outlier Detection Methods
    def _detect_z_score(self, data: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-Score method"""
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                col_outliers = z_scores > threshold
                # Map back to original indices
                outlier_mask[col_data.index] |= col_outliers
        
        return outlier_mask
    
    def _detect_iqr(self, data: pd.DataFrame, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using Interquartile Range method"""
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_mask |= col_outliers.fillna(False)
        
        return outlier_mask
    
    def _detect_isolation_forest(self, data: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())
        
        if len(numeric_data.columns) == 0 or len(numeric_data) < 2:
            return np.zeros(len(data), dtype=bool)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(numeric_data)
        
        return outlier_labels == -1
    
    def _detect_local_outlier_factor(self, data: pd.DataFrame, n_neighbors: int = 5) -> np.ndarray:
        """Detect outliers using Local Outlier Factor"""
        numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())
        
        if len(numeric_data.columns) == 0 or len(numeric_data) < n_neighbors + 1:
            return np.zeros(len(data), dtype=bool)
        
        lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(numeric_data) - 1))
        outlier_labels = lof.fit_predict(numeric_data)
        
        return outlier_labels == -1
    
    def _detect_modified_z_score(self, data: pd.DataFrame, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using Modified Z-Score (MAD-based)"""
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                median = np.median(col_data)
                mad = np.median(np.abs(col_data - median))
                
                if mad != 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    col_outliers = np.abs(modified_z_scores) > threshold
                    outlier_mask[col_data.index] |= col_outliers
        
        return outlier_mask
    
    def _detect_dbscan(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 3) -> np.ndarray:
        """Detect outliers using DBSCAN clustering"""
        numeric_data = data.select_dtypes(include=[np.number]).fillna(data.mean())
        
        if len(numeric_data.columns) == 0 or len(numeric_data) < min_samples:
            return np.zeros(len(data), dtype=bool)
        
        # Standardize data for DBSCAN
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_data)
        
        # Points labeled as -1 are considered outliers
        return cluster_labels == -1
    
    def _detect_percentile(self, data: pd.DataFrame, lower_percentile: float = 1, 
                          upper_percentile: float = 99) -> np.ndarray:
        """Detect outliers using percentile-based method"""
        outlier_mask = np.zeros(len(data), dtype=bool)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            lower_bound = data[col].quantile(lower_percentile / 100)
            upper_bound = data[col].quantile(upper_percentile / 100)
            
            col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_mask |= col_outliers.fillna(False)
        
        return outlier_mask