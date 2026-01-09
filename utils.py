import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(filepath):
    """Get basic information about uploaded file"""
    try:
        file_info = {
            'size': os.path.getsize(filepath),
            'extension': filepath.split('.')[-1].lower()
        }
        
        # Try to get row/column counts
        try:
            if file_info['extension'] == 'csv':
                # Quick row count for CSV
                with open(filepath, 'r', encoding='utf-8') as f:
                    row_count = sum(1 for line in f) - 1  # Subtract header
                file_info['rows'] = max(0, row_count)
                
                # Get column count
                df_sample = pd.read_csv(filepath, nrows=1)
                file_info['columns'] = len(df_sample.columns)
                
            elif file_info['extension'] in ['xlsx', 'xls']:
                df_sample = pd.read_excel(filepath, nrows=1)
                file_info['columns'] = len(df_sample.columns)
                
                # For Excel, we need to load more to get accurate row count
                df_full = pd.read_excel(filepath)
                file_info['rows'] = len(df_full)
                
        except Exception as e:
            logger.warning(f"Could not get detailed file info: {str(e)}")
            file_info['rows'] = 0
            file_info['columns'] = 0
        
        return file_info
        
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        return {'size': 0, 'rows': 0, 'columns': 0, 'extension': 'unknown'}

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def safe_float(value, default=0.0):
    """Safely convert value to float with default fallback"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int with default fallback"""
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def truncate_text(text, max_length=100):
    """Truncate text to specified length with ellipsis"""
    if not text:
        return ""
    
    text = str(text)
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def get_percentage(part, total):
    """Calculate percentage with proper handling of zero division"""
    try:
        if total == 0:
            return 0.0
        return (part / total) * 100
    except (ZeroDivisionError, TypeError):
        return 0.0

def format_duration(start_time, end_time):
    """Format duration between two timestamps"""
    try:
        if not start_time or not end_time:
            return "N/A"
        
        duration = end_time - start_time
        total_seconds = duration.total_seconds()
        
        if total_seconds < 60:
            return f"{total_seconds:.1f} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = total_seconds / 3600
            return f"{hours:.1f} hours"
            
    except Exception as e:
        logger.error(f"Error formatting duration: {str(e)}")
        return "N/A"

def validate_column_selection(df, column_name):
    """Validate that a column exists and is suitable for analysis"""
    if not column_name or column_name not in df.columns:
        return False, f"Column '{column_name}' not found"
    
    col_data = df[column_name]
    
    # Check if column has any non-null values
    if col_data.isnull().all():
        return False, f"Column '{column_name}' contains only null values"
    
    # Check if column has sufficient variation for analysis
    if col_data.nunique() <= 1:
        return False, f"Column '{column_name}' has insufficient variation"
    
    return True, "Valid column"

def clean_column_name(column_name):
    """Clean column name for safe usage in file names and IDs"""
    import re
    
    # Replace problematic characters
    cleaned = re.sub(r'[^\w\s-]', '', str(column_name))
    cleaned = re.sub(r'[-\s]+', '_', cleaned)
    cleaned = cleaned.strip('_')
    
    # Ensure it's not empty
    if not cleaned:
        cleaned = "unnamed_column"
    
    return cleaned.lower()

def is_numeric_column(df, column_name):
    """Check if a column contains numeric data"""
    if column_name not in df.columns:
        return False
    
    return pd.api.types.is_numeric_dtype(df[column_name])

def is_categorical_column(df, column_name, threshold=20):
    """Check if a column should be treated as categorical"""
    if column_name not in df.columns:
        return False
    
    col_data = df[column_name]
    
    # If already categorical
    if pd.api.types.is_categorical_dtype(col_data):
        return True
    
    # If text/object type with few unique values
    if col_data.dtype == 'object' and col_data.nunique() <= threshold:
        return True
    
    # If numeric but with few unique values (likely coded categories)
    if pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() <= threshold:
        return True
    
    return False
