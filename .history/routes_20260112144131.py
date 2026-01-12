import os
import logging
from datetime import datetime
from flask import render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
from app import app, db, bcrypt
from models import UploadSession, ProcessingLog, ReportGeneration, ChatSession, ChatMessage, PredictiveAnalysis, DataExport, ActivityLog, User, OTP
from data_processor import DataProcessor
from gemini_service import GeminiService
from report_generator import ReportGenerator
from outlier_detector import OutlierDetector
from utils import allowed_file, get_file_info
from json_utils import safe_json_serialize
import pandas as pd
import random
import string
from decorators import require_role
from flask_login import login_required, current_user

logger = logging.getLogger(__name__)

@app.route('/')
def homepage():
    """Futuristic homepage with SaaS design"""
    return render_template('homepage.html')

@app.route('/dashboard')
@login_required
def index():
    """Main dashboard showing recent uploads and overall statistics"""
    if current_user.role == 'ADMIN':
        recent_sessions = UploadSession.query.order_by(UploadSession.upload_time.desc()).limit(10).all()
        total_files = UploadSession.query.count()
    else:
        recent_sessions = UploadSession.query.filter_by(user_id=current_user.id).order_by(UploadSession.upload_time.desc()).limit(10).all()
        total_files = UploadSession.query.filter_by(user_id=current_user.id).count()
    
    completed_analyses = UploadSession.query.filter_by(status='completed').count()
    
    return render_template('index.html', 
                         recent_sessions=recent_sessions,
                         total_files=total_files,
                         completed_analyses=completed_analyses)

@app.route('/fetch_from_db/<int:session_id>')
@login_required
def fetch_from_db(session_id):
    """Restore file from database to uploads folder and redirect to config"""
    upload_session = UploadSession.query.get_or_404(session_id)
    if upload_session.user_id != current_user.id and current_user.role != 'ADMIN':
        flash('You do not have permission to access this file.', 'error')
        return redirect(url_for('index'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
    
    if not os.path.exists(filepath) and upload_session.file_content:
        with open(filepath, 'wb') as f:
            f.write(upload_session.file_content)
            
    session['current_session_id'] = upload_session.id
    flash(f'File "{upload_session.original_filename}" fetched from database!', 'success')
    return redirect(url_for('configure_schema', session_id=upload_session.id))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Handle file upload and initial processing"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename or 'uploaded_file')
                timestamp = str(int(pd.Timestamp.now().timestamp()))
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Ensure upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                
                # Get file content for DB storage
                file.seek(0)
                file_binary = file.read()
                
                # Get file info
                try:
                    file_info = get_file_info(filepath)
                except Exception as info_error:
                    logger.warning(f"Could not get file info: {info_error}")
                    file_info = {'size': len(file_binary), 'rows': 0, 'columns': 0}
                
                # Create upload session
                upload_session = UploadSession()
                upload_session.filename = unique_filename
                upload_session.original_filename = filename
                upload_session.file_size = len(file_binary)
                upload_session.file_content = file_binary
                upload_session.total_rows = file_info.get('rows', 0)
                upload_session.total_columns = file_info.get('columns', 0)
                upload_session.user_id = current_user.id
                db.session.add(upload_session)
                db.session.commit()
                
                # Store session ID for subsequent requests
                session['current_session_id'] = upload_session.id
                
                flash(f'File "{filename}" uploaded successfully!', 'success')
                return redirect(url_for('configure_schema', session_id=upload_session.id))
                
            except Exception as e:
                logger.error(f"Upload error: {str(e)}", exc_info=True)
                flash(f'Upload failed: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload CSV or Excel files only.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/configure/<int:session_id>')
@login_required
def configure_schema(session_id):
    """Configure schema mapping and data types"""
    upload_session = UploadSession.query.get_or_404(session_id)
    if upload_session.user_id != current_user.id and current_user.role != 'ADMIN':
        flash('You do not have permission to access this file.', 'error')
        return redirect(url_for('index'))
    
    # Load file to get column information
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
    data_processor = DataProcessor()
    
    try:
        df = data_processor.load_data(filepath)
        columns_info = data_processor.analyze_columns(df)
        
        # Convert to proper format for template
        if columns_info and isinstance(list(columns_info.values())[0], dict):
            # Already in correct format
            pass
        else:
            # Need to convert to expected format
            formatted_columns = {}
            for col_name, col_data in columns_info.items():
                if hasattr(col_data, '__dict__'):
                    formatted_columns[col_name] = col_data.__dict__
                else:
                    formatted_columns[col_name] = col_data
            columns_info = formatted_columns
        
        return render_template('analysis.html', 
                             session=upload_session,
                             columns_info=columns_info,
                             step='configure')
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        flash(f'Configuration failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/process/<int:session_id>', methods=['POST'])
@login_required
def process_data(session_id):
    """Process data with cleaning and analysis"""
    upload_session = UploadSession.query.get_or_404(session_id)
    if upload_session.user_id != current_user.id and current_user.role != 'ADMIN':
        flash('You do not have permission to access this file.', 'error')
        return redirect(url_for('index'))
    
    try:
        upload_session.status = 'processing'
        db.session.commit()
        
        # Get configuration from form
        cleaning_config = {
            'imputation_method': request.form.get('imputation_method', 'mean'),
            'outlier_method': request.form.get('outlier_method', 'iqr'),
            'validation_rules': request.form.getlist('validation_rules'),
            'apply_weights': 'apply_weights' in request.form,
            'weight_column': request.form.get('weight_column')
        }
        
        # Store configuration
        upload_session.cleaning_config = cleaning_config
        db.session.commit()
        
        # Process data
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
        data_processor = DataProcessor()
        gemini_service = GeminiService()
        
        # Log processing start
        log_entry = ProcessingLog()
        log_entry.session_id = session_id
        log_entry.step_name = 'data_processing'
        log_entry.step_status = 'started'
        log_entry.log_message = 'Starting data processing pipeline'
        db.session.add(log_entry)
        db.session.commit()
        
        # Load and process data
        df = data_processor.load_data(filepath)

        # Capture raw statistics before processing
        raw_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': int(df.isnull().sum().sum())
        }
        
        # Data cleaning steps
        if cleaning_config['imputation_method'] != 'none':
            df = data_processor.handle_missing_values(df, method=cleaning_config['imputation_method'])
        
        if cleaning_config['outlier_method'] != 'none':
            df, outlier_info = data_processor.detect_and_handle_outliers(df, method=cleaning_config['outlier_method'])
            total_outliers = 0
            for key, info in outlier_info.items():
                if isinstance(info, dict) and 'count' in info:
                    total_outliers += info['count']
            upload_session.outliers_detected = total_outliers
        
        # Apply validation rules
        validation_results = data_processor.apply_validation_rules(df, cleaning_config['validation_rules'])
        
        # Assess data quality
        quality_score = data_processor.get_data_quality_score(df)
        
        # Calculate summary statistics
        summary_stats = data_processor.calculate_summary_stats(df)
        
        # Save cleaned data for download
        cleaned_filename = f"cleaned_{upload_session.filename}"
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        
        # Save in the same format as original
        original_extension = upload_session.filename.lower().split('.')[-1]
        if original_extension == 'csv':
            df.to_csv(cleaned_filepath, index=False)
        elif original_extension in ['xlsx', 'xls']:
            df.to_excel(cleaned_filepath, index=False, engine='openpyxl')
        
        # Store cleaned filename in session for download
        upload_session.cleaning_config = safe_json_serialize({
            **cleaning_config,
            'cleaned_filename': cleaned_filename
        })
        
        # Apply weights if specified
        if cleaning_config['apply_weights'] and cleaning_config['weight_column']:
            summary_stats = data_processor.calculate_weighted_summary(df, cleaning_config['weight_column'])
        else:
            summary_stats = data_processor.calculate_summary_statistics(df)
            quality_score = data_processor.get_data_quality_score(df)
        
        # Ensure summary_stats is a dictionary
        if not isinstance(summary_stats, dict):
            summary_stats = {'error': 'Invalid summary statistics format'}
        
        # Get AI insights
        ai_insights = gemini_service.analyze_survey_data(df, summary_stats,raw_stats=raw_stats)
        
        # Update session with results
        upload_session.status = 'completed'
        upload_session.missing_values_count = raw_stats['total_missing'] # Store original missing count
        upload_session.outliers_detected = 0  # Will be updated based on actual outlier detection
        db.session.commit()
        
        # Log completion
        log_entry.step_status = 'completed'
        log_entry.end_time = pd.Timestamp.now()
        log_entry.log_message = 'Data processing completed successfully'
        log_entry.step_data = safe_json_serialize({
            'summary_stats': summary_stats,
            'quality_score': quality_score,
            'validation_results': validation_results,
            'ai_insights': ai_insights,
            'raw_stats': raw_stats,
            'cleaning_config': cleaning_config
        })
        db.session.commit()
        
        flash('Data processing completed successfully!', 'success')
        return redirect(url_for('view_results', session_id=session_id))
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        upload_session.status = 'error'
        upload_session.error_message = str(e)
        db.session.commit()
        
        flash(f'Processing failed: {str(e)}', 'error')
        return redirect(url_for('configure_schema', session_id=session_id))

@app.route('/results/<int:session_id>')
@login_required
def view_results(session_id):
    """View processing results and analysis"""
    upload_session = UploadSession.query.get_or_404(session_id)
    if upload_session.user_id != current_user.id and current_user.role != 'ADMIN':
        flash('You do not have permission to access this file.', 'error')
        return redirect(url_for('index'))
    processing_logs = ProcessingLog.query.filter_by(session_id=session_id).all()
    
    # Get the latest processing results
    latest_log = ProcessingLog.query.filter_by(
        session_id=session_id, 
        step_name='data_processing', 
        step_status='completed'
    ).order_by(ProcessingLog.end_time.desc()).first()
    
    results_data = latest_log.step_data if latest_log else {}
    
    return render_template('results.html',
                         session=upload_session,
                         processing_logs=processing_logs,
                         results_data=results_data)

@app.route('/generate_report/<int:session_id>/<report_type>')
@login_required
def generate_report(session_id, report_type):
    """Generate PDF or HTML report"""
    upload_session = UploadSession.query.get_or_404(session_id)
    if upload_session.user_id != current_user.id and current_user.role != 'ADMIN':
        flash('You do not have permission to access this file.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get processing results
        latest_log = ProcessingLog.query.filter_by(
            session_id=session_id,
            step_name='data_processing',
            step_status='completed'
        ).order_by(ProcessingLog.end_time.desc()).first()
        
        if not latest_log:
            flash('No processing results found. Please process the data first.', 'error')
            return redirect(url_for('view_results', session_id=session_id))
        
        # Generate report
        report_generator = ReportGenerator()
        report_data = {
            'session': upload_session,
            'results': latest_log.step_data,
            'processing_logs': ProcessingLog.query.filter_by(session_id=session_id).all()
        }
        
        if report_type == 'pdf':
            report_filename = report_generator.generate_pdf_report(report_data, session_id)
        else:  # html
            report_filename = report_generator.generate_html_report(report_data, session_id)
        
        # Save report record
        report_record = ReportGeneration()
        report_record.session_id = session_id
        report_record.report_type = report_type
        report_record.report_filename = report_filename
        report_path = os.path.join('reports', report_filename)
        report_record.report_size = os.path.getsize(report_path) if os.path.exists(report_path) else 0
        db.session.add(report_record)
        db.session.commit()
        
        flash(f'{report_type.upper()} report generated successfully!', 'success')
        return redirect(url_for('download_report', filename=report_filename))
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        flash(f'Report generation failed: {str(e)}', 'error')
        return redirect(url_for('view_results', session_id=session_id))

@app.route('/download/<filename>')
@login_required
def download_report(filename):
    """Download generated report"""
    report = ReportGeneration.query.filter_by(report_filename=filename).first_or_404()
    upload_session = UploadSession.query.get_or_404(report.session_id)
    if upload_session.user_id != current_user.id and current_user.role != 'ADMIN':
        flash('You do not have permission to access this file.', 'error')
        return redirect(url_for('index'))
    return send_file(
        os.path.join('reports', filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/processing_status/<int:session_id>')
def get_processing_status(session_id):
    """API endpoint to check processing status"""
    upload_session = UploadSession.query.get_or_404(session_id)
    latest_log = ProcessingLog.query.filter_by(session_id=session_id).order_by(
        ProcessingLog.start_time.desc()
    ).first()
    
    return jsonify({
        'status': upload_session.status,
        'current_step': latest_log.step_name if latest_log else None,
        'step_status': latest_log.step_status if latest_log else None,
        'message': latest_log.log_message if latest_log else None
    })

@app.route('/api/outlier_analysis/<int:session_id>')
def get_outlier_analysis(session_id):
    """API endpoint to get outlier detection method estimates"""
    try:
        upload_session = UploadSession.query.get_or_404(session_id)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
        
        # Load data for analysis
        data_processor = DataProcessor()
        df = data_processor.load_data(filepath)
        
        # Initialize outlier detector and analyze
        outlier_detector = OutlierDetector()
        analysis_results = outlier_detector.analyze_sample_and_estimate_accuracy(df, sample_size=10)
        
        return jsonify({
            'success': True,
            'analysis': analysis_results
        })
        
    except Exception as e:
        logger.error(f"Outlier analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reset_session/<int:session_id>', methods=['POST'])
def reset_session(session_id):
    """Reset a stuck processing session"""
    upload_session = UploadSession.query.get_or_404(session_id)
    
    try:
        # Update session status to failed if it's stuck
        if upload_session.status in ['processing', 'pending']:
            upload_session.status = 'uploaded'
            
            # Add a log entry for the reset
            reset_log = ProcessingLog()
            reset_log.session_id = session_id
            reset_log.step_name = 'session_reset'
            reset_log.step_status = 'completed'
            reset_log.log_message = 'Session reset by user - ready for reprocessing'
            db.session.add(reset_log)
            db.session.commit()
            
            flash('Session reset successfully. You can now process the file again.', 'success')
        else:
            flash('Session is not stuck and cannot be reset.', 'info')
            
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        flash('Failed to reset session.', 'error')
        db.session.rollback()
    
    return redirect(url_for('view_results', session_id=session_id))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('upload_file'))

@app.errorhandler(404)
def not_found(e):
    return render_template('base.html', content='<div class="text-center"><h2>Page Not Found</h2><p>The requested page could not be found.</p></div>'), 404

@app.route('/download_cleaned/<int:session_id>')
def download_cleaned_data(session_id):
    """Download cleaned dataset"""
    upload_session = UploadSession.query.get_or_404(session_id)
    
    # Check if session has been processed and has cleaned data
    if upload_session.status != 'completed':
        flash('Data processing must be completed before downloading cleaned data.', 'error')
        return redirect(url_for('view_results', session_id=session_id))
    
    # Get cleaned filename from session config
    cleaning_config = upload_session.cleaning_config or {}
    cleaned_filename = cleaning_config.get('cleaned_filename')
    
    if not cleaned_filename:
        flash('No cleaned data available for download.', 'error')
        return redirect(url_for('view_results', session_id=session_id))
    
    cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
    
    if not os.path.exists(cleaned_filepath):
        flash('Cleaned data file not found.', 'error')
        return redirect(url_for('view_results', session_id=session_id))
    
    # Determine file type and mimetype
    original_extension = upload_session.filename.lower().split('.')[-1]
    base_name = upload_session.original_filename.rsplit('.', 1)[0]
    timestamp = upload_session.upload_time.strftime('%Y%m%d_%H%M%S')
    
    if original_extension == 'csv':
        download_filename = f"{base_name}_cleaned_{timestamp}.csv"
        mimetype = 'text/csv'
    elif original_extension in ['xlsx', 'xls']:
        download_filename = f"{base_name}_cleaned_{timestamp}.xlsx"
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        download_filename = f"{base_name}_cleaned_{timestamp}.{original_extension}"
        mimetype = 'application/octet-stream'
    
    return send_file(
        cleaned_filepath,
        mimetype=mimetype,
        as_attachment=True,
        download_name=download_filename
    )

@app.route('/chat/<int:session_id>')
def chat(session_id):
    """Chat interface for conversational analysis"""
    upload_session = UploadSession.query.get_or_404(session_id)
    
    # Create or get chat session
    chat_session = ChatSession.query.filter_by(session_id=session_id).first()
    if not chat_session:
        chat_session = ChatSession(session_id=session_id)
        db.session.add(chat_session)
        db.session.commit()
    
    messages = ChatMessage.query.filter_by(chat_session_id=chat_session.id).order_by(ChatMessage.created_at).all()
    
    return render_template('chat.html',
                         session=upload_session,
                         chat_session=chat_session,
                         messages=messages)

@app.route('/api/chat/<int:session_id>', methods=['POST'])
def send_message(session_id):
    """API endpoint for sending chat messages"""
    try:
        upload_session = UploadSession.query.get_or_404(session_id)
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create chat session
        chat_session = ChatSession.query.filter_by(session_id=session_id).first()
        if not chat_session:
            chat_session = ChatSession(session_id=session_id)
            db.session.add(chat_session)
            db.session.commit()
        
        # Save user message
        user_msg = ChatMessage(
            chat_session_id=chat_session.id,
            role='user',
            content=user_message
        )
        db.session.add(user_msg)
        db.session.commit()
        
        # Get conversation context
        messages = ChatMessage.query.filter_by(chat_session_id=chat_session.id).order_by(ChatMessage.created_at).all()
        
        # Load data and get AI response
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
        data_processor = DataProcessor()
        df = data_processor.load_data(filepath)
        
        gemini_service = GeminiService()
        ai_response = gemini_service.chat_with_data(df, messages, user_message)
        
        # Save AI response
        ai_msg = ChatMessage(
            chat_session_id=chat_session.id,
            role='assistant',
            content=ai_response
        )
        db.session.add(ai_msg)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'message_id': ai_msg.id
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat_history/<int:session_id>')
def get_chat_history(session_id):
    """API endpoint to get chat history"""
    try:
        chat_session = ChatSession.query.filter_by(session_id=session_id).first()
        if not chat_session:
            return jsonify({'messages': []})
        
        messages = ChatMessage.query.filter_by(chat_session_id=chat_session.id).order_by(ChatMessage.created_at).all()
        
        return jsonify({
            'messages': [
                {
                    'id': msg.id,
                    'role': msg.role,
                    'content': msg.content,
                    'created_at': msg.created_at.isoformat()
                }
                for msg in messages
            ]
        })
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/<int:session_id>/forecast', methods=['POST'])
def get_forecast(session_id):
    """Generate time series forecast"""
    try:
        upload_session = UploadSession.query.get_or_404(session_id)
        data = request.get_json()
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
        data_processor = DataProcessor()
        df = data_processor.load_data(filepath)
        
        gemini_service = GeminiService()
        forecast_result = gemini_service.generate_forecast(df, data)
        
        # Store result
        pred_analysis = PredictiveAnalysis(
            session_id=session_id,
            analysis_type='forecast',
            metric_name=data.get('metric_column'),
            results=forecast_result
        )
        db.session.add(pred_analysis)
        
        # Log activity
        activity = ActivityLog(
            session_id=session_id,
            action='forecast_generated',
            action_type='analyze',
            details={'metric': data.get('metric_column')}
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({'success': True, 'forecast': forecast_result})
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/<int:session_id>/anomalies', methods=['POST'])
def detect_anomalies(session_id):
    """Detect anomalies in data"""
    try:
        upload_session = UploadSession.query.get_or_404(session_id)
        data = request.get_json()
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_session.filename)
        data_processor = DataProcessor()
        df = data_processor.load_data(filepath)
        
        gemini_service = GeminiService()
        anomalies = gemini_service.detect_anomalies(df, data.get('columns', []))
        
        # Store result
        pred_analysis = PredictiveAnalysis(
            session_id=session_id,
            analysis_type='anomaly',
            results=anomalies
        )
        db.session.add(pred_analysis)
        
        # Log activity
        activity = ActivityLog(
            session_id=session_id,
            action='anomalies_detected',
            action_type='analyze',
            details=anomalies
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({'success': True, 'anomalies': anomalies})
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<int:session_id>', methods=['POST'])
def export_data(session_id):
    """Export processed data"""
    try:
        upload_session = UploadSession.query.get_or_404(session_id)
        data = request.get_json()
        export_format = data.get('format', 'csv')
        
        if upload_session.status != 'completed':
            return jsonify({'error': 'Data processing must be completed first'}), 400
        
        cleaning_config = upload_session.cleaning_config or {}
        cleaned_filename = cleaning_config.get('cleaned_filename')
        
        if not cleaned_filename:
            return jsonify({'error': 'No cleaned data available'}), 400
        
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        
        if not os.path.exists(cleaned_filepath):
            return jsonify({'error': 'Cleaned data file not found'}), 404
        
        # Generate export filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        base_name = upload_session.original_filename.rsplit('.', 1)[0]
        
        if export_format == 'json':
            df = pd.read_csv(cleaned_filepath) if cleaned_filename.endswith('.csv') else pd.read_excel(cleaned_filepath)
            export_filename = f"export_{base_name}_{timestamp}.json"
            export_filepath = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
            df.to_json(export_filepath, orient='records', indent=2)
        else:
            export_filename = f"export_{base_name}_{timestamp}.{export_format}"
            export_filepath = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
            if export_format == 'csv':
                df = pd.read_csv(cleaned_filepath) if cleaned_filename.endswith('.csv') else pd.read_excel(cleaned_filepath)
                df.to_csv(export_filepath, index=False)
            elif export_format == 'excel':
                df = pd.read_csv(cleaned_filepath) if cleaned_filename.endswith('.csv') else pd.read_excel(cleaned_filepath)
                df.to_excel(export_filepath, index=False, engine='openpyxl')
        
        # Track export
        file_size = os.path.getsize(export_filepath) if os.path.exists(export_filepath) else 0
        data_export = DataExport(
            session_id=session_id,
            export_type=export_format,
            filename=export_filename,
            file_size=file_size
        )
        db.session.add(data_export)
        
        # Log activity
        activity = ActivityLog(
            session_id=session_id,
            action='data_exported',
            action_type='download',
            details={'format': export_format, 'size': file_size}
        )
        db.session.add(activity)
        db.session.commit()
        
        return send_file(
            export_filepath,
            as_attachment=True,
            download_name=export_filename
        )
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/<int:session_id>', methods=['GET'])
def get_advanced_insights(session_id):
    """Get advanced AI-powered insights"""
    try:
        upload_session = UploadSession.query.get_or_404(session_id)
        
        latest_log = ProcessingLog.query.filter_by(
            session_id=session_id,
            step_name='data_processing',
            step_status='completed'
        ).order_by(ProcessingLog.end_time.desc()).first()
        
        if not latest_log:
            return jsonify({'error': 'No analysis results found'}), 404
        
        results_data = latest_log.step_data or {}
        gemini_service = GeminiService()
        advanced_insights = gemini_service.generate_advanced_insights(results_data)
        
        return jsonify({'insights': advanced_insights})
    except Exception as e:
        logger.error(f"Advanced insights error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/activity/<int:session_id>', methods=['GET'])
def get_activity_log(session_id):
    """Get activity log for a session"""
    try:
        activities = ActivityLog.query.filter_by(session_id=session_id).order_by(
            ActivityLog.created_at.desc()
        ).all()
        
        return jsonify({
            'activities': [
                {
                    'id': a.id,
                    'action': a.action,
                    'type': a.action_type,
                    'details': a.details,
                    'timestamp': a.created_at.isoformat()
                }
                for a in activities
            ]
        })
    except Exception as e:
        logger.error(f"Activity log error: {str(e)}")
        return jsonify({'error': str(e)}), 500
