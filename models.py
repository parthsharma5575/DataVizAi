from datetime import datetime, timedelta
from app import db
from sqlalchemy import Text, JSON, ForeignKey
from flask_login import UserMixin

class User(db.Model, UserMixin):
    """Model for users"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default='USER', nullable=False)  # USER, ADMIN
    status = db.Column(db.String(50), default='UNVERIFIED', nullable=False)  # UNVERIFIED, ACTIVE, DISABLED
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    uploads = db.relationship('UploadSession', backref='owner', lazy=True)
    
    def __repr__(self):
        return f'<User {self.email}>'

class OTP(db.Model):
    """Model for one-time passwords"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    otp_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(minutes=10))
    
    user = db.relationship('User', backref='otps')

    def __repr__(self):
        return f'<OTP for User {self.user_id}>'

class ChatSession(db.Model):
    """Model to store chat conversation sessions"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('upload_session.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    upload_session = db.relationship('UploadSession', backref='chat_sessions')
    messages = db.relationship('ChatMessage', backref='chat_session', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<ChatSession {self.id}>'

class ChatMessage(db.Model):
    """Model to store individual chat messages"""
    id = db.Column(db.Integer, primary_key=True)
    chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatMessage {self.role}>'

class PredictiveAnalysis(db.Model):
    """Model to store predictive analytics results"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('upload_session.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # 'forecast', 'anomaly', 'trend'
    metric_name = db.Column(db.String(255))
    results = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    upload_session = db.relationship('UploadSession', backref='predictions')
    
    def __repr__(self):
        return f'<PredictiveAnalysis {self.analysis_type}>'

class DataExport(db.Model):
    """Model to track data exports"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('upload_session.id'), nullable=False)
    export_type = db.Column(db.String(50), nullable=False)  # 'csv', 'excel', 'json'
    filename = db.Column(db.String(255), nullable=False)
    exported_at = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer)
    
    upload_session = db.relationship('UploadSession', backref='exports')
    
    def __repr__(self):
        return f'<DataExport {self.export_type}>'

class ActivityLog(db.Model):
    """Model to track user activities"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('upload_session.id'))
    action = db.Column(db.String(100), nullable=False)
    action_type = db.Column(db.String(50))  # 'view', 'download', 'analyze', 'share'
    details = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    upload_session = db.relationship('UploadSession', backref='activities')
    
    def __repr__(self):
        return f'<ActivityLog {self.action}>'

class UploadSession(db.Model):
    """Model to track file upload sessions and their processing status"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer)
    status = db.Column(db.String(50), default='uploaded')  # uploaded, processing, completed, error
    error_message = db.Column(Text)
    
    # Storage for file content directly in DB
    file_content = db.Column(db.LargeBinary) # Store raw bytes of the file
    processed_data = db.Column(JSON) # Store processed dataframe as JSON for quick access
    
    # Data processing metadata
    total_rows = db.Column(db.Integer)
    total_columns = db.Column(db.Integer)
    missing_values_count = db.Column(db.Integer)
    outliers_detected = db.Column(db.Integer)
    
    # Processing configuration
    schema__config = db.Column(JSON)
    cleaning_config = db.Column(JSON)
    weight_config = db.Column(JSON)
    
    def __repr__(self):
        return f'<UploadSession {self.filename}>'

class ProcessingLog(db.Model):
    """Model to track processing steps and their outcomes"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('upload_session.id'), nullable=False)
    step_name = db.Column(db.String(100), nullable=False)
    step_status = db.Column(db.String(50), nullable=False)  # started, completed, failed
    start_time = db.Column(db.DateTime, default=datetime.now)
    end_time = db.Column(db.DateTime)
    log_message = db.Column(Text)
    step_data = db.Column(JSON)  # Store step-specific data
    
    session = db.relationship('UploadSession', backref='processing_logs')
    
    def __repr__(self):
        return f'<ProcessingLog {self.step_name} - {self.step_status}>'

class ReportGeneration(db.Model):
    """Model to track generated reports"""
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('upload_session.id'), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # pdf, html
    report_filename = db.Column(db.String(255), nullable=False)
    generation_time = db.Column(db.DateTime, default=datetime.utcnow)
    report_size = db.Column(db.Integer)
    
    session = db.relationship('UploadSession', backref='reports')
    
    def __repr__(self):
        return f'<Report {self.report_filename}>'
