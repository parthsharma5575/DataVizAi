import os
import logging
from flask import Flask, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
bcrypt = Bcrypt()
login_manager = LoginManager()

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORT_FOLDER'] = 'reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Database configuration (SQLite for simplicity)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///survey_analysis.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize database and extensions
db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'homepage' 

from auth_routes import auth_bp
app.register_blueprint(auth_bp, url_prefix='/auth')

from admin_routes import admin_bp
app.register_blueprint(admin_bp, url_prefix='/admin')

# Create upload and report directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

with app.app_context():
    import models
    import routes
    db.create_all()

    def create_admin_user():
        """Create admin user if not exists"""
        admin_email = os.environ.get('ADMIN_EMAIL')
        admin_password = os.environ.get('ADMIN_PASSWORD')
        if admin_email and admin_password:
            admin_user = models.User.query.filter_by(email=admin_email).first()
            if not admin_user:
                hashed_password = bcrypt.generate_password_hash(admin_password).decode('utf-8')
                admin = models.User(
                    name='Admin',
                    email=admin_email,
                    password_hash=hashed_password,
                    role='ADMIN',
                    status='ACTIVE'
                )
                db.session.add(admin)
                db.session.commit()
                logging.info(f"Admin user {admin_email} created.")

    create_admin_user()

@login_manager.user_loader
def load_user(user_id):
    return models.User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    """Handle unauthorized access"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Unauthorized'}), 401
    flash("You must be logged in to view this page.", "error")
    return redirect(url_for('homepage', login='true'))
