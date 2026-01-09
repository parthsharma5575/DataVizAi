from flask import Blueprint, request, jsonify, flash, redirect, url_for
from app import db, bcrypt
from models import User, OTP
from flask_login import login_user, logout_user, login_required
import random
import string
import logging
from datetime import datetime

auth_bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

from email_service import send_otp_email

@auth_bp.route('/signup', methods=['POST'])
def signup():
    """User signup"""
    data = request.get_json()
    logger.info(f"Signup request received: {data}")
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    password = data.get('password')

    if not all([name, email, password]):
        logger.error("Signup failed: Missing required fields")
        return jsonify({'error': 'Missing required fields'}), 400

    if User.query.filter_by(email=email).first():
        logger.error(f"Signup failed: Email {email} already registered")
        return jsonify({'error': 'Email already registered'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    new_user = User(
        name=name,
        email=email,
        phone=phone,
        password_hash=hashed_password,
        status='UNVERIFIED'
    )
    db.session.add(new_user)
    db.session.commit()
    logger.info(f"New user created: {email}")

    # Generate and send OTP
    otp_code = ''.join(random.choices(string.digits, k=6))
    otp_hash = bcrypt.generate_password_hash(otp_code).decode('utf-8')
    
    new_otp = OTP(user_id=new_user.id, otp_hash=otp_hash)
    db.session.add(new_otp)
    db.session.commit()

    send_otp_email(new_user.email, otp_code)
    logger.info(f"OTP sent to {email}")

    return jsonify({'success': True, 'message': 'OTP sent to your email.', 'userId': new_user.id}), 201

@auth_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP"""
    data = request.get_json()
    logger.info(f"OTP verification request received: {data}")
    user_id = data.get('userId')
    otp_code = data.get('otp')

    if not all([user_id, otp_code]):
        logger.error("OTP verification failed: User ID and OTP are required")
        return jsonify({'error': 'User ID and OTP are required'}), 400

    user = User.query.get(user_id)
    if not user:
        logger.error(f"OTP verification failed: User not found for ID {user_id}")
        return jsonify({'error': 'User not found'}), 404

    otp_record = OTP.query.filter_by(user_id=user_id).order_by(OTP.created_at.desc()).first()

    if not otp_record or otp_record.expires_at < datetime.utcnow():
        logger.error(f"OTP verification failed: OTP expired or invalid for user {user_id}")
        return jsonify({'error': 'OTP expired or invalid'}), 400

    if bcrypt.check_password_hash(otp_record.otp_hash, otp_code):
        user.status = 'ACTIVE'
        db.session.delete(otp_record)  # OTP is single-use
        db.session.commit()
        login_user(user)
        logger.info(f"User {user.email} verified and logged in")
        return jsonify({'success': True, 'message': 'Account verified successfully.'}), 200
    else:
        logger.error(f"OTP verification failed: Invalid OTP for user {user_id}")
        return jsonify({'error': 'Invalid OTP'}), 400

@auth_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    logger.info(f"Login request received: {data}")
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        logger.error("Login failed: Email and password are required")
        return jsonify({'error': 'Email and password are required'}), 400

    user = User.query.filter_by(email=email).first()

    if user and user.status == 'ACTIVE' and bcrypt.check_password_hash(user.password_hash, password):
        login_user(user, remember=True)
        logger.info(f"User {email} logged in successfully")
        return jsonify({'success': True, 'message': 'Logged in successfully.'}), 200
    elif user and user.status != 'ACTIVE':
        logger.error(f"Login failed: Account for {email} not active or is disabled")
        return jsonify({'error': 'Account not active or is disabled.'}), 403
    else:
        logger.error(f"Login failed: Invalid credentials for {email}")
        return jsonify({'error': 'Invalid credentials'}), 401

@auth_bp.route('/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP"""
    data = request.get_json()
    user_id = data.get('userId')
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Generate and send new OTP
    otp_code = ''.join(random.choices(string.digits, k=6))
    otp_hash = bcrypt.generate_password_hash(otp_code).decode('utf-8')
    
    # Invalidate old OTPs
    OTP.query.filter_by(user_id=user_id).delete()

    new_otp = OTP(user_id=user.id, otp_hash=otp_hash)
    db.session.add(new_otp)
    db.session.commit()

    send_otp_email(user.email, otp_code)
    logger.info(f"OTP resent to {user.email}")

    return jsonify({'success': True, 'message': 'OTP resent successfully.'}), 200

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('homepage'))
