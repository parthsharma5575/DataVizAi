from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from decorators import require_role
from models import User, UploadSession

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/dashboard')
@login_required
@require_role('ADMIN')
def dashboard():
    """Admin dashboard"""
    return render_template('admin/dashboard.html')

@admin_bp.route('/files')
@login_required
@require_role('ADMIN')
def files():
    """API endpoint for admins to get all files"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    files = UploadSession.query.order_by(UploadSession.upload_time.desc()).paginate(page=page, per_page=per_page)
    return jsonify({
        'files': [{
            'id': f.id,
            'filename': f.original_filename,
            'upload_time': f.upload_time.isoformat(),
            'file_size': f.file_size,
            'status': f.status,
            'owner': {
                'id': f.owner.id,
                'name': f.owner.name,
                'email': f.owner.email
            }
        } for f in files.items],
        'total': files.total,
        'pages': files.pages,
        'current_page': files.page
    })

@admin_bp.route('/users')
@login_required
@require_role('ADMIN')
def users():
    """API endpoint for admins to get all users"""
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'name': u.name,
        'email': u.email,
        'phone': u.phone,
        'role': u.role,
        'status': u.status,
        'created_at': u.created_at.isoformat()
    } for u in users])

@admin_bp.route('/users/<int:user_id>')
@login_required
@require_role('ADMIN')
def user(user_id):
    """API endpoint for admins to get a single user"""
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'phone': user.phone,
        'role': user.role,
        'status': user.status,
        'created_at': user.created_at.isoformat()
    })
