from functools import wraps
from flask import flash, redirect, url_for, jsonify
from flask_login import current_user

def require_role(role):
    """
    Decorator to require a specific role to access a view.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("You must be logged in to view this page.", "error")
                return redirect(url_for('homepage', login='true'))
            if current_user.role != role:
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Forbidden'}), 403
                flash("You do not have permission to access this page.", "error")
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator
