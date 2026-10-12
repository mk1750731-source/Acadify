from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from models.user import User
from models.student import StudentModel
from ml.predictor import predict_student_risk
from models.prediction import PredictionModel
import secrets, datetime
from models.db import get_db

auth_bp = Blueprint('auth', __name__)


def _dashboard_endpoint(role):
    return {
        'admin': 'admin.dashboard',
        'teacher': 'teacher.dashboard',
        'parent': 'parent.dashboard',
    }.get(role, 'student.dashboard')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    from flask import jsonify
    if request.method == 'GET':
        if current_user.is_authenticated:
            return jsonify({'authenticated': True, 'role': current_user.role})
        return jsonify({'authenticated': False})

    # POST — accept both JSON and form data
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or request.form.get('email', '')).strip()
    password = data.get('password') or request.form.get('password', '')

    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Please enter both email and password.'}), 400

    user = User.get_by_email(email)
    if user and user.check_password(password):
        login_user(user, remember=True)
        return jsonify({'status': 'success', 'user': {
            'id': str(user._id),
            'name': user.name,
            'email': user.email,
            'role': user.role,
        }})
    return jsonify({'status': 'error', 'message': 'Invalid email address or password.'}), 401

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        roll_no = request.form.get('roll_no', '').strip().upper()
        department = request.form.get('department', '').strip()
        semester = request.form.get('semester', 1)

        # Basic validations
        if not name or not email or not password or not roll_no or not department:
            flash('Please fill in all required fields.', 'danger')
            return render_template('auth/register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth/register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return render_template('auth/register.html')

        # Check existing roll no
        if StudentModel.get_by_roll_no(roll_no):
            flash(f'Roll Number "{roll_no}" is already registered.', 'danger')
            return render_template('auth/register.html')

        # Create user account
        user, err = User.create(name, email, password, role='student')
        if err:
            flash(err, 'danger')
            return render_template('auth/register.html')

        # Create student profile with real initial zeros
        student = StudentModel.create(
            user_id=user._id,
            roll_no=roll_no,
            department=department,
            semester=int(semester),
            gpa=0.0,
            attendance_pct=0.0,
            risk_level='LOW',
            risk_score=0.0,
            last_prediction=None
        )

        # Seed default subjects for attendance tracking
        from models.attendance import AttendanceModel
        AttendanceModel.seed_subjects(student['_id'], department, int(semester))

        # Seed default timetable
        try:
            from models.timetable import TimetableModel
            TimetableModel.seed_default(student['_id'], department)
        except Exception:
            pass

        # Welcome notification
        from models.notification import NotificationModel
        NotificationModel.create(student['_id'], f'Welcome to NEXUS, {name}! Your student portal is ready. Start by marking your attendance and running an AI prediction.', 'info')

        # Log activity
        from models.activity import ActivityModel
        ActivityModel.log('registration', f'New Student Registered', f'{name} ({roll_no}) joined {department}', student_id=student['_id'])

        login_user(user)
        flash('Account registered successfully! Welcome to NEXUS Student Portal.', 'success')
        return redirect(url_for('student.dashboard'))

    return render_template('auth/register.html')

@auth_bp.route('/logout')
def logout():
    from flask import jsonify
    logout_user()
    return jsonify({'status': 'success', 'message': 'Logged out successfully.'})


@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    if not email:
        return jsonify({'success': False, 'message': 'Email is required.'}), 400

    user = User.get_by_email(email)
    if not user:
        # Don't reveal if email exists
        return jsonify({'success': True, 'message': 'If this email is registered, a reset token has been generated.'})

    token = secrets.token_urlsafe(32)
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    db = get_db()
    db.password_reset_tokens.delete_many({'user_id': user._id})
    db.password_reset_tokens.insert_one({
        'user_id': user._id,
        'token': token,
        'expires_at': expires,
        'used': False
    })
    return jsonify({'success': True, 'token': token, 'message': 'Reset token generated. Use it to set a new password.'})


@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json(silent=True) or {}
    token = data.get('token', '').strip()
    new_password = data.get('password', '')

    if not token or not new_password:
        return jsonify({'success': False, 'message': 'Token and new password are required.'}), 400
    if len(new_password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'}), 400

    db = get_db()
    record = db.password_reset_tokens.find_one({'token': token, 'used': False})
    if not record:
        return jsonify({'success': False, 'message': 'Invalid or expired reset token.'}), 400

    now = datetime.datetime.now(datetime.timezone.utc)
    if record['expires_at'].replace(tzinfo=datetime.timezone.utc) < now:
        return jsonify({'success': False, 'message': 'Reset token has expired. Please request a new one.'}), 400

    user = User.get_by_id(str(record['user_id']))
    if not user:
        return jsonify({'success': False, 'message': 'User not found.'}), 404

    user.set_password(new_password)
    db.password_reset_tokens.update_one({'_id': record['_id']}, {'$set': {'used': True}})
    return jsonify({'success': True, 'message': 'Password reset successfully. You can now log in.'})
