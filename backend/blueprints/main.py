import datetime
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
from flask_login import current_user

main_bp = Blueprint('main', __name__)


@main_bp.route('/api/session')
def session_info():
    if not current_user.is_authenticated:
        return jsonify({'user': None}), 401

    return jsonify({
        'user': {
            'id': str(current_user.id),
            'name': current_user.name,
            'email': current_user.email,
            'role': current_user.role,
        }
    })


@main_bp.route('/')
def index():
    import os
    frontend = os.environ.get('FRONTEND_URL', 'https://acadify-nu-dusky.vercel.app').rstrip('/')
    return redirect(frontend)


@main_bp.route('/about')
def about():
    return render_template('about.html')


@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        flash(
            'Thank you for contacting EDU Intelligence Administration. '
            'Our team will get back to you shortly.',
            'success'
        )
        return redirect(url_for('main.contact'))

    return render_template('contact.html')


@main_bp.route('/research')
def research():
    return render_template('research.html')
