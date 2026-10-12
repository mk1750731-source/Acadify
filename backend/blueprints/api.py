from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from decorators import role_required
from models.db import get_db
from models.department import DepartmentModel
from models.student import StudentModel
from models.prediction import PredictionModel
from models.notification import NotificationModel
from ml.predictor import predict_student_risk, get_model_info

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/dashboard-stats')
def dashboard_stats():
    db = get_db()
    total_students = db.students.count_documents({})
    high_risk  = db.students.count_documents({'risk_level': 'HIGH'})
    medium_risk = db.students.count_documents({'risk_level': 'MEDIUM'})
    low_risk   = db.students.count_documents({'risk_level': 'LOW'})

    depts = DepartmentModel.get_all()
    dept_labels      = [d['name'] for d in depts]
    dept_students    = [d['total_students'] for d in depts]
    dept_gpa         = [d['avg_gpa'] for d in depts]
    dept_attendance  = [d['avg_attendance'] for d in depts]

    gpa_buckets = {
        '3.5 - 4.0 (Honors)':   db.students.count_documents({'gpa': {'$gte': 3.5}}),
        '3.0 - 3.49 (Good)':    db.students.count_documents({'gpa': {'$gte': 3.0, '$lt': 3.5}}),
        '2.5 - 2.99 (Average)': db.students.count_documents({'gpa': {'$gte': 2.5, '$lt': 3.0}}),
        'Below 2.5 (At Risk)':  db.students.count_documents({'gpa': {'$lt': 2.5}})
    }

    model_info = get_model_info()

    return jsonify({
        'status': 'success',
        'kpis': {
            'total_students': total_students,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'prediction_accuracy': model_info['accuracy']
        },
        'risk_distribution': {
            'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
            'data': [low_risk, medium_risk, high_risk],
            'colors': ['#10B981', '#F59E0B', '#F43F5E']
        },
        'department_comparison': {
            'labels': dept_labels,
            'students': dept_students,
            'gpa': dept_gpa,
            'attendance': dept_attendance
        },
        'gpa_distribution': {
            'labels': list(gpa_buckets.keys()),
            'data': list(gpa_buckets.values())
        },
        'model': model_info
    })


@api_bp.route('/ai/status')
@login_required
@role_required('admin')
def ai_model_status():
    models = {}
    try:
        model_info = get_model_info()
        models['student_risk'] = {
            'ready': True,
            'version': model_info['version'],
            'accuracy': model_info['accuracy'],
            'features': model_info['features'],
        }
        models['gpa_forecast'] = {'ready': True}
    except Exception as error:
        models['student_risk'] = {'ready': False, 'error': str(error)}
        models['gpa_forecast'] = {'ready': False, 'error': str(error)}

    try:
        from ai_service import emotion_predictor, face_recognizer
        models['facial_expression'] = {'ready': emotion_predictor.ready}
        models['face_recognition'] = {'ready': face_recognizer.ready}
    except Exception as error:
        models['facial_expression'] = {'ready': False, 'error': str(error)}
        models['face_recognition'] = {'ready': False, 'error': str(error)}

    return jsonify({'models': models})


@api_bp.route('/predict-form', methods=['POST'])
@login_required
def predict_form():
    """Run prediction from a manually filled form with all 19 model features."""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    student_id = data.get('student_id')
    student = StudentModel.get_by_id(student_id) if student_id else None

    # Build feature dict from form input
    feature_data = {
        'department':                data.get('department', 'General'),
        'gpa':                       float(data.get('gpa_current', 0)),
        'gpa_current':               float(data.get('gpa_current', 0)),
        'gpa_previous_sem':          float(data.get('gpa_previous_sem', 0)),
        'gpa_two_sems_ago':          float(data.get('gpa_two_sems_ago', 0)),
        'attendance_pct':            float(data.get('attendance_pct', 0)),
        'attendance_previous_sem':   float(data.get('attendance_previous_sem', 0)),
        'assignments_submitted_pct': float(data.get('assignments_submitted_pct', 85)),
        'avg_submission_delay_days': float(data.get('avg_submission_delay_days', 1)),
        'study_hours_weekly':        float(data.get('study_hours_weekly', 10)),
        'login_frequency_weekly':    float(data.get('login_frequency_weekly', 5)),
        'extracurricular_hours':     float(data.get('extracurricular_hours', 2)),
        'cohort_gpa_avg':            float(data.get('cohort_gpa_avg', 3.0)),
        'cohort_attendance_avg':     float(data.get('cohort_attendance_avg', 80)),
        'quiz_avg_score':            float(data.get('quiz_avg_score', 70)),
        'quiz_score_std':            float(data.get('quiz_score_std', 10)),
    }

    pred = predict_student_risk(feature_data)

    # Save to DB if student_id provided
    if student:
        PredictionModel.create(
            student_id=student['_id'],
            risk_score=pred['risk_score'],
            confidence=pred['confidence'],
            performance_forecast=pred['performance_forecast'],
            recommendations=pred['recommendations'],
            model_version=pred['model_version'],
            risk_level=pred['risk_level'],
            risk_probabilities=pred.get('risk_probabilities', {}),
            model_accuracy=pred.get('model_accuracy'),
            regressor_r2=pred.get('regressor_r2'),
        )
        StudentModel.update(student['_id'], {
            'risk_score': pred['risk_score'],
            'risk_level': pred['risk_level'],
        })
        if pred['risk_level'] == 'HIGH':
            NotificationModel.create(
                student_id=student['_id'],
                message="High Academic Risk Alert: Your current metrics have triggered an automated counseling recommendation.",
                notif_type="warning"
            )
        DepartmentModel.recalculate_stats()

    return jsonify({'status': 'success', 'prediction': {
        'risk_level':           pred['risk_level'],
        'risk_score':           pred['risk_score'],
        'confidence':           pred['confidence'],
        'risk_probabilities':   pred.get('risk_probabilities', {}),
        'performance_forecast': pred['performance_forecast'],
        'recommendations':      pred['recommendations'],
        'model_version':        pred['model_version'],
        'model_accuracy':       pred.get('model_accuracy'),
        'saved':                student is not None,
    }})


@api_bp.route('/predict/<student_id>', methods=['POST'])
@login_required
def run_prediction(student_id):
    """Re-run ML prediction for a student and save to DB dynamically."""
    student = StudentModel.get_by_id(student_id)
    if not student:
        return jsonify({'status': 'error', 'message': 'Student not found'}), 404

    # Only admin or the student themselves
    if current_user.role != 'admin':
        from models.student import StudentModel as SM
        own = SM.get_by_user_id(current_user.id)
        if not own or str(own['_id']) != student_id:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403

    pred = predict_student_risk(student)

    PredictionModel.create(
        student_id=student['_id'],
        risk_score=pred['risk_score'],
        confidence=pred['confidence'],
        performance_forecast=pred['performance_forecast'],
        recommendations=pred['recommendations'],
        model_version=pred['model_version'],
        risk_level=pred['risk_level'],
        risk_probabilities=pred.get('risk_probabilities', {}),
        model_accuracy=pred.get('model_accuracy'),
        regressor_r2=pred.get('regressor_r2'),
    )

    if pred['risk_level'] == 'HIGH':
        NotificationModel.create(
            student_id=student['_id'],
            message="High Academic Risk Alert: Your current metrics have triggered an automated counseling recommendation.",
            notif_type="warning"
        )

    DepartmentModel.recalculate_stats()

    return jsonify({
        'status': 'success',
        'prediction': {
            'risk_level':         pred['risk_level'],
            'risk_score':         pred['risk_score'],
            'confidence':         pred['confidence'],
            'risk_probabilities': pred.get('risk_probabilities', {}),
            'performance_forecast': pred['performance_forecast'],
            'recommendations':    pred['recommendations'],
            'model_version':      pred['model_version'],
            'model_accuracy':     pred.get('model_accuracy'),
        }
    })
