"""
blueprints/face.py — Face AI endpoints integrated into the existing NEXUS app.

Routes:
  GET  /face/attendance          — webcam attendance page
  GET  /face/register            — face registration page (student)
  POST /api/face/register        — register face embedding for a student
  POST /api/face/recognize       — identify face + mark attendance + expression
  POST /api/face/expression      — expression-only analysis
  GET  /api/face/students        — list students with face registration status
  GET  /api/face/attendance      — attendance history (JSON)
"""
import os
import datetime
import tempfile

from flask import (Blueprint, render_template, request, jsonify,
                   redirect, url_for, flash, send_from_directory, abort, Response)
from flask_login import login_required, current_user
from bson.objectid import ObjectId

from ai_service import emotion_predictor, face_recognizer
from ai_service.preprocessing import (
    decode_base64_image, bgr_to_rgb, resize_if_large,
    validate_image_file, secure_temp_save, allowed_file
)
from models.db import get_db
from decorators import role_required

face_bp = Blueprint('face', __name__)

_UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
os.makedirs(_UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_all_face_students(student_query=None):
    """Return list of students that have at least one face embedding."""
    db = get_db()
    query = {'face_embeddings': {'$exists': True, '$not': {'$size': 0}}}
    if student_query:
        query.update(student_query)
    students = list(db.students.find(query))
    result = []
    for s in students:
        user = db.users.find_one({'_id': s.get('user_id')})
        result.append({
            'student_id': s['_id'],
            'name': user.get('name', 'Unknown') if user else 'Unknown',
            'roll_no': s.get('roll_no', ''),
            'face_embeddings': s.get('face_embeddings', [])
        })
    return result


def _already_marked_today(student_id, session_id=None):
    db = get_db()
    today = datetime.date.today().strftime('%Y-%m-%d')
    query = {'student_id': ObjectId(student_id), 'date': today}
    if session_id:
        query['session_id'] = session_id
    return db.face_attendance.find_one(query) is not None


def _mark_face_attendance(student_id, student_name, roll_no, session_id=None):
    db = get_db()
    now = datetime.datetime.now(datetime.timezone.utc)
    doc = {
        'student_id': ObjectId(student_id),
        'student_name': student_name,
        'roll_no': roll_no,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S'),
        'status': 'present',
        'session_id': session_id or 'default',
        'marked_at': now
    }
    db.face_attendance.insert_one(doc)
    return doc


# ── Page Routes ───────────────────────────────────────────────────────────────

@face_bp.route('/api/face/status')
@login_required
def api_face_status():
    from ai_service import face_recognizer as fr, emotion_predictor as ep
    face_ready = fr.ready
    emotion_ready = ep.ready
    if not face_ready:
        return jsonify({
            'success': False,
            'ready': False,
            'face_recognition': False,
            'emotion': emotion_ready,
            'message': 'Face recognition model is still loading. Please wait a moment and try again.'
        }), 503
    return jsonify({
        'success': True,
        'ready': True,
        'face_recognition': face_ready,
        'emotion': emotion_ready,
    })

@face_bp.route('/face/attendance')
@login_required
@role_required('admin', 'teacher')
def attendance_page():
    """Webcam-based AI attendance page for staff only."""
    db = get_db()
    subjects = list(db.attendance_subjects.aggregate([
        {'$group': {'_id': '$subject'}},
        {'$sort': {'_id': 1}}
    ]))
    subjects = [{'subject': item['_id']} for item in subjects if item.get('_id')]
    return render_template('face/attendance.html', subjects=subjects)


@face_bp.route('/api/face/sessions', methods=['GET'])
@login_required
@role_required('admin', 'teacher')
def api_face_sessions():
    """List attendance sessions for today or by date."""
    db = get_db()
    date_str = request.args.get('date', datetime.date.today().strftime('%Y-%m-%d'))
    query = {'date': date_str}
    if current_user.role == 'teacher':
        query['created_by'] = str(current_user._id)
    sessions = list(db.attendance_sessions.find(query).sort('created_at', -1))
    return jsonify({'success': True, 'sessions': [
        {
            'session_id': str(s['_id']),
            'subject': s['subject'],
            'date': s['date'],
            'day': s['day'],
            'label': s.get('label', ''),
            'created_by_name': s.get('created_by_name', ''),
            'student_count': s.get('student_count', 0),
        } for s in sessions
    ]})


@face_bp.route('/api/face/session/create', methods=['POST'])
@login_required
@role_required('admin', 'teacher')
def api_create_session():
    """Create a new attendance session for a subject."""
    db = get_db()
    data = request.get_json(silent=True) or {}
    subject = data.get('subject', '').strip()
    date_str = data.get('date', datetime.date.today().strftime('%Y-%m-%d'))
    label = data.get('label', '').strip()  # e.g. "Morning", "Lab"

    if not subject:
        return jsonify({'success': False, 'message': 'Subject required.'}), 400

    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid date format.'}), 400

    day = date_obj.strftime('%A')

    # Count enrolled students
    enrolled_ids = db.attendance_subjects.distinct('student_id', {'subject': subject})
    if current_user.role == 'teacher':
        teacher_students = set(s['_id'] for s in db.students.find({'teacher_ids': current_user._id}, {'_id': 1}))
        enrolled_ids = [sid for sid in enrolled_ids if sid in teacher_students]

    session_doc = {
        'subject': subject,
        'date': date_str,
        'day': day,
        'label': label,
        'created_by': str(current_user._id),
        'created_by_name': current_user.name,
        'student_count': len(enrolled_ids),
        'created_at': datetime.datetime.now(datetime.timezone.utc)
    }
    result = db.attendance_sessions.insert_one(session_doc)
    return jsonify({
        'success': True,
        'session_id': str(result.inserted_id),
        'subject': subject,
        'date': date_str,
        'day': day,
        'label': label,
        'student_count': len(enrolled_ids),
    })


@face_bp.route('/api/face/subjects')
@login_required
@role_required('admin', 'teacher')
def api_face_subjects():
    db = get_db()
    student_query = {'teacher_ids': current_user._id} if current_user.role == 'teacher' else {}
    students = list(db.students.find(student_query, {'_id': 1, 'department': 1, 'semester': 1}))
    from models.attendance import AttendanceModel
    for student in students:
        if not AttendanceModel.get_subjects(student['_id']):
            AttendanceModel.seed_subjects(
                student['_id'], student.get('department', ''), student.get('semester', 1)
            )
    student_ids = [student['_id'] for student in students]
    subjects = db.attendance_subjects.distinct('subject', {'student_id': {'$in': student_ids}}) if student_ids else []
    return jsonify({
        'success': True,
        'subjects': [{'subject': subject} for subject in sorted(set(subjects)) if subject]
    })


@face_bp.route('/api/face/class-students')
@login_required
@role_required('admin', 'teacher')
def api_class_students():
    """Return students enrolled in a subject with their face registration status and session attendance."""
    import datetime
    db = get_db()
    subject = request.args.get('subject', '').strip()
    session_id = request.args.get('session_id', '').strip()
    if not subject:
        return jsonify({'success': False, 'message': 'Subject required.'}), 400

    student_query = {'teacher_ids': current_user._id} if current_user.role == 'teacher' else {}
    enrolled_ids = db.attendance_subjects.distinct('student_id', {'subject': subject})
    if student_query:
        teacher_students = set(s['_id'] for s in db.students.find(student_query, {'_id': 1}))
        enrolled_ids = [sid for sid in enrolled_ids if sid in teacher_students]

    today = datetime.date.today().strftime('%Y-%m-%d')
    result = []
    for sid in enrolled_ids:
        student = db.students.find_one({'_id': sid})
        if not student:
            continue
        user = db.users.find_one({'_id': student.get('user_id')})
        # Check attendance for this session specifically
        if session_id:
            att_record = db.face_attendance.find_one({'student_id': sid, 'session_id': session_id})
        else:
            att_record = db.attendance_records.find_one({
                'student_id': sid, 'subject': subject,
                'date': {'$gte': datetime.datetime.combine(datetime.date.today(), datetime.time.min),
                         '$lt': datetime.datetime.combine(datetime.date.today() + datetime.timedelta(days=1), datetime.time.min)}
            })
        face_registered = len(student.get('face_embeddings', [])) > 0
        result.append({
            '_id': str(sid),
            'name': user.get('name', 'Unknown') if user else 'Unknown',
            'roll_no': student.get('roll_no', ''),
            'department': student.get('department', ''),
            'face_registered': face_registered,
            'embedding_count': len(student.get('face_embeddings', [])),
            'today_status': att_record.get('status') if att_record else None,
            'already_marked': att_record is not None,
        })
    result.sort(key=lambda x: x['name'])
    return jsonify({'success': True, 'students': result, 'subject': subject, 'date': today})


@face_bp.route('/face/register')
@login_required
def register_page():
    """Student face registration page."""
    db = get_db()
    from models.student import StudentModel
    if current_user.role == 'admin':
        students = StudentModel.get_all(limit=500)
        for student in students:
            user = db.users.find_one({'_id': student.get('user_id')})
            student['name'] = user.get('name', 'Unknown') if user else 'Unknown'
        return render_template('admin/face_register.html', students=students)
    if current_user.role != 'student':
        return redirect(url_for('main.index'))
    student = StudentModel.get_by_user_id(current_user.id)
    face_count = len(student.get('face_embeddings', [])) if student else 0
    return render_template('face/register.html', student=student, face_count=face_count)


@face_bp.route('/face/history')
@login_required
@role_required('admin', 'teacher', 'student')
def history_page():
    """Face attendance history page."""
    db = get_db()
    query = {}
    if current_user.role == 'student':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if student:
            query['student_id'] = student['_id']
    elif current_user.role == 'teacher':
        from models.student import StudentModel
        teacher_students = list(db.students.find({'teacher_ids': current_user._id}))
        student_ids = [s['_id'] for s in teacher_students]
        if student_ids:
            query['student_id'] = {'$in': student_ids}
        else:
            query['student_id'] = None
    records = list(db.face_attendance.find(query).sort('marked_at', -1).limit(100))
    return render_template('face/history.html', records=records)


@face_bp.route('/face/wellness')
@login_required
@role_required('student')
def wellness_page():
    """Student self-check using facial-expression classification only."""
    return render_template('face/wellness.html')


@face_bp.route('/face/support-evidence/<request_id>')
@login_required
def support_evidence(request_id):
    db = get_db()
    try:
        request_doc = db.counseling.find_one({'_id': ObjectId(request_id)})
    except Exception:
        request_doc = None
    if not request_doc or not request_doc.get('evidence_filename'):
        abort(404)
    if current_user.role != 'admin':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if not student or request_doc.get('student_id') != student['_id']:
            abort(403)
    return send_from_directory(_UPLOAD_FOLDER, request_doc['evidence_filename'], as_attachment=True)


@face_bp.route('/face/self-check-report/<request_id>')
@login_required
def self_check_report(request_id):
    db = get_db()
    try:
        request_doc = db.counseling.find_one({'_id': ObjectId(request_id)})
    except Exception:
        request_doc = None
    if not request_doc or not request_doc.get('self_check_report'):
        abort(404)
    if current_user.role != 'admin':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if not student or request_doc.get('student_id') != student['_id']:
            abort(403)
    return Response(
        request_doc['self_check_report'],
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment; filename=facial-expression-self-check.txt'}
    )


# ── API: Register Face ────────────────────────────────────────────────────────

@face_bp.route('/api/face/register', methods=['POST'])
@login_required
def api_register_face():
    """
    Register a face embedding for the current student.
    Accepts: multipart image file OR JSON {image_b64: "..."}
    """
    db = get_db()
    from models.student import StudentModel

    # Resolve student
    student = None
    if current_user.role == 'student':
        student = StudentModel.get_by_user_id(current_user.id)
    elif current_user.role in ('admin', 'teacher'):
        student_id = request.form.get('student_id') or (request.get_json(silent=True) or {}).get('student_id')
        student = StudentModel.get_by_id(student_id) if student_id else None

    if not student:
        return jsonify({'success': False, 'error': 'Student profile not found.'}), 404

    if not face_recognizer.ready:
        return jsonify({'success': False,
                        'error': 'Face recognition models not loaded. '
                                 'Check server logs for details.'}), 503

    # ── Get image ─────────────────────────────────────────────────────────────
    img_bgr = None
    tmp_path = None

    if 'image' in request.files:
        f = request.files['image']
        valid, err = validate_image_file(f)
        if not valid:
            return jsonify({'success': False, 'error': err}), 400
        tmp_path = secure_temp_save(f, _UPLOAD_FOLDER)
        import cv2
        img_bgr = cv2.imread(tmp_path)
    else:
        raw = request.get_data(cache=True)
        data = {}
        try:
            import json as _json
            data = _json.loads(raw.decode('utf-8')) if raw else {}
        except Exception:
            data = request.get_json(silent=True, force=True) or {}
        b64 = data.get('image_b64', '')
        if b64:
            img_bgr = decode_base64_image(b64)

    if img_bgr is None:
        return jsonify({'success': False, 'error': 'No valid image provided.'}), 400

    try:
        img_bgr = resize_if_large(img_bgr)
        img_rgb = bgr_to_rgb(img_bgr)
        result  = face_recognizer.process_registration_image(img_rgb)

        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 422

        # Store embedding
        db.students.update_one(
            {'_id': student['_id']},
            {'$push': {'face_embeddings': result['embedding']}}
        )

        # Count total embeddings
        updated = db.students.find_one({'_id': student['_id']})
        count = len(updated.get('face_embeddings', []))

        # Log activity
        try:
            from models.activity import ActivityModel
            ActivityModel.log('face_registration', 'Face Registered',
                              f'{current_user.name} registered face embedding #{count}',
                              student_id=student['_id'])
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': f'Face registered successfully. Total embeddings: {count}',
            'embedding_count': count
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── API: Recognize + Attend + Expression ─────────────────────────────────────

@face_bp.route('/api/face/recognize', methods=['POST'])
@login_required
@role_required('admin', 'teacher')
def api_recognize():
    """
    Main webcam endpoint: detect → identify → mark attendance → expression.
    Accepts: JSON {image_b64: "data:image/...;base64,...", session_id: "optional"}
    """
    try:
        # Force-read the full body into Flask's buffer before parsing
        raw = request.get_data(cache=True)
        data = {}
        try:
            import json as _json
            data = _json.loads(raw.decode('utf-8')) if raw else {}
        except Exception:
            data = request.get_json(silent=True, force=True) or {}

        b64 = data.get('image_b64', '')
        session_id = data.get('session_id', 'default')
        subject = (data.get('subject') or '').strip()

        if not b64:
            return jsonify({'success': False, 'error': 'No image data received.',
                            'debug': {'content_type': request.content_type,
                                      'raw_len': len(raw), 'keys': list(data.keys())}}), 400
        if not subject:
            return jsonify({'success': False, 'error': 'Select a subject before scanning.',
                            'debug': {'keys': list(data.keys())}}), 400

        img_bgr = decode_base64_image(b64)
        if img_bgr is None:
            return jsonify({'success': False, 'error': 'Invalid image data.'}), 400

        img_bgr = resize_if_large(img_bgr)
        img_rgb = bgr_to_rgb(img_bgr)

        # ── Face Recognition ──────────────────────────────────────────────────
        if not face_recognizer.ready:
            return jsonify({
                'success': False,
                'error': 'Face recognition models not loaded. Check server logs.',
                'recognized': False
            }), 503

        student_query = {'teacher_ids': current_user._id} if current_user.role == 'teacher' else None
        known_students = _get_all_face_students(student_query)
        recog = face_recognizer.identify(img_rgb, known_students)

        response = {
            'success': True,
            'recognized': recog['recognized'],
            'message': recog.get('message', ''),
            'expression': None,
            'expression_confidence': None,
            'attendance': 'Not marked',
            'already_marked': False
        }

        if recog['recognized']:
            response['student_id']   = recog['student_id']
            response['student_name'] = recog['student_name']
            response['roll_no']      = recog['roll_no']

            # ── Attendance ────────────────────────────────────────────────────
            from models.attendance import AttendanceModel
            enrolled_subjects = AttendanceModel.get_subjects(recog['student_id'])
            if not enrolled_subjects:
                student = get_db().students.find_one({'_id': ObjectId(recog['student_id'])})
                if student:
                    AttendanceModel.seed_subjects(
                        student['_id'], student.get('department', ''), student.get('semester', 1)
                    )
                    enrolled_subjects = AttendanceModel.get_subjects(recog['student_id'])
            available_subjects = [
                {'subject': item['subject']} for item in enrolled_subjects if item.get('subject')
            ]
            if subject not in {item['subject'] for item in available_subjects}:
                return jsonify({
                    'success': False,
                    'recognized': True,
                    'student_name': recog['student_name'],
                    'roll_no': recog['roll_no'],
                    'message': 'This student is not enrolled in the selected subject.',
                    'available_subjects': available_subjects
                }), 409
            marked_at = datetime.datetime.now(datetime.timezone.utc)
            tracker_ok, tracker_message, tracker_record = AttendanceModel.mark_face_attendance(
                recog['student_id'], subject, marked_at
            )
            if not tracker_ok:
                response['attendance']     = 'Already marked'
                response['already_marked'] = True
            else:
                _mark_face_attendance(recog['student_id'], recog['student_name'],
                                      recog['roll_no'], session_id)
                response['attendance'] = 'Present'
                try:
                    from models.activity import ActivityModel
                    ActivityModel.log('attendance', 'Face Attendance Marked',
                                      f"{recog['student_name']} ({recog['roll_no']}) marked present via face",
                                      student_id=ObjectId(recog['student_id']))
                except Exception:
                    pass
                response['attendance_subject'] = subject
                response['attendance_date'] = marked_at.strftime('%A, %B %d, %Y')
                response['attendance_time'] = marked_at.strftime('%I:%M:%S %p UTC')

            # ── Expression on recognized face ─────────────────────────────────
            try:
                if emotion_predictor.ready:
                    face_crop = face_recognizer.crop_face(img_bgr, recog['face_location'], padding=0)
                    expr = emotion_predictor.predict(face_crop)
                    if expr['success']:
                        response['expression']               = expr['prediction']
                        response['expression_confidence']    = round(expr['confidence'] * 100, 1)
                        response['expression_probabilities'] = expr.get('probabilities', {})
            except Exception:
                pass
        else:
            try:
                if emotion_predictor.ready:
                    locs = face_recognizer.detect_faces(img_rgb)
                    if locs:
                        face_crop = face_recognizer.crop_face(img_bgr, locs[0], padding=0)
                        expr = emotion_predictor.predict(face_crop)
                        if expr['success']:
                            response['expression']               = expr['prediction']
                            response['expression_confidence']    = round(expr['confidence'] * 100, 1)
                            response['expression_probabilities'] = expr.get('probabilities', {})
            except Exception:
                pass

        return jsonify(response)

    except Exception as e:
        print(f"[api_recognize] Unhandled error: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}', 'recognized': False}), 500


# ── API: Expression Only ──────────────────────────────────────────────────────

@face_bp.route('/api/face/expression', methods=['POST'])
@login_required
@role_required('student')
def api_expression():
    """
    Run facial expression analysis only (no identity check).
    Accepts: multipart 'image' file OR JSON {image_b64: "..."}
    """
    if not emotion_predictor.ready:
        return jsonify({'success': False, 'error': 'Emotion model not loaded.'}), 503

    img_bgr = None
    tmp_path = None

    if 'image' in request.files:
        f = request.files['image']
        valid, err = validate_image_file(f)
        if not valid:
            return jsonify({'success': False, 'error': err}), 400
        tmp_path = secure_temp_save(f, _UPLOAD_FOLDER)
        import cv2
        img_bgr = cv2.imread(tmp_path)
    else:
        data = request.get_json(silent=True) or {}
        b64  = data.get('image_b64', '')
        if b64:
            img_bgr = decode_base64_image(b64)

    if img_bgr is None:
        return jsonify({'success': False, 'error': 'No valid image provided.'}), 400

    try:
        img_bgr = resize_if_large(img_bgr)
        img_rgb = bgr_to_rgb(img_bgr)

        # Try to crop face first
        if face_recognizer.ready:
            locs = face_recognizer.detect_faces(img_rgb)
            if not locs:
                return jsonify({'success': False, 'error': 'No face detected in the image.'})
            face_crop = face_recognizer.crop_face(img_bgr, locs[0], padding=0)
            result = emotion_predictor.predict(face_crop)
        else:
            # No face detector — run on full image
            result = emotion_predictor.predict(img_bgr)

        if result['success']:
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': round(result['confidence'] * 100, 1),
                'probabilities': result.get('probabilities', {})
            })
        return jsonify({'success': False, 'error': result.get('error', 'Prediction failed.')})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── API: Current student face info ───────────────────────────────────────────

@face_bp.route('/api/face/students/me')
@login_required
@role_required('student')
def api_face_students_me():
    db = get_db()
    from models.student import StudentModel
    student = StudentModel.get_by_user_id(current_user.id)
    if not student:
        return jsonify({'success': False, 'embedding_count': 0})
    return jsonify({'success': True, 'embedding_count': len(student.get('face_embeddings', []))})


# ── API: Students list ────────────────────────────────────────────────────────

@face_bp.route('/api/face/students')
@login_required
@role_required('admin')
def api_face_students():
    db = get_db()
    students = list(db.students.find({}, {'face_embeddings': 1, 'roll_no': 1, 'user_id': 1}))
    result = []
    for s in students:
        user = db.users.find_one({'_id': s.get('user_id')})
        result.append({
            'student_id': str(s['_id']),
            'name': user.get('name', 'Unknown') if user else 'Unknown',
            'roll_no': s.get('roll_no', ''),
            'face_registered': len(s.get('face_embeddings', [])) > 0,
            'embedding_count': len(s.get('face_embeddings', []))
        })
    return jsonify({'success': True, 'students': result})


# ── API: Attendance history ───────────────────────────────────────────────────

@face_bp.route('/api/face/attendance')
@login_required
@role_required('admin', 'teacher', 'student')
def api_face_attendance():
    db = get_db()
    query = {}
    if current_user.role == 'student':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if student:
            query['student_id'] = student['_id']
    records = list(db.face_attendance.find(query).sort('marked_at', -1).limit(50))
    out = []
    for r in records:
        out.append({
            'student_name': r.get('student_name', ''),
            'roll_no': r.get('roll_no', ''),
            'date': r.get('date', ''),
            'day': r.get('marked_at').strftime('%A') if r.get('marked_at') else '',
            'year': r.get('marked_at').year if r.get('marked_at') else '',
            'time': r.get('time', ''),
            'status': r.get('status', 'present'),
            'session_id': r.get('session_id', '')
        })
    return jsonify({'success': True, 'records': out})
