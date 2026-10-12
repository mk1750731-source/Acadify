"""
blueprints/face.py
Face recognition is done CLIENT-SIDE via face-api.js (browser).
Backend only: stores embeddings, compares them, marks attendance.
No AI libraries. No 503s. Always ready.
"""
import os
import json
import datetime
import numpy as np

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, abort, Response, send_from_directory
from flask_login import login_required, current_user
from bson.objectid import ObjectId

from models.db import get_db
from decorators import role_required

face_bp = Blueprint('face', __name__)

FACE_MATCH_THRESHOLD = float(os.environ.get('FACE_MATCH_THRESHOLD', '0.50'))

_UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
os.makedirs(_UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_body():
    raw = request.get_data(cache=True)
    try:
        return json.loads(raw.decode('utf-8')) if raw else {}
    except Exception:
        return request.get_json(silent=True, force=True) or {}


def _normalise(v):
    a = np.array(v, dtype='float32')
    n = np.linalg.norm(a)
    return a / n if n > 1e-10 else a


def _cosine_dist(a, b):
    return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))


def _get_face_students(student_query=None):
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


def _mark_face_attendance(student_id, student_name, roll_no, session_id):
    db = get_db()
    now = datetime.datetime.now(datetime.timezone.utc)
    db.face_attendance.insert_one({
        'student_id': ObjectId(student_id),
        'student_name': student_name,
        'roll_no': roll_no,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S'),
        'status': 'present',
        'session_id': session_id or 'default',
        'marked_at': now
    })


# ── Status — always ready (AI runs in browser) ────────────────────────────────

@face_bp.route('/api/face/status')
@login_required
def api_face_status():
    return jsonify({'success': True, 'ready': True, 'face_recognition': True, 'emotion': True})


# ── Page routes ───────────────────────────────────────────────────────────────

@face_bp.route('/face/attendance')
@login_required
@role_required('admin', 'teacher')
def attendance_page():
    db = get_db()
    subjects = list(db.attendance_subjects.aggregate([
        {'$group': {'_id': '$subject'}}, {'$sort': {'_id': 1}}
    ]))
    return render_template('face/attendance.html',
                           subjects=[{'subject': i['_id']} for i in subjects if i.get('_id')])


@face_bp.route('/face/register')
@login_required
def register_page():
    db = get_db()
    from models.student import StudentModel
    if current_user.role == 'admin':
        students = StudentModel.get_all(limit=500)
        for s in students:
            user = db.users.find_one({'_id': s.get('user_id')})
            s['name'] = user.get('name', 'Unknown') if user else 'Unknown'
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
    db = get_db()
    query = {}
    if current_user.role == 'student':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if student:
            query['student_id'] = student['_id']
    elif current_user.role == 'teacher':
        teacher_students = list(db.students.find({'teacher_ids': current_user._id}))
        ids = [s['_id'] for s in teacher_students]
        query['student_id'] = {'$in': ids} if ids else None
    records = list(db.face_attendance.find(query).sort('marked_at', -1).limit(100))
    return render_template('face/history.html', records=records)


@face_bp.route('/face/wellness')
@login_required
@role_required('student')
def wellness_page():
    return render_template('face/wellness.html')


@face_bp.route('/face/support-evidence/<request_id>')
@login_required
def support_evidence(request_id):
    db = get_db()
    try:
        doc = db.counseling.find_one({'_id': ObjectId(request_id)})
    except Exception:
        doc = None
    if not doc or not doc.get('evidence_filename'):
        abort(404)
    if current_user.role != 'admin':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if not student or doc.get('student_id') != student['_id']:
            abort(403)
    return send_from_directory(_UPLOAD_FOLDER, doc['evidence_filename'], as_attachment=True)


@face_bp.route('/face/self-check-report/<request_id>')
@login_required
def self_check_report(request_id):
    db = get_db()
    try:
        doc = db.counseling.find_one({'_id': ObjectId(request_id)})
    except Exception:
        doc = None
    if not doc or not doc.get('self_check_report'):
        abort(404)
    if current_user.role != 'admin':
        from models.student import StudentModel
        student = StudentModel.get_by_user_id(current_user.id)
        if not student or doc.get('student_id') != student['_id']:
            abort(403)
    return Response(doc['self_check_report'], mimetype='text/plain',
                    headers={'Content-Disposition': 'attachment; filename=self-check.txt'})


# ── Sessions ──────────────────────────────────────────────────────────────────

@face_bp.route('/api/face/sessions')
@login_required
@role_required('admin', 'teacher')
def api_face_sessions():
    db = get_db()
    date_str = request.args.get('date', datetime.date.today().strftime('%Y-%m-%d'))
    query = {'date': date_str}
    if current_user.role == 'teacher':
        query['created_by'] = str(current_user._id)
    sessions = list(db.attendance_sessions.find(query).sort('created_at', -1))
    return jsonify({'success': True, 'sessions': [{
        'session_id': str(s['_id']), 'subject': s['subject'],
        'date': s['date'], 'day': s['day'], 'label': s.get('label', ''),
        'created_by_name': s.get('created_by_name', ''),
        'student_count': s.get('student_count', 0),
    } for s in sessions]})


@face_bp.route('/api/face/session/create', methods=['POST'])
@login_required
@role_required('admin', 'teacher')
def api_create_session():
    db = get_db()
    data = _parse_body()
    subject = data.get('subject', '').strip()
    date_str = data.get('date', datetime.date.today().strftime('%Y-%m-%d'))
    label = data.get('label', '').strip()
    if not subject:
        return jsonify({'success': False, 'message': 'Subject required.'}), 400
    try:
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid date.'}), 400
    day = date_obj.strftime('%A')
    enrolled_ids = db.attendance_subjects.distinct('student_id', {'subject': subject})
    if current_user.role == 'teacher':
        ts = set(s['_id'] for s in db.students.find({'teacher_ids': current_user._id}, {'_id': 1}))
        enrolled_ids = [i for i in enrolled_ids if i in ts]
    result = db.attendance_sessions.insert_one({
        'subject': subject, 'date': date_str, 'day': day, 'label': label,
        'created_by': str(current_user._id), 'created_by_name': current_user.name,
        'student_count': len(enrolled_ids),
        'created_at': datetime.datetime.now(datetime.timezone.utc)
    })
    return jsonify({'success': True, 'session_id': str(result.inserted_id),
                    'subject': subject, 'date': date_str, 'day': day,
                    'label': label, 'student_count': len(enrolled_ids)})


@face_bp.route('/api/face/subjects')
@login_required
@role_required('admin', 'teacher')
def api_face_subjects():
    db = get_db()
    sq = {'teacher_ids': current_user._id} if current_user.role == 'teacher' else {}
    students = list(db.students.find(sq, {'_id': 1, 'department': 1, 'semester': 1}))
    from models.attendance import AttendanceModel
    for s in students:
        if not AttendanceModel.get_subjects(s['_id']):
            AttendanceModel.seed_subjects(s['_id'], s.get('department', ''), s.get('semester', 1))
    ids = [s['_id'] for s in students]
    subjects = db.attendance_subjects.distinct('subject', {'student_id': {'$in': ids}}) if ids else []
    # Fallback: if no subjects found, return all known default subjects
    if not subjects:
        from ai_model.attendance import DEFAULT_SUBJECTS, DEFAULT_FALLBACK
        all_defaults = set(DEFAULT_FALLBACK)
        for subj_list in DEFAULT_SUBJECTS.values():
            all_defaults.update(subj_list)
        subjects = sorted(all_defaults)
    return jsonify({'success': True, 'subjects': [{'subject': s} for s in sorted(set(subjects)) if s]})


@face_bp.route('/api/face/class-students')
@login_required
@role_required('admin', 'teacher')
def api_class_students():
    db = get_db()
    subject = request.args.get('subject', '').strip()
    session_id = request.args.get('session_id', '').strip()
    if not subject:
        return jsonify({'success': False, 'message': 'Subject required.'}), 400
    sq = {'teacher_ids': current_user._id} if current_user.role == 'teacher' else {}
    enrolled_ids = db.attendance_subjects.distinct('student_id', {'subject': subject})
    if sq:
        ts = set(s['_id'] for s in db.students.find(sq, {'_id': 1}))
        enrolled_ids = [i for i in enrolled_ids if i in ts]
    today = datetime.date.today().strftime('%Y-%m-%d')
    result = []
    for sid in enrolled_ids:
        student = db.students.find_one({'_id': sid})
        if not student:
            continue
        user = db.users.find_one({'_id': student.get('user_id')})
        att = db.face_attendance.find_one({'student_id': sid, 'session_id': session_id}) if session_id else None
        result.append({
            '_id': str(sid),
            'name': user.get('name', 'Unknown') if user else 'Unknown',
            'roll_no': student.get('roll_no', ''),
            'department': student.get('department', ''),
            'face_registered': len(student.get('face_embeddings', [])) > 0,
            'embedding_count': len(student.get('face_embeddings', [])),
            'today_status': att.get('status') if att else None,
            'already_marked': att is not None,
        })
    result.sort(key=lambda x: x['name'])
    return jsonify({'success': True, 'students': result, 'subject': subject, 'date': today})


# ── Register face (embedding sent from browser) ───────────────────────────────

@face_bp.route('/api/face/register', methods=['POST'])
@login_required
def api_register_face():
    db = get_db()
    from models.student import StudentModel
    data = _parse_body()

    student = None
    if current_user.role == 'student':
        student = StudentModel.get_by_user_id(current_user.id)
    elif current_user.role in ('admin', 'teacher'):
        sid = data.get('student_id')
        student = StudentModel.get_by_id(sid) if sid else None

    if not student:
        return jsonify({'success': False, 'error': 'Student profile not found.'}), 404

    embedding = data.get('embedding')
    if not embedding or not isinstance(embedding, list):
        return jsonify({'success': False, 'error': 'No face embedding provided.'}), 400

    emb = _normalise(embedding).tolist()
    db.students.update_one({'_id': student['_id']}, {'$push': {'face_embeddings': emb}})
    updated = db.students.find_one({'_id': student['_id']})
    count = len(updated.get('face_embeddings', []))

    try:
        from models.activity import ActivityModel
        ActivityModel.log('face_registration', 'Face Registered',
                          f'{current_user.name} registered face #{count}',
                          student_id=student['_id'])
    except Exception:
        pass

    return jsonify({'success': True,
                    'message': f'Face registered successfully. Total embeddings: {count}',
                    'embedding_count': count})


# ── Recognize (embedding sent from browser, backend matches + marks attendance)

@face_bp.route('/api/face/recognize', methods=['POST'])
@login_required
@role_required('admin', 'teacher')
def api_recognize():
    try:
        data = _parse_body()
        embedding = data.get('embedding')
        session_id = data.get('session_id', 'default')
        subject = (data.get('subject') or '').strip()

        if not embedding or not isinstance(embedding, list):
            return jsonify({'success': False, 'error': 'No face embedding provided.'}), 400
        if not subject:
            return jsonify({'success': False, 'error': 'Select a subject before scanning.'}), 400

        query = {'teacher_ids': current_user._id} if current_user.role == 'teacher' else None
        known = _get_face_students(query)

        query_emb = _normalise(embedding)
        best_dist = float('inf')
        best_student = None

        for student in known:
            for ref_raw in student.get('face_embeddings', []):
                try:
                    ref = _normalise(ref_raw)
                    if ref.shape != query_emb.shape:
                        continue
                    d = _cosine_dist(query_emb, ref)
                    if d < best_dist:
                        best_dist = d
                        best_student = student
                except Exception:
                    continue

        if best_student is None or best_dist > FACE_MATCH_THRESHOLD:
            return jsonify({'success': True, 'recognized': False,
                            'message': 'Unknown student. Face not registered in the system.'})

        recog = best_student
        response = {
            'success': True, 'recognized': True,
            'student_id': str(recog['student_id']),
            'student_name': recog['name'],
            'roll_no': recog['roll_no'],
            'distance': round(best_dist, 4),
            'attendance': 'Not marked', 'already_marked': False,
            'expression': data.get('expression'),
            'expression_confidence': data.get('expression_confidence'),
        }

        from models.attendance import AttendanceModel
        enrolled = AttendanceModel.get_subjects(str(recog['student_id']))
        if not enrolled:
            s = get_db().students.find_one({'_id': recog['student_id']})
            if s:
                AttendanceModel.seed_subjects(s['_id'], s.get('department', ''), s.get('semester', 1))
                enrolled = AttendanceModel.get_subjects(str(recog['student_id']))

        if subject not in {i['subject'] for i in enrolled if i.get('subject')}:
            return jsonify({'success': False, 'recognized': True,
                            'student_name': recog['name'], 'roll_no': recog['roll_no'],
                            'message': 'Student not enrolled in selected subject.'}), 409

        marked_at = datetime.datetime.now(datetime.timezone.utc)
        ok, _, _ = AttendanceModel.mark_face_attendance(str(recog['student_id']), subject, marked_at)
        if not ok:
            response['attendance'] = 'Already marked'
            response['already_marked'] = True
        else:
            _mark_face_attendance(str(recog['student_id']), recog['name'], recog['roll_no'], session_id)
            response['attendance'] = 'Present'
            response['attendance_subject'] = subject
            response['attendance_date'] = marked_at.strftime('%A, %B %d, %Y')
            response['attendance_time'] = marked_at.strftime('%I:%M:%S %p UTC')
            try:
                from models.activity import ActivityModel
                ActivityModel.log('attendance', 'Face Attendance Marked',
                                  f"{recog['name']} ({recog['roll_no']}) marked present via face",
                                  student_id=recog['student_id'])
            except Exception:
                pass

        return jsonify(response)

    except Exception as e:
        print(f'[api_recognize] error: {e}')
        return jsonify({'success': False, 'error': str(e), 'recognized': False}), 500


# ── Students ──────────────────────────────────────────────────────────────────

@face_bp.route('/api/face/students/me')
@login_required
@role_required('student')
def api_face_students_me():
    from models.student import StudentModel
    student = StudentModel.get_by_user_id(current_user.id)
    if not student:
        return jsonify({'success': False, 'embedding_count': 0})
    return jsonify({'success': True, 'embedding_count': len(student.get('face_embeddings', []))})


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
    return jsonify({'success': True, 'records': [{
        'student_name': r.get('student_name', ''),
        'roll_no': r.get('roll_no', ''),
        'date': r.get('date', ''),
        'day': r['marked_at'].strftime('%A') if r.get('marked_at') else '',
        'year': r['marked_at'].year if r.get('marked_at') else '',
        'time': r.get('time', ''),
        'status': r.get('status', 'present'),
        'session_id': r.get('session_id', '')
    } for r in records]})
