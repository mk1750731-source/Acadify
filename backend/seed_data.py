import argparse
import random
from app import app
from models.db import get_db
from models.user import User
from models.student import StudentModel
from models.prediction import PredictionModel
from models.department import DepartmentModel
from models.notification import NotificationModel
from ml.predictor import predict_student_risk

DEPARTMENTS = [
    "Computer Science & AI",
    "Data Science & Analytics",
    "Robotics Engineering",
    "Cybersecurity & Networks",
    "Biomedical Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Business Administration",
]

# (name, email, roll_no, department, semester, gpa, attendance_pct)
SAMPLE_STUDENTS = [
    # ── Computer Science & AI ──────────────────────────────────────────────
    ("Alexander Vance",    "alex.vance@nexus.edu",    "CS-001", "Computer Science & AI",       6, 3.85, 96.0),
    ("Elena Rostova",      "elena.r@nexus.edu",       "CS-002", "Computer Science & AI",       4, 2.15, 68.0),
    ("James Whitfield",    "james.w@nexus.edu",       "CS-003", "Computer Science & AI",       5, 3.50, 88.0),
    ("Priya Nair",         "priya.n@nexus.edu",       "CS-004", "Computer Science & AI",       3, 1.80, 55.0),
    ("Lucas Ferreira",     "lucas.f@nexus.edu",       "CS-005", "Computer Science & AI",       7, 3.95, 99.0),
    ("Aisha Okonkwo",      "aisha.o@nexus.edu",       "CS-006", "Computer Science & AI",       2, 2.60, 74.0),
    ("Ryan Nakamura",      "ryan.n@nexus.edu",        "CS-007", "Computer Science & AI",       8, 0.95, 38.0),
    ("Sofia Petrov",       "sofia.p@nexus.edu",       "CS-008", "Computer Science & AI",       1, 3.20, 82.0),

    # ── Data Science & Analytics ───────────────────────────────────────────
    ("Marcus Chen",        "marcus.c@nexus.edu",      "DS-001", "Data Science & Analytics",    6, 3.92, 98.0),
    ("Sophia Martinez",    "sophia.m@nexus.edu",      "DS-002", "Data Science & Analytics",    2, 2.40, 72.0),
    ("Ethan Brooks",       "ethan.b@nexus.edu",       "DS-003", "Data Science & Analytics",    4, 3.10, 85.0),
    ("Yuna Park",          "yuna.p@nexus.edu",        "DS-004", "Data Science & Analytics",    3, 1.55, 48.0),
    ("Carlos Mendez",      "carlos.m@nexus.edu",      "DS-005", "Data Science & Analytics",    5, 3.70, 93.0),
    ("Fatima Al-Hassan",   "fatima.h@nexus.edu",      "DS-006", "Data Science & Analytics",    7, 2.90, 79.0),

    # ── Robotics Engineering ───────────────────────────────────────────────
    ("Liam Thorne",        "liam.t@nexus.edu",        "RE-001", "Robotics Engineering",        4, 3.45, 89.0),
    ("Aria Thorne",        "aria.t@nexus.edu",        "RE-002", "Robotics Engineering",        6, 1.95, 62.0),
    ("Noah Osei",          "noah.o@nexus.edu",        "RE-003", "Robotics Engineering",        3, 3.80, 95.0),
    ("Isabella Cruz",      "isabella.c@nexus.edu",    "RE-004", "Robotics Engineering",        5, 2.20, 65.0),
    ("Kai Tanaka",         "kai.t@nexus.edu",         "RE-005", "Robotics Engineering",        2, 3.60, 91.0),
    ("Zara Ahmed",         "zara.a@nexus.edu",        "RE-006", "Robotics Engineering",        7, 1.30, 42.0),

    # ── Cybersecurity & Networks ───────────────────────────────────────────
    ("David Kim",          "david.kim@nexus.edu",     "CY-001", "Cybersecurity & Networks",    4, 3.70, 94.0),
    ("Maya Lin",           "maya.lin@nexus.edu",      "CY-002", "Cybersecurity & Networks",    2, 2.80, 78.0),
    ("Hassan Malik",       "hassan.m@nexus.edu",      "CY-003", "Cybersecurity & Networks",    6, 3.30, 87.0),
    ("Cleo Vasquez",       "cleo.v@nexus.edu",        "CY-004", "Cybersecurity & Networks",    3, 1.70, 52.0),
    ("Finn O'Brien",       "finn.ob@nexus.edu",       "CY-005", "Cybersecurity & Networks",    5, 3.55, 90.0),
    ("Leila Moradi",       "leila.mo@nexus.edu",      "CY-006", "Cybersecurity & Networks",    1, 2.50, 70.0),

    # ── Biomedical Engineering ─────────────────────────────────────────────
    ("Oliver Queen",       "oliver.q@nexus.edu",      "BE-001", "Biomedical Engineering",      8, 3.98, 99.0),
    ("Chloe Decker",       "chloe.d@nexus.edu",       "BE-002", "Biomedical Engineering",      4, 2.30, 69.0),
    ("Amara Diallo",       "amara.d@nexus.edu",       "BE-003", "Biomedical Engineering",      6, 3.65, 92.0),
    ("Victor Reyes",       "victor.r@nexus.edu",      "BE-004", "Biomedical Engineering",      2, 1.40, 45.0),
    ("Nadia Kowalski",     "nadia.k@nexus.edu",       "BE-005", "Biomedical Engineering",      5, 3.20, 83.0),

    # ── Electrical Engineering ─────────────────────────────────────────────
    ("Samuel Adeyemi",     "samuel.a@nexus.edu",      "EE-001", "Electrical Engineering",      4, 3.75, 95.0),
    ("Grace Huang",        "grace.h@nexus.edu",       "EE-002", "Electrical Engineering",      6, 2.05, 60.0),
    ("Tobias Müller",      "tobias.m@nexus.edu",      "EE-003", "Electrical Engineering",      3, 3.40, 86.0),
    ("Rania Khalil",       "rania.k@nexus.edu",       "EE-004", "Electrical Engineering",      5, 1.60, 50.0),
    ("Jin-Ho Seo",         "jinho.s@nexus.edu",       "EE-005", "Electrical Engineering",      7, 3.88, 97.0),

    # ── Mechanical Engineering ─────────────────────────────────────────────
    ("Mateo Rossi",        "mateo.r@nexus.edu",       "ME-001", "Mechanical Engineering",      4, 3.55, 90.0),
    ("Ingrid Larsen",      "ingrid.l@nexus.edu",      "ME-002", "Mechanical Engineering",      2, 2.70, 76.0),
    ("Kwame Asante",       "kwame.a@nexus.edu",       "ME-003", "Mechanical Engineering",      6, 1.20, 40.0),
    ("Valentina Russo",    "valentina.r@nexus.edu",   "ME-004", "Mechanical Engineering",      3, 3.90, 98.0),
    ("Dmitri Volkov",      "dmitri.v@nexus.edu",      "ME-005", "Mechanical Engineering",      5, 2.45, 71.0),

    # ── Extra Accounts ──────────────────────────────────────────────────────
    ("Maaz Khan",          "mk1750731@gmail.com",     "CS-009", "Computer Science & AI",       3, 3.50, 85.0),

    # ── Business Administration ────────────────────────────────────────────
    ("Penelope Shaw",      "penelope.s@nexus.edu",    "BA-001", "Business Administration",     4, 3.60, 91.0),
    ("Andre Dupont",       "andre.d@nexus.edu",       "BA-002", "Business Administration",     6, 2.10, 63.0),
    ("Mei-Ling Zhou",      "meiling.z@nexus.edu",     "BA-003", "Business Administration",     2, 3.30, 84.0),
    ("Tariq Hassan",       "tariq.h@nexus.edu",       "BA-004", "Business Administration",     5, 1.50, 47.0),
    ("Camille Dubois",     "camille.d@nexus.edu",     "BA-005", "Business Administration",     3, 3.75, 94.0),
    ("Raj Patel",          "raj.p@nexus.edu",         "BA-006", "Business Administration",     7, 2.85, 80.0),
]

ROLE_SEED_ACCOUNTS = [
    ('teacher', 'EDU Demo Teacher', 'teacher@nexus.edu', 'Teacher123!', None),
    ('parent', 'EDU Demo Parent One', 'parent@nexus.edu', 'Parent123!', 'alex.vance@nexus.edu'),
    ('parent', 'EDU Demo Parent Two', 'parent2@nexus.edu', 'Parent123!', 'elena.r@nexus.edu'),
]


def _get_or_create_role_user(name, email, password, role):
    existing = User.get_by_email(email)
    if existing:
        if existing.role != role:
            raise ValueError(f'{email} exists with role {existing.role}, expected {role}.')
        return existing, False

    user, error = User.create(name, email, password, role=role)
    if error:
        raise ValueError(error)
    return user, True


def seed_role_accounts():
    """Create development teacher/parent accounts and connect them to seeded students."""
    with app.app_context():
        db = get_db()
        teacher_record = next(item for item in ROLE_SEED_ACCOUNTS if item[0] == 'teacher')
        _, teacher_name, teacher_email, teacher_password, _ = teacher_record
        teacher, teacher_created = _get_or_create_role_user(
            teacher_name, teacher_email, teacher_password, 'teacher'
        )
        assigned = db.students.update_many({}, {'$addToSet': {'teacher_ids': teacher._id}}).modified_count
        print(f'Teacher: {teacher_email} ({"created" if teacher_created else "existing account preserved"})')
        if teacher_created:
            print(f'  Password: {teacher_password}')
        print(f'  Linked to {db.students.count_documents({"teacher_ids": teacher._id})} students.')

        for role, name, email, password, student_email in ROLE_SEED_ACCOUNTS:
            if role != 'parent':
                continue
            parent, created = _get_or_create_role_user(name, email, password, role)
            print(f'Parent: {email} ({"created" if created else "existing account preserved"})')
            if created:
                print(f'  Password: {password}')
            linked_student_user = User.get_by_email(student_email)
            student = db.students.find_one({'user_id': linked_student_user._id}) if linked_student_user else None
            if not student:
                print(f'  Student account {student_email} is not present; account remains unlinked.')
                continue
            db.students.update_one(
                {'_id': student['_id']},
                {'$addToSet': {'parent_ids': parent._id, 'teacher_ids': teacher._id}}
            )
            print(f'  Linked to {student_email}.')


def seed():
    with app.app_context():
        db = get_db()
        print("[*] Clearing existing non-admin data...")
        db.students.delete_many({})
        db.predictions.delete_many({})
        db.departments.delete_many({})
        db.notifications.delete_many({})
        db.users.delete_many({'role': 'student'})

        print("[*] Seeding Departments...")
        for dept in DEPARTMENTS:
            DepartmentModel.create(dept)

        print(f"[*] Seeding {len(SAMPLE_STUDENTS)} students with ML predictions...")
        success = 0
        STUDENT_PASSWORDS = {
            "mk1750731@gmail.com": "maazkhan",
        }
        for name, email, roll_no, dept, sem, gpa, att in SAMPLE_STUDENTS:
            password = STUDENT_PASSWORDS.get(email, "Student123!")
            user, err = User.create(name, email, password, role='student')
            if err:
                print(f"    [-] Skipping: {email} ({err})")
                continue

            student_data = {
                'gpa': gpa,
                'attendance_pct': att,
                'semester': sem,
                'department': dept,
            }
            pred = predict_student_risk(student_data)

            student = StudentModel.create(
                user_id=user._id,
                roll_no=roll_no,
                department=dept,
                semester=sem,
                gpa=gpa,
                attendance_pct=att,
                risk_level=pred['risk_level'],
                risk_score=pred['risk_score'],
                last_prediction=pred
            )

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
                    message="High Academic Risk Warning: Attendance and GPA require immediate attention. Please contact your academic advisor.",
                    notif_type="warning"
                )
            elif pred['risk_level'] == 'MEDIUM':
                NotificationModel.create(
                    student_id=student['_id'],
                    message="Medium Risk Alert: Your academic performance needs improvement. Consider scheduling a tutoring session.",
                    notif_type="info"
                )
            else:
                NotificationModel.create(
                    student_id=student['_id'],
                    message="Welcome to EDU! Your academic forecast is active and looking good. Keep it up!",
                    notif_type="info"
                )

            risk_icon = "[HIGH]" if pred['risk_level'] == 'HIGH' else ("[MED] " if pred['risk_level'] == 'MEDIUM' else "[LOW] ")
            print(f"    {risk_icon} {name:<25} GPA:{gpa}  Att:{att}%  -> {pred['risk_level']}")
            success += 1

        DepartmentModel.recalculate_stats()
        print(f"\n[OK] Seeded {success} students successfully!")
        print(f"     Login with any student: password = Student123!")
        seed_role_accounts()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roles-only', action='store_true', help='Seed teacher/parent accounts without rebuilding students.')
    args = parser.parse_args()
    seed_role_accounts() if args.roles_only else seed()
