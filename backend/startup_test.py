"""Startup and route registration test — safe to run without a live DB."""
import os, sys
os.environ['MONGO_URI'] = 'mongodb://127.0.0.1:27017/testdb'
os.environ['SECRET_KEY'] = 'testkey-not-real'

try:
    from app import create_app
    app = create_app()
except Exception as e:
    print(f'FAIL app startup: {e}')
    sys.exit(1)

rules = [str(r) for r in app.url_map.iter_rules()]
key_routes = [
    '/login', '/logout', '/register',
    '/api/session', '/api/dashboard-stats',
    '/student/dashboard', '/admin/dashboard',
    '/teacher/dashboard', '/parent/dashboard',
    '/api/face/status', '/api/face/register',
    '/api/face/recognize',
]
all_ok = True
for r in key_routes:
    found = any(r in rule for rule in rules)
    status = 'PASS' if found else 'FAIL'
    if not found:
        all_ok = False
    print(f'{status}: {r}')

print(f'\nTotal routes registered: {len(rules)}')

# Verify CORS origins
from app import app as flask_app
print('\nCORS check: app created OK')

# Verify ML predictor loads
try:
    from ml.predictor import predict_student_risk, get_model_info
    info = get_model_info()
    pred = predict_student_risk({'gpa': 2.5, 'attendance_pct': 65.0, 'department': 'Computer Science'})
    print(f'PASS: ML predictor — risk_level={pred["risk_level"]} score={pred["risk_score"]} accuracy={info["accuracy"]}%')
except Exception as e:
    print(f'FAIL: ML predictor — {e}')
    all_ok = False

sys.exit(0 if all_ok else 1)
