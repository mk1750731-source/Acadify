"""Auth flow test — login success/failure, session, logout."""
import os, sys, json
os.environ['MONGO_URI'] = 'mongodb://127.0.0.1:27017/testdb'
os.environ['SECRET_KEY'] = 'testkey-not-real'

from app import create_app
app = create_app()
client = app.test_client()

all_ok = True

# 1. GET /login — unauthenticated
resp = client.get('/login', headers={'Accept': 'application/json'})
data = resp.get_json()
ok = resp.status_code == 200 and data.get('authenticated') == False
print(f'{"PASS" if ok else "FAIL"}: GET /login unauthenticated -> {data}')
if not ok: all_ok = False

# 2. POST /login — missing fields
resp = client.post('/login', json={}, headers={'Accept': 'application/json'})
ok = resp.status_code == 400
print(f'{"PASS" if ok else "FAIL"}: POST /login empty body -> {resp.status_code}')
if not ok: all_ok = False

# 3. POST /login — wrong credentials
resp = client.post('/login', json={'email': 'nobody@test.com', 'password': 'wrong'},
                   headers={'Accept': 'application/json'})
ok = resp.status_code == 401
print(f'{"PASS" if ok else "FAIL"}: POST /login wrong creds -> {resp.status_code}')
if not ok: all_ok = False

# 4. GET /api/session — unauthenticated
resp = client.get('/api/session', headers={'Accept': 'application/json'})
ok = resp.status_code == 401
print(f'{"PASS" if ok else "FAIL"}: GET /api/session unauthenticated -> {resp.status_code}')
if not ok: all_ok = False

# 5. GET /logout — always works
resp = client.get('/logout', headers={'Accept': 'application/json'})
data = resp.get_json()
ok = resp.status_code == 200 and data.get('status') == 'success'
print(f'{"PASS" if ok else "FAIL"}: GET /logout -> {data}')
if not ok: all_ok = False

# 6. POST /forgot-password — missing email
resp = client.post('/forgot-password', json={}, headers={'Accept': 'application/json'})
ok = resp.status_code == 400
print(f'{"PASS" if ok else "FAIL"}: POST /forgot-password empty -> {resp.status_code}')
if not ok: all_ok = False

# 7. POST /forgot-password — non-existent email (should not reveal)
resp = client.post('/forgot-password', json={'email': 'nobody@test.com'},
                   headers={'Accept': 'application/json'})
data = resp.get_json()
ok = resp.status_code == 200 and data.get('success') == True
print(f'{"PASS" if ok else "FAIL"}: POST /forgot-password unknown email -> {data.get("message", "")}')
if not ok: all_ok = False

# 8. Role-protected route — unauthenticated
resp = client.get('/admin/dashboard', headers={'Accept': 'application/json'})
ok = resp.status_code == 401
print(f'{"PASS" if ok else "FAIL"}: GET /admin/dashboard unauthenticated -> {resp.status_code}')
if not ok: all_ok = False

# 9. ML predictor via API — unauthenticated predict-form
resp = client.post('/api/predict-form', json={'gpa_current': 2.5, 'attendance_pct': 65},
                   headers={'Accept': 'application/json'})
ok = resp.status_code == 401
print(f'{"PASS" if ok else "FAIL"}: POST /api/predict-form unauthenticated -> {resp.status_code}')
if not ok: all_ok = False

print(f'\nAuth tests: {"ALL PASS" if all_ok else "SOME FAILED"}')
sys.exit(0 if all_ok else 1)
