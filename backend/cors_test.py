"""CORS preflight test — verifies Access-Control-Allow-Origin headers."""
import os, sys
os.environ['MONGO_URI'] = 'mongodb://127.0.0.1:27017/testdb'
os.environ['SECRET_KEY'] = 'testkey-not-real'

from app import create_app
app = create_app()
client = app.test_client()

ORIGINS = [
    'https://acadify-nu-dusky.vercel.app',
    'https://acadify-frontend-production.up.railway.app',
    'http://localhost:3000',
]
ROUTES = ['/login', '/api/session', '/api/dashboard-stats']

all_ok = True
for origin in ORIGINS:
    for route in ROUTES:
        resp = client.options(route, headers={
            'Origin': origin,
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type',
        })
        acao = resp.headers.get('Access-Control-Allow-Origin', '')
        acac = resp.headers.get('Access-Control-Allow-Credentials', '')
        ok = origin in acao or acao == '*'
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            all_ok = False
        print(f'{status}: OPTIONS {route} from {origin[:40]} -> ACAO={acao!r} ACAC={acac!r}')

# Test rejected origin
resp = client.options('/login', headers={
    'Origin': 'https://evil.example.com',
    'Access-Control-Request-Method': 'POST',
})
acao = resp.headers.get('Access-Control-Allow-Origin', '')
rejected = 'evil.example.com' not in acao
status = 'PASS' if rejected else 'FAIL'
if not rejected:
    all_ok = False
print(f'{status}: OPTIONS /login from evil.example.com rejected -> ACAO={acao!r}')

sys.exit(0 if all_ok else 1)
