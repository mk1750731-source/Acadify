import os
os.environ['MONGO_URI'] = 'mongodb://127.0.0.1:27017/testdb'
os.environ['SECRET_KEY'] = 'testkey-not-real'
from app import create_app
app = create_app()
client = app.test_client()

resp = client.post('/api/predict-form', json={'gpa_current': 2.5}, content_type='application/json')
print('predict-form status:', resp.status_code)
print('predict-form data:', resp.get_json())

resp2 = client.get('/admin/dashboard', headers={'Accept': 'application/json'})
print('admin/dashboard status:', resp2.status_code)
print('admin/dashboard data:', resp2.get_json())
