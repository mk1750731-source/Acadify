import os
from flask import Flask, request
from flask_cors import CORS
from flask_login import LoginManager

from config import Config
from models.db import mongo
from models.user import User
import react_api

react_api.apply_patches()


def _get_origins():
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://acadify-nu-dusky.vercel.app",
        "https://acadify-frontend-production.up.railway.app",
    ]
    env_frontend = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
    if env_frontend:
        origins.append(env_frontend)
    return origins


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(
        app,
        origins=_get_origins(),
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    # Initialize MongoDB
    mongo.init_app(app)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    @login_manager.unauthorized_handler
    def unauthorized():
        from flask import request, jsonify
        if (request.is_json
                or request.headers.get('Accept') == 'application/json'
                or request.path.startswith('/api/')):
            return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
        from flask import redirect, url_for
        return redirect(url_for('auth.login', next=request.url))

    @login_manager.user_loader
    def load_user(user_id):
        return User.get_by_id(user_id)

    # Register blueprints
    from blueprints.main import main_bp
    from blueprints.auth import auth_bp
    from blueprints.student import student_bp
    from blueprints.admin import admin_bp
    from blueprints.api import api_bp
    from blueprints.face import face_bp
    from blueprints.teacher import teacher_bp
    from blueprints.parent import parent_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(face_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(parent_bp)

    # AI runs client-side — no server warmup needed

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
