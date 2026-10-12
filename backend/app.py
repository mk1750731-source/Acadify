
import os

from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager

from config import Config
from models.db import mongo
from models.user import User
import react_api


# Apply React API patches so API requests return JSON instead of HTML.
react_api.apply_patches()


def create_app():
    app = Flask(__name__)

    # Frontend origins allowed to call the backend.
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://acadify-nu-dusky.vercel.app",
        "https://acadify-frontend-production.up.railway.app",
        os.environ.get("FRONTEND_URL", ""),
    ]

    # Remove empty entries and trailing slashes.
    allowed_origins = [
        origin.rstrip("/")
        for origin in allowed_origins
        if origin and origin.strip()
    ]

    # Load Flask configuration.
    app.config.from_object(Config)

    # Configure CORS for credentialed requests.
    CORS(
        app,
        supports_credentials=True,
        origins=allowed_origins,
    )

    # Initialize MongoDB.
    mongo.init_app(app)

    # Initialize Flask-Login.
    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.get_by_id(user_id)

    # Register application blueprints.
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

    # Warm up AI models.
    # Keep the existing implementation; verify in ai_service.py
    # whether warmup_models() runs asynchronously.
    try:
        from ai_service import warmup_models
        warmup_models()
    except Exception as exc:
        app.logger.exception(
            "AI model warm-up failed: %s",
            exc,
        )

    return app


# Gunicorn imports this object in production.
app = create_app()


if __name__ == "__main__":
    # Local development only.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
