import flask
from flask import request, jsonify
from bson import ObjectId
from datetime import datetime

original_render_template = flask.render_template
original_redirect = flask.redirect
original_flash = flask.flash

def serialize_for_react(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_for_react(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_react(i) for i in obj]
    if type(obj).__name__ == 'User':
        return {"id": str(getattr(obj, '_id', '')), "name": getattr(obj, 'name', ''), "email": getattr(obj, 'email', ''), "role": getattr(obj, 'role', '')}
    
    try:
        from werkzeug.local import LocalProxy
        if isinstance(obj, LocalProxy):
            # check if it's an anonymous user
            if not obj.is_authenticated:
                return None
            return serialize_for_react(obj._get_current_object())
    except:
        pass
        
    if hasattr(obj, 'to_dict'):
        return serialize_for_react(obj.to_dict())
    
    # Primitive types
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    # Fallback string representation for other objects
    return str(obj)

def is_api_request():
    return request.headers.get('Accept') == 'application/json' or request.path.startswith('/api/') or request.is_json

def patched_render_template(template_name_or_list, **context):
    if is_api_request():
        clean_context = serialize_for_react(context)
        # Include any flashed messages
        flashes = flask.get_flashed_messages(with_categories=True)
        if flashes:
            clean_context['_flashes'] = flashes
        return jsonify(clean_context)
    return original_render_template(template_name_or_list, **context)

def patched_redirect(location, code=302, Response=None):
    if is_api_request():
        flashes = flask.get_flashed_messages(with_categories=True)
        return jsonify({"redirect": location, "_flashes": flashes})
    return original_redirect(location, code, Response)

def patched_flash(message, category='message'):
    # We still use the original flash so the session is updated,
    # but we could also intercept it if needed.
    return original_flash(message, category)

def apply_patches():
    flask.render_template = patched_render_template
    flask.redirect = patched_redirect
    flask.flash = patched_flash
