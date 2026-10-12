"""Lazy singletons for face AI services."""
import threading

_lock = threading.Lock()
_face_recognizer = None
_emotion_predictor = None


def _get_face_recognizer():
    global _face_recognizer
    if _face_recognizer is None:
        with _lock:
            if _face_recognizer is None:
                from ai_service.face_recognition import FaceRecognizer
                _face_recognizer = FaceRecognizer()
    return _face_recognizer


def _get_emotion_predictor():
    global _emotion_predictor
    if _emotion_predictor is None:
        with _lock:
            if _emotion_predictor is None:
                from ai_service.emotion_model import EmotionModel
                _emotion_predictor = EmotionModel()
    return _emotion_predictor


class _Proxy:
    def __init__(self, getter):
        object.__setattr__(self, '_getter', getter)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_getter')(), name)


face_recognizer = _Proxy(_get_face_recognizer)
emotion_predictor = _Proxy(_get_emotion_predictor)


def warmup_models():
    """Trigger instantiation (and async warmup) of both services at startup."""
    def _load():
        try:
            _get_face_recognizer()
        except Exception as e:
            print(f"[warmup] face_recognizer error: {e}")
        try:
            _get_emotion_predictor()
        except Exception as e:
            print(f"[warmup] emotion_predictor error: {e}")
    threading.Thread(target=_load, daemon=True).start()
