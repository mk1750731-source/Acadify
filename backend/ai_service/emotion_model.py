"""
EmotionModel — facial expression detection using the fer library.
No TensorFlow. Works within Railway 512MB RAM.
"""
import threading

_detector = None
_ready = False
_lock = threading.Lock()


def _load():
    global _detector, _ready
    if _ready:
        return True
    with _lock:
        if _ready:
            return True
        try:
            from fer import FER
            _detector = FER(mtcnn=False)  # mtcnn=False uses OpenCV cascade — faster, no extra deps
            _ready = True
            print("[EmotionModel] fer ready.")
            return True
        except Exception as e:
            print(f"[EmotionModel] load error: {e}")
            return False


def warmup_async():
    threading.Thread(target=_load, daemon=True).start()


class EmotionModel:
    def __init__(self):
        warmup_async()

    @property
    def ready(self):
        return _ready or _load()

    def predict(self, face_bgr_or_gray):
        if not self.ready:
            return {"success": False, "error": "Emotion model not loaded."}
        try:
            import cv2
            img = face_bgr_or_gray
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # fer expects BGR
            result = _detector.detect_emotions(img)
            if not result:
                # Try on full image without face detection
                emotions = _detector.predict_emotions(img, logits=False)
                if emotions is None:
                    return {"success": False, "error": "No emotion detected."}
                dominant = max(emotions, key=emotions.get)
                total = sum(emotions.values()) or 1.0
                probabilities = {k: round(v / total, 4) for k, v in emotions.items()}
                return {
                    "success": True,
                    "prediction": dominant,
                    "confidence": round(probabilities[dominant], 4),
                    "probabilities": probabilities,
                }
            emotions = result[0].get("emotions", {})
            if not emotions:
                return {"success": False, "error": "No emotions returned."}
            dominant = max(emotions, key=emotions.get)
            total = sum(emotions.values()) or 1.0
            probabilities = {k: round(v / total, 4) for k, v in emotions.items()}
            return {
                "success": True,
                "prediction": dominant,
                "confidence": round(probabilities[dominant], 4),
                "probabilities": probabilities,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
