"""
EmotionModel — facial expression detection using DeepFace's pretrained emotion model.
"""
import threading
from deepface import DeepFace

_lock = threading.Lock()
_warmed_up = False


def _warmup():
    global _warmed_up
    if _warmed_up:
        return
    with _lock:
        if _warmed_up:
            return
        try:
            import numpy as np
            dummy = np.zeros((48, 48, 3), dtype="uint8")
            DeepFace.analyze(img_path=dummy, actions=["emotion"], enforce_detection=False, silent=True)
            _warmed_up = True
            print("[EmotionModel] warmed up.")
        except Exception as e:
            print(f"[EmotionModel] Warmup error: {e}")


class EmotionModel:
    def __init__(self):
        self._ready = False
        try:
            _warmup()
            self._ready = _warmed_up
        except Exception as e:
            print(f"[EmotionModel] Load error: {e}")

    @property
    def ready(self):
        return self._ready

    def predict(self, face_bgr_or_gray):
        if not self._ready:
            return {"success": False, "error": "Emotion model not loaded."}
        try:
            import cv2
            img = face_bgr_or_gray
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            result = DeepFace.analyze(img_path=img, actions=["emotion"], enforce_detection=False, silent=True)
            if not result:
                return {"success": False, "error": "No face detected for emotion analysis."}
            emotions = result[0].get("emotion", {})
            if not emotions:
                return {"success": False, "error": "Emotion analysis returned no data."}
            total = sum(emotions.values()) or 1.0
            probabilities = {k: round(v / total, 4) for k, v in emotions.items()}
            dominant = result[0].get("dominant_emotion", max(emotions, key=emotions.get))
            return {
                "success": True,
                "prediction": dominant,
                "confidence": round(probabilities.get(dominant, 0.0), 4),
                "probabilities": probabilities,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
