"""
EmotionModel — facial expression detection using DeepFace's pretrained emotion model.
Same public API as before: ready, predict(face_bgr).
"""
import threading

_lock = threading.Lock()
_warmed_up = False

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def _warmup():
    global _warmed_up
    if _warmed_up:
        return
    with _lock:
        if _warmed_up:
            return
        try:
            from deepface import DeepFace
            import numpy as np
            dummy = np.zeros((48, 48, 3), dtype="uint8")
            DeepFace.analyze(
                img_path=dummy,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            _warmed_up = True
            print("[EmotionModel] DeepFace emotion model warmed up.")
        except Exception as e:
            print(f"[EmotionModel] Warmup error: {e}")


class EmotionModel:
    def __init__(self):
        self._ready = False
        self._load()

    def _load(self):
        try:
            _warmup()
            self._ready = _warmed_up
        except Exception as e:
            print(f"[EmotionModel] Load error: {e}")

    @property
    def ready(self):
        return self._ready

    def predict(self, face_bgr_or_gray):
        """
        Accept a BGR or grayscale numpy array (cropped face region).
        Returns dict: {success, prediction, confidence, probabilities}
        """
        if not self._ready:
            return {"success": False, "error": "Emotion model not loaded."}
        try:
            import cv2
            import numpy as np
            img = face_bgr_or_gray
            # Ensure BGR 3-channel for DeepFace
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            result = DeepFace_analyze(img)
            if not result:
                return {"success": False, "error": "No face detected for emotion analysis."}
            emotions = result[0].get("emotion", {})
            if not emotions:
                return {"success": False, "error": "Emotion analysis returned no data."}
            # Normalise to 0-1 probabilities
            total = sum(emotions.values()) or 1.0
            probabilities = {k: round(v / total, 4) for k, v in emotions.items()}
            dominant = result[0].get("dominant_emotion", max(emotions, key=emotions.get))
            confidence = probabilities.get(dominant, 0.0)
            return {
                "success": True,
                "prediction": dominant,
                "confidence": round(confidence, 4),
                "probabilities": probabilities,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def DeepFace_analyze(img_bgr):
    from deepface import DeepFace
    return DeepFace.analyze(
        img_path=img_bgr,
        actions=["emotion"],
        enforce_detection=False,
        silent=True,
    )
