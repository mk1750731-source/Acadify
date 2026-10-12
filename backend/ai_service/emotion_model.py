"""
EmotionModel — lightweight emotion estimation using facial geometry from MediaPipe.
No TensorFlow. No downloads. Works on Railway 512MB.

Uses mouth openness, eye openness, and eyebrow raise as heuristic signals.
Returns one of: happy, neutral, sad, surprised, angry.
"""
import threading
import numpy as np

_lock = threading.Lock()
_ready = False
_mp_face_mesh = None


def _load():
    global _ready, _mp_face_mesh
    if _ready:
        return True
    with _lock:
        if _ready:
            return True
        try:
            import mediapipe as mp
            _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            _ready = True
            print("[EmotionModel] mediapipe geometry emotion ready.")
            return True
        except Exception as e:
            print(f"[EmotionModel] load error: {e}")
            return False


def warmup_async():
    threading.Thread(target=_load, daemon=True).start()


# MediaPipe landmark indices
_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_EYE_TOP = 159
_LEFT_EYE_BOT = 145
_RIGHT_EYE_TOP = 386
_RIGHT_EYE_BOT = 374
_LEFT_BROW = 105
_RIGHT_BROW = 334
_NOSE_TIP = 1
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362


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
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = _mp_face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return {"success": False, "error": "No face detected."}

            lm = results.multi_face_landmarks[0].landmark

            def dist(a, b):
                return np.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)

            # Normalise by inter-eye distance
            eye_dist = dist(_LEFT_EYE_INNER, _RIGHT_EYE_INNER) or 0.1

            mouth_open = dist(_UPPER_LIP, _LOWER_LIP) / eye_dist
            left_eye_open = dist(_LEFT_EYE_TOP, _LEFT_EYE_BOT) / eye_dist
            right_eye_open = dist(_RIGHT_EYE_TOP, _RIGHT_EYE_BOT) / eye_dist
            eye_open = (left_eye_open + right_eye_open) / 2

            # Brow raise: brow y vs nose tip y (lower y = higher on image)
            brow_y = (lm[_LEFT_BROW].y + lm[_RIGHT_BROW].y) / 2
            nose_y = lm[_NOSE_TIP].y
            brow_raise = (nose_y - brow_y) / eye_dist

            # Mouth corners: left corner x < right corner x; y relative to lip centre
            mouth_left_y = lm[61].y
            mouth_right_y = lm[291].y
            lip_centre_y = (lm[_UPPER_LIP].y + lm[_LOWER_LIP].y) / 2
            smile = ((lip_centre_y - mouth_left_y) + (lip_centre_y - mouth_right_y)) / eye_dist

            # Heuristic scoring
            scores = {
                "happy":     max(0.0, smile * 3.0 + mouth_open * 0.5),
                "surprised": max(0.0, mouth_open * 4.0 + brow_raise * 1.5),
                "angry":     max(0.0, -brow_raise * 2.0 - smile * 1.0),
                "sad":       max(0.0, -smile * 2.0 + (0.3 - eye_open) * 2.0),
                "neutral":   1.0,
            }

            total = sum(scores.values()) or 1.0
            probabilities = {k: round(v / total, 4) for k, v in scores.items()}
            dominant = max(probabilities, key=probabilities.get)

            return {
                "success": True,
                "prediction": dominant,
                "confidence": round(probabilities[dominant], 4),
                "probabilities": probabilities,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
