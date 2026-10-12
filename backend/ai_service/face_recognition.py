"""
FaceRecognizer — face detection, embedding generation, and identity matching.
Uses DeepFace with the Facenet pretrained model.
"""
import os
import threading
import numpy as np
from deepface import DeepFace

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.40"))
_EMBEDDING_MODEL = os.environ.get("FACE_EMBEDDING_MODEL", "Facenet")
_DETECTOR_BACKEND = os.environ.get("FACE_DETECTOR_BACKEND", "opencv")

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
            dummy = np.zeros((160, 160, 3), dtype="uint8")
            DeepFace.represent(
                img_path=dummy,
                model_name=_EMBEDDING_MODEL,
                detector_backend=_DETECTOR_BACKEND,
                enforce_detection=False,
            )
            _warmed_up = True
            print(f"[FaceRecognizer] warmed up — model={_EMBEDDING_MODEL}, detector={_DETECTOR_BACKEND}")
        except Exception as e:
            print(f"[FaceRecognizer] Warmup error: {e}")


class FaceRecognizer:
    def __init__(self):
        self._ready = False
        try:
            _warmup()
            self._ready = _warmed_up
        except Exception as e:
            print(f"[FaceRecognizer] Load error: {e}")

    @property
    def ready(self):
        return self._ready

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _represent(self, image_rgb, enforce_detection=True):
        """
        Run DeepFace.represent and return (results, error_str).
        results is the raw list from DeepFace; each item has 'embedding' and 'facial_area'.
        """
        try:
            import cv2
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            results = DeepFace.represent(
                img_path=img_bgr,
                model_name=_EMBEDDING_MODEL,
                detector_backend=_DETECTOR_BACKEND,
                enforce_detection=enforce_detection,
            )
            return results, None
        except Exception as e:
            msg = str(e)
            if "Face could not be detected" in msg or "No face" in msg.lower():
                return None, "No face detected. Please use a clear frontal photo."
            return None, f"Embedding error: {msg}"

    def _normalise(self, emb_list):
        emb = np.array(emb_list, dtype="float32")
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-10 else emb

    # ── Detection (used for emotion crop when recognition already ran) ────────

    def detect_faces(self, image_rgb):
        """Return face locations as list of (top, right, bottom, left)."""
        results, _ = self._represent(image_rgb, enforce_detection=False)
        if not results:
            return []
        locations = []
        for r in results:
            fa = r.get("facial_area", {})
            x = fa.get("x", 0); y = fa.get("y", 0)
            w = fa.get("w", 0); h = fa.get("h", 0)
            locations.append((y, x + w, y + h, x))
        return locations

    # ── Registration ──────────────────────────────────────────────────────────

    def process_registration_image(self, image_rgb):
        if not self._ready:
            return {"success": False, "error": "Face recognition models not available."}
        results, err = self._represent(image_rgb, enforce_detection=True)
        if results is None:
            return {"success": False, "error": err}
        if len(results) > 1:
            return {"success": False, "error": f"{len(results)} faces detected. Ensure only one person is in frame."}
        emb = self._normalise(results[0]["embedding"])
        fa = results[0].get("facial_area", {})
        x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", image_rgb.shape[1]), fa.get("h", image_rgb.shape[0])
        return {
            "success": True,
            "embedding": emb.tolist(),
            "face_location": (y, x + w, y + h, x),
        }

    # ── Recognition ───────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_distance(a, b):
        return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))

    def identify(self, image_rgb, known_students):
        if not self._ready:
            return {"recognized": False, "message": "Face recognition models not available."}

        results, err = self._represent(image_rgb, enforce_detection=True)
        if results is None:
            return {"recognized": False, "message": err}
        if len(results) > 1:
            return {"recognized": False, "message": "Multiple faces detected. Ensure only one person is in frame."}

        emb = self._normalise(results[0]["embedding"])
        fa = results[0].get("facial_area", {})
        x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", image_rgb.shape[1]), fa.get("h", image_rgb.shape[0])
        face_location = (y, x + w, y + h, x)

        best_distance = float("inf")
        best_student = None

        for student in known_students:
            for embedding in student.get("face_embeddings", []):
                try:
                    ref = self._normalise(embedding)
                    if ref.shape != emb.shape:
                        continue
                    dist = self._cosine_distance(emb, ref)
                    if dist < best_distance:
                        best_distance = dist
                        best_student = student
                except Exception:
                    continue

        if best_student is None or best_distance > FACE_MATCH_THRESHOLD:
            return {"recognized": False, "message": "Unknown student. Face not registered in the system."}

        return {
            "recognized": True,
            "student_id": str(best_student["student_id"]),
            "student_name": best_student.get("name", ""),
            "roll_no": best_student.get("roll_no", ""),
            "distance": round(best_distance, 4),
            "face_location": face_location,
        }

    # ── Crop ──────────────────────────────────────────────────────────────────

    def crop_face(self, image_bgr, face_location, padding=10):
        top, right, bottom, left = face_location
        h, w = image_bgr.shape[:2]
        return image_bgr[
            max(0, top - padding):min(h, bottom + padding),
            max(0, left - padding):min(w, right + padding),
        ]
