"""
FaceRecognizer — face detection, embedding generation, and identity matching.
Uses DeepFace with the Facenet pretrained model (downloaded once on first use).
Same public API as before: process_registration_image, identify, detect_faces, crop_face.
"""
import os
import threading
import numpy as np

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.40"))
_EMBEDDING_MODEL = os.environ.get("FACE_EMBEDDING_MODEL", "Facenet")  # or "VGG-Face", "ArcFace"
_DETECTOR_BACKEND = os.environ.get("FACE_DETECTOR_BACKEND", "opencv")  # lightweight, no extra downloads

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
            from deepface import DeepFace
            import numpy as np
            dummy = np.zeros((160, 160, 3), dtype="uint8")
            DeepFace.represent(
                img_path=dummy,
                model_name=_EMBEDDING_MODEL,
                detector_backend=_DETECTOR_BACKEND,
                enforce_detection=False,
            )
            _warmed_up = True
            print(f"[FaceRecognizer] DeepFace warmed up — model={_EMBEDDING_MODEL}, detector={_DETECTOR_BACKEND}")
        except Exception as e:
            print(f"[FaceRecognizer] Warmup error: {e}")


class FaceRecognizer:
    def __init__(self):
        self._ready = False
        self._load()

    def _load(self):
        try:
            _warmup()
            self._ready = _warmed_up
        except Exception as e:
            print(f"[FaceRecognizer] Load error: {e}")

    @property
    def ready(self):
        return self._ready

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_faces(self, image_rgb):
        """Return face locations as list of (top, right, bottom, left)."""
        try:
            import cv2
            from deepface.detectors import DetectorWrapper
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            faces = DetectorWrapper.detect_faces(
                detector_backend=_DETECTOR_BACKEND,
                img=img_bgr,
                align=False,
            )
            locations = []
            for face_obj in faces:
                region = face_obj.facial_area if hasattr(face_obj, 'facial_area') else face_obj.get('facial_area', {})
                x = region.x if hasattr(region, 'x') else region.get('x', 0)
                y = region.y if hasattr(region, 'y') else region.get('y', 0)
                w = region.w if hasattr(region, 'w') else region.get('w', 0)
                h = region.h if hasattr(region, 'h') else region.get('h', 0)
                locations.append((y, x + w, y + h, x))
            return locations
        except Exception as e:
            print(f"[FaceRecognizer] detect_faces error: {e}")
            return []

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _get_embedding(self, image_rgb):
        """Return L2-normalised embedding array from an RGB image, or None."""
        try:
            from deepface import DeepFace
            import cv2
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            results = DeepFace.represent(
                img_path=img_bgr,
                model_name=_EMBEDDING_MODEL,
                detector_backend=_DETECTOR_BACKEND,
                enforce_detection=True,
            )
            if not results:
                return None, "No face detected."
            if len(results) > 1:
                return None, f"{len(results)} faces detected. Please ensure only one person is in the frame."
            emb = np.array(results[0]["embedding"], dtype="float32")
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            return emb, None
        except Exception as e:
            msg = str(e)
            if "Face could not be detected" in msg or "No face" in msg.lower():
                return None, "No face detected. Please use a clear frontal photo."
            return None, f"Embedding error: {msg}"

    # ── Registration ──────────────────────────────────────────────────────────

    def process_registration_image(self, image_rgb):
        if not self._ready:
            return {"success": False, "error": "Face recognition models not available."}
        emb, err = self._get_embedding(image_rgb)
        if emb is None:
            return {"success": False, "error": err}
        # Detect location for crop reference
        locs = self.detect_faces(image_rgb)
        return {
            "success": True,
            "embedding": emb.tolist(),
            "face_location": locs[0] if locs else (0, image_rgb.shape[1], image_rgb.shape[0], 0),
        }

    # ── Recognition ───────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_distance(a, b):
        return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))

    def identify(self, image_rgb, known_students):
        if not self._ready:
            return {"recognized": False, "message": "Face recognition models not available."}

        emb, err = self._get_embedding(image_rgb)
        if emb is None:
            return {"recognized": False, "message": err}

        best_distance = float("inf")
        best_student = None

        for student in known_students:
            for embedding in student.get("face_embeddings", []):
                try:
                    ref = np.asarray(embedding, dtype="float32")
                    norm = np.linalg.norm(ref)
                    if norm > 1e-10:
                        ref = ref / norm
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

        locs = self.detect_faces(image_rgb)
        return {
            "recognized": True,
            "student_id": str(best_student["student_id"]),
            "student_name": best_student.get("name", ""),
            "roll_no": best_student.get("roll_no", ""),
            "distance": round(best_distance, 4),
            "face_location": locs[0] if locs else (0, image_rgb.shape[1], image_rgb.shape[0], 0),
        }

    # ── Crop ──────────────────────────────────────────────────────────────────

    def crop_face(self, image_bgr, face_location, padding=10):
        top, right, bottom, left = face_location
        h, w = image_bgr.shape[:2]
        return image_bgr[
            max(0, top - padding):min(h, bottom + padding),
            max(0, left - padding):min(w, right + padding),
        ]
