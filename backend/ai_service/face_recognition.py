"""
FaceRecognizer — dlib HOG + ResNet128 embeddings via face_recognition library.
No model downloads at runtime. Works within Railway 512MB RAM.
Embeddings are 128-dim float arrays — stored in MongoDB as before.
"""
import os
import numpy as np

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.55"))

_fr = None
_ready = False


def _load():
    global _fr, _ready
    if _ready:
        return True
    try:
        import face_recognition as _face_recognition
        _fr = _face_recognition
        _ready = True
        print("[FaceRecognizer] face_recognition (dlib) ready.")
        return True
    except Exception as e:
        print(f"[FaceRecognizer] load error: {e}")
        return False


def warmup_async():
    import threading
    threading.Thread(target=_load, daemon=True).start()


class FaceRecognizer:
    def __init__(self):
        warmup_async()

    @property
    def ready(self):
        return _ready or _load()

    def _get_fr(self):
        if not self.ready:
            return None
        return _fr

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_rgb(self, image_rgb):
        """Ensure uint8 RGB numpy array."""
        arr = np.asarray(image_rgb, dtype="uint8")
        if arr.ndim == 2:
            import cv2
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _get_encodings(self, image_rgb, num_jitters=1):
        """Return (encodings, locations) or (None, error_str)."""
        fr = self._get_fr()
        if fr is None:
            return None, "Face recognition not ready."
        img = self._to_rgb(image_rgb)
        locations = fr.face_locations(img, model="hog")
        if not locations:
            return None, "No face detected. Please use a clear frontal photo."
        encodings = fr.face_encodings(img, known_face_locations=locations, num_jitters=num_jitters)
        return encodings, locations

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_faces(self, image_rgb):
        """Return face locations as list of (top, right, bottom, left)."""
        fr = self._get_fr()
        if fr is None:
            return []
        try:
            img = self._to_rgb(image_rgb)
            return fr.face_locations(img, model="hog")
        except Exception as e:
            print(f"[FaceRecognizer] detect_faces error: {e}")
            return []

    def process_registration_image(self, image_rgb):
        if not self.ready:
            return {"success": False, "error": "Face recognition models are still loading. Please wait."}
        try:
            encodings, locations = self._get_encodings(image_rgb, num_jitters=2)
            if encodings is None:
                return {"success": False, "error": locations}  # locations holds error str here
            if len(encodings) > 1:
                return {"success": False, "error": f"{len(encodings)} faces detected. Ensure only one person is in frame."}
            emb = np.array(encodings[0], dtype="float32")
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            return {
                "success": True,
                "embedding": emb.tolist(),
                "face_location": locations[0],
            }
        except Exception as e:
            return {"success": False, "error": f"Registration error: {e}"}

    def identify(self, image_rgb, known_students):
        if not self.ready:
            return {"recognized": False, "message": "Face recognition models are still loading. Please wait."}
        try:
            encodings, locations = self._get_encodings(image_rgb, num_jitters=1)
            if encodings is None:
                return {"recognized": False, "message": locations}
            if len(encodings) > 1:
                return {"recognized": False, "message": "Multiple faces detected. Ensure only one person is in frame."}

            query_enc = np.array(encodings[0], dtype="float32")
            norm = np.linalg.norm(query_enc)
            if norm > 1e-10:
                query_enc = query_enc / norm

            best_distance = float("inf")
            best_student = None

            for student in known_students:
                for embedding in student.get("face_embeddings", []):
                    try:
                        ref = np.array(embedding, dtype="float32")
                        ref_norm = np.linalg.norm(ref)
                        if ref_norm > 1e-10:
                            ref = ref / ref_norm
                        if ref.shape != query_enc.shape:
                            continue
                        # Euclidean distance on normalised vectors
                        dist = float(np.linalg.norm(query_enc - ref))
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
                "face_location": locations[0],
            }
        except Exception as e:
            return {"recognized": False, "message": f"Recognition error: {e}"}

    def crop_face(self, image_bgr, face_location, padding=10):
        top, right, bottom, left = face_location
        h, w = image_bgr.shape[:2]
        return image_bgr[
            max(0, top - padding):min(h, bottom + padding),
            max(0, left - padding):min(w, right + padding),
        ]
