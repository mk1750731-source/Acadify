"""
FaceRecognizer — MediaPipe BlazeFace detection + Face Mesh landmark embeddings.
Pre-built wheels, no compilation, ~100MB RAM. Works on Railway free tier.

Embedding: 468 face landmarks (x,y,z) = 1404-dim vector, L2-normalised.
Stored in MongoDB as float arrays (replaces old dlib/DeepFace embeddings).
"""
import os
import threading
import numpy as np

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.12"))

_lock = threading.Lock()
_mp_face_mesh = None
_mp_face_detection = None
_ready = False
_load_error = None


def _load():
    global _mp_face_mesh, _mp_face_detection, _ready, _load_error
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
                refine_landmarks=False,
                min_detection_confidence=0.5,
            )
            _mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )
            _ready = True
            _load_error = None
            print("[FaceRecognizer] mediapipe ready.")
            return True
        except Exception as e:
            _load_error = str(e)
            print(f"[FaceRecognizer] load error: {e}")
            return False


def warmup_async():
    threading.Thread(target=_load, daemon=True).start()


class FaceRecognizer:
    def __init__(self):
        warmup_async()

    @property
    def ready(self):
        return _ready or _load()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _to_rgb(self, img):
        """Ensure uint8 RGB numpy array."""
        import cv2
        arr = np.asarray(img, dtype="uint8")
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _get_landmarks_embedding(self, image_rgb):
        """
        Returns (embedding_1404_dim, bbox_tuple) or (None, error_str).
        bbox = (top, right, bottom, left) in pixels.
        """
        if not self.ready:
            return None, "Face recognition not ready."
        img = self._to_rgb(image_rgb)
        results = _mp_face_mesh.process(img)
        if not results.multi_face_landmarks:
            return None, "No face detected. Please use a clear frontal photo."
        if len(results.multi_face_landmarks) > 1:
            return None, f"{len(results.multi_face_landmarks)} faces detected. Ensure only one person is in frame."

        h, w = img.shape[:2]
        lm = results.multi_face_landmarks[0].landmark

        # Build 1404-dim embedding from all 468 landmarks (x, y, z)
        coords = np.array([[p.x, p.y, p.z] for p in lm], dtype="float32").flatten()

        # Centre + scale normalise (make pose-invariant)
        coords_xy = coords.reshape(-1, 3)[:, :2]
        centre = coords_xy.mean(axis=0)
        coords_xy -= centre
        scale = np.linalg.norm(coords_xy) or 1.0
        coords_xy /= scale
        coords_norm = coords.reshape(-1, 3).copy()
        coords_norm[:, :2] = coords_xy
        emb = coords_norm.flatten()

        # L2 normalise
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm

        # Bounding box from landmarks
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        pad = 10
        bbox = (max(0, min(ys) - pad), min(w, max(xs) + pad),
                min(h, max(ys) + pad), max(0, min(xs) - pad))

        return emb, bbox

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_faces(self, image_rgb):
        """Return face locations as list of (top, right, bottom, left)."""
        if not self.ready:
            return []
        try:
            img = self._to_rgb(image_rgb)
            results = _mp_face_detection.process(img)
            if not results.detections:
                return []
            h, w = img.shape[:2]
            locs = []
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                left = max(0, int(bb.xmin * w))
                top = max(0, int(bb.ymin * h))
                right = min(w, int((bb.xmin + bb.width) * w))
                bottom = min(h, int((bb.ymin + bb.height) * h))
                locs.append((top, right, bottom, left))
            return locs
        except Exception as e:
            print(f"[FaceRecognizer] detect_faces error: {e}")
            return []

    def process_registration_image(self, image_rgb):
        if not self.ready:
            return {"success": False, "error": "Face recognition models are still loading. Please wait."}
        emb, result = self._get_landmarks_embedding(image_rgb)
        if emb is None:
            return {"success": False, "error": result}
        return {
            "success": True,
            "embedding": emb.tolist(),
            "face_location": result,
        }

    @staticmethod
    def _cosine_distance(a, b):
        return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))

    def identify(self, image_rgb, known_students):
        if not self.ready:
            return {"recognized": False, "message": "Face recognition models are still loading. Please wait."}

        emb, result = self._get_landmarks_embedding(image_rgb)
        if emb is None:
            return {"recognized": False, "message": result}

        face_location = result
        best_distance = float("inf")
        best_student = None

        for student in known_students:
            for embedding in student.get("face_embeddings", []):
                try:
                    ref = np.array(embedding, dtype="float32")
                    if ref.shape != emb.shape:
                        continue
                    norm = np.linalg.norm(ref)
                    if norm > 1e-10:
                        ref = ref / norm
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

    def crop_face(self, image_bgr, face_location, padding=10):
        top, right, bottom, left = face_location
        h, w = image_bgr.shape[:2]
        return image_bgr[
            max(0, top - padding):min(h, bottom + padding),
            max(0, left - padding):min(w, right + padding),
        ]
