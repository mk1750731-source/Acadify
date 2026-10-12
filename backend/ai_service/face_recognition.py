
"""
FaceRecognizer — face detection, embedding generation, and identity matching.
Uses:
  - MTCNN for face detection
  - Custom FaceNet Keras model for embeddings
  - Cosine similarity for identity matching
"""
import os
import threading
import numpy as np

FACE_MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.40"))

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "face_attendance_models",
)
_FACENET_PATH = os.path.join(_MODELS_DIR, "best_facenet_model.keras")
_MODEL_INPUT_SIZE = (160, 160)

_model = None
_detector = None
_model_lock = threading.Lock()
_detector_lock = threading.Lock()
_inference_lock = threading.RLock()


def _build_model_without_lambda():
    """
    Load FaceNet, remove its Lambda normalization layer, and return
    the 128-dimensional embedding model.

    Uses model_compat.load_compatible_model to handle compatible
    Dense-layer configuration differences between Keras versions.
    """
    import zipfile
    import json
    import shutil
    import tempfile
    import tensorflow as tf
    from model_compat import load_compatible_model

    fixed_path = _FACENET_PATH.replace(".keras", "_fixed.keras")

    # Reuse the fixed model if it already exists.
    if os.path.exists(fixed_path):
        return load_compatible_model(fixed_path, compile=False)

    if not os.path.exists(_FACENET_PATH):
        raise FileNotFoundError(
            f"FaceNet model not found: {_FACENET_PATH}"
        )

    # Read the original Keras archive.
    with zipfile.ZipFile(_FACENET_PATH, "r") as z:
        cfg = json.loads(z.read("config.json"))
        weights_data = z.read("model.weights.h5")
        metadata = z.read("metadata.json")

    layers = cfg["config"]["layers"]

    # Locate the Lambda layer that performs L2 normalization.
    lambda_idx = next(
        (
            i for i, layer in enumerate(layers)
            if layer.get("class_name") == "Lambda"
        ),
        None,
    )

    if lambda_idx is None:
        raise ValueError(
            "No Lambda layer found in the FaceNet model config."
        )

    if lambda_idx == 0:
        raise ValueError(
            "The Lambda layer has no preceding embedding layer."
        )

    # Keep the layers before Lambda. The preceding layer supplies embeddings.
    pre_lambda_name = layers[lambda_idx - 1]["config"]["name"]
    cfg["config"]["layers"] = layers[:lambda_idx]
    cfg["config"]["output_layers"] = [pre_lambda_name, 0, 0]

    # Write a modified .keras archive without altering the original model.
    tmp_dir = tempfile.mkdtemp()

    try:
        files_to_write = {
            "config.json": json.dumps(cfg).encode("utf-8"),
            "model.weights.h5": weights_data,
            "metadata.json": metadata,
        }

        for filename, data in files_to_write.items():
            with open(os.path.join(tmp_dir, filename), "wb") as f:
                f.write(data)

        with zipfile.ZipFile(
            fixed_path, "w", zipfile.ZIP_DEFLATED
        ) as zout:
            for filename in files_to_write:
                zout.write(
                    os.path.join(tmp_dir, filename),
                    arcname=filename,
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    model = load_compatible_model(fixed_path, compile=False)
    print(f"[FaceRecognizer] Fixed model saved -> {fixed_path}")
    return model


def _get_model():
    """Return the shared FaceNet model, loading it only once per process."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _build_model_without_lambda()

    return _model


def _get_detector():
    """Return the shared MTCNN detector, loading it only once per process."""
    global _detector

    if _detector is None:
        with _detector_lock:
            if _detector is None:
                from mtcnn import MTCNN
                _detector = MTCNN()

    return _detector


class FaceRecognizer:
    def __init__(self):
        self._detector_ready = False
        self._model_ready = False
        self._load()

    def _load(self):
        # Load MTCNN face detector.
        try:
            _get_detector()
            self._detector_ready = True
            print("[FaceRecognizer] MTCNN detector loaded.")
        except Exception as e:
            print(
                f"[FaceRecognizer] WARNING: "
                f"could not load MTCNN detector: {e}"
            )

        # Load and warm up FaceNet.
        if os.path.exists(_FACENET_PATH):
            try:
                model = _get_model()
                dummy = np.zeros(
                    (1, *_MODEL_INPUT_SIZE, 3),
                    dtype="float32",
                )

                with _inference_lock:
                    model.predict(dummy, verbose=0)

                self._model_ready = True
                print(
                    "[FaceRecognizer] FaceNet model loaded "
                    "and warmed up."
                )
            except Exception as e:
                print(
                    f"[FaceRecognizer] WARNING: "
                    f"could not load FaceNet model: {e}"
                )
        else:
            print(
                f"[FaceRecognizer] WARNING: model not found "
                f"at {_FACENET_PATH}"
            )

    @property
    def ready(self):
        return self._detector_ready and self._model_ready

    # Detection

    def detect_faces(self, image_rgb):
        """
        Return face locations as (top, right, bottom, left).
        Input image must be in RGB format.
        """
        if not self._detector_ready:
            return []

        try:
            detector = _get_detector()

            with _inference_lock:
                results = detector.detect_faces(image_rgb)

            locations = []

            for result in results:
                if result.get("confidence", 0) < 0.85:
                    continue

                x, y, w, h = result["box"]
                x, y = max(0, x), max(0, y)

                locations.append((y, x + w, y + h, x))

            return locations

        except Exception as e:
            print(f"[FaceRecognizer] Detection error: {e}")
            return []

    # Embeddings

    def _preprocess_face(self, image_rgb, face_location):
        """
        Crop and resize the face to 160x160, then standardize it.
        This preprocessing must match the model's training procedure.
        """
        import cv2

        top, right, bottom, left = face_location
        face = image_rgb[top:bottom, left:right]

        if face.size == 0:
            raise ValueError("Empty face crop")

        face = cv2.resize(face, _MODEL_INPUT_SIZE).astype("float32")

        mean = face.mean()
        std = face.std()
        std = max(std, 1.0 / np.sqrt(face.size))

        face = (face - mean) / std
        return np.expand_dims(face, axis=0)

    def _embed(self, model_input):
        """Run inference and return an L2-normalized embedding."""
        model = _get_model()

        with _inference_lock:
            embedding = model.predict(model_input, verbose=0)[0]

        norm = np.linalg.norm(embedding)

        if norm > 1e-10:
            return embedding / norm

        return embedding

    def get_embedding(self, image_rgb, face_location=None):
        """Return a face embedding as a list, or None if unavailable."""
        if not self.ready:
            return None

        try:
            locations = (
                [face_location]
                if face_location
                else self.detect_faces(image_rgb)
            )

            if not locations:
                return None

            return self._embed(
                self._preprocess_face(image_rgb, locations[0])
            ).tolist()

        except Exception as e:
            print(f"[FaceRecognizer] Embedding error: {e}")
            return None

    # Registration

    def process_registration_image(self, image_rgb):
        """
        Validate a registration image.
        Exactly one face must be detected.
        """
        if not self.ready:
            return {
                "success": False,
                "error": "Face recognition models not available.",
            }

        locations = self.detect_faces(image_rgb)

        if len(locations) == 0:
            return {
                "success": False,
                "error": (
                    "No face detected. Please use a clear frontal photo."
                ),
            }

        if len(locations) > 1:
            return {
                "success": False,
                "error": (
                    f"{len(locations)} faces detected. "
                    "Please ensure only one person is in the frame."
                ),
            }

        try:
            embedding = self._embed(
                self._preprocess_face(image_rgb, locations[0])
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Could not generate face embedding: {e}",
            }

        return {
            "success": True,
            "embedding": embedding.tolist(),
            "face_location": locations[0],
        }

    # Recognition

    @staticmethod
    def _cosine_distance(a, b):
        """Calculate cosine distance between normalized vectors."""
        return 1.0 - float(np.clip(np.dot(a, b), -1.0, 1.0))

    def identify(self, image_rgb, known_students):
        """
        Identify a face against registered students.

        Each student may contain:
        student_id, name, roll_no, face_embeddings.
        """
        if not self.ready:
            return {
                "recognized": False,
                "message": "Face recognition models not available.",
            }

        locations = self.detect_faces(image_rgb)

        if len(locations) == 0:
            return {
                "recognized": False,
                "message": (
                    "No face detected. Please move closer to the camera."
                ),
            }

        if len(locations) > 1:
            return {
                "recognized": False,
                "message": (
                    "Multiple faces detected. "
                    "Please ensure only one student is in frame."
                ),
            }

        try:
            probe = self._embed(
                self._preprocess_face(image_rgb, locations[0])
            )
        except Exception as e:
            return {
                "recognized": False,
                "message": f"Could not read face clearly: {e}",
            }

        best_distance = float("inf")
        best_student = None

        for student in known_students:
            for embedding in student.get("face_embeddings", []):
                try:
                    reference = np.asarray(embedding, dtype="float32")
                    norm = np.linalg.norm(reference)

                    if norm > 1e-10:
                        reference = reference / norm

                    if reference.shape != probe.shape:
                        continue

                    distance = self._cosine_distance(probe, reference)

                    if distance < best_distance:
                        best_distance = distance
                        best_student = student

                except Exception:
                    continue

        if (
            best_student is None
            or best_distance > FACE_MATCH_THRESHOLD
        ):
            return {
                "recognized": False,
                "message": (
                    "Unknown student. Face not registered in the system."
                ),
            }

        return {
            "recognized": True,
            "student_id": str(best_student["student_id"]),
            "student_name": best_student.get("name", ""),
            "roll_no": best_student.get("roll_no", ""),
            "distance": round(best_distance, 4),
            "face_location": locations[0],
        }

    # Crop face region

    def crop_face(self, image_bgr, face_location, padding=10):
        """Return a cropped face region in BGR format."""
        top, right, bottom, left = face_location
        height, width = image_bgr.shape[:2]

        return image_bgr[
            max(0, top - padding):min(height, bottom + padding),
            max(0, left - padding):min(width, right + padding),
        ]
