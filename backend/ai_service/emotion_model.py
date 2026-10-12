"""
EmotionModel — loads face_expression_model.keras using metadata from face_expression_model.pkl.
Never reloads on each request; instantiate once and reuse.
"""
import os
import pickle

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIRS = [
    os.path.join(_BASE, 'ai_model'),
]


class EmotionModel:
    def __init__(self):
        self.model    = None
        self.metadata = None
        self.img_size = (48, 48)
        self.classes  = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self._load()

    def _load(self):
        model_dir = next((path for path in _MODEL_DIRS if os.path.exists(
            os.path.join(path, 'face_expression_model.keras'))), _MODEL_DIRS[-1])
        keras_path = os.path.join(model_dir, 'face_expression_model.keras')
        pkl_path = os.path.join(model_dir, 'face_expression_model.pkl')

        # Always initialise idx_to_class so predict() never crashes
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        # ── 1. Load PKL metadata ──────────────────────────────────────────────
        if not os.path.exists(pkl_path):
            print(f"[EmotionModel] WARNING: PKL not found at {pkl_path}, using defaults")
        else:
            try:
                with open(pkl_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.img_size = self.metadata.get('image_size', (48, 48))
                classes_raw = self.metadata.get('classes',
                              self.metadata.get('index_to_class',
                              {i: c for i, c in enumerate(self.classes)}))
                # Normalise: support both list and dict formats
                if isinstance(classes_raw, dict):
                    self.idx_to_class = {int(k): v for k, v in classes_raw.items()}
                    self.classes = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]
                else:
                    self.classes = list(classes_raw)
                    self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
                print(f"[EmotionModel] PKL loaded — classes: {self.classes}, img_size: {self.img_size}")
            except Exception as e:
                print(f"[EmotionModel] PKL load error: {e}, using defaults")

        # ── 2. Load Keras model ───────────────────────────────────────────────
        if not os.path.exists(keras_path):
            print(f"[EmotionModel] WARNING: Keras model not found at {keras_path}")
            return
        try:
            import tensorflow as tf
            from model_compat import load_compatible_model
            self.model = load_compatible_model(keras_path)
            print(f"[EmotionModel] Keras model loaded successfully from {keras_path}")
        except Exception as e:
            print(f"[EmotionModel] Keras load error: {e}")

    @property
    def ready(self):
        return self.model is not None

    def predict(self, face_bgr_or_gray):
        """
        Accept a BGR or grayscale numpy array (cropped face region).
        Returns dict: {success, prediction, confidence} or {success, error}
        """
        if not self.ready:
            return {'success': False, 'error': 'Emotion model not loaded'}
        try:
            import cv2
            import numpy as np
            img = face_bgr_or_gray
            # Convert to grayscale if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.img_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)  # (1,H,W,1)
            preds = self.model.predict(img, verbose=0)
            probabilities = np.asarray(preds[0], dtype='float32')
            idx   = int(np.argmax(preds))
            conf  = float(np.max(probabilities))
            label = self.idx_to_class.get(idx, 'unknown')
            distribution = {
                self.idx_to_class.get(i, 'unknown'): round(float(probability), 4)
                for i, probability in enumerate(probabilities)
            }
            return {
                'success': True,
                'prediction': label,
                'confidence': round(conf, 4),
                'probabilities': distribution
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_from_path(self, image_path):
        """Convenience: load image from disk then predict."""
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return {'success': False, 'error': 'Cannot read image file'}
            return self.predict(img)
        except Exception as e:
            return {'success': False, 'error': str(e)}
