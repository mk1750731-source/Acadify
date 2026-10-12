# model_compat.py — no longer needed, DeepFace handles model loading.
# Kept as stub to avoid ImportError if referenced elsewhere.

def load_compatible_model(path, **kwargs):
    import tensorflow as tf
    return tf.keras.models.load_model(path, **kwargs)
