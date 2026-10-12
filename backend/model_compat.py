
import tensorflow as tf


def load_compatible_model(path, **kwargs):
    """Load a Keras model after removing nullable Dense quantization config."""

    class CompatibleDense(tf.keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config = dict(config)
            config.pop("quantization_config", None)
            return cls(**config)

    custom_objects = {
        "Dense": CompatibleDense,
        "keras.layers.Dense": CompatibleDense,
        "keras>layers>Dense": CompatibleDense,
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(path, **kwargs)
