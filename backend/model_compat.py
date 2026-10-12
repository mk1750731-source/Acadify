
import tensorflow as tf


class CompatibleDense(tf.keras.layers.Dense):
    """Handle a nullable quantization_config in serialized Dense layers."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        if config.get("quantization_config") is None:
            config.pop("quantization_config", None)

        return cls(**config)


def load_compatible_model(path, **kwargs):
    """Load a Keras model with a compatibility override for Dense."""
    custom_objects = {
        "Dense": CompatibleDense,
        "keras.layers.Dense": CompatibleDense,
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        return tf.keras.models.load_model(path, **kwargs)
