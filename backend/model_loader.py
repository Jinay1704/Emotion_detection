"""
backend/model_loader.py

Loads the fine-tuned EfficientNet-B4 Keras model.
Key optimizations:
  - model() instead of model.predict()  -> no TF overhead per call
  - predict_batch()                     -> one forward pass for all faces in a frame
  - preprocess_input cached import
"""
import os
import sys
import logging
import numpy as np
import cv2

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import MODEL_PATH, CLASS_NAMES, IM_SIZE

logger = logging.getLogger(__name__)

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "\n Model file not found: " + MODEL_PATH + "\n"
            "   In Colab: finetuned_model.save('efficientnet_full_model.keras')\n"
            "   Then copy to your local models/ folder."
        )

    import tensorflow as tf
    logger.info("Loading EfficientNet-B4 from " + MODEL_PATH)
    _model = tf.keras.models.load_model(MODEL_PATH)
    _model.trainable = False
    logger.info("Model ready. Input shape: " + str(_model.input_shape))
    return _model


def _preprocess_crop(face_bgr):
    """Preprocess one BGR crop -> float32 array (256, 256, 3)."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)
    return preprocess_input(resized.astype("float32"))


def predict(face_bgr, top_k=3):
    """Predict emotion from a single BGR face crop."""
    model   = load_model()
    inp     = np.expand_dims(_preprocess_crop(face_bgr), axis=0)
    probs   = model(inp, training=False).numpy()[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i], round(float(probs[i]), 4)) for i in top_idx]


def predict_batch(crops_bgr, top_k=3):
    """
    Predict emotions for multiple face crops in ONE forward pass.
    10-30x faster than calling predict() in a loop for video processing.

    Args:
        crops_bgr : list of BGR numpy arrays
        top_k     : top predictions to return per crop

    Returns:
        List of [(emotion, prob), ...] per crop
    """
    if not crops_bgr:
        return []

    model     = load_model()
    batch     = np.stack([_preprocess_crop(c) for c in crops_bgr], axis=0)
    all_probs = model(batch, training=False).numpy()

    results = []
    for probs in all_probs:
        top_idx = np.argsort(probs)[::-1][:top_k]
        results.append([(CLASS_NAMES[i], round(float(probs[i]), 4)) for i in top_idx])
    return results


def warmup():
    """Compile model graph at startup so first real request is fast."""
    dummy = np.zeros((60, 60, 3), dtype=np.uint8)
    predict_batch([dummy, dummy], top_k=1)
    logger.info("Model warmup complete.")