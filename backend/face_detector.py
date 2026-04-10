"""
backend/face_detector.py
MediaPipe face detector — updated for MediaPipe 0.10.x new API.
mp.solutions was removed in 0.10.x — now uses mediapipe.tasks.
"""
import cv2
import numpy as np
import logging
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import MP_CONFIDENCE, FACE_PAD_RATIO, MIN_FACE_PX

logger = logging.getLogger(__name__)

# Lazy-loaded detector instance
_detector = None


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    import mediapipe as mp

    # ── MediaPipe 0.10.x new Tasks API ────────────────────────────────────────
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # Download model file if not present
        model_path = os.path.join(_BACKEND_DIR, "blaze_face_short_range.tflite")
        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe face detection model...")
            import urllib.request
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/1/"
                "blaze_face_short_range.tflite"
            )
            urllib.request.urlretrieve(url, model_path)
            logger.info("Model downloaded to " + model_path)

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options   = mp_vision.FaceDetectorOptions(
            base_options=base_opts,
            min_detection_confidence=MP_CONFIDENCE,
        )
        _detector = mp_vision.FaceDetector.create_from_options(options)
        logger.info("MediaPipe FaceDetector (Tasks API) ready.")
        return _detector

    except Exception as e:
        logger.warning("MediaPipe Tasks API failed (%s), trying legacy API...", str(e))

    # ── Fallback: legacy solutions API (MediaPipe < 0.10) ─────────────────────
    try:
        _detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=MP_CONFIDENCE,
        )
        logger.info("MediaPipe FaceDetection (legacy solutions API) ready.")
        return _detector
    except Exception as e2:
        raise RuntimeError(
            "Could not initialise MediaPipe face detector.\n"
            "Tasks API error: " + str(e) + "\n"
            "Legacy API error: " + str(e2)
        )


def _pad_and_clamp(x1, y1, x2, y2, h, w):
    pw = int((x2 - x1) * FACE_PAD_RATIO)
    ph = int((y2 - y1) * FACE_PAD_RATIO)
    return (
        max(0, x1 - pw),
        max(0, y1 - ph),
        min(w,  x2 + pw),
        min(h,  y2 + ph),
    )


def _parse_new_api(detection, h, w):
    """Parse detection result from MediaPipe Tasks API (0.10.x)."""
    bb   = detection.bounding_box
    x1   = max(0, bb.origin_x)
    y1   = max(0, bb.origin_y)
    x2   = min(w, bb.origin_x + bb.width)
    y2   = min(h, bb.origin_y + bb.height)
    conf = round(detection.categories[0].score, 3) if detection.categories else 0.0
    return x1, y1, x2, y2, conf


def _parse_legacy_api(detection, h, w):
    """Parse detection result from MediaPipe solutions API (< 0.10)."""
    bb   = detection.location_data.relative_bounding_box
    x1   = max(0, int(bb.xmin * w))
    y1   = max(0, int(bb.ymin * h))
    x2   = min(w, int((bb.xmin + bb.width)  * w))
    y2   = min(h, int((bb.ymin + bb.height) * h))
    conf = round(detection.score[0], 3)
    return x1, y1, x2, y2, conf


def detect_faces(frame_bgr):
    """
    Detect all faces in a BGR frame.
    Works with both MediaPipe 0.10.x (Tasks API) and older versions.

    Returns list of dicts sorted by area (largest first):
    {
        "bbox":       (x1, y1, x2, y2),   padded + clamped
        "bbox_tight": (x1, y1, x2, y2),   raw output
        "confidence": float
    }
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    h, w = frame_bgr.shape[:2]
    detector = _get_detector()

    # Detect which API we are using by checking detector type name
    detector_type = type(detector).__name__

    if detector_type == "FaceDetector":
        # ── New Tasks API ────────────────────────────────────────────────────
        import mediapipe as mp
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )
        result      = detector.detect(mp_image)
        raw_dets    = result.detections if result.detections else []
        parse_fn    = _parse_new_api
    else:
        # ── Legacy solutions API ─────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result   = detector.process(rgb)
        raw_dets = result.detections if result.detections else []
        parse_fn = _parse_legacy_api

    faces = []
    for det in raw_dets:
        x1, y1, x2, y2, conf = parse_fn(det, h, w)

        if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
            continue

        px1, py1, px2, py2 = _pad_and_clamp(x1, y1, x2, y2, h, w)

        faces.append({
            "bbox":       (px1, py1, px2, py2),
            "bbox_tight": (x1,  y1,  x2,  y2),
            "confidence": conf,
        })

    faces.sort(
        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        reverse=True,
    )
    return faces