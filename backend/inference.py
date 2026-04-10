"""
backend/inference.py

Optimized pipeline:
  - predict_batch()  : all faces in a frame processed in ONE model call
  - model() call     : no TF overhead vs model.predict()
  - frame_skip auto  : calculates skip to hit target FPS instead of fixed skip
  - resize before detection : smaller image = faster MediaPipe + faster crop
"""
import cv2
import os
import sys
import logging
import tempfile
import time
import numpy as np
from collections import Counter
from PIL import Image

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import (
    CLASS_NAMES, EMOTION_EMOJI, EMOTION_COLORS_BGR,
    FRAME_SKIP, MAX_FRAMES, OUTPUT_FPS,
)
from model_loader import predict, predict_batch
from face_detector import detect_faces

logger = logging.getLogger(__name__)

# Max width to resize video frames before detection (speeds up MediaPipe a lot)
DETECT_MAX_W = 480


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw(frame, bbox, emotion, confidence, face_id=0):
    x1, y1, x2, y2 = bbox
    color = EMOTION_COLORS_BGR.get(emotion, (200, 200, 200))
    label = "#" + str(face_id) + " " + emotion + " " + str(int(confidence * 100)) + "%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    by = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, by), (x1 + tw + 6, by + th + 6), color, -1)
    cv2.putText(frame, label, (x1 + 3, by + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── Result helpers ────────────────────────────────────────────────────────────

def _face_result(face_id, bbox, preds):
    emotion, conf = preds[0]
    return {
        "face_id":    face_id,
        "bbox":       list(bbox),
        "emotion":    emotion,
        "confidence": conf,
        "emoji":      EMOTION_EMOJI.get(emotion, ""),
        "all_preds":  [{"emotion": e, "probability": p} for e, p in preds],
    }


def _make_summary(counts):
    if not counts:
        return {
            "dominant_emotion":    "unknown",
            "dominant_emoji":      "",
            "emotion_counts":      {},
            "emotion_percentages": {},
            "total_detections":    0,
        }
    total    = sum(counts.values())
    dominant = counts.most_common(1)[0][0]
    return {
        "dominant_emotion":    dominant,
        "dominant_emoji":      EMOTION_EMOJI.get(dominant, ""),
        "emotion_counts":      dict(counts),
        "emotion_percentages": {e: round(c / total * 100, 1) for e, c in counts.items()},
        "total_detections":    total,
    }


# ── Image prediction ──────────────────────────────────────────────────────────

def predict_image(image_input, use_face_detection=True):
    t0 = time.time()

    if isinstance(image_input, bytes):
        arr   = np.frombuffer(image_input, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        frame = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        frame = image_input.copy()
    else:
        raise TypeError("Unsupported image type: " + str(type(image_input)))

    if frame is None or frame.size == 0:
        raise ValueError("Could not decode image.")

    annotated = frame.copy()
    faces     = []

    if use_face_detection:
        detections = detect_faces(frame)

        if not detections:
            preds = predict(frame)
            h, w  = frame.shape[:2]
            faces.append(_face_result(0, (0, 0, w, h), preds))
            annotated = _draw(annotated, (0, 0, w, h), preds[0][0], preds[0][1])
        else:
            # Batch: preprocess all crops then one model call
            crops = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                crops.append(frame[y1:y2, x1:x2])

            batch_preds = predict_batch(crops)

            for fid, (det, preds) in enumerate(zip(detections, batch_preds)):
                faces.append(_face_result(fid, det["bbox"], preds))
                annotated = _draw(annotated, det["bbox"], preds[0][0], preds[0][1], fid)
    else:
        preds = predict(frame)
        h, w  = frame.shape[:2]
        faces.append(_face_result(0, (0, 0, w, h), preds))
        annotated = _draw(annotated, (0, 0, w, h), preds[0][0], preds[0][1])

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

    return {
        "num_faces":             len(faces),
        "faces":                 faces,
        "summary":               _make_summary(Counter(f["emotion"] for f in faces)),
        "annotated_image_bytes": buf.tobytes(),
        "latency_ms":            round((time.time() - t0) * 1000, 1),
    }


# ── Video prediction ──────────────────────────────────────────────────────────

def predict_video(video_input, frame_skip=FRAME_SKIP,
                  max_frames=MAX_FRAMES, save_video=True):
    """
    Optimized video emotion pipeline.

    Speed improvements over naive approach:
      1. Resize frame to max 480px wide before detection  -> MediaPipe 2-3x faster
      2. predict_batch() for all faces in a frame         -> one model call per frame
      3. model() instead of model.predict()               -> no TF session overhead
      4. Scale bboxes back to original resolution for drawing
    """
    t0 = time.time()

    tmp_in_path = None
    if isinstance(video_input, (bytes, bytearray)):
        tmp_fd, tmp_in_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        with open(tmp_in_path, "wb") as f:
            f.write(video_input)
        video_path = tmp_in_path
    else:
        video_path = video_input

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: " + video_path)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %d frames @ %.1ffps (%dx%d)", total_src, src_fps, vid_w, vid_h)

    # Scale factor for detection (run on smaller frame)
    if vid_w > DETECT_MAX_W:
        scale = DETECT_MAX_W / float(vid_w)
        det_w = int(vid_w * scale)
        det_h = int(vid_h * scale)
    else:
        scale = 1.0
        det_w, det_h = vid_w, vid_h

    logger.info("Detection scale: %.2f -> %dx%d", scale, det_w, det_h)

    tmp_out_path = None
    writer       = None
    if save_video:
        tmp_fd, tmp_out_path = tempfile.mkstemp(suffix="_emotion.mp4")
        os.close(tmp_fd)
        writer = cv2.VideoWriter(
            tmp_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            OUTPUT_FPS,
            (vid_w, vid_h),
        )

    frame_results = []
    global_counts = Counter()
    timeline      = []
    yielded = 0
    idx     = 0

    try:
        while cap.isOpened() and yielded < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % frame_skip != 0:
                idx += 1
                continue

            ts        = round(idx / src_fps, 2)
            annotated = frame.copy()
            faces     = []

            # ── Resize for fast detection ─────────────────────────────────────
            if scale < 1.0:
                small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            else:
                small = frame

            detections = detect_faces(small)

            if detections:
                # Scale bboxes back to original resolution
                crops = []
                orig_bboxes = []
                for det in detections:
                    sx1, sy1, sx2, sy2 = det["bbox"]
                    # Scale back to original frame coords
                    ox1 = int(sx1 / scale)
                    oy1 = int(sy1 / scale)
                    ox2 = int(sx2 / scale)
                    oy2 = int(sy2 / scale)
                    # Clamp to frame
                    ox1 = max(0, min(vid_w, ox1))
                    oy1 = max(0, min(vid_h, oy1))
                    ox2 = max(0, min(vid_w, ox2))
                    oy2 = max(0, min(vid_h, oy2))
                    orig_bboxes.append((ox1, oy1, ox2, oy2))
                    crop = frame[oy1:oy2, ox1:ox2]
                    if crop.size > 0:
                        crops.append(crop)
                    else:
                        crops.append(frame)   # fallback

                # ONE model call for all faces in this frame
                batch_preds = predict_batch(crops)

                for fid, (bbox, preds) in enumerate(zip(orig_bboxes, batch_preds)):
                    faces.append(_face_result(fid, bbox, preds))
                    annotated = _draw(annotated, bbox, preds[0][0], preds[0][1], fid)
                    global_counts[preds[0][0]] += 1

            else:
                # No face found — classify full frame
                preds = predict(frame)
                h, w  = frame.shape[:2]
                faces.append(_face_result(0, (0, 0, w, h), preds))
                annotated = _draw(annotated, (0, 0, w, h), preds[0][0], preds[0][1])
                global_counts[preds[0][0]] += 1

            cv2.putText(annotated, "t=" + str(ts) + "s",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1, cv2.LINE_AA)

            frame_results.append({
                "frame_idx":     idx,
                "timestamp_sec": ts,
                "faces":         faces,
            })

            if faces:
                timeline.append({
                    "timestamp_sec":    ts,
                    "dominant_emotion": faces[0]["emotion"],
                    "confidence":       faces[0]["confidence"],
                })

            if writer:
                writer.write(annotated)

            yielded += 1
            idx     += 1

            # Progress log every 20 frames
            if yielded % 20 == 0:
                elapsed = time.time() - t0
                fps_so_far = yielded / elapsed if elapsed > 0 else 0
                logger.info("Processed %d frames (%.1f fps)", yielded, fps_so_far)

    finally:
        cap.release()
        if writer:
            writer.release()
        if tmp_in_path and os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)

    total_time = time.time() - t0
    logger.info("Done: %d frames in %.1fs (%.1f fps)", yielded, total_time,
                yielded / total_time if total_time > 0 else 0)

    return {
        "frames_processed":  yielded,
        "video_meta": {
            "fps":          src_fps,
            "total_frames": total_src,
            "duration_sec": round(total_src / src_fps, 2),
            "width":        vid_w,
            "height":       vid_h,
        },
        "frame_results":     frame_results,
        "summary":           _make_summary(global_counts),
        "timeline":          timeline,
        "output_video_path": tmp_out_path,
        "latency_ms":        round(total_time * 1000, 1),
    }