import base64
import logging
import os
import sys
import tempfile

from flask import Flask, jsonify, request, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FLASK_HOST, FLASK_PORT, DEBUG,
    CLASS_NAMES, MODEL_PATH,
    FRAME_SKIP, MAX_FRAMES,
    MAX_CONTENT_LENGTH,
)
from model_loader import warmup
from inference import predict_image, predict_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Temp video store: filename -> full path
_video_store = {}

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska"}


# ── Startup warmup ────────────────────────────────────────────────────────────

def _startup():
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model not found at %s\n"
            "  Save from Colab with: finetuned_model.save('efficientnet_full_model.keras')\n"
            "  Then copy it to your local models/ folder.",
            MODEL_PATH,
        )
    else:
        try:
            warmup()
            logger.info("EfficientNet-B4 warmed up and ready.")
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", str(e))


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "model":       "EfficientNet-B4 (fine-tuned)",
        "classes":     CLASS_NAMES,
        "model_ready": os.path.exists(MODEL_PATH),
    })


# ── Image prediction ──────────────────────────────────────────────────────────

@app.route("/predict/image", methods=["POST"])
def api_predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send as multipart/form-data with key 'file'."}), 400

    f = request.files["file"]
    if f.content_type not in ALLOWED_IMAGE_TYPES:
        return jsonify({"error": "Unsupported file type: " + str(f.content_type)}), 400

    image_bytes = f.read()
    if not image_bytes:
        return jsonify({"error": "Empty file."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not loaded. Copy efficientnet_full_model.keras to models/"}), 503

    use_detection = request.form.get("use_face_detection", "true").lower() == "true"

    try:
        result = predict_image(image_bytes, use_face_detection=use_detection)
    except Exception as e:
        logger.exception("Image prediction failed")
        return jsonify({"error": str(e)}), 500

    img_b64 = base64.b64encode(result.pop("annotated_image_bytes")).decode("utf-8")
    result["annotated_image_b64"] = img_b64
    return jsonify(result)


# ── Video prediction ──────────────────────────────────────────────────────────

@app.route("/predict/video", methods=["POST"])
def api_predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    f = request.files["file"]
    video_bytes = f.read()
    if not video_bytes:
        return jsonify({"error": "Empty file."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not loaded."}), 503

    try:
        frame_skip = int(request.form.get("frame_skip", FRAME_SKIP))
        max_frames = int(request.form.get("max_frames", MAX_FRAMES))
        save_video = request.form.get("save_video", "true").lower() == "true"
    except ValueError:
        return jsonify({"error": "frame_skip and max_frames must be integers."}), 400

    try:
        result = predict_video(
            video_bytes,
            frame_skip=frame_skip,
            max_frames=max_frames,
            save_video=save_video,
        )
    except Exception as e:
        logger.exception("Video prediction failed")
        return jsonify({"error": str(e)}), 500

    video_path = result.pop("output_video_path", None)
    if video_path and os.path.exists(video_path):
        fname = os.path.basename(video_path)
        _video_store[fname] = video_path
        result["annotated_video_url"] = "/download/" + fname

    return jsonify(result)


# ── Video download (✅ FIXED FOR STREAMING) ───────────────────────────────────

@app.route("/download/<filename>", methods=["GET"])
def download_video(filename):
    path = _video_store.get(filename)
    if not path or not os.path.exists(path):
        return jsonify({"error": "Video not found or expired."}), 404

    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range", None)

    if range_header:
        # Parse Range header: bytes=start-end
        byte1, byte2 = 0, None
        m = range_header.replace("bytes=", "").split("-")
        byte1 = int(m[0])
        if len(m) > 1 and m[1]:
            byte2 = int(m[1])

        if byte2 is None or byte2 >= file_size:
            byte2 = file_size - 1

        length = byte2 - byte1 + 1

        with open(path, "rb") as f:
            f.seek(byte1)
            data = f.read(length)

        resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
        resp.headers.add("Content-Range", f"bytes {byte1}-{byte2}/{file_size}")
        resp.headers.add("Accept-Ranges", "bytes")
        resp.headers.add("Content-Length", str(length))
        return resp

    # No Range header → send full file
    return send_file(path, mimetype="video/mp4", as_attachment=False)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _startup()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
