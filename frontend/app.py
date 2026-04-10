"""
frontend/app.py  -  Streamlit UI
Run: streamlit run app.py
Compact layout: upload + preview + detect button + result all in one view.
"""
import base64
import io
import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:5000")

CLASS_NAMES = ["angry", "happy", "sad"]
EMOJI  = {"angry": "😠", "happy": "😊", "sad": "😢"}
COLORS = {"angry": "#e74c3c", "happy": "#f1c40f", "sad": "#2980b9"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def backend_ok():
    try:
        return requests.get(API_URL + "/health", timeout=3).status_code == 200
    except Exception:
        return False


def b64_to_pil(s):
    return Image.open(io.BytesIO(base64.b64decode(s)))


def resize_for_display(pil_img, max_width=340):
    """Resize PIL image to max_width keeping aspect ratio."""
    w, h = pil_img.size
    if w <= max_width:
        return pil_img
    ratio = max_width / float(w)
    return pil_img.resize((max_width, int(h * ratio)), Image.LANCZOS)


def confidence_bars(all_preds):
    emotions = [d["emotion"] for d in all_preds]
    probs    = [d["probability"] * 100 for d in all_preds]
    fig = go.Figure(go.Bar(
        x=probs,
        y=[EMOJI.get(e, "") + " " + e for e in emotions],
        orientation="h",
        marker_color=[COLORS.get(e, "#aaa") for e in emotions],
        text=[str(round(p, 1)) + "%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        height=130,
        margin=dict(l=5, r=45, t=5, b=5),
        xaxis=dict(range=[0, 115], showticklabels=False, showgrid=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def donut_chart(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    fig = go.Figure(go.Pie(
        labels=[EMOJI.get(l, "") + " " + l for l in labels],
        values=values,
        hole=0.55,
        marker=dict(colors=[COLORS.get(l, "#aaa") for l in labels]),
        textinfo="percent+label",
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=5, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def timeline_chart(timeline):
    if not timeline:
        return go.Figure()
    df = pd.DataFrame(timeline)
    fig = px.scatter(
        df,
        x="timestamp_sec",
        y="dominant_emotion",
        color="dominant_emotion",
        color_discrete_map=COLORS,
        size="confidence",
        size_max=14,
        labels={"timestamp_sec": "Time (s)", "dominant_emotion": "Emotion"},
        height=240,
    )
    fig.update_layout(
        margin=dict(l=5, r=5, t=10, b=30),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(categoryorder="array", categoryarray=CLASS_NAMES),
    )
    return fig


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed by default = more space
)

st.markdown("""
<style>
/* Remove default top padding so title starts at very top */
.block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }

/* Tighter tab padding */
.stTabs [data-baseweb="tab-panel"] { padding-top: 0.5rem; }

/* Compact file uploader */
[data-testid="stFileUploader"] { padding: 0; }
[data-testid="stFileUploader"] section { padding: 0.4rem 0.6rem; min-height: 0; }
[data-testid="stFileUploader"] section > div { padding: 0.2rem 0; }

/* Face result card */
.face-card {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
    margin-bottom: 6px;
}
.face-card .big-emoji { font-size: 2.2rem; line-height: 1.1; }
.face-card .label     { font-size: 1.1rem; font-weight: 700; margin-top: 4px; }
.face-card .sub       { font-size: 0.8rem; color: #999; margin-top: 2px; }

/* Tighter metric */
[data-testid="stMetric"] { padding: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar (collapsed by default, still accessible) ─────────────────────────

with st.sidebar:
    st.markdown("### Settings")

    if backend_ok():
        st.success("Backend connected")
    else:
        st.error("Backend offline\n\nRun:\n```\ncd backend\npython app.py\n```")

    st.divider()
    st.info("Model: EfficientNet-B4")

    use_detection = st.toggle("Face Detection (MediaPipe)", value=True)

    st.divider()
    with st.expander("Video Settings"):
        frame_skip = st.slider("Every Nth frame", 1, 10, 10)
        max_frames = st.slider("Max frames",      50, 500, 200)
        save_video = st.checkbox("Save annotated video", value=True)

    st.divider()
    for cls in CLASS_NAMES:
        st.markdown(EMOJI[cls] + "  " + cls.capitalize())


# ── Header (compact) ─────────────────────────────────────────────────────────

st.markdown("### 😊 Face Emotion Detection")
st.caption("MediaPipe → EfficientNet-B4 → 😠 Angry / 😊 Happy / 😢 Sad")

tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])


# ════════════════════════════════════════════════════
# IMAGE TAB  — everything in one row, no scrolling
# Layout: [upload + button | original img | result img | face cards]
# ════════════════════════════════════════════════════

with tab_img:

    # Row 1: uploader left, images right — all in one view
    up_col, orig_col, ann_col, info_col = st.columns([1.2, 1, 1, 1], gap="small")

    with up_col:
        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_upload",
            label_visibility="collapsed",
        )
        st.caption("JPG / PNG / WEBP")

        detect_clicked = st.button(
            "🔍 Detect Emotions",
            type="primary",
            use_container_width=True,
            disabled=(uploaded is None),
        )

        # Show small thumbnail in upload column
        if uploaded:
            pil_preview = Image.open(uploaded)
            st.image(
                resize_for_display(pil_preview, max_width=220),
                caption="Uploaded",
                use_container_width=False,
            )

    # Placeholders for result columns
    with orig_col:
        orig_placeholder = st.empty()

    with ann_col:
        ann_placeholder  = st.empty()

    with info_col:
        info_placeholder = st.empty()

    # ── Run prediction ────────────────────────────────────────────────────────
    if uploaded and detect_clicked:
        with st.spinner("Running..."):
            try:
                uploaded.seek(0)
                resp = requests.post(
                    API_URL + "/predict/image",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                    data={"use_face_detection": str(use_detection).lower()},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.ConnectionError:
                st.error("Cannot reach backend. Run: python backend/app.py")
                st.stop()
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        ann_pil  = b64_to_pil(result["annotated_image_b64"])
        summary  = result.get("summary", {})
        dominant = summary.get("dominant_emotion", "?")
        faces    = result.get("faces", [])

        # Original image (fixed height)
        with orig_placeholder.container():
            st.markdown("**Original**")
            uploaded.seek(0)
            st.image(
                resize_for_display(Image.open(uploaded), max_width=300),
                use_container_width=False,
            )

        # Annotated image
        with ann_placeholder.container():
            st.markdown("**Detected**")
            st.image(
                resize_for_display(ann_pil, max_width=300),
                use_container_width=False,
            )

        # Info: metrics + face cards
        with info_placeholder.container():
            st.markdown("**Results**")
            st.metric("Faces",   result["num_faces"])
            st.metric("Emotion", EMOJI.get(dominant, "") + " " + dominant)
            st.metric("Time",    str(result["latency_ms"]) + " ms")

            if faces:
                st.divider()
                for i, face in enumerate(faces):
                    e = face["emotion"]
                    c = face["confidence"]
                    st.markdown(
                        "<div class='face-card'>"
                        "<div class='big-emoji'>" + EMOJI.get(e, "") + "</div>"
                        "<div class='label'>" + e.capitalize() + "</div>"
                        "<div class='sub'>" + str(round(c * 100, 1)) + "% confident</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        confidence_bars(face["all_preds"]),
                        use_container_width=True,
                        key="bar_" + str(i),
                    )

        # Download (below the row)
        st.divider()
        buf = io.BytesIO()
        ann_pil.save(buf, format="JPEG")
        st.download_button(
            "⬇️ Download annotated image",
            buf.getvalue(),
            "emotion_result.jpg",
            "image/jpeg",
        )


# ════════════════════════════════════════════════════
# VIDEO TAB
# ════════════════════════════════════════════════════

with tab_vid:

    v_up_col, v_prev_col = st.columns([1, 2], gap="small")

    with v_up_col:
        uploaded_v = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
            key="vid_upload",
            label_visibility="collapsed",
        )
        st.caption("MP4 / AVI / MOV")

        analyse_clicked = st.button(
            "🔍 Analyse Video",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_v is None),
        )

    with v_prev_col:
        if uploaded_v:
            st.video(uploaded_v)

    if uploaded_v and analyse_clicked:
        with st.spinner(
            "Processing every " + str(frame_skip) +
            " frame(s), up to " + str(max_frames) + " frames..."
        ):
            try:
                uploaded_v.seek(0)
                resp = requests.post(
                    API_URL + "/predict/video",
                    files={"file": (uploaded_v.name, uploaded_v.read(), uploaded_v.type)},
                    data={
                        "frame_skip": frame_skip,
                        "max_frames": max_frames,
                        "save_video": str(save_video).lower(),
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.ConnectionError:
                st.error("Cannot reach backend.")
                st.stop()
            except Exception as e:
                st.error("Failed: " + str(e))
                st.stop()

        if "error" in result:
            st.error(result["error"])
            st.stop()

        summary  = result.get("summary", {})
        dominant = summary.get("dominant_emotion", "?")
        counts   = summary.get("emotion_counts", {})
        meta     = result.get("video_meta", {})

        # Compact metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames",   result["frames_processed"])
        c2.metric("Duration", str(meta.get("duration_sec", "?")) + "s")
        c3.metric("Emotion",  EMOJI.get(dominant, "") + " " + dominant)
        c4.metric("Time",     str(result["latency_ms"]) + " ms")

        st.divider()

        # Charts side by side
        ch1, ch2 = st.columns([2, 1], gap="small")
        with ch1:
            st.markdown("**Emotion over time**")
            st.plotly_chart(
                timeline_chart(result.get("timeline", [])),
                use_container_width=True,
                key="timeline",
            )
        with ch2:
            st.markdown("**Overall split**")
            if counts:
                st.plotly_chart(
                    donut_chart(counts),
                    use_container_width=True,
                    key="donut",
                )

        st.divider()

        # Breakdown table
        pct = summary.get("emotion_percentages", {})
        if pct:
            rows = []
            for e, v in sorted(pct.items(), key=lambda x: -x[1]):
                rows.append({
                    "Emotion":    EMOJI.get(e, "") + " " + e.capitalize(),
                    "Count":      counts.get(e, 0),
                    "Percentage": str(v) + "%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        import tempfile

        video_url = result.get("annotated_video_url")
        if video_url and save_video:
            try:
                vr = requests.get(API_URL + video_url, timeout=60)
                vr.raise_for_status()

                video_bytes = vr.content
                st.write("Video size (bytes):", len(video_bytes))  # debug info

                # Save to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_bytes)
                    tmp_path = tmp.name

                col_dl, col_prev = st.columns([1, 2])

                with col_dl:
                    st.download_button(
                        "⬇️ Download annotated video",
                        video_bytes,
                        "emotion_video.mp4",
                        "video/mp4",
                    )

                with col_prev:
                    st.markdown("**Preview (annotated)**")
                    st.video(video_bytes) 

            except Exception as e:
                st.warning("Could not fetch annotated video: " + str(e))

        # Per-frame detail
        with st.expander("Per-frame detail"):
            rows = []
            for fr in result.get("frame_results", []):
                for face in fr["faces"]:
                    rows.append({
                        "Frame":      fr["frame_idx"],
                        "Time (s)":   fr["timestamp_sec"],
                        "Face #":     face["face_id"],
                        "Emotion":    face["emoji"] + " " + face["emotion"],
                        "Confidence": str(round(face["confidence"] * 100, 1)) + "%",
                    })
            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No face detections recorded.")