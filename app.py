# ------------------ SYSTEM FIXES ------------------
import os
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------ IMPORTS ------------------
import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2

# ------------------ PAGE SETUP ------------------
st.set_page_config(
    page_title="A Real-Time Fruit Detection System Using YOLO",
    layout="wide"
)

# ------------------ CUSTOM STYLING ------------------
st.markdown("""
<style>
.main { padding: 1rem !important; }

.stApp {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe);
    font-family: 'Poppins', sans-serif;
}

h1 {
    text-align: center;
    font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section {
    background: #ffffff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.count-card {
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    color: white;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0px 10px 25px rgba(124,58,237,0.4);
}

.count-card span {
    font-size: 36px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.title("A Real-Time Fruit Detection System Using YOLO")
st.markdown(
    "<h4 style='text-align:center;'>Detect fruits in images and videos using YOLOv9</h4>",
    unsafe_allow_html=True
)

# ------------------ SIDEBAR ------------------
# ------------------ SIDEBAR (ENHANCED CONFIGURATION) ------------------
st.sidebar.markdown("## ⚙ System Configuration")
st.sidebar.markdown("---")

model_path = st.sidebar.text_input(
    "📂 YOLO Model Path",
    "C:/Users/Pavan Kumar/Downloads/appp/best.pt",
    help="Path to the trained YOLO model (.pt file)"
)

conf = st.sidebar.slider(
    "🎯 Detection Confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

st.sidebar.markdown(
    f"<p style='color:#7c3aed;font-weight:600;'>Current Confidence: {conf:.2f}</p>",
    unsafe_allow_html=True
)

mode = st.sidebar.radio(
    "🧠 Detection Mode",
    ["Image", "Video"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ Adjust confidence to control detection accuracy.\n\n"
    "Lower value → more detections\n\n"
    "Higher value → more precise detections"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ------------------ IMAGE MODE ------------------
if mode == "Image":
    uploaded_image = st.file_uploader(
        "📤 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_image.read())
            img_path = tmp.name

        # Original Image
        st.markdown("<div class='section'><h3>📷 Original Image</h3></div>", unsafe_allow_html=True)
        original_img = Image.open(img_path)
        st.image(original_img, use_container_width=True)

        st.info("🔍 Detecting fruits...")

        # YOLO Prediction
        results = model.predict(source=img_path, conf=conf, save=False)

        # Convert BGR → RGB
        result_bgr = results[0].plot()
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # Detected Image
        st.markdown("<div class='section'><h3>🎯 Detected Image</h3></div>", unsafe_allow_html=True)
        st.image(result_rgb, use_container_width=True)

        # Fruit Counts
        labels = [model.names[int(box.cls)] for box in results[0].boxes]

        if labels:
            counts = {label: labels.count(label) for label in set(labels)}

            st.markdown(
                "<div class='section'><h3>🍎 Fruit Detection Summary</h3></div>",
                unsafe_allow_html=True
            )

            cols = st.columns(len(counts))
            for col, (fruit, count) in zip(cols, counts.items()):
                col.markdown(
                    f"""
                    <div class="count-card">
                        {fruit}
                        <span>{count}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("❌ No fruits detected")

        os.unlink(img_path)

# ------------------ VIDEO MODE ------------------
elif mode == "Video":
    uploaded_video = st.file_uploader(
        "📤 Upload a Video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video:
        ext = uploaded_video.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name

        st.markdown("<div class='section'><h3>🎥 Original Video</h3></div>", unsafe_allow_html=True)
        st.video(video_path)

        st.info("⏳ Processing video...")

        results = model.predict(
            source=video_path,
            conf=conf,
            save=True,
            project="runs/detect",
            name="fruit_ui",
            vid_stride=1,
            stream=False
        )

        pred_dir = results[0].save_dir
        pred_videos = [
            f for f in os.listdir(pred_dir)
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]

        if pred_videos:
            output_path = os.path.join(pred_dir, pred_videos[0])

            st.markdown("<div class='section'><h3>✅ Processed Video</h3></div>", unsafe_allow_html=True)
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Processed Video",
                    data=f,
                    file_name="detected_fruit_video.mp4",
                    mime="video/mp4"
                )
        else:
            st.warning("⚠ Processed video not found")

        os.unlink(video_path)
