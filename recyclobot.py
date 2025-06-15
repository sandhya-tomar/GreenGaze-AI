import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Page Setup & Styling
st.set_page_config(page_title="♻️ RecycloBot by Sandhya", layout="centered", page_icon="🪴")

st.markdown("""
<style>
/* Title Section Glow + Background */
h1 {
    background: linear-gradient(90deg, #a8e6cf, #dcedc1);
    padding: 12px 24px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(100, 200, 150, 0.3);
    transition: all 0.3s ease;
    display: inline-block;
    color: #0d1b2a !important;
}
h1:hover {
    transform: scale(1.03);
    box-shadow: 0 0 25px rgba(76, 175, 80, 0.4);
}

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: transparent;
    color: #0d1b2a;
}

[data-testid="stAppViewContainer"] > .main {
    background: rgba(255,255,255,0.98);
    backdrop-filter: blur(10px);
}

.stApp {
    background: transparent;
}

#bgvid {
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%;
    min-height: 100%;
    object-fit: cover;
    z-index: -1;
    opacity: 0.25;
}

.stButton>button {
    background-color: #4caf50;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    padding: 0.6em 1.2em;
    transition: 0.3s ease;
    border: none;
}
.stButton>button:hover {
    background-color: #00796b;
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(38, 166, 154, 0.6);
}
.block-container {
    border-radius: 15px;
    padding: 2rem;
    background-color: #ffffff;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-top: 1rem;
    color: #0d1b2a;
}
.confidence-bar {
    height: 20px;
    border-radius: 8px;
    background: linear-gradient(to right, #66bb6a, #43a047);
    margin-top: 0.5rem;
}
/* Fix success/info/warning visibility */
.stAlert>div {
    color: #0d1b2a !important;
    font-weight: 500;
    background-color: rgba(230, 255, 230, 0.9);
}
.stAlert[data-baseweb="notification"] .stMarkdown {
    color: #0d1b2a !important;
    font-weight: 500;
}
</style>
<video autoplay muted loop id="bgvid">
    <source src="https://cdn.pixabay.com/video/2018/02/04/14117-254487693_tiny.mp4" type="video/mp4">
</video>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
    <h1 style='
        background: linear-gradient(90deg, #a7ffeb, #b2fef7);
        display: inline-block;
        padding: 14px 28px;
        border-radius: 16px;
        color: #004d40;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0, 150, 136, 0.25);
        transition: 0.3s ease;
    '>🌍 GreenGaze AI</h1>
</div>
""", unsafe_allow_html=True)

st.subheader("♻️ Is this waste recyclable? Upload and find out!")

st.markdown("## 🖼 Upload an Image to Begin")
st.markdown("---")

st.markdown("<label style='color: black; font-size: 18px; font-weight: 500;'>📁 Upload an image (jpg, png)</label>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Load model and labels
    try:
        model = load_model("model/keras_model.h5", compile=False)
        class_names = [line.strip() for line in open("model/labels.txt")]
    except:
        st.error("🛑 Model load error. Check if keras_model.h5 and labels.txt are present in 'model/' folder.")
        st.stop()

    # Preprocess image
    size = (224, 224)
    img = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized, axis=0)

    # Predict
    prediction = model.predict(data)[0]
    index = np.argmax(prediction)
    confidence = prediction[index] * 100
    label = class_names[index]

    # Display result
    st.markdown("### 🤖 AI Prediction Result")
    st.markdown(f"""
<div style='
    display: inline-block;
    padding: 8px 18px;
    background-color: {"#a5d6a7" if label.lower() == "recyclable" else "#ef9a9a"};
    color: {"#1b5e20" if label.lower() == "recyclable" else "#b71c1c"};
    border-radius: 30px;
    font-weight: 600;
    font-size: 16px;
    margin-top: 12px;
    text-align: center;
'>
✅ Prediction: {label}
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class=\"confidence-bar\" style=\"width:{int(confidence)}%;\"></div>
    """, unsafe_allow_html=True)
    st.info(f"💡 Confidence Score: **{confidence:.2f}%**")

    if label.lower() == "recyclable":
        st.success("♻️ Tip: Rinse items before recycling!")
    else:
        st.warning("🚫 Not recyclable. Look for sustainable alternatives.")

st.markdown("---")
st.markdown("""
<center style="font-size: 16px;">
Made with 💚 by <b>Sandhya</b> — <a href="https://github.com/sandhya-tomar" target="_blank" style="color: #2e7d32; font-weight: 600; text-decoration: none;">Visit GitHub ↗</a>
</center>
""", unsafe_allow_html=True)





