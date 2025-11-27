import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
import sounddevice as sd
import tempfile
from tensorflow.keras.models import load_model
import pandas as pd

# ---------------------------
# CONSTANTS – must match training
# ---------------------------
SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128

# ---------------------------
# LOAD MODEL + LABEL ENCODER
# ---------------------------
@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_cnn_best.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_emotion_model()
emotion_classes = list(le.classes_)

# ---------------------------
# AUDIO PROCESSING
# ---------------------------
def load_audio_fixed(path, sr=SAMPLE_RATE, duration=DURATION):
    audio, sr = librosa.load(path, sr=sr)
    expected_len = int(sr * duration)
    if len(audio) > expected_len:
        audio = audio[:expected_len]
    else:
        audio = np.pad(audio, (0, expected_len - len(audio)))
    return audio, sr

def audio_to_logmel(path):
    audio, sr = load_audio_fixed(path)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel

def preprocess_for_model(path):
    logmel = audio_to_logmel(path)
    logmel = logmel[..., np.newaxis]
    return np.expand_dims(logmel, axis=0)

def predict_emotion(path):
    x = preprocess_for_model(path)
    pred = model.predict(x)
    probs = pred[0]
    idx = np.argmax(probs)
    emotion = le.inverse_transform([idx])[0]
    return emotion, probs


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Aour Kya Haal Chal",
    page_icon="✨",
    layout="wide",
)

# ---------------------------
# PREMIUM CSS + ANIMATIONS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at top, #4b2f9b 0, #241734 40%, #05030c 100%);
    color: #ffffff;
}

/* center main block */
.main-block {
    max-width: 950px;
    margin: 0 auto;
    padding: 10px 10px 60px 10px;
}

/* header */
.app-title {
    text-align: center;
    font-size: 54px;
    font-weight: 800;
    margin-bottom: -5px;
    color: #FFE082;
    text-shadow: 0 0 35px #FFC107, 0 0 70px rgba(255,193,7,0.7);
    animation: fadeDown 0.9s ease-out forwards;
    opacity: 0;
}

.app-subtitle {
    text-align: center;
    font-size: 20px;
    color: #E0D7FF;
    margin-bottom: 35px;
    animation: fadeIn 1.1s ease-out forwards;
    opacity: 0;
}

/* cards */
.card {
    background: rgba(18, 18, 40, 0.85);
    border-radius: 22px;
    padding: 24px 26px;
    box-shadow: 0 22px 45px rgba(0,0,0,0.55);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.05);
}

/* mode card animations */
.upload-card { animation: slideLeft 0.8s ease-out forwards; opacity: 0; }
.record-card { animation: slideRight 0.8s ease-out forwards; opacity: 0; }

/* result card */
.result-card {
    margin-top: 24px;
    border-radius: 24px;
    padding: 22px;
    background: radial-gradient(circle at top left, rgba(0, 229, 255, 0.25), rgba(58, 29, 112, 0.95));
    box-shadow: 0 0 50px rgba(0, 229, 255, 0.4);
    border: 1px solid rgba(0, 229, 255, 0.4);
    animation: fadeIn 0.9s ease-out forwards;
}

/* emotion text */
.prediction-text {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #00F5FF;
    text-shadow: 0 0 25px #00F5FF, 0 0 50px rgba(0,245,255,0.8);
}

/* probability heading */
.prob-title {
    font-weight: 600;
    margin-bottom: 8px;
    color: #E1E6FF;
}

/* radio */
div.row-widget.stRadio > div { justify-content: center; }
label[data-baseweb="radio"] > span {
    font-weight: 500;
}

/* mic button */
.mic-wrapper {
    display: flex;
    justify-content: center;
    margin-top: 10px;
    margin-bottom: 8px;
}
.mic-button {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: conic-gradient(from 210deg, #ff0066, #ff9800, #ffc107, #ff0066);
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 35px rgba(255,0,102,0.6);
    animation: pulse 2.2s infinite;
}
.mic-inner {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: radial-gradient(circle at top, #2b133f, #170b26 70%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 700;
    font-size: 20px;
}

/* animations */
@keyframes fadeDown {
    0% { transform: translateY(-30px); opacity: 0; }
    100% { transform: translateY(0px); opacity: 1; }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideLeft {
    0% { transform: translateX(-45px); opacity: 0; }
    100% { transform: translateX(0px); opacity: 1; }
}
@keyframes slideRight {
    0% { transform: translateX(45px); opacity: 0; }
    100% { transform: translateX(0px); opacity: 1; }
}
@keyframes pulse {
    0%   { transform: scale(1); box-shadow: 0 0 30px rgba(255,0,102,0.5); }
    50%  { transform: scale(1.06); box-shadow: 0 0 55px rgba(255,153,0,0.8); }
    100% { transform: scale(1); box-shadow: 0 0 30px rgba(255,0,102,0.5); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<div class='main-block'>", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>✨ Aour Kya Haal Chal ✨</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='app-subtitle'>Premium Speech Emotion Recognition </p>",
    unsafe_allow_html=True
)

# ---------------------------
# MODE SELECTION
# ---------------------------
mode = st.radio("", ["🎧 Upload Audio", "🎙️ Record Using Microphone"])

# container with two columns (on large screens)
col1, col2 = st.columns([1, 1])

# a small helper to render the prediction nicely
def render_prediction(emotion, probs):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction-text'>{emotion.upper()}</div>", unsafe_allow_html=True)

    # show probabilities
    prob_df = pd.DataFrame({
        "Emotion": emotion_classes,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    st.markdown("<p class='prob-title'>Emotion probabilities</p>", unsafe_allow_html=True)
    st.bar_chart(prob_df.set_index("Emotion"))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# UPLOAD MODE
# ---------------------------
if mode == "🎧 Upload Audio":
    with col1:
        st.markdown("<div class='card upload-card'>", unsafe_allow_html=True)
        st.subheader("Upload Audio")
        st.caption("Drop a `.wav` file and I’ll analyse the emotion in your voice.")

        uploaded = st.file_uploader("Upload a WAV file", type=["wav"], label_visibility="collapsed")

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded.read())
                audio_path = tmp.name

            st.audio(audio_path)

            emotion, probs = predict_emotion(audio_path)
            render_prediction(emotion, probs)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.empty()  # keep layout balanced

# ---------------------------
# MICROPHONE MODE
# ---------------------------
else:
    with col1:
        st.markdown("<div class='card record-card'>", unsafe_allow_html=True)
        st.subheader("Record Using Microphone")
        st.caption("Click the glowing mic, speak for 3 seconds, and I’ll tell you how you sound. 💬")

        # fancy mic button
        st.markdown("<div class='mic-wrapper'>", unsafe_allow_html=True)
        if st.button("", key="mic_button"):
            # record
            st.info("🎙️ Recording... please speak now for 3 seconds.")
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()

            temp_wav = "mic_input.wav"
            sf.write(temp_wav, recording, SAMPLE_RATE)

            st.success("✅ Recording complete!")
            st.audio(temp_wav)

            emotion, probs = predict_emotion(temp_wav)
            render_prediction(emotion, probs)
        else:
            # just draw the mic visual
            st.markdown(
                "<div class='mic-button'><div class='mic-inner'>Tap to Record</div></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.empty()

st.markdown("</div>", unsafe_allow_html=True)
