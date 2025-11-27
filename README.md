# ✨ Aour Kya Haal Chal — Speech Emotion Recognition App

Aour Kya Haal Chal is a **premium, animated Speech Emotion Recognition (SER)** web application built using **Streamlit**, **TensorFlow**, and **Librosa**.  
It allows users to **upload audio** or **record live voice** and predicts the **emotion** along with a probability distribution.

---

## ⭐ Features

### 🎧 Upload Audio
Upload `.wav` files and get instant emotion predictions.

### 🎙️ Microphone Recording (3 seconds)
Record audio directly through your browser and analyse your emotion.

### 🔮 Deep Learning Model
- CNN model trained on log-mel spectrograms  
- High accuracy  
- Uses TensorFlow/Keras backend  

### 🎨 Premium UI / UX
- Glowing neon mic animation  
- Radial gradients  
- Glassmorphism cards  
- Smooth fade + slide animations  
- Clean Poppins typography  

---

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── emotion_cnn_best.h5    # Trained CNN model
├── label_encoder.pkl      # Label encoder for emotion classes
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## 🛠️ Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 2. Install Dependencies

Add this to your `requirements.txt`:

```
streamlit
numpy
librosa
soundfile
sounddevice
tensorflow
pandas
pickle-mixin
```

Install:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

Run the following:

```bash
streamlit run app.py
```

App opens at:

```
http://localhost:8501
```

---

## 🔬 How It Works

### 1. Audio Preprocessing  
- Loads audio at **22050 Hz**
- Trims/pads to **3 seconds**
- Converts to **128-MEL** log-mel spectrogram

### 2. Model Prediction  
- CNN predicts emotion class  
- Highest probability emotion is shown  
- Bar chart displays probability for all emotions  

### 3. UI Rendering  
Custom CSS adds:  
- Animated headers  
- Sliding cards  
- Neon glowing mic button  
- Gradient backgrounds  

---

## 🎤 Microphone Recording Logic

The app records 3 seconds using:

```python
sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
```

Then saves as a temporary `.wav` file and sends it to the model.

---

## 📊 Output

You get:
- ⭐ **Predicted Emotion** (large glowing text)
- 📊 **Probability Bar Chart** for all emotions

---

## 🧠 Required Model Files

Keep these in the same folder as `app.py`:

- `emotion_cnn_best.h5`
- `label_encoder.pkl`

The app loads them automatically at startup.

---

## 🏆 Technologies Used

- Streamlit  
- TensorFlow / Keras  
- Librosa  
- SoundDevice  
- SoundFile  
- Pandas  
- Custom CSS Animations  

---

## 📣 Author

**Aour Kya Haal Chal – Premium Speech Emotion Recognition App**

Made with ❤️ using Deep Learning + Beautiful UI.

