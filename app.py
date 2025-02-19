import streamlit as st
import numpy as np
import joblib
import librosa
import tempfile
import os

# Load trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("speaker_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

speaker_model, scaler = load_model()

def extract_features(audio_file):
    """Extract MFCC features from the uploaded audio file."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.vstack((mfccs, delta_mfccs, delta2_mfccs))
        return np.mean(features.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def identify_speaker(audio_path):
    """Predict if the uploaded audio belongs to the trained speaker."""
    features = extract_features(audio_path)
    if features is None:
        return None, None

    features = scaler.transform(features.reshape(1, -1))

    try:
        log_likelihood = speaker_model.score(features)
        return log_likelihood, features
    except Exception as e:
        st.error(f"Error in speaker identification: {e}")
        return None, None

# Streamlit UI
st.title("🎤 Speaker Verification System")
st.subheader("Upload an audio file to verify the speaker")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filename = tmp_file.name

    # Perform speaker identification
    log_likelihood, features = identify_speaker(tmp_filename)
    
    # Define a likelihood threshold (adjust as needed)
    threshold = -40  # Example threshold, tune based on your model performance
    
    if log_likelihood is not None:
        st.write(f"Log-likelihood Score: {log_likelihood:.2f}")
        
        if log_likelihood > threshold:
            st.success("✅ Speaker identified!")
        else:
            st.warning("❌ Speaker not identified.")

    # Cleanup temp file
    os.remove(tmp_filename)
