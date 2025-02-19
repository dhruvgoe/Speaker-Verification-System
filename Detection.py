import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d  # For interpolation
from scipy.optimize import brentq      # For EER calculation

def extract_features(audio_file):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features = np.vstack((mfccs, delta_mfccs, delta2_mfccs))
        mfccs_mean = np.mean(features.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return None

def train_speaker_model(audio_files, n_components=8, reg_covar=1e-2):
    """Train a GMM for a speaker with cross-validation."""
    features = []
    for file in audio_files:
        feats = extract_features(file)
        if feats is not None:
            features.append(feats)
    features = np.array(features)

    if features.size == 0:
        print("Error: No valid features extracted. Check your audio files.")
        return None, None

    # Feature scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    try:
        gmm = GaussianMixture(n_components=n_components, covariance_type='tied',
                              random_state=42, reg_covar=reg_covar)
        gmm.fit(features)
        return gmm, scaler
    except ValueError as e:
        print(f"GMM fitting error: {e}")
        return None, None

def identify_speaker(audio_file, speaker_model, scaler):
    """Identify if the speaker in the audio file matches the trained model."""
    test_features = extract_features(audio_file)
    if test_features is None:
        return None

    # Scale the test features
    test_features = scaler.transform(test_features.reshape(1, -1))

    try:
        log_likelihood = speaker_model.score(test_features)
        return log_likelihood
    except Exception as e:
        print(f"Error scoring audio file {audio_file}: {e}")
        return None

def evaluate_model(speaker_model, scaler, positive_samples, negative_samples):
    """Evaluate the model using AUC and EER."""
    positive_likelihoods = [identify_speaker(sample, speaker_model, scaler) for sample in positive_samples]
    positive_likelihoods = [ll for ll in positive_likelihoods if ll is not None]

    negative_likelihoods = [identify_speaker(sample, speaker_model, scaler) for sample in negative_samples]
    negative_likelihoods = [ll for ll in negative_likelihoods if ll is not None]

    if positive_likelihoods and negative_likelihoods:
        y_true = np.concatenate([np.ones(len(positive_likelihoods)), np.zeros(len(negative_likelihoods))])
        y_scores = np.concatenate([positive_likelihoods, negative_likelihoods])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Calculate EER
        f = interp1d(fpr, tpr)
        eer = brentq(lambda x: 1. - x - f(x), 0., 1.)
        thresh = f(eer)
        return roc_auc, eer, thresh
    else:
        return None, None, None

# Example usage:
data_dir = "./dataset"
audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

test_positive_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav") and 'Speaker0041' in f]
test_negative_files = ["Test-negative1.wav", "Test-negative2.wav", "Test-negative3.wav"]

kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_auc = 0
best_model = None
best_scaler = None

for train_index, val_index in kf.split(audio_files):
    train_files = [audio_files[i] for i in train_index]
    val_files = [audio_files[i] for i in val_index]

    speaker_model, scaler = train_speaker_model(train_files)
    if speaker_model is None:
        exit()

    roc_auc, eer, _ = evaluate_model(speaker_model, scaler, val_files, test_negative_files)
    if roc_auc is not None and roc_auc > best_auc:
        best_auc = roc_auc
        best_model = speaker_model
        best_scaler = scaler

if best_model is None:
    print("Model training failed.")
    exit()

# Evaluate on the held-out test set
roc_auc, eer, threshold = evaluate_model(best_model, best_scaler, test_positive_files, test_negative_files)
if roc_auc is not None:
    print(f"Test ROC AUC: {roc_auc}, EER: {eer}, Threshold: {threshold}")
else:
    print("Evaluation failed.")

# Example of making a prediction:
test_audio = "Test-positive.wav"
log_likelihood = identify_speaker(test_audio, best_model, best_scaler)
if log_likelihood is not None:
    print(f"Log-likelihood for {test_audio}: {log_likelihood}")
    if log_likelihood > threshold:
        print("Speaker identified.")
    else:
        print("Speaker not identified.")
else:
    print("Prediction failed.")