# Speaker Verification System

This project demonstrates a speaker verification system using Gaussian Mixture Models (GMMs) and Mel-Frequency Cepstral Coefficients (MFCCs).  It provides a practical example of how to train and evaluate a GMM-based speaker verification model.

## Overview

This speaker verification system utilizes **Gaussian Mixture Models (GMMs)** to model the unique vocal characteristics of individual speakers.  The system extracts **Mel-Frequency Cepstral Coefficients (MFCCs)**, including their first and second derivatives (delta and delta-delta), from audio samples to capture the spectral envelope of speech.  These features are then used to train a GMM for each enrolled speaker.  
During verification, the system compares the likelihood of a given audio sample belonging to a claimed speaker's GMM with a threshold.  This comparison allows the system to accept or reject the speaker's identity claim.  The performance of the system is rigorously evaluated using metrics such as the **Area Under the ROC Curve (AUC)** and the **Equal Error Rate (EER)**, providing a comprehensive assessment of its accuracy and reliability.  The implementation incorporates k-fold cross-validation during training to enhance model robustness and generalization to unseen data, and feature scaling to optimize the performance of the GMMs.


## Installation

To get this project running on your machine, follow these installation instructions:

(1) Clone the Repository: git clone https://github.com/dhruvgoe/Speaker-Verification-System.git

(2) Navigate to the project directory: cd Speaker-Verification-System

(3) Install the required dependencies: pip install -r requirements.txt


## Usage

**(1) Training Speaker Models:**

Before verifying speakers, you must train GMMs for each enrolled speaker. Use the `Detection.py` script for this.

```bash
python Detection.py --data_dir ./dataset --negative_files Test-negative1.wav Test-negative2.wav Test-negative3.wav --n_components 8 --reg_covar 0.01
```
Details:

* --data_dir: Path to the directory containing audio files, organized by speaker (see Dataset). Required.

* --negative_files: Paths to the audio files used as negative samples during evaluation. Separate multiple files with spaces. Required.

* --n_components: Number of GMM components. Experiment with this value. Default is 8.

* --reg_covar: Regularization parameter for covariance matrices. Helps prevent overfitting. Default is 0.01.

**(2) Verifying a Speaker:**

After training, verify a speaker's identity using the same script.

```
python Detection.py --test_audio Test-positive.wav --model_file speaker_models.pkl --scaler_file scaler.pkl --threshold 0.5
```
Details: 
* --test_audio: Path to the audio file to verify. Required.

* --model_file: Path to the saved GMM models (output from training). Required.

* --scaler_file: Path to the saved scaler object (output from training). Required.

* --threshold: The decision threshold. If not provided the threshold from the training stage is used.

**(3) Output:**

The script outputs the log-likelihood of the test audio belonging to the claimed speaker's model and the verification decision.  For example:

```
Log-likelihood for Test-positive.wav: -22.567
Speaker identified.
```

Or:

```
Log-likelihood for Test-negative.wav: -35.892
Speaker not identified.
```

## Dataset

This project utilizes the [Speaker Recognition Audio Dataset](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset) from Kaggle. This dataset contains audio samples from multiple speakers, providing a diverse range of vocal characteristics for training and evaluation.

For this specific implementation, the model is trained using voice samples from a *single* target speaker within this dataset.  The audio files for this target speaker are located within their designated folder in the dataset.  The specific speaker ID used for training can be configured within the `Detection.py` script by adjusting the path to the speaker's directory.

Negative samples, representing impostor speakers, were also drawn from the same Kaggle dataset.  Specifically, audio samples from *different* speakers (i.e., speakers *other* than the target speaker) were used as negative examples.  The specific files used as negative samples are defined in the main directory of project variable within the script.

## Code Explaination

**(1) Libraries used:**

* **librosa**: A Python library for music and audio analysis, used here for loading audio files and extracting MFCC features.

* **numpy**: A fundamental library for numerical computing in Python, used for array manipulation and calculations.

* **scikit-learn** (sklearn): A machine learning library in Python, used here for feature scaling (StandardScaler), GMM modeling (GaussianMixture), and k-fold cross-validation (KFold).

* **scipy**: A library for scientific computing in Python, used here for calculating the Equal Error Rate (EER) via root finding (brentq).

**(2) Functions Implemented:**

**1. extract_features(audio_file):**

**Purpose**: This function extracts Mel-Frequency Cepstral Coefficients (MFCCs) and their delta and delta-delta (first and second derivative) features from a given audio file. These features represent the spectral envelope of the audio signal and are crucial for speaker recognition.

**Functionality**:
* Loads the audio file using librosa.load().

* Computes MFCCs using librosa.feature.mfcc().

* Calculates delta MFCCs (first derivative) using librosa.feature.delta().

* Calculates delta-delta MFCCs (second derivative) using librosa.feature.delta() with order=2.

* Stacks the MFCCs, delta MFCCs, and delta-delta MFCCs together.

* Calculates the mean of the features across time frames. This is a common way to summarize the features for a given audio clip.

* Returns the mean of the feature vector.

* Includes a try-except block to catch potential errors during audio loading or feature extraction and prints an error message, returning None if an error occurs. This is important for robustness.

**2. train_speaker_model(audio_files, n_components=8, reg_covar=1e-2):**

**Purpose**: This function trains a Gaussian Mixture Model (GMM) for a speaker using the provided audio files. It also performs feature scaling and k-fold cross-validation.

**Functionality:**
* Calls extract_features() for each audio file to get the feature vectors.

* Handles cases where feature extraction fails for some audio files.

* Uses StandardScaler from sklearn.preprocessing to scale the features. Feature scaling is essential for GMM performance.
  
* Creates a GaussianMixture model from sklearn.mixture with the specified n_components (number of Gaussian components) and reg_covar (regularization for covariance matrices).

* Fits the GMM to the scaled features.

* Returns the trained GMM and the StandardScaler object. The scaler is needed later for processing test data.

* Includes a try-except block to catch potential errors during GMM fitting.

**3. identify_speaker(audio_file, speaker_model, scaler):**

**Purpose**: This function takes an audio file and a trained speaker model (GMM) and calculates the log-likelihood of the audio belonging to that speaker's model.

**Functionality**:

* Extracts features from the input audio file using extract_features().

* Handles cases where feature extraction fails.

* Scales the extracted features using the provided scaler (which was trained on the training data). It's crucial to use the same scaler.

* Calculates the log-likelihood of the scaled features under the speaker's GMM using speaker_model.score().

* Returns the log-likelihood.

* Includes a try-except block to catch potential errors during scoring.

**4. evaluate_model(speaker_model, scaler, positive_samples, negative_samples):**

**Purpose**: This function evaluates the performance of the trained speaker model using a set of positive (audio from the target speaker) and negative (audio from other speakers or background noise) samples. It calculates the Area Under the ROC Curve (AUC) and the Equal Error Rate (EER).

**Functionality**:

* Calculates log-likelihoods for all positive and negative samples using identify_speaker().

* Creates true labels (y_true) and predicted scores (y_scores).

* Calculates the ROC curve and AUC using roc_curve and auc from sklearn.metrics.

* Calculates the EER using interp1d (from scipy.interpolate for interpolation) and brentq (from scipy.optimize for root finding). The EER is the point where the false positive rate equals the false negative rate.

* Returns the AUC, EER, and the threshold at which the EER is achieved.

* Handles cases where likelihood calculation fails for some samples.

## Features

*   **Gaussian Mixture Models (GMMs):**  Utilizes GMMs to model the probability distribution of features extracted from a speaker's voice. GMMs are well-suited for this task as they can capture complex patterns in speech data.

*   **Mel-Frequency Cepstral Coefficients (MFCCs):**  Employs MFCCs as the primary feature representation. MFCCs are widely used in speech recognition and speaker recognition due to their effectiveness in representing the spectral characteristics of speech.  Delta and delta-delta features are also calculated to incorporate temporal information.

*   **K-fold Cross-Validation:**  Implements k-fold cross-validation during model training. This technique helps to assess the model's performance on unseen data and reduces the risk of overfitting.

*   **Performance Metrics (AUC and EER):**  Evaluates the system's performance using the Area Under the ROC Curve (AUC) and the Equal Error Rate (EER).  AUC provides an overall measure of the system's ability to distinguish between speakers, while EER indicates the point at which the false positive and false negative rates are equal.  These are standard metrics used in speaker recognition research.

## Example 

**(1) When Input is Target Voice**

![image](https://github.com/user-attachments/assets/16e21fca-d38c-4015-b192-d57dd582dd88)

**(2) When Input is Non-Target Voice**

![image](https://github.com/user-attachments/assets/4f79e733-c7c0-443f-8bf9-71081b5561ec)

## Future Work

This project provides a solid foundation for speaker verification.  Potential future enhancements include:

*   Exploring different feature representations (e.g., i-vectors, x-vectors).
*   Implementing more advanced GMM training techniques.
*   Integrating the system with a real-world application.
*   Improving the system's robustness to noise and other audio variations.

## Conclusion

This project provides a working implementation of a speaker verification system using GMMs and MFCCs.  It demonstrates the key steps involved in feature extraction, model training, and verification.  We hope this project serves as a useful starting point for further exploration in speaker recognition.
