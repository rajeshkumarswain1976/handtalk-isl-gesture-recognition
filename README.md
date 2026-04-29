# HandTalk — ISL Gesture Recognition

Real-time Indian Sign Language (ISL) gesture recognition system using
MediaPipe hand landmark extraction and machine learning.
Achieves 99.65% test accuracy across 23 gesture classes.

---

### What it does

- Captures webcam frames for each ISL gesture class
- Extracts hand landmark features using MediaPipe (21 keypoints per hand)
- Trains and compares Random Forest and HistGradientBoosting classifiers
- Applies feature-level augmentation (rotation + scale) to improve robustness
- Deploys a real-time Streamlit web app for live gesture prediction

---

### Pipeline
Webcam Capture (capture_gestures.py)
→ MediaPipe Landmark Extraction (extract_features.py)
→ Feature Augmentation + Model Training (build_model.py)
→ Real-time Streamlit App (gesture_app.py)

---

### Dataset

ISL Gesture Image Dataset — [Download from Kaggle](<YOUR KAGGLE LINK HERE>)

- 23 ISL gesture classes
- 1,000 images per class
- Both hands captured per gesture

Place the downloaded folder as:
isl_data/

---

### Results

| Model | Test Accuracy | CV Accuracy (5-fold) |
|---|---|---|
| **HistGradientBoosting** | **99.65%** | 99.71±0.10% |
| Random Forest | 99.42% | 99.82±0.04% |

- 22,077 feature samples after augmentation
- Feature augmentation: 7 rotations × 5 scales per sample
- Stratified train/test split preserves class balance

---

### Stack

| | |
|---|---|
| **Hand Tracking** | MediaPipe |
| **ML Models** | scikit-learn · Random Forest · HistGradientBoosting |
| **Data** | NumPy · joblib · pickle |
| **Visualization** | Matplotlib · ConfusionMatrixDisplay |
| **App** | Streamlit |
| **Language** | Python 3.13 |

---

### Project Structure
handtalk-isl-gesture-recognition/
│
├── build_model.py          # Training pipeline
├── capture_gestures.py     # Gesture image capture via webcam
├── extract_features.py     # MediaPipe landmark extraction
├── gesture_app.py          # Streamlit production app
├── hand_utils.py           # Shared utilities
├── config.py               # Project configuration
├── labels.txt              # Class labels
├── requirements.txt        # Python dependencies
│
├── model_ht.pkl            # HistGradientBoosting model (99.65%)
├── model_rf.pkl            # Random Forest model (99.42%)
│
├── confusion_matrix_histgradientboosting.png
└── model_comparison.png

---

### Setup & Run

```bash
# Clone the repo
git clone https://github.com/yourusername/handtalk-isl-gesture-recognition
cd handtalk-isl-gesture-recognition

# Install dependencies
pip install -r requirements.txt

# Step 1 — Capture gesture images
python capture_gestures.py --out ./isl_data --count 1000

# Step 2 — Extract MediaPipe features
python extract_features.py

# Step 3 — Train models
python build_model.py

# Step 4 — Run the app
streamlit run gesture_app.py
```

---

### How Feature Extraction Works

- MediaPipe detects 21 hand landmarks per hand (x, y coordinates)
- Landmarks are wrist-normalised to remove position dependency
- Left and right hand vectors concatenated into a single feature vector
- Images with no detected hand are skipped and logged

---

### How Augmentation Works

- Each sample augmented 3 times with random rotation (±15°) and scale (0.85–1.15)
- Augmentation applied at feature level — no image processing needed
- Stratified split preserves class balance after augmentation

---

> Note: isl_data/ and model .pkl files are not included in the repo due to size.
> Download the dataset from Kaggle and run the pipeline to generate models.
