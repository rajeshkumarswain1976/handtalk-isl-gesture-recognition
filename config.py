# config.py - Central configuration for HandTalk Gesture Recognition System

import os

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "isl_data")
FEATURES_FILE  = os.path.join(BASE_DIR, "data_isl.pickle")
MODEL_FILE     = os.path.join(BASE_DIR, "model_ht.pkl")
LABELS_FILE    = os.path.join(BASE_DIR, "labels.txt")
ERROR_LOG      = os.path.join(BASE_DIR, "errors.log")
CM_OUTPUT      = os.path.join(BASE_DIR, "confusion_matrix.png")

GESTURES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'W',
    'X', 'Y', 'Z',
]
IMAGES_PER_CLASS   = 1000
CAPTURE_INTERVAL_S = 0.1
HAND_SWITCH_AT     = 500

MP_STATIC_MODE       = True
MP_MIN_DETECTION     = 0.30
MP_MIN_TRACKING      = 0.30
MP_MAX_HANDS         = 2

TEST_SPLIT    = 0.20
RANDOM_SEED   = 7
CV_FOLDS      = 5

AUGMENTATION_ENABLED  = True
AUG_ROTATION_DEG      = 15
AUG_SCALE_FACTOR      = 0.15
AUG_TRANSLATION_PCT   = 0.05
AUG_NOISE_STD         = 0.005

GB_MAX_ITER      = 300
GB_MAX_DEPTH     = 6
GB_LEARNING_RATE = 0.08
GB_L2_REG        = 0.05

RF_N_ESTIMATORS  = 300
RF_MAX_DEPTH     = None

DEFAULT_CONF_THRESHOLD = 0.50
DEFAULT_FRAME_SKIP     = 2
DEFAULT_SMOOTH_WINDOW  = 5
