"""
extract_features.py – Extract MediaPipe landmarks
"""

import os
import pickle
import argparse
import logging

import cv2
import mediapipe as mp
from tqdm import tqdm

from config import (DATASET_DIR, FEATURES_FILE, LABELS_FILE, ERROR_LOG,
                     MP_STATIC_MODE, MP_MIN_DETECTION, MP_MAX_HANDS, MP_MIN_TRACKING)
from hand_utils import (FEATURES_PER_HAND, TOTAL_FEATURES,
                        fix_vector_length, read_class_labels, wrist_normalise)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("extract_features")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def landmarks_to_vec(lm_proto) -> list:
    raw_x = [p.x for p in lm_proto.landmark]
    raw_y = [p.y for p in lm_proto.landmark]
    flat  = []
    for x, y in zip(raw_x, raw_y):
        flat.extend([x, y])
    return wrist_normalise(flat)


def build_feature_vector(mp_results) -> list:
    vec = [0.0] * TOTAL_FEATURES
    if not mp_results.multi_hand_landmarks:
        return vec

    hand_labels = []
    for h in (mp_results.multi_handedness or []):
        try:
            hand_labels.append(h.classification[0].label)
        except Exception:
            hand_labels.append("Unknown")

    hand_map: dict[str, list] = {}
    for idx, lm in enumerate(mp_results.multi_hand_landmarks):
        label = hand_labels[idx] if idx < len(hand_labels) else f"hand{idx}"
        if label not in hand_map:
            coords = landmarks_to_vec(lm)
            hand_map[label] = fix_vector_length(coords, FEATURES_PER_HAND).tolist()

    left  = hand_map.get("Left",  [0.0] * FEATURES_PER_HAND)
    right = hand_map.get("Right", [0.0] * FEATURES_PER_HAND)
    return left + right


def extract_landmarks(data_dir: str = DATASET_DIR,
                      output_path: str = FEATURES_FILE,
                      labels_path: str = LABELS_FILE) -> tuple:

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=MP_STATIC_MODE,
        min_detection_confidence=MP_MIN_DETECTION,
        min_tracking_confidence=MP_MIN_TRACKING,
        max_num_hands=MP_MAX_HANDS,
    )

    try:
        classes = read_class_labels(data_dir, labels_path)
        log.info("Classes found (%d): %s", len(classes), classes)

        image_index: list[tuple[str, str]] = []
        for cls in classes:
            folder = os.path.join(data_dir, cls)
            for fn in os.listdir(folder):
                if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                    image_index.append((os.path.join(folder, fn), cls))

        log.info("Total images to process: %d", len(image_index))

        features, labels, errors = [], [], []

        for img_path, cls in tqdm(image_index, desc="Extracting", unit="img"):
            try:
                bgr = cv2.imread(img_path)
                if bgr is None:
                    errors.append(f"UNREADABLE  {img_path}")
                    continue
                rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result  = detector.process(rgb)
                vec     = build_feature_vector(result)
                if all(v == 0.0 for v in vec):
                    errors.append(f"NO_HAND     {img_path}")
                    continue
                features.append(vec)
                labels.append(cls)
            except Exception as exc:
                errors.append(f"ERROR       {img_path}  →  {exc}")

        with open(output_path, "wb") as fh:
            pickle.dump({"data": features, "labels": labels}, fh)

        log.info("Saved %d feature vectors → %s", len(features), output_path)
        log.info("Skipped / errored: %d (see %s)", len(errors), ERROR_LOG)

        if errors:
            with open(ERROR_LOG, "w") as fh:
                fh.write("\n".join(errors))

        return features, labels
    finally:
        detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HandTalk – Feature Extraction")
    parser.add_argument("--data_dir", default=DATASET_DIR)
    parser.add_argument("--out",      default=FEATURES_FILE)
    parser.add_argument("--labels_path", default=LABELS_FILE,
                        help="Path to class labels file")
    args = parser.parse_args()
    extract_landmarks(args.data_dir, args.out, args.labels_path)
