# hand_utils.py - Utility functions for HandTalk

import os
import numpy as np

NUM_KEYPOINTS     = 21
AXES_PER_KP       = 2
FEATURES_PER_HAND = NUM_KEYPOINTS * AXES_PER_KP
TOTAL_FEATURES    = FEATURES_PER_HAND * 2
PER_HAND_LEN      = FEATURES_PER_HAND
TARGET_VECTOR_LEN = TOTAL_FEATURES


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_class_labels(data_dir: str, output_file: str = "labels.txt") -> list:
    classes = sorted(
        entry for entry in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, entry))
    )
    with open(output_file, "w") as fh:
        fh.write("\n".join(classes) + "\n")
    return classes


def load_labels_from_dir(data_dir, out_labels_file="labels_isl.txt"):
    return read_class_labels(data_dir, out_labels_file)


def fix_vector_length(vector, target_len: int) -> np.ndarray:
    arr = list(vector)
    if len(arr) < target_len:
        arr += [0.0] * (target_len - len(arr))
    return np.asarray(arr[:target_len], dtype=np.float32)


def pad_or_truncate(vector, target_len: int) -> np.ndarray:
    return fix_vector_length(vector, target_len)


def wrist_normalise(landmarks_xy: list) -> list:
    if not landmarks_xy:
        return landmarks_xy
    xs = landmarks_xy[0::2]
    ys = landmarks_xy[1::2]
    ox, oy = xs[0], ys[0]
    span = max(max(xs) - min(xs), 1e-6)
    normalised = []
    for x, y in zip(xs, ys):
        normalised.extend([(x - ox) / span, (y - oy) / span])
    return normalised


def augment_vector(vec, rotation=0.0, scale=1.0, tx=0.0, ty=0.0, noise_std=0.0):
    result = np.array(vec, dtype=np.float64).copy()
    theta = np.radians(rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for hand_offset in [0, FEATURES_PER_HAND]:
        if hand_offset + 1 >= len(result):
            break
        for i in range(FEATURES_PER_HAND // 2):
            idx_x = hand_offset + i * 2
            idx_y = idx_x + 1
            if idx_y >= len(result):
                break
            x, y = result[idx_x], result[idx_y]
            x = x * cos_t - y * sin_t
            y = x * sin_t + y * cos_t
            x = x * scale + tx
            y = y * scale + ty
            if noise_std > 0:
                x += np.random.normal(0, noise_std)
                y += np.random.normal(0, noise_std)
            result[idx_x] = x
            result[idx_y] = y
    return result.astype(np.float32)
