"""
build_model.py – Train with feature-level augmentation
"""

import pickle
import argparse
import logging
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, classification_report,
                                      ConfusionMatrixDisplay, confusion_matrix)
from joblib import dump

from config     import (FEATURES_FILE, MODEL_FILE, CM_OUTPUT,
                         TEST_SPLIT, RANDOM_SEED, CV_FOLDS,
                         GB_MAX_ITER, GB_MAX_DEPTH,
                         GB_LEARNING_RATE, GB_L2_REG,
                         RF_N_ESTIMATORS, RF_MAX_DEPTH)
from hand_utils import TOTAL_FEATURES, FEATURES_PER_HAND, fix_vector_length

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("build_model")

RF_MODEL_FILE = "model_rf.pkl"

AUG_ROTATIONS = [-15, -10, -5, 0, 5, 10, 15]
AUG_SCALES    = [0.85, 0.92, 1.0, 1.08, 1.15]


def sanitise_matrix(X_raw: np.ndarray) -> np.ndarray:
    if X_raw.shape[1] == TOTAL_FEATURES:
        return X_raw.astype(np.float32)
    return np.vstack([fix_vector_length(row, TOTAL_FEATURES) for row in X_raw])


def augment_features(X, y, n_aug=3):
    rng = np.random.RandomState(RANDOM_SEED)
    X_aug, y_aug = [], []
    for i in range(len(X)):
        rotations = rng.choice(AUG_ROTATIONS, n_aug, replace=False)
        scales = rng.choice(AUG_SCALES, n_aug, replace=False)
        for rot, sc in zip(rotations, scales):
            aug = X[i].copy()
            theta = np.radians(rot)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for hand_off in [0, FEATURES_PER_HAND]:
                if hand_off + 1 >= len(aug):
                    break
                for j in range(FEATURES_PER_HAND // 2):
                    xi = hand_off + j * 2
                    yi = xi + 1
                    if yi >= len(aug):
                        break
                    x, yv = aug[xi], aug[yi]
                    x_new = x * cos_t - yv * sin_t
                    yv_new = x * sin_t + yv * cos_t
                    aug[xi] = x_new * sc
                    aug[yi] = yv_new * sc
            X_aug.append(aug)
            y_aug.append(y[i])
    return np.vstack(X_aug), np.array(y_aug)


def save_confusion_matrix(y_true, y_pred, classes, model_name, path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, colorbar=True, cmap="YlGnBu", xticks_rotation=90)
    ax.set_title(f"HandTalk – Confusion Matrix ({model_name})", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: dict, out_path: str = "model_comparison.png"):
    names     = list(results.keys())
    test_accs = [results[n]["test_acc"] * 100 for n in names]
    cv_accs   = [results[n]["cv_mean"]  * 100 for n in names]
    times     = [results[n]["train_time"] for n in names]

    x = np.arange(len(names))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_accs = test_accs + cv_accs
    y_min = max(0, min(all_accs) - 2)
    y_max = min(100, max(all_accs) + 5)

    ax = axes[0]
    ax.bar(x - w/2, test_accs, w, label="Test Accuracy",    color="#7c3aed", alpha=0.9)
    ax.bar(x + w/2, cv_accs,   w, label=f"CV ({CV_FOLDS}-fold) Accuracy", color="#06b6d4", alpha=0.9)
    for i, (ta, ca) in enumerate(zip(test_accs, cv_accs)):
        ax.text(i - w/2, ta + 0.3, f"{ta:.1f}%", ha="center", fontsize=9, color="#7c3aed", fontweight="bold")
        ax.text(i + w/2, ca + 0.3, f"{ca:.1f}%", ha="center", fontsize=9, color="#06b6d4", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(y_min, y_max)
    ax.set_title("Accuracy: RF vs HistGradientBoosting", fontsize=12)
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    bars = ax2.bar(names, times, color=["#7c3aed", "#06b6d4"], alpha=0.85, width=0.4)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{t:.1f}s", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Training time (seconds)"); ax2.set_title("Training Time Comparison", fontsize=12)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("HandTalk – Model Comparison Report", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Comparison chart saved → %s", out_path)


def train_and_evaluate(clf, name, X_train, X_test, y_train, y_test,
                       X_scaled_orig, y_orig, skf):
    log.info("Training %s …", name)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred   = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    log.info("Running %d-fold CV for %s on original (non-augmented) data …", CV_FOLDS, name)
    cv_scores = cross_val_score(clf, X_scaled_orig, y_orig, cv=skf,
                                scoring="accuracy", n_jobs=-1)

    log.info("%-30s  test=%.2f%%  cv=%.2f±%.2f%%  time=%.1fs",
             name, test_acc*100, cv_scores.mean()*100, cv_scores.std()*100, train_time)
    print(f"\n── {name} ── Classification Report ──────────────────────")
    print(classification_report(y_test, y_pred))

    return {
        "clf"        : clf,
        "test_acc"   : test_acc,
        "cv_mean"    : cv_scores.mean(),
        "cv_std"     : cv_scores.std(),
        "train_time" : train_time,
        "y_pred"     : y_pred,
    }


def train(data_path=FEATURES_FILE, out_gb=MODEL_FILE, out_rf=RF_MODEL_FILE):

    log.info("Loading features from %s …", data_path)
    with open(data_path, "rb") as fh:
        bundle = pickle.load(fh)

    X = sanitise_matrix(np.asarray(bundle["data"]))
    y = np.asarray(bundle["labels"])
    classes = sorted(set(y))
    log.info("Dataset before aug: X=%s  classes=%d", X.shape, len(classes))

    scaler = StandardScaler()
    X_scaled_orig = scaler.fit_transform(X)

    log.info("Applying feature-level augmentation…")
    X_aug, y_aug = augment_features(X, y, n_aug=3)
    log.info("Dataset after aug: X=%s  classes=%d", X_aug.shape, len(classes))

    X_train, X_test, y_train, y_test = train_test_split(
        scaler.transform(X_aug), y_aug,
        test_size=TEST_SPLIT, stratify=y_aug,
        random_state=RANDOM_SEED, shuffle=True,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    log.info("Train=%d  Test=%d  CV_folds=%d", len(X_train), len(X_test), CV_FOLDS)

    rf_clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )

    gb_clf = HistGradientBoostingClassifier(
        max_iter=GB_MAX_ITER,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        l2_regularization=GB_L2_REG,
        random_state=RANDOM_SEED,
    )

    results = {}
    results["Random Forest"] = train_and_evaluate(
        rf_clf, "Random Forest",
        X_train, X_test, y_train, y_test, X_scaled_orig, y, skf)

    results["HistGradientBoosting"] = train_and_evaluate(
        gb_clf, "HistGradientBoosting",
        X_train, X_test, y_train, y_test, X_scaled_orig, y, skf)

    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║           MODEL COMPARISON SUMMARY                       ")
    print("╠═══════════════════════════════════════════════════════════╣")
    for name, r in results.items():
        print(f"║  {name:<25}  test={r['test_acc']*100:5.2f}%"
              f"  cv={r['cv_mean']*100:5.2f}±{r['cv_std']*100:.2f}%"
              f"  ({r['train_time']:.1f}s) ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    winner_name = max(results, key=lambda n: results[n]["test_acc"])
    log.info("Best model: %s (%.2f%%)", winner_name, results[winner_name]["test_acc"]*100)

    for model_name, save_path, clf_obj in [
        ("Random Forest",        out_rf, rf_clf),
        ("HistGradientBoosting", out_gb, gb_clf),
    ]:
        dump({
            "classifier"  : clf_obj,
            "scaler"      : scaler,
            "classes"     : classes,
            "feature_len" : TOTAL_FEATURES,
            "model_name"  : model_name,
        }, save_path)
        log.info("%s saved → %s", model_name, save_path)

    cm_path = CM_OUTPUT.replace(".png", f"_{winner_name.replace(' ', '_').lower()}.png")
    save_confusion_matrix(
        y_test, results[winner_name]["y_pred"], classes, winner_name, cm_path)

    plot_model_comparison(results, "model_comparison.png")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HandTalk – Train & Compare Models")
    parser.add_argument("--data",   default=FEATURES_FILE)
    parser.add_argument("--out_gb", default=MODEL_FILE)
    parser.add_argument("--out_rf", default=RF_MODEL_FILE)
    args = parser.parse_args()
    train(args.data, args.out_gb, args.out_rf)
