"""
capture_gestures.py  –  HandTalk Gesture Recognition System
============================================================
Capture webcam frames for each ISL gesture class and store them as JPEG files.

Usage
-----
    python capture_gestures.py [--out ./isl_data] [--count 1000]
"""

import os
import cv2
import time
import argparse

from config import GESTURES, IMAGES_PER_CLASS, CAPTURE_INTERVAL_S, HAND_SWITCH_AT

FONT       = cv2.FONT_HERSHEY_DUPLEX
CLR_GREEN  = (50, 220, 100)
CLR_ORANGE = (0, 165, 255)
CLR_WHITE  = (240, 240, 240)


def overlay_text(frame, lines, start_y=55, dy=45, colour=CLR_GREEN, scale=0.9, thickness=2):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (40, start_y + i * dy),
                    FONT, scale, colour, thickness, cv2.LINE_AA)


def wait_for_ready(cap, gesture_name: str) -> bool:
    """Show a 'Get ready' screen until the user presses Q (or ESC to quit)."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        overlay_text(frame, [
            f"  Gesture : {gesture_name}",
            "  Press Q to start capturing",
            "  Press ESC to quit",
        ], colour=CLR_GREEN)
        cv2.imshow("HandTalk  |  Capture Mode", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            return True
        if key == 27:
            return False


def wait_for_switch(cap, gesture_name: str) -> bool:
    """Ask user to switch hands mid-capture."""
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        overlay_text(frame, [
            f"  Halfway done – {gesture_name}",
            "  Switch hands, then press Q",
        ], colour=CLR_ORANGE)
        cv2.imshow("HandTalk  |  Capture Mode", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            return True
        if key == 27:
            return False


def capture_all(output_dir: str, images_per_class: int) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera – check device permissions.")

    os.makedirs(output_dir, exist_ok=True)
    for gesture in GESTURES:
        os.makedirs(os.path.join(output_dir, gesture), exist_ok=True)

    for gesture in GESTURES:
        print(f"\n[HandTalk] Ready to capture: {gesture}")
        if not wait_for_ready(cap, gesture):
            print("[HandTalk] Aborted by user.")
            break

        count   = 0
        t_last  = time.time()
        switched = False

        while count < images_per_class:
            ok, frame = cap.read()
            if not ok:
                print("[HandTalk] Frame read failed – retrying…")
                continue

            frame = cv2.flip(frame, 1)

            if count == HAND_SWITCH_AT and not switched:
                switched = True
                if not wait_for_switch(cap, gesture):
                    break
                t_last = time.time()
                continue

            # Draw progress overlay BEFORE writing so display stays in sync
            pct  = count / images_per_class
            bar_w = int(frame.shape[1] * 0.7)
            cv2.rectangle(frame, (40, 30), (40 + bar_w, 60), (60, 60, 60), -1)
            cv2.rectangle(frame, (40, 30), (40 + int(bar_w * pct), 60), CLR_GREEN, -1)
            overlay_text(frame, [f"  {gesture}  {count}/{images_per_class}"],
                         start_y=90, colour=CLR_WHITE)
            cv2.imshow("HandTalk  |  Capture Mode", frame)

            now = time.time()
            if now - t_last >= CAPTURE_INTERVAL_S:
                path = os.path.join(output_dir, gesture, f"{count:04d}.jpg")
                cv2.imwrite(path, frame)
                count += 1
                t_last = now

            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"[HandTalk] Captured {count} images for '{gesture}'.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[HandTalk] Capture session complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HandTalk – Gesture Image Capture")
    parser.add_argument("--out",   default="./isl_data",   help="Output root directory")
    parser.add_argument("--count", default=IMAGES_PER_CLASS, type=int,
                        help="Images per gesture class")
    args = parser.parse_args()
    capture_all(args.out, args.count)
