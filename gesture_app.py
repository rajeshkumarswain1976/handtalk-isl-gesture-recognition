"""
gesture_app.py – HandTalk Production ISL Recognition
"""

import os
import sys
import time
import streamlit as st

if sys.version_info >= (3, 13):
    st.error("Python 3.13+ not supported by MediaPipe.")
    st.stop()

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from joblib import load

from config import MODEL_FILE, LABELS_FILE, DEFAULT_CONF_THRESHOLD, \
                       DEFAULT_FRAME_SKIP, DEFAULT_SMOOTH_WINDOW
from hand_utils import FEATURES_PER_HAND, TOTAL_FEATURES, fix_vector_length, wrist_normalise

# ── Load Model First ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    if not os.path.exists(path):
        return None, None
    try:
        bundle = load(path)
        return bundle.get("classifier"), bundle.get("scaler")
    except Exception:
        return None, None

clf, scaler = load_model(MODEL_FILE)

# ── Page Config & Styles ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="HandTalk",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0c1220 0%, #111827 50%, #0c1220 100%) !important;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] .stSlider label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(30, 41, 59, 0.5) !important;
        padding: 6px !important;
        border-radius: 14px !important;
        border: 1px solid #334155 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stFileUploader {
        border: 2px dashed #475569 !important;
        border-radius: 14px !important;
        background: rgba(30, 41, 59, 0.5) !important;
        padding: 24px !important;
    }
    
    .stFileUploader * { color: #e2e8f0 !important; }
    
    .header-container {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #6366f1, #3b82f6);
    }
    
    .header-container h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-container p {
        color: #94a3b8;
        margin: 8px 0 0;
        font-size: 1rem;
    }
    
    .pred-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 20px;
        padding: 36px 28px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .pred-card .label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    .pred-card .gesture {
        font-size: 5.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #60a5fa, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin: 12px 0;
    }
    
    .pred-card .conf-text { color: #94a3b8; font-size: 1rem; font-weight: 500; }
    
    .pred-card .conf-bar-bg {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 8px; height: 10px; width: 100%; margin-top: 16px;
        overflow: hidden; border: 1px solid #334155;
    }
    
    .pred-card .conf-bar-fill {
        height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 8px; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
    }
    
    .pred-card .top3 {
        margin-top: 20px; text-align: left; padding-top: 16px; border-top: 1px solid #1e293b;
    }
    
    .pred-card .top3-title {
        color: #64748b; font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 2px; font-weight: 700; margin-bottom: 10px;
    }
    
    .pred-card .top3-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; margin: 4px 0; background: rgba(15, 23, 42, 0.5);
        border-radius: 8px; border: 1px solid #1e293b;
    }
    
    .pred-card .top3-label { color: #e2e8f0; font-weight: 600; font-size: 1rem; }
    .pred-card .top3-conf { color: #60a5fa; font-weight: 700; font-size: 0.95rem; }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border: 1px solid #334155; border-radius: 14px; padding: 16px 20px; text-align: center;
    }
    
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #60a5fa; line-height: 1.2; }
    .metric-card .lbl {
        font-size: 0.7rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 700; margin-bottom: 4px;
    }
    
    .no-hand {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border: 1px solid #334155; border-radius: 14px; padding: 28px; text-align: center;
    }
    
    .no-hand .icon { font-size: 3rem; margin-bottom: 12px; }
    .no-hand .text { color: #94a3b8; font-size: 1rem; line-height: 1.5; }
    
    .stAlert { background: rgba(30, 41, 59, 0.8) !important; border: 1px solid #334155 !important; border-radius: 12px !important; }
    .stAlert * { color: #e2e8f0 !important; }
    .stSuccess { background: rgba(16, 185, 129, 0.15) !important; border: 1px solid rgba(16, 185, 129, 0.4) !important; }
    .stSuccess * { color: #6ee7b7 !important; }
    .stError { background: rgba(239, 68, 68, 0.15) !important; border: 1px solid rgba(239, 68, 68, 0.4) !important; }
    .stError * { color: #fca5a5 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1>🤟 HandTalk</h1>
    <p>Real-time Indian Sign Language Recognition — 23 Static Gestures</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    
    conf_thresh = st.slider(
        "Confidence Threshold", 0.0, 1.0, DEFAULT_CONF_THRESHOLD, 0.05,
        help="Minimum confidence to display a prediction"
    )
    
    frame_skip = st.slider(
        "Frame Skip", 1, 10, DEFAULT_FRAME_SKIP,
        help="Process every N frames"
    )
    
    smooth_window = st.slider(
        "Smoothing Window", 1, 20, DEFAULT_SMOOTH_WINDOW,
        help="Frames for majority vote smoothing"
    )
    
    stability = st.slider(
        "Stability Threshold", 1, 10, 3,
        help="Consistent frames needed before showing prediction"
    )
    
    st.divider()
    
    st.markdown("### 💡 Usage Tips")
    st.markdown("""
- ✋ **Good lighting** on your hand
- 📏 **Center** your hand in frame  
- ⏱️ **Hold steady** for 2-3s
- 🔄 Try both **left & right** hands
    """)
    
    st.divider()
    
    if clf is not None:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Found")

# ── MediaPipe Setup ───────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

detector = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3,
    max_num_hands=2
)

# ── Core Functions ────────────────────────────────────────────────────────────

def get_vector(res):
    vec = np.zeros(TOTAL_FEATURES, dtype=np.float32)
    if not res.multi_hand_landmarks:
        return vec, 0

    labels = []
    for h in (res.multi_handedness or []):
        try:
            labels.append(h.classification[0].label)
        except Exception:
            labels.append("Unknown")

    hm = {}
    for i, lm in enumerate(res.multi_hand_landmarks):
        lab = labels[i] if i < len(labels) else "Unknown"
        if lab not in hm:
            raw = [c for p in lm.landmark for c in (p.x, p.y)]
            hm[lab] = fix_vector_length(wrist_normalise(raw), FEATURES_PER_HAND).tolist()

    left = hm.get("Left", [0.0] * FEATURES_PER_HAND)
    right = hm.get("Right", [0.0] * FEATURES_PER_HAND)
    vec[:FEATURES_PER_HAND] = left
    vec[FEATURES_PER_HAND:] = right
    return vec, len(res.multi_hand_landmarks)


def predict(vec):
    if clf is None:
        return None, 0.0, []
    try:
        def eval_vec(v):
            sv = scaler.transform([v])[0] if scaler else v
            proba = clf.predict_proba([sv])[0]
            pred = clf.predict([sv])[0]
            top3 = [(str(clf.classes_[j]), float(proba[j])) for j in proba.argsort()[::-1][:3]]
            return str(pred), float(proba.max()), top3

        l1, c1, t1 = eval_vec(vec)
        sw = np.concatenate([vec[FEATURES_PER_HAND:], vec[:FEATURES_PER_HAND]])
        l2, c2, t2 = eval_vec(sw)

        return (l1, c1, t1) if c1 >= c2 else (l2, c2, t2)
    except Exception:
        return None, 0.0, []

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_img, tab_cam = st.tabs(["📷 Upload Image", "🎥 Live Camera"])

with tab_img:
    st.markdown("### Upload an image for gesture recognition")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded:
        raw = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)
        vec, hc = get_vector(res)

        disp = frame.copy()
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(disp, lm, mp_hands.HAND_CONNECTIONS,
                                       mp_style.get_default_hand_landmarks_style(),
                                       mp_style.get_default_hand_connections_style())

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Preview")
            st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_column_width=True)
            
        with col2:
            if hc == 0:
                st.markdown("""
<div class="no-hand">
    <div class="icon">✋</div>
    <div class="text">No hand detected.<br>Try improving lighting or angle.</div>
</div>""", unsafe_allow_html=True)
            else:
                label, conf, top3 = predict(vec)
                if label and conf >= conf_thresh:
                    top_items = "".join([
                        f"""<div class="top3-item">
                            <span class="top3-label">{l}</span>
                            <span class="top3-conf">{c:.1%}</span>
                        </div>""" 
                        for l, c in top3
                    ])
                    
                    st.markdown(f"""
<div class="pred-card">
    <div class="label">Detected Gesture</div>
    <div class="gesture">{label}</div>
    <div class="conf-text">Confidence Score</div>
    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{int(conf*100)}%"></div></div>
    <div class="top3">
        <div class="top3-title">Top 3 Predictions</div>
        {top_items}
    </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div class="no-hand">
    <div class="icon">🤷</div>
    <div class="text">Low confidence: {conf:.0%}<br>Adjust threshold in settings.</div>
</div>""", unsafe_allow_html=True)

with tab_cam:
    st.markdown("### Live camera recognition")
    
    use_cam = st.checkbox("Enable Camera", value=False)
    
    if use_cam:
        if st.button("▶ Start Recognition", type="primary"):
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                st.error("Cannot open camera.")
                st.stop()

            frame_count = 0
            pred_buf = deque(maxlen=smooth_window)
            current_stable = None
            current_conf = 0.0
            hold_count = 0
            t_start = time.time()

            metrics = st.empty()
            frame_view = st.empty()
            info_box = st.empty()

            try:
                while True:
                    ok, raw_frame = cap.read()
                    if not ok:
                        break

                    flipped = cv2.flip(raw_frame, 1)
                    rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
                    res = detector.process(rgb)

                    disp = flipped.copy()
                    if res.multi_hand_landmarks:
                        for lm in res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(disp, lm, mp_hands.HAND_CONNECTIONS,
                                                   mp_style.get_default_hand_landmarks_style(),
                                                   mp_style.get_default_hand_connections_style())

                    vec, hand_count = get_vector(res)

                    if frame_count % frame_skip == 0 and hand_count > 0:
                        label, conf, _ = predict(vec)
                        if conf >= conf_thresh and label:
                            pred_buf.append(label)
                            current_conf = conf
                        else:
                            pred_buf.append(None)

                    valid_preds = [p for p in pred_buf if p]
                    if valid_preds:
                        counts = {}
                        for p in valid_preds:
                            counts[p] = counts.get(p, 0) + 1
                        
                        best_pred = max(counts, key=counts.get)
                        best_count = counts[best_pred]
                        total_valid = len(valid_preds)
                        
                        if best_count > total_valid * 0.5:
                            if best_pred == current_stable:
                                hold_count += 1
                            else:
                                if current_stable is None:
                                    current_stable = best_pred
                                    hold_count = 1
                                else:
                                    hold_count += 1
                                    if hold_count >= stability:
                                        current_stable = best_pred
                                        hold_count = 0
                        elif current_stable:
                            hold_count = max(0, hold_count - 1)
                            if hold_count <= 0:
                                current_stable = None

                    display_label = current_stable if current_stable else "—"
                    display_conf = current_conf if current_stable else 0.0

                    if display_label != "—":
                        color = (96, 165, 250)
                        cv2.rectangle(disp, (15, 15), (220, 120), (0, 0, 0), -1)
                        cv2.rectangle(disp, (15, 15), (220, 120), color, 3)
                        cv2.putText(disp, f"{display_label}", (35, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 3, cv2.LINE_AA)
                        cv2.putText(disp, f"{display_conf:.0%}", (35, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                    else:
                        color = (100, 116, 139)
                        cv2.rectangle(disp, (15, 15), (280, 60), (0, 0, 0), -1)
                        cv2.putText(disp, "Detecting...", (30, 45),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

                    frame_view.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

                    fps = frame_count / max(time.time() - t_start, 1e-3)
                    stability_pct = int((hold_count / max(stability, 1)) * 100) if current_stable else 0
                    
                    status_text = "Ready" if display_label == "—" else f"Detected: {display_label}"
                    info_box.markdown(f"""
<div style="color: #94a3b8; font-size: 0.9rem; padding: 8px 0;">
    <strong style="color: #e2e8f0;">Status:</strong> {status_text} &nbsp;|&nbsp; 
    <strong style="color: #e2e8f0;">Frame:</strong> {frame_count}
</div>""", unsafe_allow_html=True)

                    metrics.markdown(f"""
<div style="display:flex; gap:12px; margin-bottom:16px;">
    <div class="metric-card"><div class="lbl">FPS</div><div class="val">{fps:.1f}</div></div>
    <div class="metric-card"><div class="lbl">Hands</div><div class="val">{hand_count}</div></div>
    <div class="metric-card"><div class="lbl">Gesture</div><div class="val">{display_label}</div></div>
    <div class="metric-card"><div class="lbl">Stability</div><div class="val">{stability_pct}%</div></div>
</div>""", unsafe_allow_html=True)

                    frame_count += 1

            except Exception as exc:
                st.error(f"Stream error: {exc}")
            finally:
                cap.release()
