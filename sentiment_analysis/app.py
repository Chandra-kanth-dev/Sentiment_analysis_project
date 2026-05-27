import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import gdown
import os
import kagglehub

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONSTANTS  (must match training exactly)
# ─────────────────────────────────────────────
VOCAB_SIZE  = 10000
MAX_LENGTH  = 50
MODEL_PATH  = "mental_health_rnn_model.h5"
GDRIVE_ID   = "1whOvUTQlR3oQ5bL46awDL5HQs1QfTLrA"

CLASSES = [
    "Anxiety", "Bipolar", "Depression",
    "Normal", "Personality disorder", "Stress", "Suicidal"
]

CLASS_META = {
    "Anxiety":              {"color": "#F59E0B", "icon": "😰", "tip": "Try breathing exercises, mindfulness, or speaking with a therapist."},
    "Bipolar":              {"color": "#8B5CF6", "icon": "🔄", "tip": "Mood tracking and professional psychiatric support can be very helpful."},
    "Depression":           {"color": "#3B82F6", "icon": "😔", "tip": "You're not alone — please reach out to a trusted person or mental health professional."},
    "Normal":               {"color": "#10B981", "icon": "😊", "tip": "Your mental state appears balanced. Keep up healthy habits!"},
    "Personality disorder": {"color": "#EC4899", "icon": "🧩", "tip": "Structured therapy (DBT / CBT) can provide meaningful, lasting support."},
    "Stress":               {"color": "#F97316", "icon": "⚡", "tip": "Prioritize rest, set limits, and consider stress-management techniques."},
    "Suicidal":             {"color": "#EF4444", "icon": "🆘", "tip": "⚠️ Please reach out NOW — iCall: 9152987821 | Vandrevala: 1860-2662-345"},
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MindScan · Mental Health AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #060910 !important;
    color: #e2e8f0;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 760px !important; }

/* ── hero ── */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 60%, #0f0a1e 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin-bottom: 2.2rem;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 70% 50% at 50% 0%, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 5vw, 2.6rem);
    font-weight: 800;
    color: #fff;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    letter-spacing: -0.5px;
}
.hero h1 span { color: #818cf8; }
.hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    max-width: 480px;
    margin: 0 auto;
    font-weight: 300;
}

/* ── section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.6rem;
}

/* ── textarea override ── */
.stTextArea > label { display: none; }
.stTextArea textarea {
    background: #0d1424 !important;
    color: #f1f5f9 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    transition: border-color .2s !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
}

/* ── button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.75rem 2rem !important;
    transition: opacity .2s, transform .15s !important;
    cursor: pointer;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── result card ── */
.result-card {
    background: #0d1424;
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1.5rem 0 0.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent);
    border-radius: 16px 16px 0 0;
}
.result-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.4rem;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.result-conf {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 1.2rem;
}
.result-conf b { color: #94a3b8; }
.advice {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    font-size: 0.9rem;
    color: #cbd5e1;
    line-height: 1.6;
}

/* ── probability bars ── */
.prob-section { margin-top: 1.5rem; }
.prob-row { margin-bottom: 0.9rem; }
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.prob-name { font-size: 0.82rem; color: #94a3b8; }
.prob-pct  { font-size: 0.82rem; color: #64748b; font-variant-numeric: tabular-nums; }
.prob-track {
    background: #111827;
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.prob-fill {
    height: 6px;
    border-radius: 999px;
    background: var(--bar-color);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}

/* ── crisis box ── */
.crisis {
    background: rgba(220,38,38,.08);
    border: 1px solid rgba(220,38,38,.35);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-top: 1rem;
    color: #fca5a5;
    font-size: 0.9rem;
    line-height: 1.6;
}
.crisis strong { color: #f87171; display: block; margin-bottom: 0.3rem; font-family: 'Syne', sans-serif; }

/* ── disclaimer ── */
.disclaimer {
    background: #080d17;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #475569;
    font-size: 0.78rem;
    text-align: center;
    margin-top: 2.5rem;
    line-height: 1.6;
}

/* ── loader dots ── */
@keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40% { transform: translateY(-6px); }
}
.dots span {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #6366f1;
    margin: 0 3px;
    animation: bounce 1.2s infinite ease-in-out;
}
.dots span:nth-child(2) { animation-delay: .15s; }
.dots span:nth-child(3) { animation-delay: .3s; }

/* ── spinner override ── */
.stSpinner > div { border-top-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NLTK
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    nltk.download('punkt',      quiet=True)
    nltk.download('stopwords',  quiet=True)
    nltk.download('punkt_tab',  quiet=True)

download_nltk()
stop_words = set(stopwords.words('english'))

# ─────────────────────────────────────────────
# LOAD DATASET + FIT TOKENIZER  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tokenizer_from_dataset():
    path  = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
    files = os.listdir(path)
    df    = pd.read_csv(os.path.join(path, files[0]))

    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Same preprocessing as training
    df['clean_text'] = df['statement'].str.lower()
    df['clean_text'] = df['clean_text'].apply(
        lambda t: re.sub(f"[{re.escape(string.punctuation)}]", "", str(t))
    )
    df['clean_text'] = df['clean_text'].apply(
        lambda t: " ".join(w for w in t.split() if w not in stop_words)
    )

    tok = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(df['clean_text'])

    le = LabelEncoder()
    le.fit(df['status'])

    return tok, le

# ─────────────────────────────────────────────
# LOAD MODEL  (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rnn_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# PREPROCESS + PREDICT
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text

def predict(model, tokenizer, label_encoder, text: str):
    clean  = preprocess(text)
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    probs  = model.predict(padded, verbose=0)[0]
    idx    = int(np.argmax(probs))
    # Use label_encoder classes if available, else fallback to CLASSES
    classes = list(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else CLASSES
    label  = classes[idx]
    return label, float(probs[idx]), probs, classes

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI · Mental Health · NLP</div>
  <h1>Mind<span>Scan</span></h1>
  <p>Paste any statement below — our SimpleRNN model will classify its mental health sentiment across 7 categories.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
with st.spinner("Loading model & tokenizer…"):
    try:
        tokenizer, label_encoder = load_tokenizer_from_dataset()
        model = load_rnn_model()
        resources_ok = True
    except Exception as err:
        st.error(f"Failed to load resources: {err}")
        resources_ok = False

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Your Statement</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="input",
    placeholder="e.g.  I haven't been able to sleep properly. Everything feels overwhelming and pointless…",
    height=150,
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    analyze = st.button("🔍  Analyze Statement", use_container_width=True)
with col_clear:
    clear = st.button("✕  Clear", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if analyze and resources_ok:
    text = user_input.strip()
    if not text:
        st.warning("Please type or paste a statement first.")
    else:
        with st.spinner("Analyzing…"):
            label, confidence, probs, classes = predict(model, tokenizer, label_encoder, text)

        meta  = CLASS_META.get(label, {"color": "#6366f1", "icon": "🔵", "tip": ""})
        color = meta["color"]
        icon  = meta["icon"]

        # ── result card ──
        st.markdown(f"""
        <div class="result-card" style="--accent:{color}">
          <div class="result-eyebrow">Detected Sentiment</div>
          <div class="result-class">{icon} {label}</div>
          <div class="result-conf">Confidence: <b>{confidence*100:.1f}%</b></div>
          <div class="advice">{meta['tip']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── probability bars ──
        st.markdown('<div class="section-label" style="margin-top:1.8rem">All Class Probabilities</div>', unsafe_allow_html=True)
        sorted_pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        bars_html = '<div class="prob-section">'
        for cls, prob in sorted_pairs:
            pct  = prob * 100
            clr  = CLASS_META.get(cls, {}).get("color", "#6366f1")
            ico  = CLASS_META.get(cls, {}).get("icon", "")
            bars_html += f"""
            <div class="prob-row">
              <div class="prob-header">
                <span class="prob-name">{ico} {cls}</span>
                <span class="prob-pct">{pct:.1f}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" style="width:{pct}%; --bar-color:{clr};"></div>
              </div>
            </div>"""
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)

        # ── crisis alert ──
        suicidal_idx = classes.index("Suicidal") if "Suicidal" in classes else -1
        if label == "Suicidal" or (suicidal_idx >= 0 and probs[suicidal_idx] > 0.30):
            st.markdown("""
            <div class="crisis">
              <strong>🆘 Crisis Alert — Immediate Help Available</strong>
              If you or someone you know is in danger, please reach out right now:<br>
              <b>iCall (India):</b> 9152987821 &nbsp;|&nbsp;
              <b>Vandrevala Foundation:</b> 1860-2662-345 &nbsp;|&nbsp;
              <b>AASRA:</b> 91-22-27546669
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <b>Disclaimer:</b> MindScan is an academic project for educational purposes only.
  It is <b>not</b> a substitute for professional mental health diagnosis, advice, or treatment.
  If you are experiencing mental health challenges, please consult a qualified professional.
</div>
""", unsafe_allow_html=True)