import streamlit as st
import torch
import time
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from googletrans import Translator

# ── Model ─────────────────────────────────────────────────────────────────────
model_name = "Salesforce/blip-vqa-base"
model_path = "best_vqa_models_NLP.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained(model_name)
model     = BlipForQuestionAnswering.from_pretrained(model_name)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VQA · Multimodal AI", layout="wide", initial_sidebar_state="collapsed")

# ── Theme ─────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

D = st.session_state.theme == "dark"

if D:
    BG          = "#080B12"
    ACCENT      = "#00E5FF"
    ACCENT2     = "#FF6B35"
    ACCENT_DIM  = "rgba(0,229,255,0.14)"
    ACCENT_DIM2 = "rgba(255,107,53,0.12)"
    BORDER      = "rgba(0,229,255,0.13)"
    BORDER2     = "rgba(255,255,255,0.06)"
    TEXT_H      = "#FFFFFF"
    TEXT_B      = "#A8B5C8"
    TEXT_DIM    = "rgba(168,181,200,0.45)"
    TEXT_MONO   = "rgba(0,229,255,0.65)"
    GLOW        = "rgba(0,229,255,0.18)"
    GLOW2       = "rgba(255,107,53,0.15)"
    BTN_BG      = "#00E5FF"
    BTN_TXT     = "#080B12"
    BTN_HOV     = "#33EAFF"
    PILL_BG     = "rgba(0,229,255,0.07)"
    ANSWER_BG   = "rgba(0,229,255,0.05)"
    NAV_BG      = "rgba(8,11,18,0.88)"
    FOOT_BG     = "rgba(8,11,18,0.7)"
    BADGE_CLR   = "#FF6B35"
    BADGE_BG    = "rgba(255,107,53,0.1)"
    PIPE_BG     = "rgba(13,17,23,0.7)"
    UPLOAD_BG   = "rgba(0,229,255,0.025)"
    TGL_ICON    = "○"
    TGL_LABEL   = "Light Mode"
    IMG_SHADOW  = "rgba(0,0,0,0.55)"
    HERO_WM     = "rgba(0,229,255,0.04)"
    STEP2_CLR   = "#FF6B35"
    STEP2_BG    = "rgba(255,107,53,0.12)"
    STEP2_BR    = "rgba(255,107,53,0.2)"
    STEP2_GLOW  = "rgba(255,107,53,0.18)"
else:
    BG          = "#F2F5FB"
    ACCENT      = "#0057FF"
    ACCENT2     = "#E84E00"
    ACCENT_DIM  = "rgba(0,87,255,0.1)"
    ACCENT_DIM2 = "rgba(232,78,0,0.08)"
    BORDER      = "rgba(0,87,255,0.14)"
    BORDER2     = "rgba(0,0,0,0.07)"
    TEXT_H      = "#04101F"
    TEXT_B      = "#3A4B60"
    TEXT_DIM    = "rgba(58,75,96,0.5)"
    TEXT_MONO   = "rgba(0,87,255,0.6)"
    GLOW        = "rgba(0,87,255,0.1)"
    GLOW2       = "rgba(232,78,0,0.1)"
    BTN_BG      = "#0057FF"
    BTN_TXT     = "#FFFFFF"
    BTN_HOV     = "#0046D0"
    PILL_BG     = "rgba(0,87,255,0.06)"
    ANSWER_BG   = "rgba(0,87,255,0.04)"
    NAV_BG      = "rgba(242,245,251,0.9)"
    FOOT_BG     = "rgba(242,245,251,0.8)"
    BADGE_CLR   = "#E84E00"
    BADGE_BG    = "rgba(232,78,0,0.08)"
    PIPE_BG     = "rgba(225,230,240,0.8)"
    UPLOAD_BG   = "rgba(0,87,255,0.02)"
    TGL_ICON    = "●"
    TGL_LABEL   = "Dark Mode"
    IMG_SHADOW  = "rgba(0,87,255,0.1)"
    HERO_WM     = "rgba(0,87,255,0.04)"
    STEP2_CLR   = "#E84E00"
    STEP2_BG    = "rgba(232,78,0,0.08)"
    STEP2_BR    = "rgba(232,78,0,0.2)"
    STEP2_GLOW  = "rgba(232,78,0,0.12)"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, .stApp {{
    background-color: {BG} !important;
    color: {TEXT_B} !important;
    font-family: 'Syne', sans-serif;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

/* Ambient glow bg */
.stApp::before {{
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 700px 500px at 10% 15%, {GLOW}, transparent),
        radial-gradient(ellipse 500px 400px at 90% 80%, {GLOW2}, transparent);
    pointer-events: none; z-index: 0;
    animation: ambientDrift 14s ease-in-out infinite alternate;
}}
@keyframes ambientDrift {{
    0%   {{ opacity: 0.6; }}
    100% {{ opacity: 1; }}
}}

/* ── NAV ── */
.vqa-nav {{
    position: relative; z-index: 20;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 52px; height: 58px;
    border-bottom: 1px solid {BORDER2};
    background: {NAV_BG};
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}}
.nav-brand {{ display: flex; align-items: center; gap: 14px; }}
.nav-logomark {{
    width: 34px; height: 34px;
    border: 2px solid {ACCENT}; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 500;
    color: {ACCENT}; letter-spacing: -0.05em;
    box-shadow: 0 0 14px {GLOW};
}}
.nav-name {{
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700;
    letter-spacing: 0.08em; color: {TEXT_H}; text-transform: uppercase;
}}
.nav-pills {{ display: flex; gap: 10px; align-items: center; }}
.nav-pill {{
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 0.15em; color: {TEXT_MONO};
    padding: 4px 10px; border-radius: 4px;
    background: {PILL_BG}; border: 1px solid {BORDER};
    text-transform: uppercase;
}}

/* ── PIPELINE ── */
.pipeline {{
    position: relative; z-index: 10;
    display: flex; align-items: center; justify-content: center;
    gap: 4px; padding: 18px 52px;
    border-bottom: 1px solid {BORDER2};
    background: {PIPE_BG};
    backdrop-filter: blur(8px); overflow: hidden;
}}
.pipeline::before {{
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(
        90deg, transparent 0, transparent 44px,
        {BORDER2} 44px, {BORDER2} 45px
    );
    opacity: 0.5;
}}
.p-node {{
    position: relative; z-index: 2;
    display: flex; flex-direction: column; align-items: center; gap: 7px;
    padding: 10px 22px;
}}
.p-dot {{
    width: 9px; height: 9px; border-radius: 50%;
    background: {ACCENT};
    box-shadow: 0 0 0 4px {ACCENT_DIM}, 0 0 14px {GLOW};
    animation: pdot 2.6s ease-in-out infinite;
}}
.p-dot.o {{ background: {ACCENT2}; box-shadow: 0 0 0 4px {ACCENT_DIM2}, 0 0 14px {GLOW2}; }}
@keyframes pdot {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50%       {{ transform: scale(1.4); opacity: 0.65; }}
}}
.p-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 0.22em; text-transform: uppercase; color: {TEXT_MONO};
}}
.p-arrow {{
    position: relative; z-index: 2;
    color: {BORDER}; font-size: 16px;
    padding-bottom: 16px; flex-shrink: 0; margin: 0 2px;
}}

/* ── HERO ── */
.hero {{
    position: relative; z-index: 5;
    padding: 54px 52px 44px;
    border-bottom: 1px solid {BORDER2};
    overflow: hidden;
}}
.hero::after {{
    content: 'VQA';
    position: absolute; right: 40px; top: 50%; transform: translateY(-50%);
    font-family: 'Syne', sans-serif; font-size: 200px; font-weight: 800;
    line-height: 1; color: {HERO_WM};
    pointer-events: none; user-select: none; letter-spacing: -0.06em;
}}
.hero-kicker {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.38em; text-transform: uppercase; color: {ACCENT};
    margin-bottom: 20px; display: flex; align-items: center; gap: 12px;
}}
.hero-kicker::after {{
    content: ''; flex: 0 0 52px; height: 1px; background: {ACCENT}; opacity: 0.5;
}}
.hero-h1 {{
    font-family: 'Syne', sans-serif; font-size: clamp(38px, 5.5vw, 78px);
    font-weight: 800; line-height: 0.92; color: {TEXT_H};
    letter-spacing: -0.04em; margin-bottom: 6px;
}}
.hero-h1 span {{
    color: {ACCENT};
    font-style: italic; font-family: 'Instrument Serif', serif; font-weight: 400;
}}
.hero-p {{
    max-width: 540px; font-size: 13.5px; line-height: 1.78; color: {TEXT_DIM};
    margin-top: 22px; letter-spacing: 0.01em;
}}
.hero-stats {{
    display: flex; gap: 44px; margin-top: 34px;
}}
.h-stat {{ display: flex; flex-direction: column; gap: 5px; }}
.h-num {{
    font-family: 'Syne', sans-serif; font-size: 30px; font-weight: 800;
    color: {TEXT_H}; letter-spacing: -0.05em; line-height: 1;
}}
.h-num s {{ color: {ACCENT}; text-decoration: none; }}
.h-lbl {{
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 0.22em; text-transform: uppercase; color: {TEXT_DIM};
}}
.tag-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 28px; }}
.tag {{
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 0.15em; text-transform: uppercase;
    padding: 5px 12px; border-radius: 4px;
    border: 1px solid {BORDER}; color: {TEXT_MONO}; background: {PILL_BG};
}}
.tag.o {{ border-color: {ACCENT_DIM2}; color: {BADGE_CLR}; background: {BADGE_BG}; }}

/* ── WORKSPACE ── */
.ws-col {{ position: relative; z-index: 5; padding: 40px 52px 52px; }}
.ws-col.right {{ border-left: 1px solid {BORDER2}; }}

/* ── Step header ── */
.s-head {{ display: flex; align-items: flex-start; gap: 16px; margin-bottom: 26px; }}
.s-num {{
    flex-shrink: 0; width: 42px; height: 42px;
    border: 1px solid {BORDER}; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500;
    color: {ACCENT}; background: {ACCENT_DIM};
    box-shadow: 0 0 14px {GLOW};
}}
.s-num.two {{
    color: {STEP2_CLR}; background: {STEP2_BG};
    border-color: {STEP2_BR}; box-shadow: 0 0 14px {STEP2_GLOW};
}}
.s-info {{ padding-top: 7px; }}
.s-title {{
    font-family: 'Syne', sans-serif; font-size: 17px; font-weight: 700;
    color: {TEXT_H}; letter-spacing: 0.01em;
}}
.s-desc {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.15em; color: {TEXT_DIM}; text-transform: uppercase;
    margin-top: 2px;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{ background: transparent !important; }}
[data-testid="stFileUploader"] > div {{ background: transparent !important; border: none !important; }}
[data-testid="stFileUploadDropzone"] {{
    background: {UPLOAD_BG} !important;
    border: 1px dashed {BORDER} !important;
    border-radius: 12px !important;
    padding: 44px 32px !important;
    transition: all 0.3s !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {ACCENT} !important; background: {ACCENT_DIM} !important;
    box-shadow: 0 0 28px {GLOW} !important;
}}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {{
    color: {TEXT_MONO} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 0.1em !important;
}}
[data-testid="stFileUploadDropzone"] button {{
    background: {ACCENT_DIM} !important; border: 1px solid {BORDER} !important;
    color: {ACCENT} !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.18em !important;
    border-radius: 6px !important; padding: 8px 22px !important; transition: all 0.25s !important;
}}
[data-testid="stFileUploadDropzone"] button:hover {{
    background: {BORDER} !important; box-shadow: 0 0 14px {GLOW} !important;
}}

/* ── Image ── */
.stImage img {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
    box-shadow: 0 6px 28px {IMG_SHADOW} !important;
}}
.stImage > div > div > p {{
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 0.15em !important; color: {TEXT_MONO} !important;
    text-align: center !important; margin-top: 10px !important;
}}

/* ── Textarea ── */
.stTextArea textarea {{
    background: {UPLOAD_BG} !important;
    border: 1px solid {BORDER} !important; border-radius: 10px !important;
    color: {TEXT_H} !important;
    font-family: 'Instrument Serif', serif !important;
    font-size: 20px !important; line-height: 1.65 !important;
    caret-color: {ACCENT} !important; padding: 20px 22px !important;
    resize: none !important; transition: all 0.25s !important;
}}
.stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_DIM}, 0 0 24px {GLOW} !important;
    outline: none !important;
}}
.stTextArea textarea::placeholder {{ color: {TEXT_DIM} !important; font-style: italic; }}
.stTextArea label {{
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
    letter-spacing: 0.28em !important; color: {TEXT_MONO} !important;
    text-transform: uppercase !important; margin-bottom: 8px !important;
}}

/* ── Primary button ── */
[data-testid="stBaseButton-primary"] {{
    background: {BTN_BG} !important; border: none !important;
    border-radius: 8px !important; color: {BTN_TXT} !important;
    font-family: 'Syne', sans-serif !important; font-size: 12px !important;
    font-weight: 700 !important; letter-spacing: 0.22em !important;
    text-transform: uppercase !important; padding: 14px 40px !important;
    width: 100% !important; transition: all 0.25s !important; margin-top: 16px !important;
    box-shadow: 0 0 0 1px {ACCENT_DIM}, 0 6px 22px {GLOW} !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
    background: {BTN_HOV} !important; transform: translateY(-2px) !important;
    box-shadow: 0 0 0 2px {ACCENT_DIM}, 0 10px 34px {GLOW} !important;
}}
[data-testid="stBaseButton-primary"]:active {{ transform: translateY(0) !important; }}

/* ── Secondary button (toggle) ── */
[data-testid="stBaseButton-secondary"] {{
    background: transparent !important; border: 1px solid {BORDER} !important;
    color: {TEXT_B} !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.2em !important;
    padding: 6px 18px !important; border-radius: 20px !important;
    transition: all 0.25s !important; white-space: nowrap !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{
    border-color: {ACCENT} !important; color: {ACCENT} !important;
    background: {ACCENT_DIM} !important;
}}

/* ── Answer ── */
.ans-wrap {{
    margin-top: 28px; border-radius: 12px;
    background: {ANSWER_BG}; border: 1px solid {BORDER};
    overflow: hidden;
    animation: ansIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
    box-shadow: 0 0 40px {GLOW};
}}
@keyframes ansIn {{
    from {{ opacity: 0; transform: translateY(18px) scale(0.97); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
.ans-head {{
    padding: 12px 20px; border-bottom: 1px solid {BORDER2};
    display: flex; align-items: center; gap: 10px;
    background: {ACCENT_DIM};
}}
.ans-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: {ACCENT}; box-shadow: 0 0 8px {GLOW};
    animation: pdot 1.6s ease infinite;
}}
.ans-tag {{
    font-family: 'IBM Plex Mono', monospace; font-size: 9px;
    letter-spacing: 0.28em; color: {ACCENT}; text-transform: uppercase;
}}
.ans-body {{ padding: 26px 26px 30px; }}
.ans-text {{
    font-family: 'Instrument Serif', serif; font-size: 27px;
    line-height: 1.42; color: {TEXT_H}; letter-spacing: 0.01em;
}}

/* ── Success badge ── */
.ok-badge {{
    display: inline-flex; align-items: center; gap: 8px;
    background: {PILL_BG}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 14px; font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.2em; color: {ACCENT}; margin-bottom: 18px;
}}
.ok-dot {{
    width: 6px; height: 6px; border-radius: 50%; background: {ACCENT};
    animation: pdot 1.8s ease infinite; display: inline-block;
}}

/* ── Empty ── */
.empty-z {{
    border: 1px dashed {BORDER}; border-radius: 12px;
    padding: 60px 32px; text-align: center; background: {UPLOAD_BG};
}}
.empty-ico {{ font-size: 34px; margin-bottom: 14px; opacity: 0.22; }}
.empty-txt {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.22em; color: {TEXT_DIM}; line-height: 2.4; text-transform: uppercase;
}}

/* ── Alerts ── */
.stAlert {{
    background: {PILL_BG} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 0.05em !important;
}}
.stAlert p {{ color: {ACCENT} !important; }}

/* ── Progress ── */
.stProgress > div > div {{
    background: {BORDER} !important; border-radius: 0 !important; height: 2px !important;
}}
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2}) !important;
    border-radius: 0 !important; transition: width 0.35s ease !important;
    box-shadow: 0 0 10px {GLOW} !important;
}}

/* ── Footer ── */
.vqa-foot {{
    position: relative; z-index: 10;
    padding: 18px 52px; border-top: 1px solid {BORDER2};
    display: flex; align-items: center; justify-content: space-between;
    background: {FOOT_BG}; backdrop-filter: blur(12px);
}}
.f-txt {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    letter-spacing: 0.14em; color: {TEXT_DIM};
}}
.f-sep {{ width: 1px; height: 14px; background: {BORDER2}; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}
</style>
""", unsafe_allow_html=True)

# ── NAV ───────────────────────────────────────────────────────────────────────
n1, n2, n3 = st.columns([3, 4, 2])
with n1:
    st.markdown(f"""
    <div class="vqa-nav" style="border-right:none; justify-content:flex-start;">
        <div class="nav-brand">
            <div class="nav-logomark">VQ</div>
            <span class="nav-name">Multimodal VQA</span>
        </div>
    </div>""", unsafe_allow_html=True)
with n2:
    st.markdown(f"""
    <div class="vqa-nav" style="border-left:none; border-right:none; justify-content:center;">
        <div class="nav-pills">
            <span class="nav-pill">CV + NLP Fusion</span>
            <span class="nav-pill">Bangla &amp; English</span>
            <span class="nav-pill">BLIP Architecture</span>
        </div>
    </div>""", unsafe_allow_html=True)
with n3:
    st.markdown(f'<div class="vqa-nav" style="border-left:none; justify-content:flex-end; gap:12px;">', unsafe_allow_html=True)
    st.button(f"{TGL_ICON}  {TGL_LABEL}", key="theme_toggle", on_click=toggle_theme, type="secondary")
    st.markdown("</div>", unsafe_allow_html=True)

# ── PIPELINE STRIP ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="pipeline">
    <div class="p-node"><div class="p-dot"></div><span class="p-label">Image Input</span></div>
    <div class="p-arrow">→</div>
    <div class="p-node"><div class="p-dot o"></div><span class="p-label">Visual Encoder</span></div>
    <div class="p-arrow">→</div>
    <div class="p-node"><div class="p-dot"></div><span class="p-label">Text Encoder</span></div>
    <div class="p-arrow">→</div>
    <div class="p-node"><div class="p-dot o"></div><span class="p-label">Cross-Attn Fusion</span></div>
    <div class="p-arrow">→</div>
    <div class="p-node"><div class="p-dot"></div><span class="p-label">Decode Answer</span></div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-kicker">AI-Complete Multimodal Task</div>
    <h1 class="hero-h1">See.<br><span>Understand.</span><br>Answer.</h1>
    <p class="hero-p">
        Given an image and a natural language question, this system bridges computer vision
        and NLP — through object detection, scene understanding, counting, and semantic
        reasoning — to produce a correct natural language answer.
    </p>
    <div class="hero-stats">
        <div class="h-stat"><div class="h-num">2<s>×</s></div><div class="h-lbl">Modalities Fused</div></div>
        <div class="h-stat"><div class="h-num">∞</div><div class="h-lbl">Question Types</div></div>
        <div class="h-stat"><div class="h-num">bn<s>+en</s></div><div class="h-lbl">Languages Supported</div></div>
    </div>
    <div class="tag-row">
        <span class="tag">Object Detection</span>
        <span class="tag">Scene Classification</span>
        <span class="tag">Spatial Reasoning</span>
        <span class="tag">Counting</span>
        <span class="tag o">Bengali NLP</span>
        <span class="tag o">Cultural Heritage Sites</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── WORKSPACE ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(f"""
    <div class="ws-col">
        <div class="s-head">
            <div class="s-num">01</div>
            <div class="s-info">
                <div class="s-title">Upload Image</div>
                <div class="s-desc">Visual modality input</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    up_prog = st.empty()
    uploaded_file = st.file_uploader("Drop image here or browse files", type=["jpg","png","jpeg"], label_visibility="visible")

    if uploaded_file:
        p = up_prog.progress(0)
        for i in range(0, 101, 20):
            time.sleep(0.07); p.progress(i)
        up_prog.empty()
        st.markdown(f'<div class="ok-badge"><span class="ok-dot"></span>Image loaded into pipeline</div>', unsafe_allow_html=True)

        ip = st.progress(0); ist = st.empty()
        for i in range(0, 101, 20):
            time.sleep(0.05); ip.progress(i); ist.text(f"Processing visual input...  {i}%")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="VISUAL INPUT · READY FOR INFERENCE", use_column_width=True)
        ip.empty(); ist.empty()
    else:
        st.markdown(f"""
        <div class="empty-z">
            <div class="empty-ico">🖼</div>
            <div class="empty-txt">JPG · PNG · JPEG<br>Drop image to begin</div>
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="ws-col right">
        <div class="s-head">
            <div class="s-num two">02</div>
            <div class="s-info">
                <div class="s-title">Ask Your Question</div>
                <div class="s-desc">Natural language query</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    question = st.text_area(
        "Type your question — Bangla or English",
        height=148,
        placeholder="What is happening in this image? · এই ছবিতে কী দেখা যাচ্ছে?",
    )

    translator = Translator()

    if st.button("⟡  RUN INFERENCE", key="ask_btn", type="primary"):
        if uploaded_file is None:
            st.warning("⚠  Please upload an image first.")
        elif not question.strip():
            st.warning("⚠  Please enter a question.")
        else:
            pb = st.progress(0); st_txt = st.empty()
            stages = [
                (20, "Encoding visual features..."),
                (45, "Tokenising language input..."),
                (68, "Running cross-attention fusion..."),
                (88, "Decoding answer tokens..."),
                (100, "Inference complete."),
            ]
            cur = 0
            for val, msg in stages:
                for v in range(cur, val, 4):
                    time.sleep(0.07); pb.progress(v)
                pb.progress(val); st_txt.text(msg); cur = val; time.sleep(0.18)

            detected_lang = translator.detect(question).lang

            if detected_lang == 'en':
                inputs = processor(images=image, text=question, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=200, temperature=0.7)
                pb.empty(); st_txt.empty()
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                st.markdown(f"""
                <div class="ans-wrap">
                    <div class="ans-head"><span class="ans-dot"></span><span class="ans-tag">Response · English · Inference Complete</span></div>
                    <div class="ans-body"><div class="ans-text">{answer}</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                bq = translator.translate(question, src='bn', dest='en')
                inputs = processor(images=image, text=bq.text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=200, temperature=0.7)
                pb.empty(); st_txt.empty()
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                tr = translator.translate(answer, src='en', dest='bn')
                st.markdown(f"""
                <div class="ans-wrap">
                    <div class="ans-head"><span class="ans-dot"></span><span class="ans-tag">উত্তর · বাংলা · Inference Complete</span></div>
                    <div class="ans-body"><div class="ans-text">{tr.text}</div></div>
                </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="vqa-foot">
    <div style="display:flex; align-items:center; gap:18px;">
        <span class="f-txt">© 2025 Multimodal VQA System</span>
        <div class="f-sep"></div>
        <span class="f-txt">BLIP · Fine-tuned on Cultural Heritage Dataset</span>
    </div>
    <span class="f-txt">Tridib Chakraborty · AI/ML Engineer</span>
</div>
""", unsafe_allow_html=True)