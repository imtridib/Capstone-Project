import html
import time

import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from deep_translator import GoogleTranslator
from langdetect import detect as detect_language

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

MODEL_NAME = "Salesforce/blip-vqa-base"
MODEL_WEIGHTS = "best_vqaEnglish_models.pth"
GROQ_MODEL = "llama-3.3-70b-versatile"  # change here if Groq deprecates this model


# ── Cached resources (load ONCE, not on every rerun / theme toggle) ──────────
@st.cache_resource(show_spinner="Loading VQA model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    if device.type == "cuda":
        model.half()  # half precision only pays off on GPU
    model.eval()
    return processor, model, device


@st.cache_resource
def get_groq_client():
    """Reads GROQ_API_KEY from .streamlit/secrets.toml. Returns None if absent/unavailable."""
    if not GROQ_SDK_AVAILABLE:
        return None
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        return None
    return Groq(api_key=api_key)


processor, model, device = load_model()
groq_client = get_groq_client()


def run_vqa(image: Image.Image, question: str) -> str:
    """Talks to the BLIP model, making full use of the processor + device."""
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device.type == "cuda":
        inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_length=200)
    return processor.decode(output_ids[0], skip_special_tokens=True)


def enrich_with_groq(question_en: str, raw_answer: str, target_lang: str):
    """
    Sends the user's question (in English) + the BLIP model's raw (often terse) answer
    to Groq, asking it to reason about what that answer actually means in context and
    produce a properly written response IN THE SAME LANGUAGE the visitor originally
    asked in. Returns (answer_text, used_groq: bool).
    target_lang: "en" or "bn" — the language of the visitor's original question.
    Falls back to a direct/translated answer if Groq isn't configured or the call fails.
    """
    lang_name = "English" if target_lang == "en" else "Bengali (বাংলা)"

    if groq_client is not None:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a heritage guide for Bangladeshi cultural and historical sites. "
                            "You are given a visitor's question about a photo of a heritage site and a "
                            "short, raw label produced by a visual question-answering model. That raw "
                            "label is your only evidence about what is actually in the image — treat it "
                            "as ground truth, but expand it into a natural, informative, well-formed answer "
                            "that directly addresses the visitor's question, adding relevant historical or "
                            "architectural context where you can do so accurately. Do not contradict the "
                            "raw label and do not invent specific facts (names, dates) you aren't given. "
                            f"Write your entire answer in {lang_name} only. "
                            "Respond ONLY with a JSON object with exactly one key, \"answer\", containing "
                            "the complete answer text. No other text."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Visitor's question: {question_en}\nRaw model label: {raw_answer}",
                    },
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            import json
            parsed = json.loads(completion.choices[0].message.content)
            answer = parsed.get("answer", "").strip()
            if answer:
                return answer, True
        except Exception:
            pass  # fall through to translator-based fallback

    # Fallback: no Groq key configured, SDK missing, or the call failed
    if target_lang == "en":
        return raw_answer, False
    return GoogleTranslator(source="en", target="bn").translate(raw_answer), False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heritage VQA · Bangladesh",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Persistent state ──────────────────────────────────────────────────────────
st.session_state.setdefault("theme", "night")
st.session_state.setdefault("last_file_id", None)
st.session_state.setdefault("answer", None)  # (label, answer_text, used_groq)


def toggle_theme():
    st.session_state.theme = "day" if st.session_state.theme == "night" else "night"


is_night = st.session_state.theme == "night"

# ── Theme tokens ──────────────────────────────────────────────────────────────
THEMES = {
    "night": dict(
        BG="#0A0806", BG_GRAD="radial-gradient(ellipse 80% 60% at 50% -10%, rgba(200,170,110,0.08), transparent)",
        ACCENT="#D4AF6A", ACCENT_2="#E8845C", ACCENT_DIM="rgba(212,175,106,0.25)",
        ACCENT_MID="rgba(212,175,106,0.5)", ACCENT_FAINT="rgba(212,175,106,0.07)",
        TEXT_PRI="#F5ECD8", TEXT_DIM="rgba(235,224,204,0.5)", TEXT_MONO="rgba(212,175,106,0.55)",
        BORDER="rgba(212,175,106,0.2)", BORDER_FAINT="rgba(212,175,106,0.1)", GRAIN_OP="0.3",
        ANSWER_BG="linear-gradient(135deg, rgba(212,175,106,0.09), rgba(212,175,106,0.02))",
        UPLOAD_BG="#161209", CARD_BG="rgba(255,255,255,0.015)",
        HERO_WM="rgba(212,175,106,0.05)", BTN_TEXT="#0A0806", BTN_HOVER="#EAC98C",
        SHADOW="0 20px 60px -20px rgba(0,0,0,0.6)", GLOW="0 0 40px rgba(212,175,106,0.12)",
        TOGGLE_ICON="☀", TOGGLE_LABEL="Day Mode",
    ),
    "day": dict(
        BG="#FBF7EF", BG_GRAD="radial-gradient(ellipse 80% 60% at 50% -10%, rgba(139,69,19,0.08), transparent)",
        ACCENT="#7A3B10", ACCENT_2="#A0431A", ACCENT_DIM="rgba(122,59,16,0.32)",
        ACCENT_MID="rgba(122,59,16,0.55)", ACCENT_FAINT="rgba(122,59,16,0.07)",
        TEXT_PRI="#2C1A0E", TEXT_DIM="rgba(44,26,14,0.72)", TEXT_MONO="rgba(122,59,16,0.85)",
        BORDER="rgba(122,59,16,0.35)", BORDER_FAINT="rgba(122,59,16,0.18)", GRAIN_OP="0.15",
        ANSWER_BG="linear-gradient(135deg, rgba(122,59,16,0.1), rgba(122,59,16,0.03))",
        UPLOAD_BG="#F4EBDC", CARD_BG="rgba(122,59,16,0.03)",
        HERO_WM="rgba(122,59,16,0.14)", BTN_TEXT="#FBF7EF", BTN_HOVER="#8F4014",
        SHADOW="0 20px 60px -20px rgba(139,69,19,0.25)", GLOW="0 0 40px rgba(139,69,19,0.1)",
        TOGGLE_ICON="☽", TOGGLE_LABEL="Night Mode",
    ),
}
T = THEMES["night" if is_night else "day"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Josefin+Sans:wght@200;300;400;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, .stApp {{
    background-color: {T['BG']} !important;
    background-image: {T['BG_GRAD']} !important;
    color: {T['TEXT_PRI']} !important;
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    transition: background-color 0.2s ease, color 0.2s ease;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: {T['GRAIN_OP']};
}}

/* ── Topbar ── */
.heritage-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 52px 16px;
    border-bottom: 1px solid {T['BORDER']};
}}
.topbar-left {{ display: flex; align-items: center; gap: 20px; }}
.topbar-mark {{
    width: 34px; height: 34px;
    border: 1.5px solid {T['ACCENT']};
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 17px; font-weight: 700;
    color: {T['ACCENT']};
    background: {T['ACCENT_FAINT']};
    transform: rotate(45deg);
}}
.topbar-mark span {{ transform: rotate(-45deg); }}
.topbar-eyebrow {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.35em;
    color: {T['ACCENT']};
    text-transform: uppercase;
    font-weight: 500;
}}
.topbar-badge {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: {T['ACCENT_MID']};
    border: 1px solid {T['ACCENT_DIM']};
    padding: 4px 12px;
    border-radius: 20px;
    margin-left: 4px;
}}

[data-testid="stBaseButton-secondary"] {{
    background: {T['ACCENT_FAINT']} !important;
    border: 1px solid {T['ACCENT_DIM']} !important;
    color: {T['ACCENT']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.22em !important;
    padding: 8px 20px !important;
    border-radius: 20px !important;
    transition: all 0.15s !important;
    font-weight: 500 !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{
    background: {T['ACCENT_DIM']} !important;
    border-color: {T['ACCENT']} !important;
    box-shadow: {T['GLOW']} !important;
}}

/* ── Hero ── */
.heritage-hero {{
    padding: 68px 52px 40px;
    border-bottom: 1px solid {T['BORDER_FAINT']};
    position: relative;
    overflow: hidden;
}}
.heritage-hero::after {{
    content: 'বাংলাদেশ';
    position: absolute;
    right: 30px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Cormorant Garamond', serif;
    font-size: 110px;
    font-weight: 300;
    color: {T['HERO_WM']};
    pointer-events: none;
    line-height: 1;
}}
.capstone-strip {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 30px;
    padding-bottom: 24px;
    border-bottom: 1px dashed {T['BORDER_FAINT']};
}}
.capstone-kicker {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {T['BTN_TEXT']};
    background: linear-gradient(135deg, {T['ACCENT']}, {T['ACCENT_2']});
    padding: 6px 14px;
    border-radius: 20px;
    flex-shrink: 0;
    box-shadow: {T['GLOW']};
}}
.capstone-name {{
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-weight: 400;
    font-size: 16px;
    color: {T['TEXT_DIM']};
    line-height: 1.5;
    max-width: 620px;
}}
.hero-label {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.4em;
    color: {T['ACCENT']};
    text-transform: uppercase;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 500;
}}
.hero-label::before {{
    content: '';
    display: inline-block;
    width: 32px;
    height: 1px;
    background: linear-gradient(90deg, {T['ACCENT']}, transparent);
}}
.hero-title {{
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(32px, 4.4vw, 60px);
    font-weight: 600;
    line-height: 1.12;
    color: {T['TEXT_PRI']};
    max-width: 680px;
    margin-bottom: 16px;
    letter-spacing: -0.01em;
}}
.hero-title em {{
    font-style: italic;
    font-weight: 600;
    background: linear-gradient(120deg, {T['ACCENT']}, {T['ACCENT_2']});
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hero-subtitle {{
    font-size: 14px;
    letter-spacing: 0.03em;
    color: {T['TEXT_DIM']};
    max-width: 500px;
    line-height: 1.8;
    margin-bottom: 26px;
}}
.hero-chips {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.hero-chip {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: {T['TEXT_MONO']};
    border: 1px solid {T['BORDER']};
    background: {T['CARD_BG']};
    padding: 7px 14px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}
.hero-chip .dot {{
    width: 5px; height: 5px; border-radius: 50%;
    background: {T['ACCENT']};
    box-shadow: 0 0 8px {T['ACCENT']};
}}

/* ── Panel headers ── */
.panel-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 22px;
    padding-top: 44px;
}}
.panel-number {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {T['BTN_TEXT']};
    background: linear-gradient(135deg, {T['ACCENT']}, {T['ACCENT_2']});
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    flex-shrink: 0;
    font-weight: 500;
    box-shadow: {T['GLOW']};
}}
.panel-title {{
    font-family: 'Josefin Sans', sans-serif;
    font-size: 12px;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: {T['TEXT_PRI']};
    font-weight: 400;
}}
.panel-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.05em;
    color: {T['TEXT_MONO']};
    margin-left: auto;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{ background: transparent !important; }}
[data-testid="stFileUploader"] > div {{ background: transparent !important; border: none !important; }}
[data-testid="stFileUploadDropzone"] {{
    background: {T['UPLOAD_BG']} !important;
    border: 1.5px dashed {T['ACCENT_DIM']} !important;
    border-radius: 10px !important;
    padding: 40px !important;
    transition: all 0.25s !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {T['ACCENT_MID']} !important;
    background: {T['ACCENT_FAINT']} !important;
    box-shadow: {T['GLOW']} !important;
}}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {{
    color: {T['TEXT_MONO']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
}}
[data-testid="stFileUploadDropzone"] button {{
    background: transparent !important;
    border: 1px solid {T['ACCENT_DIM']} !important;
    color: {T['ACCENT']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    border-radius: 20px !important;
    padding: 8px 20px !important;
}}
[data-testid="stFileUploadDropzone"] button:hover {{
    background: {T['ACCENT_DIM']} !important;
    border-color: {T['ACCENT']} !important;
}}

/* ── Progress bars ── */
.stProgress {{
    margin: 4px 0 2px 0 !important;
}}
.stProgress > div > div {{
    background: {T['BORDER_FAINT']} !important;
    border-radius: 20px !important;
    height: 6px !important;
}}
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {T['ACCENT']}, {T['ACCENT_2']}) !important;
    border-radius: 20px !important;
    transition: width 0.15s ease !important;
    box-shadow: {T['GLOW']} !important;
}}
.stProgress p {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    color: {T['TEXT_MONO']} !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}}

/* ── Image card ── */
[data-testid="stImage"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid {T['BORDER']} !important;
    box-shadow: {T['SHADOW']} !important;
}}

/* ── Textarea ── */
.stTextArea textarea {{
    background: {T['UPLOAD_BG']} !important;
    border: 1.5px solid {T['ACCENT_DIM']} !important;
    border-radius: 10px !important;
    color: {T['TEXT_PRI']} !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 19px !important;
    font-weight: 400 !important;
    line-height: 1.7 !important;
    caret-color: {T['ACCENT']} !important;
    padding: 20px !important;
    resize: none !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
.stTextArea textarea:focus {{
    border-color: {T['ACCENT_MID']} !important;
    box-shadow: 0 0 0 3px {T['ACCENT_FAINT']} !important;
    outline: none !important;
}}
.stTextArea textarea::placeholder {{ color: {T['ACCENT_DIM']} !important; }}
.stTextArea label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.3em !important;
    color: {T['TEXT_MONO']} !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}}

/* ── Analyse button ── */
[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {T['ACCENT']}, {T['ACCENT_2']}) !important;
    border: none !important;
    border-radius: 10px !important;
    color: {T['BTN_TEXT']} !important;
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.32em !important;
    text-transform: uppercase !important;
    padding: 16px 40px !important;
    width: 100% !important;
    transition: all 0.25s !important;
    margin-top: 16px !important;
    box-shadow: {T['GLOW']} !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px {T['ACCENT_DIM']} !important;
    filter: brightness(1.08);
}}

/* ── Answer card ── */
.answer-box {{
    padding: 24px 26px;
    border: 1px solid {T['BORDER']};
    background: {T['ANSWER_BG']};
    border-radius: 12px;
    box-shadow: {T['SHADOW']};
    margin-top: 24px;
    animation: fadeSlide 0.35s ease forwards;
}}
@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.answer-label {{
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.35em;
    color: {T['ACCENT']};
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
}}
.answer-label::before {{ content: '◆'; font-size: 8px; }}
.answer-text {{
    font-family: 'Cormorant Garamond', serif;
    font-size: 21px;
    font-weight: 400;
    color: {T['TEXT_PRI']};
    line-height: 1.55;
}}
.groq-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 24px;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    color: {T['ACCENT']};
    text-transform: uppercase;
    opacity: 0.85;
}}

/* ── Success pill ── */
.success-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {T['ANSWER_BG']};
    border: 1px solid {T['ACCENT_DIM']};
    border-radius: 20px;
    padding: 7px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: {T['ACCENT']};
    margin-bottom: 16px;
}}

/* ── Empty placeholder ── */
.empty-zone {{
    border: 1.5px dashed {T['ACCENT_DIM']};
    border-radius: 12px;
    padding: 68px 32px;
    text-align: center;
    background: {T['UPLOAD_BG']};
}}
.empty-icon {{
    font-size: 30px;
    color: {T['ACCENT']};
    opacity: 0.4;
    margin-bottom: 14px;
}}
.empty-text {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: {T['TEXT_MONO']};
    line-height: 2.2;
}}

/* ── Alerts ── */
.stAlert {{
    background: {T['ANSWER_BG']} !important;
    border: 1px solid {T['ACCENT_DIM']} !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
}}
.stAlert p {{ color: {T['ACCENT']} !important; }}

/* ── Image caption ── */
.stImage > div > div > p {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    color: {T['TEXT_MONO']} !important;
    text-align: center !important;
    margin-top: 12px !important;
}}

/* ── Col2 left border ── */
[data-testid="column"]:nth-child(2) {{
    border-left: 1px solid {T['BORDER_FAINT']} !important;
    padding-left: 48px !important;
}}

/* ── Footer ── */
.heritage-footer {{
    padding: 26px 52px;
    border-top: 1px solid {T['BORDER_FAINT']};
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 48px;
}}
.footer-text {{
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    letter-spacing: 0.1em;
    color: {T['TEXT_MONO']};
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {T['BG']}; }}
::-webkit-scrollbar-thumb {{ background: {T['ACCENT_DIM']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T['ACCENT_MID']}; }}
</style>
""", unsafe_allow_html=True)

# ── Topbar ────────────────────────────────────────────────────────────────────
tb1, tb2, tb3 = st.columns([5, 1.5, 1.5])
with tb1:
    st.markdown("""
    <div class="heritage-topbar" style="border-right: none; padding-right: 0;">
        <div class="topbar-left">
            <div class="topbar-mark"><span>H</span></div>
            <span class="topbar-eyebrow">Heritage Intelligence System</span>
            <span class="topbar-badge">VQA · Groq · NLP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with tb3:
    st.markdown(f"<div style='padding-top: 10px; padding-bottom: 10px; border-bottom: 1px solid {T['BORDER']};'>", unsafe_allow_html=True)
    st.button(f"{T['TOGGLE_ICON']}  {T['TOGGLE_LABEL']}", key="theme_toggle", on_click=toggle_theme, type="secondary")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
groq_ready = groq_client is not None
groq_chip = "Groq contextual reasoning" if groq_ready else "Groq not configured"
st.markdown(f"""
<div class="heritage-hero">
    <div class="capstone-strip">
        <span class="capstone-kicker">Capstone Project</span>
        <span class="capstone-name">A Multilingual Visual Question Answering System for Cultural Heritage Sites of Bangladesh</span>
    </div>
    <div class="hero-label">Multilingual Visual Question Answering</div>
    <h1 class="hero-title">
        Uncover the <em>stories</em> behind Bangladesh's Cultural Sites<br>with AI-powered Image-Text Understanding
    </h1>
    <p class="hero-subtitle">
        Upload an image of a heritage site and ask anything — in Bangla or English.
        A vision model reads the image, and Groq turns that reading into a full,
        context-aware answer in both languages.
    </p>
    <div class="hero-chips">
        <span class="hero-chip"><span class="dot"></span>BLIP VQA</span>
        <span class="hero-chip"><span class="dot"></span>{groq_chip}</span>
        <span class="hero-chip"><span class="dot"></span>English + বাংলা</span>
        <span class="hero-chip"><span class="dot"></span>{"GPU" if device.type == "cuda" else "CPU"} inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not groq_ready:
    st.info(
        "Groq isn't configured yet — answers will use the raw model output translated with "
        "Google Translate as a fallback. Add a `.streamlit/secrets.toml` file next to `app.py` "
        "containing `GROQ_API_KEY = \"your-key\"` to enable Groq's contextual answers.",
        icon="⚠️",
    )

# ── Workspace ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

image = None

with col1:
    st.markdown("""
    <div class="panel-header">
        <div class="panel-number">01</div>
        <div class="panel-title">Upload Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop image here or browse files",
        type=["jpg", "png", "jpeg"],
        label_visibility="visible",
    )

    if uploaded_file:
        is_new_upload = uploaded_file.file_id != st.session_state.last_file_id
        if is_new_upload:
            st.session_state.last_file_id = uploaded_file.file_id
            st.session_state.answer = None  # clear stale answer for the new image

            load_bar = st.progress(0, text="Loading image… 0%")
            for pct in range(0, 101, 10):
                time.sleep(0.05)
                load_bar.progress(pct, text=f"Loading image… {pct}%")
            load_bar.empty()

        image = Image.open(uploaded_file).convert("RGB")
        if is_new_upload:
            st.markdown('<div class="success-pill">✦ &nbsp;Image received</div>', unsafe_allow_html=True)
        st.image(image, caption="UPLOADED ARTIFACT · click to enlarge", width=340)
    else:
        st.session_state.last_file_id = None
        st.markdown("""
        <div class="empty-zone">
            <div class="empty-icon">◈</div>
            <div class="empty-text">JPG · PNG · JPEG<br>awaiting image</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="panel-header">
        <div class="panel-number">02</div>
        <div class="panel-title">Ask a Question</div>
    </div>
    """, unsafe_allow_html=True)

    # Keying the widget to the current image's file_id makes it remount empty
    # whenever a new image is uploaded, instead of keeping the old question around.
    question_key = f"question_input_{st.session_state.last_file_id}"
    question = st.text_area(
        "Your question — Bangla or English",
        height=140,
        placeholder="e.g. What is the architectural style of this monument? · এই স্থানটি কোথায় অবস্থিত?",
        key=question_key,
    )

    if st.button("ANALYSE IMAGE", key="ask_btn", type="primary"):
        if image is None:
            st.warning("Please upload an image to proceed.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            bar1 = st.progress(0, text="Reading the image… 0%")
            for pct in range(0, 71, 10):
                time.sleep(0.05)
                bar1.progress(pct, text=f"Reading the image… {pct}%")

            try:
                detected_lang = detect_language(question)
            except Exception:
                detected_lang = "en"  # langdetect can fail on very short strings

            # Only English vs Bengali are supported as answer languages; anything
            # else detected still gets translated to English for the model,
            # but the answer defaults to English rather than a random language.
            target_lang = "bn" if detected_lang == "bn" else "en"

            en_question = (
                question if detected_lang == "en"
                else GoogleTranslator(source=detected_lang, target="en").translate(question)
            )
            raw_answer = run_vqa(image, en_question)
            bar1.progress(100, text="Reading the image… 100%")
            time.sleep(0.1)
            bar1.empty()

            bar2 = st.progress(0, text="Building a contextual answer… 0%")
            for pct in range(0, 71, 10):
                time.sleep(0.05)
                bar2.progress(pct, text=f"Building a contextual answer… {pct}%")

            answer_text, used_groq = enrich_with_groq(en_question, raw_answer, target_lang)
            label = "Response · English" if target_lang == "en" else "উত্তর · বাংলা"
            st.session_state.answer = (label, answer_text, used_groq)

            bar2.progress(100, text="Building a contextual answer… 100%")
            time.sleep(0.1)
            bar2.empty()

    if st.session_state.answer:
        label, answer_text, used_groq = st.session_state.answer
        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">{html.escape(label)}</div>
            <div class="answer-text">{html.escape(answer_text)}</div>
        </div>
        """, unsafe_allow_html=True)
        if used_groq:
            st.markdown('<div class="groq-tag">✦ Enriched with Groq contextual reasoning</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="heritage-footer">
    <span class="footer-text">© 2025 Heritage Intelligence · Bangladesh</span>
    <span class="footer-text">Developed by Tridib Chakraborty · AI/ML Engineer</span>
</div>
""", unsafe_allow_html=True)
