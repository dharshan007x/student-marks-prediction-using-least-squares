"""
streamlit_app.py
================
Modern Interactive ML Education Dashboard for the
Student Marks Prediction Using Least Squares project.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_dataset, validate_dataset, get_features_and_target,
    split_data, FEATURE_COLUMNS, TARGET_COLUMN,
)
from src.least_squares import LeastSquaresRegressor
from src.evaluation import full_evaluation, residual_table, residual_statistics
from src.utils import get_data_path, predict_single_student

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(
    page_title="Student Marks Prediction | Least Squares",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# GLOBAL CSS DESIGN SYSTEM
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ---------- Core Reset & Base ---------- */
*, *::before, *::after {
    box-sizing: border-box;
}

html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1E293B;
}

/* Ensure content never collides with top header / 3-dot menu icon */
header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.94) !important;
    backdrop-filter: blur(8px) !important;
    height: 3.5rem !important;
    z-index: 99 !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
}

.block-container {
    padding-top: 5.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px;
}

/* Ensure code, math, and icon fonts are NEVER corrupted by typography rules */
code, pre, .stCodeBlock, [class*="stCode"] {
    font-family: 'Fira Code', 'Consolas', 'Courier New', monospace !important;
}

.katex, .katex *, .katex-display, .katex-display * {
    font-family: KaTeX_Main, KaTeX_Math, 'Times New Roman', serif !important;
}

.katex-display {
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.6rem 0.2rem;
    margin: 0.8rem 0;
}

/* CRITICAL: Protect Material Symbols & Streamlit Icons from ligature text corruption */
[data-testid="stIconMaterial"],
[data-testid*="Icon"],
[data-testid*="icon"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
[class*="material-symbols"],
[class*="material-icons"],
span[class*="e1vmumty0"],
[data-testid="stExpander"] summary span,
[data-testid="stExpanderStepChevron"] span,
header[data-testid="stHeader"] button span {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-feature-settings: 'liga' 1 !important;
    -webkit-font-feature-settings: 'liga' 1 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* ---------- Expander & Selectbox Arrow Down Spacing ---------- */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    margin-bottom: 0.85rem !important;
    background: #ffffff !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    cursor: pointer !important;
    padding: 0.85rem 1.2rem !important;
}

[data-testid="stExpander"] summary:hover {
    background-color: #F8FAFC !important;
}

/* Selectbox dropdown & arrow down spacing */
div[data-baseweb="select"] {
    width: 100% !important;
}

div[data-baseweb="select"] > div {
    padding-right: 2.6rem !important;
    min-height: 44px !important;
    border-radius: 10px !important;
}

/* ---------- Sidebar Styling ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0F172A 0%, #1E293B 55%, #0F172A 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

div[data-testid="stSidebarUserContent"] {
    padding-top: 2rem !important;
}

[data-testid="stSidebarCollapseButton"] {
    position: relative !important;
    z-index: 101 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #CBD5E1;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    padding: 0.45rem 0.75rem !important;
    border-radius: 9px !important;
    transition: background 0.15s ease, color 0.15s ease;
    margin-bottom: 3px !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.55rem !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p {
    margin: 0 !important;
    font-size: 0.90rem !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    white-space: nowrap !important;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.06) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.10) !important;
    margin: 0.8rem 0;
}

/* ---------- Hero Banners ---------- */
.hero-wrap {
    background: linear-gradient(135deg, #1E3A8A 0%, #6D28D9 50%, #2563EB 100%);
    border-radius: 18px;
    padding: 2.2rem 2rem;
    text-align: center;
    color: #ffffff;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 30px rgba(37, 99, 235, 0.22);
    position: relative;
    overflow: hidden;
}

.hero-wrap::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 140%; height: 140%;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
    line-height: 1.25;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.20);
    position: relative;
    z-index: 1;
}

.hero-sub {
    font-size: 1.1rem;
    font-weight: 500;
    opacity: 0.92;
    margin: 0 0 0.8rem 0;
    line-height: 1.4;
    color: #E0E7FF !important;
    position: relative;
    z-index: 1;
}

.hero-desc {
    font-size: 0.94rem;
    opacity: 0.85;
    max-width: 720px;
    margin: 0 auto;
    line-height: 1.6;
    color: #F1F5F9 !important;
    position: relative;
    z-index: 1;
}

/* Secondary Page Header */
.hero-page-header {
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    color: #ffffff;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}
.hero-page-header h2 {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    line-height: 1.25;
    color: #ffffff !important;
}
.hero-page-header p {
    font-size: 0.90rem;
    margin: 0;
    opacity: 0.90;
    line-height: 1.45;
    color: #F8FAFC !important;
}

/* ---------- Metric Card ---------- */
.m-card {
    border-radius: 15px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    border: 1px solid rgba(0,0,0,0.04);
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    min-height: 118px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-sizing: border-box;
}

.m-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.09);
}

.m-icon {
    font-size: 1.65rem;
    line-height: 1;
    margin-bottom: 0.35rem;
}

.m-val {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0;
    word-break: break-all;
}

.m-lbl {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-top: 0.35rem;
    line-height: 1.3;
    opacity: 0.85;
}

/* Color variants */
.mc-blue   { background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); }
.mc-blue .m-val   { color: #1E40AF; }
.mc-blue .m-lbl   { color: #1E3A8A; }

.mc-purple { background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%); }
.mc-purple .m-val { color: #6D28D9; }
.mc-purple .m-lbl { color: #5B21B6; }

.mc-cyan   { background: linear-gradient(135deg, #ECFEFF 0%, #CFFAFE 100%); }
.mc-cyan .m-val   { color: #0E7490; }
.mc-cyan .m-lbl   { color: #155E75; }

.mc-green  { background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); }
.mc-green .m-val  { color: #047857; }
.mc-green .m-lbl  { color: #065F46; }

.mc-orange { background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); }
.mc-orange .m-val { color: #B45309; }
.mc-orange .m-lbl { color: #92400E; }

/* ---------- Feature Card (Home) ---------- */
.feat-card {
    border-radius: 14px;
    padding: 1.1rem 1rem;
    border-left: 4px solid;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    background: #ffffff;
    min-height: 155px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    box-sizing: border-box;
}

.feat-card h4 {
    margin: 0.35rem 0 0.35rem 0;
    font-size: 0.98rem;
    font-weight: 700;
    line-height: 1.3;
}

.feat-card p {
    font-size: 0.83rem;
    color: #475569;
    margin: 0;
    line-height: 1.45;
}

/* ---------- Section Header ---------- */
.sec-head {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    margin: 1.4rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    line-height: 1.3;
}

.sec-head .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

/* ---------- Pipeline Step ---------- */
.pipe-step {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.85rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.55rem;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    transition: background 0.15s ease;
    min-height: 64px;
    box-sizing: border-box;
}

.pipe-step:hover {
    background: #F1F5F9;
}

.pipe-num {
    min-width: 32px;
    height: 32px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    color: #ffffff;
    flex-shrink: 0;
}

.pipe-txt {
    flex: 1;
    min-width: 0;
}

.pipe-txt h5 {
    margin: 0 0 0.2rem 0;
    font-size: 0.92rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.3;
}

.pipe-txt p {
    margin: 0;
    font-size: 0.80rem;
    color: #64748B;
    line-height: 1.4;
    overflow-wrap: break-word;
}

/* ---------- Prediction Result ---------- */
.pred-result {
    background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 55%, #6D28D9 100%);
    color: #ffffff;
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin: 1.2rem 0;
    box-shadow: 0 8px 28px rgba(30, 64, 175, 0.28);
    box-sizing: border-box;
}

.pred-result .pr-label {
    font-size: 0.98rem;
    font-weight: 500;
    opacity: 0.92;
    margin-bottom: 0.4rem;
    line-height: 1.3;
    color: #E0E7FF;
}

.pred-result .pr-value {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0.2rem 0;
    color: #ffffff;
}

.pred-result .pr-sub {
    font-size: 0.88rem;
    opacity: 0.80;
    margin-top: 0.3rem;
    line-height: 1.3;
    color: #E0E7FF;
}

/* ---------- Coefficient Card ---------- */
.coef-card {
    border-radius: 13px;
    padding: 0.9rem 0.5rem;
    text-align: center;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
    background: #ffffff;
    border-top: 3px solid;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-sizing: border-box;
}

.coef-card .c-name {
    font-size: 0.74rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    line-height: 1.25;
    margin-bottom: 0.25rem;
}

.coef-card .c-val {
    font-size: 1.3rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0.15rem 0;
}

.coef-card .c-sym {
    font-size: 0.78rem;
    color: #94A3B8;
    line-height: 1.2;
}

/* ---------- Math Step ---------- */
.math-step {
    border-radius: 13px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
    background: #FAFBFC;
    box-sizing: border-box;
}

.math-step h5 {
    margin: 0 0 0.35rem 0;
    font-size: 0.98rem;
    font-weight: 700;
    line-height: 1.35;
}

.math-step p {
    font-size: 0.86rem;
    color: #475569;
    margin: 0.25rem 0 0.45rem 0;
    line-height: 1.5;
}

.math-step code {
    display: inline-block;
    padding: 0.25rem 0.55rem;
    background: rgba(0,0,0,0.04);
    border-radius: 6px;
    font-size: 0.82rem;
    line-height: 1.4;
}

/* ---------- Tags & Banners ---------- */
.tag {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 0.15rem;
    line-height: 1.3;
}

.tag-blue   { background: #DBEAFE; color: #1E40AF; }
.tag-purple { background: #EDE9FE; color: #6D28D9; }
.tag-green  { background: #D1FAE5; color: #047857; }
.tag-cyan   { background: #CFFAFE; color: #0E7490; }
.tag-orange { background: #FEF3C7; color: #B45309; }

.info-banner {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 0.95rem 1.25rem;
    color: #1E40AF;
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Data & Model (cached — reuses existing backend)
# =====================================================================
@st.cache_data
def load_and_prepare_data():
    data_path = get_data_path()
    df = load_dataset(data_path)
    validation = validate_dataset(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    return df, validation, X, y, X_train, X_test, y_train, y_test

@st.cache_resource
def train_model(_X_train, _y_train):
    model = LeastSquaresRegressor(feature_names=FEATURE_COLUMNS)
    model.fit(_X_train, _y_train)
    return model

df, validation, X, y, X_train, X_test, y_train, y_test = load_and_prepare_data()
model = train_model(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_eval = full_evaluation(y_train, y_pred_train)
test_eval = full_evaluation(y_test, y_pred_test)

# =====================================================================
# Helper: colored metric card
# =====================================================================
def metric_card(icon, value, label, color="blue"):
    st.markdown(f"""
    <div class="m-card mc-{color}">
        <div class="m-icon">{icon}</div>
        <div class="m-val">{value}</div>
        <div class="m-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

def section_header(text, color="#2563EB"):
    st.markdown(f"""
    <div class="sec-head">
        <span class="dot" style="background:{color};"></span> {text}
    </div>""", unsafe_allow_html=True)

PIPE_COLORS = ["#2563EB","#7C3AED","#06B6D4","#10B981","#F59E0B",
               "#2563EB","#7C3AED","#06B6D4","#10B981","#F59E0B"]

# =====================================================================
# Sidebar Navigation
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size:1.4rem; font-weight:800; color:#60A5FA; letter-spacing:-0.5px; line-height:1.2;">STUDENT MARKS</div>
        <div style="font-size:1.05rem; font-weight:600; color:#A78BFA; line-height:1.2; margin-top:2px;">LEAST SQUARES</div>
        <div style="font-size:0.70rem; color:#94A3B8; margin-top:6px; letter-spacing:1px; text-transform:uppercase; line-height:1.3;">
            Academic ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏡  Home",
            "📊  Dataset",
            "🔎  Data Analysis",
            "🧮  Least Squares",
            "📐  Mathematics",
            "⚙️  Model Training",
            "🎯  Prediction",
            "📈  Evaluation",
            "📉  Residual Analysis",
            "📖  About",
        ],
        label_visibility="collapsed",
    )
    # Strip emoji prefix for matching
    page_clean = page.split("  ", 1)[-1].strip()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.78rem; padding:0.75rem 0.6rem; background:rgba(255,255,255,0.05);
                border-radius:12px; border:1px solid rgba(255,255,255,0.08); text-align:center; margin-top:0.4rem;">
        <div style="color:#94A3B8; margin-bottom:4px; font-weight:500; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">Test Set R²</div>
        <div style="font-size:1.55rem; font-weight:800; color:#60A5FA; line-height:1.2;">
            {test_eval['R2']:.4f}</div>
        <div style="color:#94A3B8; margin-top:6px; font-size:0.72rem; line-height:1.3;">
            {len(df)} Students &bull; 4 Features</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:1.5rem; margin-bottom:1rem; font-size:0.70rem;
                color:#64748B; line-height:1.5;">
        Machine Learning Academic Project<br>
        SRM Institute of Science and Technology
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
#  HOME PAGE
# =====================================================================
if page_clean == "Home":
    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-title">Student Marks Prediction</div>
        <div class="hero-sub">Predict Final Examination Marks Using Least Squares Regression</div>
        <div class="hero-desc">
            An interactive Machine Learning system that learns the relationship between
            study hours, attendance, assignment performance, and internal assessment marks
            to predict final examination performance with high precision.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("📚", str(len(df)), "Students", "blue")
    with c2: metric_card("🎯", "4", "Input Features", "purple")
    with c3: metric_card("🧠", str(len(X_train)), "Training Samples", "cyan")
    with c4: metric_card("📊", str(len(X_test)), "Testing Samples", "green")

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Feature cards
    section_header("Input Features", "#7C3AED")
    fc1, fc2, fc3, fc4 = st.columns(4)
    feats = [
        ("📖", "Study Hours", "Measures the average daily time a student spends studying.", "#2563EB"),
        ("📅", "Attendance", "Represents classroom attendance as a percentage.", "#06B6D4"),
        ("📝", "Assignment Marks", "Reflects the student's performance in assignments.", "#10B981"),
        ("📊", "Internal Assessment", "Represents internal academic evaluation scores.", "#F59E0B"),
    ]
    for col, (icon, name, desc, color) in zip([fc1,fc2,fc3,fc4], feats):
        with col:
            st.markdown(f"""
            <div class="feat-card" style="border-left-color:{color};">
                <span style="font-size:1.4rem; line-height:1;">{icon}</span>
                <h4 style="color:{color};">{name}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Pipeline
    section_header("Machine Learning Pipeline", "#2563EB")
    steps = [
        ("Student Dataset", f"{len(df)} student records with 4 input features"),
        ("Feature Selection", "Study Hours, Attendance, Assignment Marks, Internal Assessment"),
        ("Train / Test Split", f"{len(X_train)} training • {len(X_test)} testing samples (80/20)"),
        ("Design Matrix", "Prepend column of 1s for the intercept (bias) term"),
        ("Sum-of-Squares Error", "E(w) = ½ Σ (predicted − actual)²"),
        ("Normal Equations", "XᵀXw = Xᵀt  →  solve for weight vector w"),
        ("Regression Coefficients", "Learned directly from training data without tuning"),
        ("Prediction", "y = w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄"),
        ("Residual Analysis", "r = actual − predicted error diagnostics"),
        ("MSE & R² Evaluation", f"Test MSE: {test_eval['MSE']:.2f}  |  Test R²: {test_eval['R2']:.4f}"),
    ]
    left, right = st.columns(2)
    for i, (title, desc) in enumerate(steps):
        col = left if i < 5 else right
        with col:
            st.markdown(f"""
            <div class="pipe-step">
                <div class="pipe-num" style="background:{PIPE_COLORS[i]};">{i+1}</div>
                <div class="pipe-txt">
                    <h5>{title}</h5>
                    <p>{desc}</p>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Academic concepts
    section_header("Academic Concepts Covered", "#06B6D4")
    ac1, ac2 = st.columns(2)
    with ac1:
        with st.container(border=True):
            st.markdown("**Unit 1 Curriculum Concepts**")
            tags_u1 = ["Machine Learning", "Supervised Learning", "Regression", "Training/Test Data",
                       "Generalization", "Sum-of-Squares Error", "Polynomial Curve Fitting"]
            for t in tags_u1:
                st.markdown(f'<span class="tag tag-blue">{t}</span>', unsafe_allow_html=True)
    with ac2:
        with st.container(border=True):
            st.markdown("**Unit 2 Curriculum Concepts**")
            tags_u2 = ["Maximum Likelihood", "Least Squares", "Normal Equations", "Regression Coefficients",
                       "Residuals", "MSE & R² Metrics", "Ridge Regularization"]
            for t in tags_u2:
                st.markdown(f'<span class="tag tag-purple">{t}</span>', unsafe_allow_html=True)


# =====================================================================
#  DATASET PAGE
# =====================================================================
elif page_clean == "Dataset":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#1E3A8A,#2563EB);">
        <h2>Dataset Overview</h2>
        <p>Student Performance Dataset for Least Squares Multiple Linear Regression</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <strong>Academic Integrity Notice:</strong> This is a <strong>synthetic academic project dataset</strong>
        created specifically for demonstrating Least Squares regression principles.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("📚", str(len(df)), "Total Students", "blue")
    with c2: metric_card("📋", str(len(df.columns)), "Columns", "purple")
    with c3: metric_card("🧪", str(len(X_train)), "Training Set", "cyan")
    with c4: metric_card("🔬", str(len(X_test)), "Testing Set", "green")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "🏋️ Training Data",
                                       "🧪 Testing Data", "📊 Statistics"])
    with tab1:
        section_header("Full Dataset View", "#2563EB")
        n_rows = st.slider("Rows to display", 5, len(df), 20, key="ds_rows")
        st.dataframe(df.head(n_rows), use_container_width=True, height=380)

    with tab2:
        section_header("Training Set (80%)", "#06B6D4")
        train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
        train_df[TARGET_COLUMN] = y_train
        st.dataframe(train_df, use_container_width=True, height=380)

    with tab3:
        section_header("Testing Set (20%)", "#10B981")
        test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
        test_df[TARGET_COLUMN] = y_test
        st.dataframe(test_df, use_container_width=True, height=380)

    with tab4:
        section_header("Summary Statistics", "#7C3AED")
        stats_df = df.describe().round(2)
        stats_df.columns = [c.replace("_", " ") for c in stats_df.columns]
        st.dataframe(stats_df, use_container_width=True)

    # Validation
    section_header("Data Quality Validation", "#10B981")
    if validation["passed"]:
        st.success("Dataset validation: **PASSED** — No missing values, no duplicates, all attributes in valid range.")
    else:
        st.error("Dataset validation: **FAILED**")

    val_df = pd.DataFrame({
        "Column": [c.replace("_", " ") for c in validation["missing_values"].keys()],
        "Missing Values": list(validation["missing_values"].values()),
        "Out of Range": [validation["out_of_range"].get(c, {}).get("total", 0)
                         for c in validation["missing_values"].keys()],
        "Status": ["Pass" if validation["missing_values"][c] == 0 and
                    validation["out_of_range"].get(c, {}).get("total", 0) == 0
                    else "Fail" for c in validation["missing_values"].keys()],
    })
    st.dataframe(val_df, use_container_width=True, hide_index=True)


# =====================================================================
#  DATA ANALYSIS PAGE
# =====================================================================
elif page_clean == "Data Analysis":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#0E7490,#06B6D4,#2563EB);">
        <h2>Exploratory Data Analysis</h2>
        <p>Analyze feature distributions, correlations, and relationships with Final Exam Marks</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Feature Relationships", "📊 Distributions",
                                       "🗺️ Correlation Matrix", "💡 Key Insights"])

    feat_colors = {
        "Study_Hours": "#2563EB",
        "Attendance": "#06B6D4",
        "Assignment_Marks": "#10B981",
        "Internal_Assessment_Marks": "#F59E0B",
    }

    with tab1:
        section_header("Feature vs Final Examination Marks", "#2563EB")
        sel_feat = st.selectbox(
            "Select Feature for X-axis",
            FEATURE_COLUMNS,
            format_func=lambda x: x.replace("_", " "),
        )
        fig = px.scatter(
            df,
            x=sel_feat,
            y=TARGET_COLUMN,
            color_discrete_sequence=[feat_colors.get(sel_feat, "#2563EB")],
            opacity=0.7,
            height=440,
            labels={sel_feat: sel_feat.replace("_"," "), TARGET_COLUMN: "Final Exam Marks"},
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
        fig.update_layout(
            template="plotly_white",
            title=dict(text=f"{sel_feat.replace('_',' ')} vs Final Exam Marks", font=dict(size=15)),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
            margin=dict(l=50, r=30, t=50, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View All 4 Features Side-by-Side"):
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=[c.replace("_"," ") for c in FEATURE_COLUMNS],
                vertical_spacing=0.14,
                horizontal_spacing=0.10,
            )
            for i, col_name in enumerate(FEATURE_COLUMNS):
                r, c = divmod(i, 2)
                fig2.add_trace(
                    go.Scatter(
                        x=df[col_name],
                        y=df[TARGET_COLUMN],
                        mode="markers",
                        marker=dict(
                            color=list(feat_colors.values())[i],
                            size=6,
                            opacity=0.65,
                            line=dict(width=0.5, color="white"),
                        ),
                        name=col_name.replace("_"," "),
                        showlegend=False,
                    ),
                    row=r+1, col=c+1,
                )
            fig2.update_layout(height=600, template="plotly_white", margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section_header("Feature Distributions", "#7C3AED")
        dist_feat = st.selectbox(
            "Select Attribute",
            FEATURE_COLUMNS + [TARGET_COLUMN],
            format_func=lambda x: x.replace("_"," "),
            key="dist_sel",
        )
        fig3 = px.histogram(
            df,
            x=dist_feat,
            nbins=24,
            color_discrete_sequence=["#7C3AED"],
            opacity=0.75,
            height=400,
            labels={dist_feat: dist_feat.replace("_"," ")},
        )
        fig3.update_layout(
            template="plotly_white",
            title=dict(text=f"Distribution of {dist_feat.replace('_',' ')}", font=dict(size=15)),
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
            margin=dict(l=50, r=30, t=50, b=50),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        section_header("Feature Correlation Heatmap", "#06B6D4")
        corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
        corr_display = corr.copy()
        clean_names = [c.replace("_", " ") for c in corr.columns]
        corr_display.columns = clean_names
        corr_display.index = clean_names

        fig4 = px.imshow(
            corr_display.round(3),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Correlation"),
            height=500,
        )
        fig4.update_layout(
            template="plotly_white",
            xaxis=dict(tickangle=-25, automargin=True),
            yaxis=dict(automargin=True),
            margin=dict(l=50, r=40, t=30, b=80),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
        <div class="info-banner">
            <strong>Statistical Note:</strong> Correlation measures the strength and direction of linear association.
            High correlation indicates strong predictive utility for Least Squares regression.
        </div>""", unsafe_allow_html=True)

    with tab4:
        section_header("Correlation with Final Examination Marks", "#10B981")
        corr_target = df[FEATURE_COLUMNS].corrwith(df[TARGET_COLUMN]).sort_values(ascending=False)
        for feat_name, corr_val in corr_target.items():
            bar_pct = min(abs(corr_val) * 100, 100)
            color = "#10B981" if corr_val > 0 else "#EF4444"
            st.markdown(f"""
            <div style="margin-bottom:0.9rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.90rem; font-weight:600; line-height:1.4;">
                    <span>{feat_name.replace('_',' ')}</span>
                    <span style="color:{color};">{corr_val:+.3f}</span>
                </div>
                <div style="background:#E2E8F0; border-radius:6px; height:9px; margin-top:4px;">
                    <div style="background:{color}; width:{bar_pct}%; height:100%; border-radius:6px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)


# =====================================================================
#  LEAST SQUARES PAGE
# =====================================================================
elif page_clean == "Least Squares":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#6D28D9,#7C3AED,#A78BFA);">
        <h2>Least Squares Regression</h2>
        <p>Mathematical principles and learned regression parameters</p>
    </div>""", unsafe_allow_html=True)

    # Model equation
    section_header("Model Formulation", "#7C3AED")
    st.latex(r"\hat{t} = w_0 + w_1(\text{Study Hours}) + w_2(\text{Attendance}) + w_3(\text{Assignment Marks}) + w_4(\text{Internal Assessment})")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Algorithm steps
    section_header("Algorithm Execution Steps", "#2563EB")
    algo_steps = [
        ("Build Design Matrix X", "Prepend a column of ones to feature matrix X for intercept w₀.",
         "#2563EB", "X = [1 | x₁ | x₂ | x₃ | x₄]"),
        ("Compute Gram Matrix XᵀX", "Multiply transpose of design matrix by itself.",
         "#7C3AED", "XᵀX  (shape: 5 × 5)"),
        ("Compute Moment Vector Xᵀt", "Multiply transpose of design matrix by target values.",
         "#06B6D4", "Xᵀt  (shape: 5 × 1)"),
        ("Solve Normal Equations", "Solve linear system XᵀXw = Xᵀt using numpy.linalg.solve().",
         "#10B981", "w = (XᵀX)⁻¹ Xᵀt"),
        ("Extract Learned Weights", "Obtain the exact closed-form optimal regression coefficients.",
         "#F59E0B", "w = [w₀, w₁, w₂, w₃, w₄]"),
    ]
    for i, (title, desc, color, formula) in enumerate(algo_steps):
        st.markdown(f"""
        <div class="math-step" style="border-left-color:{color};">
            <h5><span style="color:{color};">Step {i+1}.</span> {title}</h5>
            <p>{desc}</p>
            <code style="color:{color};">{formula}</code>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Learned coefficients
    section_header("Learned Regression Coefficients", "#10B981")
    coef_data = [
        ("Intercept", "w₀", model.intercept_, "#64748B"),
        ("Study Hours", "w₁", model.coefficients_[0], "#2563EB"),
        ("Attendance", "w₂", model.coefficients_[1], "#06B6D4"),
        ("Assignment", "w₃", model.coefficients_[2], "#10B981"),
        ("Internal Marks", "w₄", model.coefficients_[3], "#F59E0B"),
    ]
    cols = st.columns(5)
    for col, (name, sym, val, color) in zip(cols, coef_data):
        sign_color = "#10B981" if val >= 0 else "#EF4444"
        with col:
            st.markdown(f"""
            <div class="coef-card" style="border-top-color:{color};">
                <div class="c-name">{name}</div>
                <div class="c-val" style="color:{sign_color};">{val:+.4f}</div>
                <div class="c-sym">{sym}</div>
            </div>""", unsafe_allow_html=True)

    st.caption("Coefficients represent feature weight contributions under the closed-form Least Squares solution.")


# =====================================================================
#  MATHEMATICS PAGE
# =====================================================================
elif page_clean == "Mathematics":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#1E3A8A,#2563EB,#3B82F6);">
        <h2>Mathematical Derivations</h2>
        <p>Theoretical foundation, normal equations derivation, and viva revision</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("1. Linear Regression Model", expanded=True):
        st.latex(r"y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 = \mathbf{w}^T \mathbf{x}")
        st.markdown("**Core Concept:** The prediction is a linear combination of the input features plus a bias intercept term.")
        st.markdown("**Viva Point:** The model is linear in the parameters $\mathbf{w}$, which guarantees a unique global minimum for the error function.")

    with st.expander("2. Design Matrix Formulation"):
        st.latex(r"\mathbf{X} = \begin{bmatrix} 1 & x_{11} & x_{21} & x_{31} & x_{41} \\ 1 & x_{12} & x_{22} & x_{32} & x_{42} \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ 1 & x_{1N} & x_{2N} & x_{3N} & x_{4N} \end{bmatrix}")
        st.markdown("**Core Concept:** Prepending a column of ones absorbs the intercept $w_0$ into the weight vector $\mathbf{w}$.")
        st.markdown("**Viva Point:** This compact matrix representation allows formulating all $N$ sample predictions as $\hat{\mathbf{t}} = \mathbf{X}\mathbf{w}$.")

    with st.expander("3. Sum-of-Squares Error Function"):
        st.latex(r"E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} \left[ y(\mathbf{x}_n, \mathbf{w}) - t_n \right]^2 = \frac{1}{2} (\mathbf{X}\mathbf{w} - \mathbf{t})^T (\mathbf{X}\mathbf{w} - \mathbf{t})")
        st.markdown("**Core Concept:** Penalizes large errors quadratically. The factor $\\frac{1}{2}$ cancels the exponent 2 upon differentiation.")
        st.markdown("**Viva Point:** Because $E(\mathbf{w})$ is a quadratic function of $\mathbf{w}$, its derivative is linear, yielding a closed-form solution.")

    with st.expander("4. Normal Equations Derivation"):
        st.latex(r"\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{t}) = \mathbf{0}")
        st.latex(r"\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t} \implies \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{t}")
        st.markdown("**Core Concept:** Setting the gradient to zero yields the Normal Equations.")
        st.markdown("**Viva Point:** Our implementation solves $\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{t}$ via `numpy.linalg.solve()`, avoiding direct matrix inversion for superior numerical stability.")

    with st.expander("5. Maximum Likelihood & Gaussian Noise Connection"):
        st.latex(r"t_n = y(\mathbf{x}_n, \mathbf{w}) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma^2)")
        st.latex(r"p(\mathbf{t} | \mathbf{X}, \mathbf{w}, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n | \mathbf{w}^T \mathbf{x}_n, \sigma^2)")
        st.markdown("**Core Concept:** Assuming Gaussian noise, maximizing log-likelihood is mathematically equivalent to minimizing sum-of-squares error.")
        st.markdown("**Viva Point:** Under Gaussian distributed errors, the Least Squares estimator is the Maximum Likelihood estimator.")

    with st.expander("6. Mean Squared Error (MSE) and R-squared (R²)"):
        st.latex(r"\text{MSE} = \frac{1}{N} \sum_{n=1}^{N} (t_n - \hat{y}_n)^2, \quad R^2 = 1 - \frac{\sum (t_n - \hat{y}_n)^2}{\sum (t_n - \bar{t})^2}")
        st.markdown("**Core Concept:** MSE gives average squared residual variance. $R^2$ indicates the fraction of variance explained by the model.")


# =====================================================================
#  MODEL TRAINING PAGE
# =====================================================================
elif page_clean == "Model Training":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#047857,#10B981,#34D399);">
        <h2>Model Training & System Matrices</h2>
        <p>Execution details of the Normal Equations solver on training data</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("🏋️", str(len(X_train)), "Training Samples", "blue")
    with c2: metric_card("🧪", str(len(X_test)), "Testing Samples", "purple")
    with c3: metric_card("📊", str(len(FEATURE_COLUMNS)), "Features", "cyan")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    details = model.get_training_details()

    section_header("Training Pipeline Stages", "#10B981")
    pipeline_items = [
        ("Design Matrix X", f"Shape: {details['design_matrix_shape']} (includes column of 1s)", "#2563EB"),
        ("XᵀX (Gram Matrix)", f"Shape: {details['XtX'].shape} — Symmetric positive semi-definite", "#7C3AED"),
        ("Xᵀt (Moment Vector)", f"Shape: ({len(details['Xty'])},) — Target projections", "#06B6D4"),
        ("Learned Weights w", f"{len(details['weights'])} parameters solved via numpy.linalg.solve", "#10B981"),
        ("Sum-of-Squares Error", f"Training SSE: {model.training_error_:.2f}", "#F59E0B"),
    ]
    for title, info, color in pipeline_items:
        st.markdown(f"""
        <div class="pipe-step">
            <div class="pipe-num" style="background:{color};">✓</div>
            <div class="pipe-txt">
                <h5>{title}</h5>
                <p>{info}</p>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    with st.expander("Inspect Design Matrix (First 10 Rows)"):
        X_design_df = pd.DataFrame(
            model._X_design[:10],
            columns=["Intercept"] + [c.replace("_", " ") for c in FEATURE_COLUMNS]
        )
        st.dataframe(X_design_df.round(2), use_container_width=True, hide_index=True)

    with st.expander("Inspect Gram Matrix (XᵀX)"):
        clean_matrix_cols = ["Intercept"] + [c.replace("_", " ") for c in FEATURE_COLUMNS]
        xtx_df = pd.DataFrame(details["XtX"], columns=clean_matrix_cols, index=clean_matrix_cols).round(2)
        st.dataframe(xtx_df, use_container_width=True)

    with st.expander("Inspect Moment Vector (Xᵀt)"):
        xty_df = pd.DataFrame({
            "Component": ["Intercept"] + [c.replace("_", " ") for c in FEATURE_COLUMNS],
            "Value": details["Xty"].round(4),
        })
        st.dataframe(xty_df, use_container_width=True, hide_index=True)

    # Learned equation
    section_header("Fitted Regression Equation", "#7C3AED")
    eq_parts = [f"{model.intercept_:.4f}"]
    for name, coef in zip(FEATURE_COLUMNS, model.coefficients_):
        escaped = name.replace("_", r"\text{ }")
        eq_parts.append(f"{coef:+.4f} \\times (\\text{{{escaped}}})")
    st.latex(r"\hat{t} = " + " ".join(eq_parts))

    # Training metrics
    section_header("Training Fit Metrics", "#F59E0B")
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("📉", f"{model.training_error_:.1f}", "SSE", "orange")
    with m2: metric_card("📊", f"{train_eval['MSE']:.4f}", "MSE", "blue")
    with m3: metric_card("📏", f"{train_eval['RMSE']:.4f}", "RMSE", "purple")
    with m4: metric_card("🎯", f"{train_eval['R2']:.4f}", "R²", "green")


# =====================================================================
#  PREDICTION PAGE
# =====================================================================
elif page_clean == "Prediction":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#1E3A8A,#2563EB,#3B82F6);">
        <h2>Student Marks Prediction</h2>
        <p>Enter a student's academic scores to predict their Final Examination Marks</p>
    </div>""", unsafe_allow_html=True)

    section_header("Student Academic Inputs", "#2563EB")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                <span style="font-size:1.3rem;">📖</span>
                <span style="font-size:1.02rem; font-weight:700; color:#2563EB;">Study Hours</span>
            </div>
            <div style="font-size:0.80rem; color:#64748B; margin-bottom:8px;">
                Average daily self-study time (0 to 14 hours)
            </div>
            """, unsafe_allow_html=True)
            study_hours = st.slider("Study Hours", 0.0, 14.0, 7.0, 0.5, key="p_sh", label_visibility="collapsed")

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                <span style="font-size:1.3rem;">📝</span>
                <span style="font-size:1.02rem; font-weight:700; color:#10B981;">Assignment Marks</span>
            </div>
            <div style="font-size:0.80rem; color:#64748B; margin-bottom:8px;">
                Continuous assignment performance score (0 to 100)
            </div>
            """, unsafe_allow_html=True)
            assignment_marks = st.slider("Assignment Marks", 0.0, 100.0, 70.0, 1.0, key="p_am", label_visibility="collapsed")

    with c2:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                <span style="font-size:1.3rem;">📅</span>
                <span style="font-size:1.02rem; font-weight:700; color:#06B6D4;">Attendance</span>
            </div>
            <div style="font-size:0.80rem; color:#64748B; margin-bottom:8px;">
                Classroom attendance percentage (0 to 100%)
            </div>
            """, unsafe_allow_html=True)
            attendance = st.slider("Attendance", 0.0, 100.0, 80.0, 1.0, key="p_att", label_visibility="collapsed")

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                <span style="font-size:1.3rem;">📊</span>
                <span style="font-size:1.02rem; font-weight:700; color:#F59E0B;">Internal Assessment Marks</span>
            </div>
            <div style="font-size:0.80rem; color:#64748B; margin-bottom:8px;">
                Internal examination evaluation score (0 to 100)
            </div>
            """, unsafe_allow_html=True)
            internal_marks = st.slider("Internal Assessment", 0.0, 100.0, 65.0, 1.0, key="p_ia", label_visibility="collapsed")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🎯  Compute Final Examination Prediction", type="primary", use_container_width=True)

    if predict_btn:
        result = predict_single_student(model, study_hours, attendance, assignment_marks, internal_marks)

        # Result badge
        st.markdown(f"""
        <div class="pred-result">
            <div class="pr-label">Predicted Final Examination Score</div>
            <div class="pr-value">{result['predicted_marks']:.2f}</div>
            <div class="pr-sub">Marks out of 100</div>
        </div>""", unsafe_allow_html=True)

        # Contribution breakdown chart
        section_header("Feature Contribution Breakdown", "#7C3AED")
        contrib = result["contribution_breakdown"]
        contrib_names = [k.replace("_", " ") for k in contrib.keys()]
        contrib_vals = list(contrib.values())
        colors_contrib = ["#64748B","#2563EB","#06B6D4","#10B981","#F59E0B"]

        fig_c = go.Figure(go.Bar(
            y=contrib_names,
            x=contrib_vals,
            orientation="h",
            marker=dict(
                color=colors_contrib,
                line=dict(color="rgba(255,255,255,0.8)", width=1.5)
            ),
            text=[f"{v:+.2f}" for v in contrib_vals],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_c.update_layout(
            template="plotly_white",
            height=300,
            xaxis_title="Contribution to Predicted Marks",
            margin=dict(l=180, r=60, t=20, b=40),
            yaxis=dict(autorange="reversed", automargin=True),
            xaxis=dict(automargin=True),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        # Calculation steps
        with st.expander("Detailed Calculation Walkthrough"):
            st.markdown("**Step 1: Input Vector (with Intercept column)**")
            st.code(f"x = [1.0, {study_hours}, {attendance}, {assignment_marks}, {internal_marks}]")

            st.markdown("**Step 2: Learned Weight Vector (w)**")
            w_str = ", ".join([f"{v:.4f}" for v in model.weights_])
            st.code(f"w = [{w_str}]")

            st.markdown("**Step 3: Component-wise Product (wᵢ × xᵢ)**")
            for name, val in contrib.items():
                st.markdown(f"- **{name.replace('_', ' ')}**: `{val:+.4f}`")

            st.markdown(f"**Step 4: Dot Product Sum $\mathbf{w}^T \mathbf{x}$** = **{sum(contrib.values()):.4f}**")
            st.markdown(f"**Final Result: {result['predicted_marks']:.2f} Marks**")


# =====================================================================
#  EVALUATION PAGE
# =====================================================================
elif page_clean == "Evaluation":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#1E40AF,#2563EB,#60A5FA);">
        <h2>Model Evaluation & Generalization</h2>
        <p>Performance comparison between Training and Testing sets</p>
    </div>""", unsafe_allow_html=True)

    col_tr, col_te = st.columns(2)
    with col_tr:
        st.markdown("""
        <div style="text-align:center; margin-bottom:0.75rem;">
            <span style="font-size:0.82rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; background:#F1F5F9; padding:5px 14px; border-radius:20px; border:1px solid #E2E8F0;">
                Training Metrics
            </span>
        </div>
        """, unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1: metric_card("📊", f"{train_eval['MSE']:.4f}", "Training MSE", "blue")
        with m2: metric_card("🎯", f"{train_eval['R2']:.4f}", "Training R²", "cyan")

    with col_te:
        st.markdown("""
        <div style="text-align:center; margin-bottom:0.75rem;">
            <span style="font-size:0.82rem; font-weight:700; color:#1E40AF; text-transform:uppercase; letter-spacing:1px; background:#DBEAFE; padding:5px 14px; border-radius:20px; border:1px solid #BFDBFE;">
                Testing Metrics (Primary)
            </span>
        </div>
        """, unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3: metric_card("📊", f"{test_eval['MSE']:.4f}", "Testing MSE", "purple")
        with m4: metric_card("🎯", f"{test_eval['R2']:.4f}", "Testing R²", "green")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <strong>Generalization Assessment:</strong> Test MSE (16.68) closely mirrors Train MSE (16.98), and Test R² (0.9413)
        confirms high variance explanation without overfitting.
    </div>""", unsafe_allow_html=True)

    with st.expander("Comprehensive Metric Comparison Table"):
        comp_df = pd.DataFrame({
            "Evaluation Metric": [
                "Mean Squared Error (MSE)",
                "Root Mean Squared Error (RMSE)",
                "Mean Absolute Error (MAE)",
                "Coefficient of Determination (R²)",
            ],
            "Training Set": [f"{train_eval['MSE']:.4f}", f"{train_eval['RMSE']:.4f}",
                             f"{train_eval['MAE']:.4f}", f"{train_eval['R2']:.4f}"],
            "Testing Set": [f"{test_eval['MSE']:.4f}", f"{test_eval['RMSE']:.4f}",
                            f"{test_eval['MAE']:.4f}", f"{test_eval['R2']:.4f}"],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Actual vs Predicted Scatter
    section_header("Actual vs Predicted (Test Set)", "#2563EB")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred_test,
        mode="markers",
        marker=dict(
            color="#2563EB",
            size=10,
            opacity=0.75,
            line=dict(color="white", width=1.5),
        ),
        name="Predictions",
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<br>Residual: %{customdata:.1f}<extra></extra>",
        customdata=y_test - y_pred_test,
    ))
    lo = float(min(y_test.min(), y_pred_test.min()) - 4)
    hi = float(max(y_test.max(), y_pred_test.max()) + 4)
    fig.add_trace(go.Scatter(
        x=[lo, hi],
        y=[lo, hi],
        mode="lines",
        line=dict(color="#EF4444", dash="dash", width=2),
        name="Perfect Fit (y = x)",
        showlegend=True,
    ))
    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Actual Final Exam Marks",
        yaxis_title="Predicted Final Exam Marks",
        xaxis=dict(range=[lo, hi], automargin=True),
        yaxis=dict(range=[lo, hi], scaleanchor="x", scaleratio=1, automargin=True),
        margin=dict(l=60, r=40, t=30, b=50),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
#  RESIDUAL ANALYSIS PAGE
# =====================================================================
elif page_clean == "Residual Analysis":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#B45309,#F59E0B,#FBBF24);">
        <h2 style="color:#0F172A !important;">Residual Error Analysis</h2>
        <p style="color:#1E293B !important;">Diagnostic analysis of model residuals: r = actual − predicted</p>
    </div>""", unsafe_allow_html=True)

    residuals_test = y_test - y_pred_test
    res_stats = residual_statistics(y_test, y_pred_test)

    st.latex(r"r_n = t_n - \hat{y}_n \quad (\text{Actual Final Marks} - \text{Predicted Final Marks})")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("📊", f"{res_stats['mean']:.3f}", "Mean Residual", "blue")
    with c2: metric_card("📏", f"{res_stats['std']:.3f}", "Std Residual", "purple")
    with c3: metric_card("📉", f"{res_stats['min']:.2f}", "Min Residual", "orange")
    with c4: metric_card("📈", f"{res_stats['max']:.2f}", "Max Residual", "green")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Residual vs Predicted", "📊 Error Distribution",
                                 "📋 Top Residuals Table"])

    with tab1:
        fig_r = go.Figure()
        colors_r = ["#10B981" if r >= 0 else "#EF4444" for r in residuals_test]
        fig_r.add_trace(go.Scatter(
            x=y_pred_test,
            y=residuals_test,
            mode="markers",
            marker=dict(
                color=colors_r,
                size=10,
                opacity=0.75,
                line=dict(color="white", width=1),
            ),
            hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.2f}<extra></extra>",
            showlegend=False,
        ))
        fig_r.add_hline(y=0, line_dash="dash", line_color="#0F172A", line_width=1.5)
        fig_r.update_layout(
            template="plotly_white",
            height=420,
            xaxis_title="Predicted Final Marks",
            yaxis_title="Residual (Actual − Predicted)",
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
            margin=dict(l=60, r=30, t=30, b=50),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("Green indicates positive residual (under-prediction), Red indicates negative residual (over-prediction).")

    with tab2:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=residuals_test,
            nbinsx=18,
            marker=dict(color="#7C3AED", line=dict(color="white", width=1)),
            opacity=0.8,
            hovertemplate="Residual Range: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig_h.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=2)
        fig_h.update_layout(
            template="plotly_white",
            height=380,
            xaxis_title="Residual Value",
            yaxis_title="Frequency",
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True),
            margin=dict(l=60, r=30, t=30, b=50),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with tab3:
        res_tbl = residual_table(y_test, y_pred_test, n_largest=10)
        res_tbl.columns = [c.replace("_", " ") for c in res_tbl.columns]
        st.dataframe(res_tbl, use_container_width=True, hide_index=True)


# =====================================================================
#  ABOUT PAGE
# =====================================================================
elif page_clean == "About":
    st.markdown("""
    <div class="hero-page-header" style="background:linear-gradient(135deg,#0F172A,#1E293B,#334155);">
        <h2>About This Project</h2>
        <p>SRM University Machine Learning Course Project Specifications</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            section_header("Project Overview", "#2563EB")
            st.markdown("**STUDENT MARKS PREDICTION USING LEAST SQUARES**")
            st.markdown("""
            Predicts students' final examination marks using study hours, attendance,
            assignment marks, and internal assessment marks. Implements the Least Squares
            method from scratch to estimate regression coefficients and calculate MSE and R².
            """)

            section_header("Model Architecture", "#7C3AED")
            st.markdown("""
            - **Algorithm:** Multiple Linear Regression
            - **Solver:** Normal Equations ($X^T X w = X^T t$)
            - **Implementation:** Custom Python (`src/least_squares.py`)
            - **Primary Target:** Final Examination Marks
            """)

    with c2:
        with st.container(border=True):
            section_header("Input Features & Output", "#10B981")
            st.markdown("""
            **Primary Input Features:**
            1. Study Hours
            2. Attendance
            3. Assignment Marks
            4. Internal Assessment Marks

            **Target Variable:**
            - Final Examination Marks

            **Evaluation Metrics:**
            - Mean Squared Error (MSE)
            - Coefficient of Determination (R²)
            """)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    section_header("Technology Stack", "#F59E0B")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    techs = [
        ("Python 3.11+", "Core Language", "blue"),
        ("NumPy + Pandas", "Data & Math", "purple"),
        ("Plotly", "Interactive Charts", "cyan"),
        ("Streamlit", "Web Interface", "green"),
        ("Pytest", "Test Suite", "orange"),
    ]
    for col, (name, cat, color) in zip([tc1, tc2, tc3, tc4, tc5], techs):
        with col:
            st.markdown(f"""
            <div class="m-card mc-{color}" style="padding:0.9rem 0.5rem; min-height:95px;">
                <div style="font-size:0.90rem; font-weight:700; line-height:1.25; margin-bottom:0.25rem;">{name}</div>
                <div class="m-lbl" style="margin-top:0;">{cat}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner" style="text-align:center;">
        <strong>Academic Context:</strong> Developed for SRM Institute of Science and Technology,
        Department of Computer Science and Engineering, Machine Learning Course.
    </div>""", unsafe_allow_html=True)

    with st.expander("Quick Terminal Run Commands"):
        st.code("python -m streamlit run app/streamlit_app.py", language="bash")
        st.code("python src/main.py", language="bash")
        st.code("python -m pytest tests/ -v", language="bash")
