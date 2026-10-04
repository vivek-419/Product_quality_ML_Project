"""
Production Quality Prediction Using Machine Learning
Dataset: UCI SECOM Semiconductor Manufacturing Dataset
Demonstration & Evaluation Dashboard
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Production Quality Prediction | UCI SECOM ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DESIGN SYSTEM & PROFESSIONAL STYLING (Light Themed Tables & Metrics)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"], p, div:not([data-testid="stIconMaterial"]):not([class*="material"]), 
    h1, h2, h3, h4, h5, h6, span:not([data-testid="stIconMaterial"]), button, input, label {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #F8FAFC !important;
    }

    .main .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
        max-width: 1280px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
        padding-top: 0.8rem !important;
    }

    /* Brand Header */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0.5rem 1.0rem 0.5rem;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 0.85rem;
    }
    .brand-logo-text {
        font-size: 1.0rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #F8FAFC !important;
    }
    .brand-tag {
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #94A3B8;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Sidebar Nav Buttons */
    [data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        color: #94A3B8 !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.55rem 0.75rem !important;
        border-radius: 0 5px 5px 0 !important;
        margin-bottom: 2px !important;
        box-shadow: none !important;
        transition: all 0.12s ease-in-out !important;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        border-left: 3px solid #64748B !important;
    }
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-left: 3px solid #2563EB !important;
    }

    /* Sidebar Meta Box */
    .sidebar-meta-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px 12px;
        margin-top: 10px;
        font-size: 0.77rem;
    }
    .sidebar-meta-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        border-bottom: 1px solid #283548;
    }
    .sidebar-meta-row:last-child {
        border-bottom: none;
    }
    .sidebar-meta-key {
        color: #94A3B8;
        font-weight: 500;
    }
    .sidebar-meta-val {
        color: #F8FAFC;
        font-weight: 600;
    }

    /* Top Banner Header */
    .app-header-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .app-header-left {
        max-width: 820px;
    }
    .app-header-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
    }
    .app-header-subtitle {
        font-size: 0.84rem;
        font-weight: 600;
        color: #475569;
    }
    .app-header-badges {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .badge-pill {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        color: #334155;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .badge-pill-accent {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Section Subtitle Banner */
    .section-banner {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.6px;
        color: #475569;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    /* Metric Cards */
    .metric-grid-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }
    .metric-grid-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 3px;
    }
    .metric-grid-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.15;
    }
    .metric-grid-meta {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 3px;
        font-weight: 500;
    }

    /* Content Box / Section Card */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        margin-bottom: 14px;
    }
    .content-box-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        padding-bottom: 6px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .content-box-title::before {
        content: "";
        display: inline-block;
        width: 3px;
        height: 13px;
        background: #2563EB;
        border-radius: 2px;
    }

    /* Result Badges */
    .result-card-pass {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 5px solid #16A34A;
        border-radius: 6px;
        padding: 16px 18px;
        margin: 10px 0;
    }
    .result-card-fail {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 6px;
        padding: 16px 18px;
        margin: 10px 0;
    }
    .result-badge-pass {
        display: inline-block;
        background-color: #16A34A;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.88rem;
        letter-spacing: 0.05em;
        padding: 4px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .result-badge-fail {
        display: inline-block;
        background-color: #DC2626;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.88rem;
        letter-spacing: 0.05em;
        padding: 4px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }

    /* Note banners */
    .callout-info {
        background-color: #F8FAFC;
        border-left: 3px solid #2563EB;
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        font-size: 0.8rem;
        color: #334155;
        margin: 8px 0;
        line-height: 1.4;
    }
    .callout-warning {
        background-color: #FFFBEB;
        border-left: 3px solid #D97706;
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        font-size: 0.8rem;
        color: #92400E;
        margin: 8px 0;
        line-height: 1.4;
    }

    /* Native Streamlit Table/DataFrame Light Theme Override */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. CLEAN LIGHT-THEME TABLE RENDERER
# -----------------------------------------------------------------------------
def render_styled_table(df, align_right_cols=None, max_rows=None):
    """
    Renders a clean, light-themed HTML table matching the application UI.
    """
    align_right_cols = align_right_cols or []
    display_df = df.head(max_rows) if max_rows else df
    
    html = ['<div style="overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 6px; margin: 4px 0 10px 0;">']
    html.append('<table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; color: #1E293B; background: #FFFFFF;">')
    
    # Table Header
    html.append('<thead style="background-color: #F8FAFC;">')
    html.append('<tr>')
    for col in display_df.columns:
        align = 'right' if col in align_right_cols else 'left'
        html.append(f'<th style="padding: 8px 12px; font-weight: 700; color: #475569; text-transform: uppercase; font-size: 0.71rem; letter-spacing: 0.5px; border-bottom: 1px solid #E2E8F0; text-align: {align}; white-space: nowrap;">{col}</th>')
    html.append('</tr></thead>')
    
    # Table Body
    html.append('<tbody>')
    for i, (_, row) in enumerate(display_df.iterrows()):
        bg = '#FFFFFF' if i % 2 == 0 else '#FAFAFA'
        html.append(f'<tr style="background-color: {bg}; border-bottom: 1px solid #F1F5F9;">')
        for col in display_df.columns:
            val = row[col]
            align = 'right' if col in align_right_cols else 'left'
            val_str = str(val)
            
            is_mono = isinstance(val, (int, float)) or any(c in val_str for c in ['%', 'Feature_', '0.', '±'])
            font_mono = "font-family: 'JetBrains Mono', monospace;" if is_mono else ""
            
            # Semantic highlight for Pass / Fail / Decision Tree
            if val_str == "PASS":
                cell_content = '<span style="color: #16A34A; font-weight: 700; background: #DCFCE7; padding: 2px 7px; border-radius: 4px; font-size: 0.76rem;">PASS</span>'
                style = f"padding: 6px 12px; text-align: {align};"
            elif val_str == "FAIL":
                cell_content = '<span style="color: #DC2626; font-weight: 700; background: #FEE2E2; padding: 2px 7px; border-radius: 4px; font-size: 0.76rem;">FAIL</span>'
                style = f"padding: 6px 12px; text-align: {align};"
            elif "Decision Tree" in val_str and "Selected" in val_str:
                cell_content = f'<strong>{val_str}</strong>'
                style = f"padding: 7px 12px; text-align: {align}; color: #2563EB; {font_mono}"
            else:
                cell_content = val_str
                style = f"padding: 7px 12px; text-align: {align}; {font_mono}"
                
            html.append(f'<td style="{style}">{cell_content}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    
    st.markdown(''.join(html), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. PIPELINE LOADING & COMPATIBILITY LAYER
# -----------------------------------------------------------------------------
MODEL_FILE = "production_quality_model.joblib"
MODEL_BACKUP = "production_quality_model (1).joblib"

@st.cache_resource(show_spinner=False)
def load_production_pipeline():
    """
    Loads the trained Scikit-Learn pipeline from the final notebook execution.
    Applies cross-version compatibility for SimpleImputer internal attributes.
    """
    target_path = None
    if os.path.exists(MODEL_FILE):
        target_path = MODEL_FILE
    elif os.path.exists(MODEL_BACKUP):
        target_path = MODEL_BACKUP
        
    if target_path is None:
        return None, "The trained model file 'production_quality_model.joblib' was not found in the project root."
        
    try:
        pipeline = joblib.load(target_path)
        
        # Compatibility patch for SimpleImputer attribute naming differences across versions
        if hasattr(pipeline, "named_steps"):
            for step in pipeline.named_steps.values():
                if hasattr(step, "_fit_dtype") and not hasattr(step, "_fill_dtype"):
                    step._fill_dtype = step._fit_dtype
                elif hasattr(step, "_fill_dtype") and not hasattr(step, "_fit_dtype"):
                    step._fit_dtype = step._fill_dtype
                    
        return pipeline, None
    except Exception as e:
        return None, f"The trained model could not be loaded. ({str(e)})"


pipeline, load_error = load_production_pipeline()


# -----------------------------------------------------------------------------
# 5. SENSOR INPUT VALIDATION & PREPROCESSING ALIGNMENT
# -----------------------------------------------------------------------------
EXPECTED_RAW_FEATURES = 590
EXPECTED_MODEL_FEATURES = 474

def get_pipeline_feature_names(model_pipeline):
    if model_pipeline is not None and hasattr(model_pipeline, "feature_names_in_"):
        return list(model_pipeline.feature_names_in_)
    return [f"Feature_{i+1}" for i in range(EXPECTED_MODEL_FEATURES)]


EXPECTED_FEATURE_NAMES = get_pipeline_feature_names(pipeline)


def validate_and_prepare_input(df_raw, model_pipeline):
    """
    Validates sensor data and aligns features:
    1. Accepts 590-feature raw SECOM data (drops the 116 non-informative features).
    2. Accepts 474-feature data.
    3. Handles extra trailing labels/timestamps if present (e.g., 591/592 cols).
    4. Allows missing values (imputed by pipeline's internal SimpleImputer).
    """
    if df_raw is None or len(df_raw) == 0:
        return None, "Empty dataset provided. Please provide valid semiconductor sensor measurements."
        
    df = df_raw.copy()
    num_cols = df.shape[1]
    
    # Handle SECOM datasets with trailing label/timestamp
    if num_cols in [591, 592]:
        df = df.iloc[:, :590]
        num_cols = 590

    # Case A: 590 raw SECOM sensor channels
    if num_cols == EXPECTED_RAW_FEATURES:
        if not str(df.columns[0]).startswith("Feature_"):
            df.columns = [f"Feature_{i+1}" for i in range(EXPECTED_RAW_FEATURES)]
            
        missing_expected = [c for c in EXPECTED_FEATURE_NAMES if c not in df.columns]
        if not missing_expected:
            df_for_model = df[EXPECTED_FEATURE_NAMES].copy()
        else:
            all_590 = [f"Feature_{i+1}" for i in range(EXPECTED_RAW_FEATURES)]
            dropped_116 = [c for c in all_590 if c not in EXPECTED_FEATURE_NAMES]
            dropped_indices = [int(c.split('_')[1]) - 1 for c in dropped_116]
            keep_indices = [i for i in range(EXPECTED_RAW_FEATURES) if i not in dropped_indices]
            df_for_model = df.iloc[:, keep_indices].copy()
            df_for_model.columns = EXPECTED_FEATURE_NAMES

    # Case B: 474 cleaned features
    elif num_cols == EXPECTED_MODEL_FEATURES:
        df_for_model = df.copy()
        df_for_model.columns = EXPECTED_FEATURE_NAMES

    else:
        return None, f"Invalid input structure: Expected {EXPECTED_RAW_FEATURES} raw SECOM sensor features or {EXPECTED_MODEL_FEATURES} pipeline features, but received {num_cols} columns."

    # Validate numeric conversion
    try:
        df_numeric = df_for_model.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        return None, f"Data parsing error: Sensor measurements could not be parsed as numeric values. ({str(e)})"

    missing_count = int(df_numeric.isna().sum().sum())
    
    return {
        "df_model": df_numeric,
        "row_count": df_numeric.shape[0],
        "col_count": num_cols,
        "features_to_model": df_numeric.shape[1],
        "missing_count": missing_count
    }, None


# -----------------------------------------------------------------------------
# 6. REAL SECOM SAMPLE DATA LOADER
# -----------------------------------------------------------------------------
def load_real_secom_sample(sample_type="random"):
    secom_data_path = "secom/secom.data" if os.path.exists("secom/secom.data") else "secom.data"
    labels_path = "secom/secom_labels.data" if os.path.exists("secom/secom_labels.data") else "secom_labels.data"

    if os.path.exists(secom_data_path) and os.path.exists(labels_path):
        try:
            df_raw = pd.read_csv(secom_data_path, sep=r"\s+", header=None)
            df_raw.columns = [f"Feature_{i+1}" for i in range(df_raw.shape[1])]
            y_series = pd.read_csv(labels_path, sep=r"\s+", header=None)[0].map({-1: 0, 1: 1})

            if sample_type == "pass":
                pass_indices = y_series[y_series == 0].index
                idx = int(np.random.choice(pass_indices))
                return df_raw.iloc[[idx]].copy(), 0, f"SECOM Verified Pass Wafer (Row #{idx})"
            elif sample_type == "fail":
                fail_indices = y_series[y_series == 1].index
                idx = int(np.random.choice(fail_indices))
                return df_raw.iloc[[idx]].copy(), 1, f"SECOM Verified Fail Wafer (Row #{idx})"
            elif sample_type == "random":
                idx = int(np.random.randint(0, len(df_raw)))
                return df_raw.iloc[[idx]].copy(), int(y_series.iloc[idx]), f"SECOM Random Wafer Run (Row #{idx})"
            elif sample_type == "batch_10":
                indices = np.random.choice(len(df_raw), size=10, replace=False)
                return df_raw.iloc[indices].copy(), y_series.iloc[indices].tolist(), "SECOM 10-Wafer Inspection Batch"
        except Exception:
            pass

    # Fallback sample files
    if sample_type == "pass" and os.path.exists("sample_pass_wafer.csv"):
        return pd.read_csv("sample_pass_wafer.csv"), 0, "Benchmark Pass Wafer (CSV)"
    elif sample_type == "fail" and os.path.exists("sample_fail_wafer.csv"):
        return pd.read_csv("sample_fail_wafer.csv"), 1, "Benchmark Fail Wafer (CSV)"
    elif sample_type in ["batch_10", "random"] and os.path.exists("sample_batch_wafers.csv"):
        df = pd.read_csv("sample_batch_wafers.csv")
        if sample_type == "random":
            return df.sample(1), None, "Benchmark Wafer Sample (CSV)"
        return df, None, "Benchmark 10-Wafer Batch (CSV)"

    return None, None, "Sample data could not be loaded."


# -----------------------------------------------------------------------------
# 7. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "nav_section" not in st.session_state:
    st.session_state.nav_section = "Overview"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "active_input_df" not in st.session_state:
    st.session_state.active_input_df = None

if "active_source_label" not in st.session_state:
    st.session_state.active_source_label = None

if "active_ground_truth" not in st.session_state:
    st.session_state.active_ground_truth = None


# -----------------------------------------------------------------------------
# 8. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="brand-container">
            <div>
                <div class="brand-logo-text">SECOM YIELD AI</div>
                <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 500;">Quality Prediction System</div>
            </div>
            <span class="brand-tag">Academic</span>
        </div>
    """, unsafe_allow_html=True)

    nav_pages = [
        ("Overview", "Overview", "⊞"),
        ("Prediction", "Quality Prediction", "◈"),
        ("Model Performance", "Model Performance", "▥"),
        ("Pipeline & Features", "Pipeline & Features", "⌥"),
        ("Dataset & About", "Dataset & About", "ℹ")
    ]

    for page_key, label, symbol in nav_pages:
        is_active = (st.session_state.nav_section == page_key)
        btn_text = f"{symbol}   {label}"
        
        if st.button(
            btn_text,
            key=f"nav_{page_key}",
            type="primary" if is_active else "secondary"
        ):
            st.session_state.nav_section = page_key
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # Model Artifact Status Box
    st.markdown("""
    <div class="sidebar-meta-box">
        <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 6px; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.5px;">
            System Architecture
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Dataset:</span>
            <span class="sidebar-meta-val">UCI SECOM</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Raw Features:</span>
            <span class="sidebar-meta-val">590 Sensors</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Clean Features:</span>
            <span class="sidebar-meta-val">474 (116 dropped)</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Selected:</span>
            <span class="sidebar-meta-val">Top 60 (MI)</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Final Model:</span>
            <span class="sidebar-meta-val">Decision Tree</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Pipeline Status:</span>
            <span class="sidebar-meta-val" style="color: """ + ("#10B981" if pipeline is not None else "#EF4444") + """;">""" + ("Online" if pipeline is not None else "Missing") + """</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 9. TOP HEADER NAVBAR
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header-card">
    <div class="app-header-left">
        <div class="app-header-title">Production Quality Prediction Using Machine Learning</div>
        <div class="app-header-subtitle">UCI SECOM Semiconductor Manufacturing Dataset — Inference & Evaluation System</div>
    </div>
    <div class="app-header-badges">
        <span class="badge-pill">Dataset: UCI SECOM</span>
        <span class="badge-pill">Task: Binary Class (0/1)</span>
        <span class="badge-pill-accent">Model: Decision Tree</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
if st.session_state.nav_section == "Overview":
    
    st.markdown('<div class="section-banner">Project Metrics & Evaluation Protocol</div>', unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    with kpi1:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Dataset</div>
                <div class="metric-grid-value">UCI SECOM</div>
            </div>
            <div class="metric-grid-meta">1,567 Total Wafers</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Classification</div>
                <div class="metric-grid-value">Binary</div>
            </div>
            <div class="metric-grid-meta">0: Pass | 1: Fail</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Sensor Space</div>
                <div class="metric-grid-value">590 → 474</div>
            </div>
            <div class="metric-grid-meta">116 dropped (NaN/Const)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Selected Features</div>
                <div class="metric-grid-value">Top 60</div>
            </div>
            <div class="metric-grid-meta">Mutual Information</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Models Compared</div>
                <div class="metric-grid-value">5 Models</div>
            </div>
            <div class="metric-grid-meta">5-Fold Stratified CV</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi6:
        st.markdown("""
        <div class="metric-grid-card" style="border-top: 2px solid #2563EB;">
            <div>
                <div class="metric-grid-label">Final Model</div>
                <div class="metric-grid-value" style="font-size: 1.15rem; color: #2563EB;">Decision Tree</div>
            </div>
            <div class="metric-grid-meta">Selected by CV F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Executive Summary & Manufacturing Problem</div>
            <p style="font-size: 0.84rem; color: #334155; line-height: 1.5; margin-bottom: 8px;">
                In semiconductor wafer fabrication, continuous sensors monitor complex deposition, etching, chemical-mechanical polishing (CMP), and photolithography processes across tools. Identifying defective wafers early in the inline manufacturing flow prevents downstream packaging scrap and costly field failures.
            </p>
            <p style="font-size: 0.84rem; color: #334155; font-weight: 600; margin-bottom: 4px;">
                Analytical Characteristics of the UCI SECOM Dataset:
            </p>
            <ul style="font-size: 0.82rem; color: #475569; line-height: 1.55; margin-top: 0;">
                <li><strong>High Dimensionality:</strong> 590 continuous inline sensor signals per wafer run.</li>
                <li><strong>Missing Data & Inactive Sensors:</strong> 116 completely empty or constant features removed before modeling.</li>
                <li><strong>Extreme Class Imbalance:</strong> Only <strong>6.6%</strong> of production runs result in defect failures (104 Fail vs 1,463 Pass).</li>
                <li><strong>The Accuracy Paradox:</strong> A naive baseline predicting all samples as "Pass" achieves 93.4% accuracy but captures 0% of defects.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">End-to-End Pipeline Workflow</div>
            <div style="font-size: 0.82rem; color: #334155; line-height: 1.55;">
                1. <strong>Pre-Pipeline Cleaning:</strong> 590 raw features → Drop 116 all-missing/zero-variance features → 474 usable features.<br>
                2. <strong>Partitioning:</strong> Stratified 80/20 train/test split (1,253 training records / 314 held-out test records).<br>
                3. <strong>Encapsulated Pipeline:</strong> SimpleImputer(median) → VarianceThreshold(0.0) → SelectKBest(Mutual Info, k=60) → DecisionTreeClassifier(balanced).<br>
                4. <strong>Cross-Validation:</strong> 5-Fold Stratified CV used for model selection, prioritizing F1 score over raw accuracy.<br>
                5. <strong>Model Selection:</strong> Decision Tree selected because it achieved the highest cross-validation F1 score (0.1625) and defect recall (40.51%).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Project Verification Matrix</div>
            <table style="width: 100%; font-size: 0.81rem; color: #334155; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Dataset</td>
                    <td style="font-weight: 700; text-align: right;">UCI SECOM Dataset</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Total Production Runs</td>
                    <td style="font-weight: 700; text-align: right;">1,567 Wafers</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Overall Failure Rate</td>
                    <td style="font-weight: 700; text-align: right;">6.64% (104 Fail / 1,463 Pass)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Training Partition (80%)</td>
                    <td style="font-weight: 700; text-align: right;">1,253 samples</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Held-Out Test Partition (20%)</td>
                    <td style="font-weight: 700; text-align: right;">314 samples</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Raw Sensor Channels</td>
                    <td style="font-weight: 700; text-align: right;">590 Features</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Pre-Filtered Features</td>
                    <td style="font-weight: 700; text-align: right;">116 Removed</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Features Entering Pipeline</td>
                    <td style="font-weight: 700; text-align: right;">474 Features</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Selected Pipeline Features</td>
                    <td style="font-weight: 700; text-align: right;">Top 60 (Mutual Info)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Final Selected Model</td>
                    <td style="font-weight: 700; color: #2563EB; text-align: right;">Decision Tree</td>
                </tr>
                <tr style="height: 26px;">
                    <td style="color: #64748B; font-weight: 600;">Held-Out Test Accuracy</td>
                    <td style="font-weight: 700; text-align: right;">76.43%</td>
                </tr>
            </table>
            
            <div class="callout-info" style="margin-top: 10px;">
                Navigate to <strong>Quality Prediction</strong> to run live inferences on authentic SECOM wafer records.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: QUALITY PREDICTION (MAIN LIVE DEMONSTRATION)
# =============================================================================
elif st.session_state.nav_section == "Prediction":
    
    if load_error:
        st.error(load_error)
        st.stop()

    st.markdown('<div class="section-banner">Sensor Ingestion & Real-Time Assessment</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Select Ingestion Mode:",
        options=[
            "Option A: Load Authentic SECOM Benchmark Samples (Instant Demo)",
            "Option B: Upload Sensor Telemetry File (CSV / Space-Separated .data)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------------------
    # Mode A: Benchmark Sample Loader
    # -------------------------------------------------------------------------
    if "Option A" in input_mode:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Load Verified SECOM Wafer Records</div>
            <div style="font-size: 0.81rem; color: #64748B; margin-bottom: 10px;">
                Select from verified UCI SECOM production runs to test the Decision Tree pipeline on real sensor data.
            </div>
        """, unsafe_allow_html=True)
        
        b1, b2, b3, b4 = st.columns(4)
        
        with b1:
            if st.button("Pass Wafer Sample", use_container_width=True):
                sample_df, truth, label = load_real_secom_sample("pass")
                if sample_df is not None:
                    st.session_state.active_input_df = sample_df
                    st.session_state.active_source_label = label
                    st.session_state.active_ground_truth = truth
                    st.rerun()

        with b2:
            if st.button("Fail Wafer Sample", use_container_width=True):
                sample_df, truth, label = load_real_secom_sample("fail")
                if sample_df is not None:
                    st.session_state.active_input_df = sample_df
                    st.session_state.active_source_label = label
                    st.session_state.active_ground_truth = truth
                    st.rerun()

        with b3:
            if st.button("Random Wafer Run", use_container_width=True):
                sample_df, truth, label = load_real_secom_sample("random")
                if sample_df is not None:
                    st.session_state.active_input_df = sample_df
                    st.session_state.active_source_label = label
                    st.session_state.active_ground_truth = truth
                    st.rerun()

        with b4:
            if st.button("10-Wafer Batch Lot", use_container_width=True):
                sample_df, truth, label = load_real_secom_sample("batch_10")
                if sample_df is not None:
                    st.session_state.active_input_df = sample_df
                    st.session_state.active_source_label = label
                    st.session_state.active_ground_truth = truth
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Mode B: CSV File Upload
    # -------------------------------------------------------------------------
    else:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Upload Sensor Data File</div>
            <div style="font-size: 0.81rem; color: #64748B; margin-bottom: 10px;">
                Upload a CSV or space-delimited <code>secom.data</code> file containing sensor observations (590 or 474 columns).
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Sensor File",
            type=["csv", "data", "txt"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                try:
                    df_up = pd.read_csv(uploaded_file)
                    if df_up.shape[1] < 100:
                        uploaded_file.seek(0)
                        df_up = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
                except Exception:
                    uploaded_file.seek(0)
                    df_up = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

                st.session_state.active_input_df = df_up
                st.session_state.active_source_label = f"Uploaded File: '{uploaded_file.name}'"
                st.session_state.active_ground_truth = None
            except Exception as e:
                st.error(f"Prediction could not be completed. Please verify the uploaded data format. ({str(e)})")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Active Sample Preview & Validation
    # -------------------------------------------------------------------------
    if st.session_state.active_input_df is not None:
        raw_input = st.session_state.active_input_df
        prep_result, val_err = validate_and_prepare_input(raw_input, pipeline)

        if val_err:
            st.error(val_err)
        else:
            st.markdown(f"""
            <div class="content-box">
                <div class="content-box-title">Active Data Ingestion: {st.session_state.active_source_label}</div>
                <div style="display: flex; gap: 20px; font-size: 0.81rem; color: #475569; margin-bottom: 8px; flex-wrap: wrap;">
                    <span>Rows: <strong>{prep_result['row_count']}</strong></span>
                    <span>Sensor Columns: <strong>{prep_result['col_count']}</strong></span>
                    <span>Features Entering Model: <strong>{prep_result['features_to_model']}</strong></span>
                    <span>Missing Values: <strong>{prep_result['missing_count']}</strong> (imputed by pipeline)</span>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("Inspect Raw Sensor Input Matrix", expanded=False):
                preview_cols = prep_result["df_model"].iloc[:, :10].copy()
                preview_cols.columns = [f"Sensor_{i+1}" for i in range(preview_cols.shape[1])]
                render_styled_table(preview_cols.round(3), align_right_cols=list(preview_cols.columns), max_rows=5)

            # Trigger Prediction Button
            if st.button("Run Quality Assessment", type="primary", use_container_width=True):
                try:
                    df_model_in = prep_result["df_model"]
                    preds = pipeline.predict(df_model_in)
                    
                    has_proba = hasattr(pipeline, "predict_proba")
                    probs = pipeline.predict_proba(df_model_in) if has_proba else None

                    # ---------------------------------------------------------
                    # Single Wafer Result Display
                    # ---------------------------------------------------------
                    if len(preds) == 1:
                        pred_val = int(preds[0])
                        prob_pass = float(probs[0][0]) if probs is not None else (1.0 if pred_val == 0 else 0.0)
                        prob_fail = float(probs[0][1]) if probs is not None else (0.0 if pred_val == 0 else 1.0)
                        
                        is_pass = (pred_val == 0)
                        status_str = "PASS" if is_pass else "FAIL"

                        # Log to session history
                        st.session_state.prediction_history.append({
                            "Run ID": f"#{len(st.session_state.prediction_history) + 1}",
                            "Data Source": st.session_state.active_source_label,
                            "Result": status_str,
                            "Pass Probability": f"{prob_pass * 100:.1f}%",
                            "Fail Probability": f"{prob_fail * 100:.1f}%"
                        })

                        r_col1, r_col2 = st.columns([1.3, 1])

                        with r_col1:
                            if is_pass:
                                st.markdown(f"""
                                <div class="result-card-pass">
                                    <div class="result-badge-pass">PASS — COMPLIANT</div>
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #166534; margin-bottom: 2px;">
                                        Production Quality: Compliant Wafer Run
                                    </div>
                                    <div style="font-size: 0.81rem; color: #15803D; line-height: 1.4;">
                                        Sensor telemetry satisfies nominal manufacturing yield criteria. Process variables remain within standard operational tolerances.
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-card-fail">
                                    <div class="result-badge-fail">FAIL — DEFECT RISK</div>
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #991B1B; margin-bottom: 2px;">
                                        Production Quality: Defect Risk Detected
                                    </div>
                                    <div style="font-size: 0.81rem; color: #B91C1C; line-height: 1.4;">
                                        Sensor telemetry indicates anomalous process signatures. Wafer is flagged for metrology containment and physical defect review.
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        with r_col2:
                            st.markdown("""
                            <div class="content-box" style="margin-top: 10px;">
                                <div class="content-box-title">Predicted Class Probabilities</div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div style="margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: #0F172A; margin-bottom: 2px;">
                                    <span>Pass Probability (Class 0)</span>
                                    <span style="color: #16A34A;">{prob_pass * 100:.1f}%</span>
                                </div>
                                <div style="width: 100%; height: 8px; background-color: #E2E8F0; border-radius: 4px; overflow: hidden;">
                                    <div style="width: {prob_pass * 100}%; height: 100%; background-color: #16A34A;"></div>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600; color: #0F172A; margin-bottom: 2px;">
                                    <span>Fail Probability (Class 1)</span>
                                    <span style="color: #DC2626;">{prob_fail * 100:.1f}%</span>
                                </div>
                                <div style="width: 100%; height: 8px; background-color: #E2E8F0; border-radius: 4px; overflow: hidden;">
                                    <div style="width: {prob_fail * 100}%; height: 100%; background-color: #DC2626;"></div>
                                </div>
                            </div>
                            
                            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.35;">
                                Note: Class probabilities are derived directly from leaf distributions in the trained Decision Tree estimator.
                            </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ---------------------------------------------------------
                    # Multi-Row Batch Prediction Display
                    # ---------------------------------------------------------
                    else:
                        total_count = len(preds)
                        pass_count = int((preds == 0).sum())
                        fail_count = int((preds == 1).sum())
                        pass_pct = (pass_count / total_count) * 100
                        fail_pct = (fail_count / total_count) * 100

                        st.session_state.prediction_history.append({
                            "Run ID": f"#{len(st.session_state.prediction_history) + 1}",
                            "Data Source": st.session_state.active_source_label,
                            "Result": f"{pass_count} Pass / {fail_count} Fail",
                            "Pass Probability": f"{pass_pct:.1f}%",
                            "Fail Probability": f"{fail_pct:.1f}%"
                        })

                        bm1, bm2, bm3, bm4, bm5 = st.columns(5)
                        with bm1:
                            st.metric("Total Evaluated", total_count)
                        with bm2:
                            st.metric("Compliant (Pass)", pass_count)
                        with bm3:
                            st.metric("Defects (Fail)", fail_count)
                        with bm4:
                            st.metric("Yield Pass %", f"{pass_pct:.1f}%")
                        with bm5:
                            st.metric("Defect Rate %", f"{fail_pct:.1f}%")

                        out_df = pd.DataFrame({
                            "Sample Row": [f"Wafer_{i+1}" for i in range(total_count)],
                            "Predicted Class": preds,
                            "Result": np.where(preds == 1, "FAIL", "PASS"),
                            "Pass Probability": probs[:, 0].round(4) if probs is not None else np.where(preds == 0, 1.0, 0.0),
                            "Fail Probability": probs[:, 1].round(4) if probs is not None else np.where(preds == 1, 1.0, 0.0)
                        })

                        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                        render_styled_table(out_df, align_right_cols=["Predicted Class", "Pass Probability", "Fail Probability"], max_rows=25)

                        csv_data = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Full Prediction Report (CSV)",
                            data=csv_data,
                            file_name="secom_quality_predictions.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Prediction could not be completed. ({str(e)})")

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Session Prediction History
    # -------------------------------------------------------------------------
    if len(st.session_state.prediction_history) > 0:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Session Prediction History</div>
        """, unsafe_allow_html=True)
        
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        render_styled_table(hist_df, align_right_cols=["Pass Probability", "Fail Probability"])
        
        if st.button("Clear Session History"):
            st.session_state.prediction_history = []
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 3: MODEL PERFORMANCE & CROSS-VALIDATION BENCHMARK
# =============================================================================
elif st.session_state.nav_section == "Model Performance":
    
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">Final Selected Model: Decision Tree (Trained Pipeline)</div>
        <div style="font-size: 0.82rem; color: #475569; margin-bottom: 10px;">
            The final model was selected using <strong>5-Fold Stratified Cross-Validation</strong>, prioritizing F1 score, followed by PR-AUC and Recall. 
            Decision Tree was selected because it achieved the highest cross-validation F1 score among all evaluated models.
        </div>
    """, unsafe_allow_html=True)

    # Held-Out Test Set Headline Performance (20% Split = 314 Samples)
    st.markdown("""
    <div style="font-size: 0.76rem; font-weight: 700; color: #2563EB; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">
        20% Held-Out Test Set Performance (314 Unseen Samples)
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1:
        st.markdown("""
        <div class="metric-grid-card" style="border-top: 2px solid #2563EB;">
            <div>
                <div class="metric-grid-label">Test Accuracy</div>
                <div class="metric-grid-value">76.43%</div>
            </div>
            <div class="metric-grid-meta">0.7643 Overall</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Test Precision</div>
                <div class="metric-grid-value">9.23%</div>
            </div>
            <div class="metric-grid-meta">0.0923 on Fail (1)</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Test Recall</div>
                <div class="metric-grid-value">28.57%</div>
            </div>
            <div class="metric-grid-meta">6 / 21 Defects Caught</div>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Test F1-Score</div>
                <div class="metric-grid-value">13.95%</div>
            </div>
            <div class="metric-grid-meta">0.1395 Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)
    with t5:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Test ROC-AUC</div>
                <div class="metric-grid-value">55.13%</div>
            </div>
            <div class="metric-grid-meta">0.5513 Discriminability</div>
        </div>
        """, unsafe_allow_html=True)
    with t6:
        st.markdown("""
        <div class="metric-grid-card">
            <div>
                <div class="metric-grid-label">Test PR-AUC</div>
                <div class="metric-grid-value">10.03%</div>
            </div>
            <div class="metric-grid-meta">0.1003 Precision-Recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Confusion Matrix & Breakdown
    cm_left, cm_right = st.columns([1, 1.2])

    with cm_left:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Held-Out Confusion Matrix (Test Partition)</div>
        """, unsafe_allow_html=True)
        
        cm_matrix = np.array([[234, 59], [15, 6]])
        
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F8FAFC')
        
        ax.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
        
        labels = [
            [f"TN = 234\n(True Pass)", f"FP = 59\n(False Alarm)"],
            [f"FN = 15\n(Escaped Defect)", f"TP = 6\n(Captured Defect)"]
        ]
        
        for i in range(2):
            for j in range(2):
                color = "white" if cm_matrix[i, j] > 100 else "#0F172A"
                ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontweight="700", fontsize=8.5)
                
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Pass (0)", "Predicted Fail (1)"], fontsize=8, fontweight="600")
        ax.set_yticklabels(["Actual Pass (0)", "Actual Fail (1)"], fontsize=8, fontweight="600")
        ax.set_xlabel("Predicted Quality", fontsize=8.5, fontweight="700", color="#334155")
        ax.set_ylabel("Actual Ground Truth", fontsize=8.5, fontweight="700", color="#334155")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with cm_right:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-title">Held-Out Test Partition Metrology Breakdown</div>
            <table style="width: 100%; font-size: 0.81rem; color: #334155; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #F1F5F9; height: 30px;">
                    <td style="font-weight: 600; color: #0F172A;">True Negatives (TN)</td>
                    <td style="font-weight: 700; color: #16A34A; text-align: right;">234 Wafers</td>
                    <td style="font-size: 0.74rem; color: #64748B; text-align: right;">Correctly approved compliant runs</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 30px;">
                    <td style="font-weight: 600; color: #0F172A;">False Positives (FP)</td>
                    <td style="font-weight: 700; color: #D97706; text-align: right;">59 Wafers</td>
                    <td style="font-size: 0.74rem; color: #64748B; text-align: right;">False alarms (unnecessary inspections)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 30px;">
                    <td style="font-weight: 600; color: #0F172A;">False Negatives (FN)</td>
                    <td style="font-weight: 700; color: #DC2626; text-align: right;">15 Wafers</td>
                    <td style="font-size: 0.74rem; color: #64748B; text-align: right;">Escaped defects (missed quality fails)</td>
                </tr>
                <tr style="border-bottom: 1px solid #F1F5F9; height: 30px;">
                    <td style="font-weight: 600; color: #0F172A;">True Positives (TP)</td>
                    <td style="font-weight: 700; color: #16A34A; text-align: right;">6 Wafers</td>
                    <td style="font-size: 0.74rem; color: #64748B; text-align: right;">Defective wafers caught before scrap</td>
                </tr>
                <tr style="height: 30px;">
                    <td style="font-weight: 700; color: #0F172A;">Total Test Samples</td>
                    <td style="font-weight: 800; color: #2563EB; text-align: right;">314 Wafers</td>
                    <td style="font-size: 0.74rem; color: #64748B; text-align: right;">293 Pass (93.3%) / 21 Fail (6.7%)</td>
                </tr>
            </table>
            
            <div class="callout-warning" style="margin-top: 10px;">
                <strong>Honest Academic Interpretation:</strong><br>
                The model demonstrates limited ability to detect the minority Fail class. Accuracy is higher than the Fail-class recall because the dataset is strongly imbalanced.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5-Fold Stratified Cross-Validation Benchmark Table & Comparison
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">5-Fold Stratified Cross-Validation — Training Data (1,253 Samples)</div>
        <div style="font-size: 0.81rem; color: #64748B; margin-bottom: 8px;">
            Cross-validation comparison conducted on the training split using identical preprocessing pipelines.
        </div>
    """, unsafe_allow_html=True)

    cv_df = pd.DataFrame([
        {
            "Model": "Decision Tree (Selected)",
            "Accuracy": "73.99% ± 4.5%",
            "Precision": "10.30% ± 2.9%",
            "Recall": "40.51% ± 20.0%",
            "F1-Score": "0.1625 ± 0.05",
            "ROC-AUC": "0.6036 ± 0.07",
            "PR-AUC": "0.1144 ± 0.02",
            "Selection Rationale": "Highest CV F1 score; best balance between recall and precision"
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": "68.55% ± 4.4%",
            "Precision": "9.72% ± 3.3%",
            "Recall": "44.49% ± 13.4%",
            "F1-Score": "0.1591 ± 0.05",
            "ROC-AUC": "0.6141 ± 0.06",
            "PR-AUC": "0.1599 ± 0.05",
            "Selection Rationale": "High recall but lower accuracy and lower F1 than Decision Tree"
        },
        {
            "Model": "Support Vector Machine (SVM)",
            "Accuracy": "90.26% ± 1.6%",
            "Precision": "19.22% ± 16.0%",
            "Recall": "13.09% ± 9.3%",
            "F1-Score": "0.1536 ± 0.12",
            "ROC-AUC": "0.6112 ± 0.07",
            "PR-AUC": "0.1372 ± 0.05",
            "Selection Rationale": "High accuracy but poor recall on minority failure class"
        },
        {
            "Model": "XGBoost",
            "Accuracy": "92.34% ± 0.5%",
            "Precision": "20.00% ± 16.3%",
            "Recall": "6.03% ± 5.4%",
            "F1-Score": "0.0911 ± 0.08",
            "ROC-AUC": "0.6924 ± 0.06",
            "PR-AUC": "0.1682 ± 0.02",
            "Selection Rationale": "Severely biased towards majority class; low defect recall"
        },
        {
            "Model": "Random Forest",
            "Accuracy": "93.30% ± 0.3%",
            "Precision": "0.00% ± 0.0%",
            "Recall": "0.00% ± 0.0%",
            "F1-Score": "0.0000 ± 0.00",
            "ROC-AUC": "0.7173 ± 0.06",
            "PR-AUC": "0.1984 ± 0.05",
            "Selection Rationale": "Predicted 100% Pass; failed to detect any defects (F1 = 0)"
        }
    ])

    render_styled_table(cv_df, align_right_cols=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"])

    # Comparative Metric Chart
    fig_cv, ax_cv = plt.subplots(figsize=(8.5, 3.2))
    fig_cv.patch.set_facecolor('white')
    ax_cv.set_facecolor('#F8FAFC')

    models_list = ["Decision Tree\n(Selected)", "Logistic\nRegression", "SVM", "XGBoost", "Random\nForest"]
    f1_scores = [0.1625, 0.1591, 0.1536, 0.0911, 0.0000]
    recall_scores = [0.4051, 0.4449, 0.1309, 0.0603, 0.0000]
    
    x = np.arange(len(models_list))
    width = 0.35

    rects1 = ax_cv.bar(x - width/2, f1_scores, width, label='F1-Score (Primary Metric)', color='#2563EB')
    rects2 = ax_cv.bar(x + width/2, recall_scores, width, label='Recall (Defect Capture)', color='#94A3B8')

    ax_cv.set_ylabel('Score (0.0 to 1.0)', fontsize=8, fontweight='600')
    ax_cv.set_title('5-Fold Cross-Validation: F1-Score & Recall Comparison', fontsize=9, fontweight='700', color='#0F172A')
    ax_cv.set_xticks(x)
    ax_cv.set_xticklabels(models_list, fontsize=8)
    ax_cv.set_ylim(0, 0.6)
    ax_cv.grid(axis='y', linestyle='--', alpha=0.35)
    ax_cv.legend(frameon=False, fontsize=8)

    for rect in rects1:
        height = rect.get_height()
        ax_cv.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='700', color='#2563EB')

    plt.tight_layout()
    st.pyplot(fig_cv)

    st.markdown("""
    <div style="font-size: 0.8rem; color: #475569; margin-top: 8px;">
        <strong>The Accuracy Paradox in Imbalanced Semiconductor Manufacturing:</strong><br>
        Notice how Random Forest achieved the highest overall accuracy (93.30%) yet had an F1-score and Recall of 0.0000 because it defaulted to predicting every single wafer as Pass. In industrial quality assurance, such a model is useless. Decision Tree with balanced class weights was chosen because it achieved the highest F1 score (0.1625) and meaningful defect recall (40.51%).
    </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: PIPELINE & FEATURE INTEL
# =============================================================================
elif st.session_state.nav_section == "Pipeline & Features":
    
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">Machine Learning Pipeline Architecture</div>
        <div style="font-size: 0.82rem; color: #475569; margin-bottom: 12px;">
            The model pipeline is fully encapsulated inside a Scikit-Learn <code>Pipeline</code> object, ensuring consistent transformation on new production inputs without data leakage.
        </div>
    """, unsafe_allow_html=True)

    # Pipeline Steps Flow
    st.markdown("""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px; margin-bottom: 14px;">
        <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.81rem;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #0F172A; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">INGESTION</span>
                <span><strong>590 Raw SECOM Continuous Sensor Channels</strong></span>
            </div>
            <div style="color: #94A3B8; padding-left: 24px;">↓ <em>Pre-filter: Remove 116 completely missing and constant features</em></div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #334155; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">STAGE 0</span>
                <span><strong>474 Clean Sensor Variables Entering Pipeline</strong></span>
            </div>
            <div style="color: #94A3B8; padding-left: 24px;">↓ <em>Scikit-Learn Pipeline Execution Begins</em></div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #2563EB; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">STAGE 1</span>
                <span><code>SimpleImputer(strategy='median')</code> — Imputes sensor dropouts using training medians</span>
            </div>
            <div style="color: #94A3B8; padding-left: 24px;">↓</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #2563EB; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">STAGE 2</span>
                <span><code>VarianceThreshold(threshold=0.0)</code> — Eliminates zero-variance non-informative signals</span>
            </div>
            <div style="color: #94A3B8; padding-left: 24px;">↓</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #2563EB; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">STAGE 3</span>
                <span><code>SelectKBest(score_func=mutual_info_classif, k=60)</code> — Selects top 60 non-linear informative features</span>
            </div>
            <div style="color: #94A3B8; padding-left: 24px;">↓</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #16A34A; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">STAGE 4</span>
                <span><code>DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight='balanced')</code></span>
            </div>
            <div style="color: #94A3B8; padding-left: 24px;">↓</div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #0F172A; color: #FFF; padding: 2px 7px; border-radius: 3px; font-weight: 700; font-size: 0.7rem;">OUTPUT</span>
                <span><strong>Binary Classification: 0 (Pass) vs 1 (Fail) + Class Probabilities</strong></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance Section
    st.markdown("""
    <div class="content-box-title">Feature Selection & Importance Metrics</div>
    <div style="font-size: 0.82rem; color: #475569; margin-bottom: 10px;">
        Features were selected using <strong>Mutual Information (MI)</strong>, measuring statistical dependence between sensor telemetry and wafer yield without assuming Gaussian distributions.
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("""
        <div style="font-weight: 600; font-size: 0.82rem; color: #0F172A; margin-bottom: 4px;">
            Top Mutual Information Selected Sensor Signals
        </div>
        """, unsafe_allow_html=True)

        if pipeline is not None and "feature_selection" in pipeline.named_steps:
            fs_step = pipeline.named_steps["feature_selection"]
            cols = pipeline.feature_names_in_
            support = fs_step.get_support()
            scores = fs_step.scores_
            
            selected_features = cols[support]
            selected_scores = scores[support]
            top_idx = np.argsort(selected_scores)[::-1][:10]
            
            top_mi_df = pd.DataFrame({
                "Rank": [f"#{i+1}" for i in range(len(top_idx))],
                "Sensor Feature": selected_features[top_idx],
                "Mutual Information Score": selected_scores[top_idx].round(4)
            })
            render_styled_table(top_mi_df, align_right_cols=["Mutual Information Score"])

    with col_f2:
        st.markdown("""
        <div style="font-weight: 600; font-size: 0.82rem; color: #0F172A; margin-bottom: 4px;">
            Decision Tree Split Importance (Gini Importance)
        </div>
        """, unsafe_allow_html=True)

        if pipeline is not None and "model" in pipeline.named_steps:
            tree_model = pipeline.named_steps["model"]
            fs_step = pipeline.named_steps["feature_selection"]
            selected_features = pipeline.feature_names_in_[fs_step.get_support()]
            
            importances = tree_model.feature_importances_
            top_imp_idx = np.argsort(importances)[::-1][:10]
            
            top_tree_df = pd.DataFrame({
                "Rank": [f"#{i+1}" for i in range(len(top_imp_idx))],
                "Sensor Feature": selected_features[top_imp_idx],
                "Gini Importance": importances[top_imp_idx].round(4)
            })
            render_styled_table(top_tree_df, align_right_cols=["Gini Importance"])

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 5: DATASET & ABOUT
# =============================================================================
elif st.session_state.nav_section == "Dataset & About":
    
    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">UCI SECOM Dataset Characteristics</div>
        <div style="font-size: 0.82rem; color: #475569; margin-bottom: 12px;">
            The SECOM dataset is a benchmark dataset collected from a semiconductor fabrication production line.
        </div>
    """, unsafe_allow_html=True)

    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("""
        <table style="width: 100%; font-size: 0.81rem; color: #334155; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Total Samples (Wafers)</td>
                <td style="font-weight: 700; text-align: right;">1,567 Process Runs</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Original Sensor Signals</td>
                <td style="font-weight: 700; text-align: right;">590 Features</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Empty / Constant Features</td>
                <td style="font-weight: 700; text-align: right;">116 Removed Features</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Features Entering Pipeline</td>
                <td style="font-weight: 700; text-align: right;">474 Features</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Selected Pipeline Features</td>
                <td style="font-weight: 700; text-align: right;">Top 60 (Mutual Info)</td>
            </tr>
            <tr style="height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Total Missing Values in Raw Data</td>
                <td style="font-weight: 700; text-align: right;">41,951 values (4.5%)</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown("""
        <table style="width: 100%; font-size: 0.81rem; color: #334155; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Training Split Size (80%)</td>
                <td style="font-weight: 700; text-align: right;">1,253 samples</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Training Class Breakdown</td>
                <td style="font-weight: 700; text-align: right;">1,170 Pass (93.4%) | 83 Fail (6.6%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Held-Out Test Split (20%)</td>
                <td style="font-weight: 700; text-align: right;">314 samples</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Test Class Breakdown</td>
                <td style="font-weight: 700; text-align: right;">293 Pass (93.3%) | 21 Fail (6.7%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #F1F5F9; height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Target Class 0 (Pass)</td>
                <td style="font-weight: 700; color: #16A34A; text-align: right;">Compliant Production Yield</td>
            </tr>
            <tr style="height: 28px;">
                <td style="font-weight: 600; color: #64748B;">Target Class 1 (Fail)</td>
                <td style="font-weight: 700; color: #DC2626; text-align: right;">Wafer Defect / Out of Spec</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="content-box">
        <div class="content-box-title">Academic Attribution & Context</div>
        <div style="font-size: 0.82rem; color: #334155; line-height: 1.55;">
            <strong>Project Title:</strong> Production Quality Prediction Using Machine Learning<br>
            <strong>Context:</strong> Machine Learning Final Project (Semester 5)<br>
            <strong>Dataset Citation:</strong> UCI Machine Learning Repository — SECOM Dataset (McCann, Johnston, & Bickerstaff, 2008).<br>
            <strong>Implementation Stack:</strong> Python 3.12, Scikit-Learn 1.6.1, Streamlit, Pandas, NumPy, Matplotlib.<br>
            <strong>Trained Artifact:</strong> Encapsulated Scikit-Learn Pipeline (<code>production_quality_model.joblib</code>).
        </div>
    </div>
    """, unsafe_allow_html=True)