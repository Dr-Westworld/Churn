"""
Churn AI — Telco Customer Churn Prediction Platform
=========================================================
Standalone Streamlit app. Loads the pkl produced by ProductionXGBoostPipeline
but re-implements preprocessing inline so Ray is never needed at inference time.

Advanced features beyond the reference examples:
  1. Churn probability gauge (Plotly)
  2. Risk tier system  (Low / Medium / High / Critical)
  3. Customer Lifetime Value at-risk estimate
  4. Rule-based risk driver analysis with domain-knowledge annotations
  5. Retention action playbook (recommendation engine)
  6. Retention impact — tornado chart showing probability drop per action
  7. Batch CSV scoring with portfolio analytics & downloadable results
  8. What-If simulator — live re-scoring as you tweak inputs
"""

# ── Imports ────────────────────────────────────────────────────────────────
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — dark-panel, data-terminal aesthetic ──────────────────────────────
# Signature element: the probability gauge + risk tier badge as a unified
# "mission control" readout. Everything else is deliberately quiet.
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0d1117; }
  [data-testid="stHeader"]           { background:#0d1117; }

  /* metric cards */
  .cg-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:16px 20px; margin:6px 0;
  }
  .cg-card-title  { font-size:0.75rem; color:#8b949e; text-transform:uppercase;
                    letter-spacing:0.08em; margin-bottom:4px; }
  .cg-card-value  { font-size:1.6rem; font-weight:700; color:#e6edf3; }
  .cg-card-sub    { font-size:0.8rem; color:#8b949e; margin-top:2px; }

  /* risk badge */
  .risk-badge {
    display:inline-block; padding:6px 18px; border-radius:20px;
    font-weight:700; font-size:1rem; letter-spacing:0.05em;
  }

  /* risk driver rows */
  .driver-row {
    display:flex; align-items:center; gap:10px;
    padding:8px 12px; border-radius:8px; margin:4px 0;
    background:#161b22; border:1px solid #30363d;
  }
  .driver-label { flex:1; font-size:0.9rem; color:#e6edf3; }
  .driver-value { font-size:0.85rem; color:#8b949e; font-family:monospace; }

  /* recommendation cards */
  .rec-card {
    padding:10px 14px; border-radius:8px; margin:5px 0;
    background:#161b22; border-left:4px solid #388bfd;
  }
  .rec-title  { font-weight:600; color:#e6edf3; font-size:0.9rem; }
  .rec-detail { color:#8b949e; font-size:0.82rem; margin-top:2px; }
  .rec-badge  { font-size:0.72rem; font-weight:700; padding:2px 8px;
                border-radius:10px; float:right; margin-left:8px; }

  /* section heading */
  .cg-section { color:#388bfd; font-size:0.78rem; font-weight:700;
                text-transform:uppercase; letter-spacing:0.1em;
                margin:18px 0 8px; }

  /* tab strip */
  div[data-testid="stTabs"] button { font-size:0.95rem; font-weight:600; }

  /* scrollable table */
  .scrolltable { max-height:380px; overflow-y:auto; }

  hr.cg-divider { border-color:#30363d; margin:18px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

TENURE_GROUP_MAP = {
    "0-1yr": 0, "1-2yr": 1, "2-3yr": 2,
    "3-4yr": 3, "4-5yr": 4, "5-6yr": 5,
}
CHARGE_GROUP_MAP = {"low": 0, "medium": 1, "high": 2, "very_high": 3}

SERVICE_COLS = [
    "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

RISK_TIERS = {
    "LOW":      ("#238636", "🟢", "Low Risk"),
    "MEDIUM":   ("#9e6a03", "🟡", "Medium Risk"),
    "HIGH":     ("#bd561d", "🟠", "High Risk"),
    "CRITICAL": ("#da3633", "🔴", "Critical Risk"),
}

SELECTBOX_OPTIONS = {
    "gender":           ["Male", "Female"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["Yes", "No"],
    "PhoneService":     ["Yes", "No"],
    "MultipleLines":    ["Yes", "No", "No phone service"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["Yes", "No", "No internet service"],
    "OnlineBackup":     ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport":      ["Yes", "No", "No internet service"],
    "StreamingTV":      ["Yes", "No", "No internet service"],
    "StreamingMovies":  ["Yes", "No", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

DEFAULT_ROW = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0, "TotalCharges": 840.0,
}

# ══════════════════════════════════════════════════════════════════════════
# Model loading — pure pkl, no Ray
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading Churn AI model …")
def load_pkg():
    candidates = sorted(
        Path("model").glob("xgboost_model_*.pkl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        st.error("❌ No `xgboost_model_*.pkl` found in **model/**. Train the model first.")
        st.stop()
    return joblib.load(candidates[0]), str(candidates[0].name)


try:
    PKG, MODEL_NAME = load_pkg()
except Exception as exc:
    st.error(f"❌ Could not load model: {exc}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# Preprocessing — mirrors pipeline.preprocess_features without Ray
# ══════════════════════════════════════════════════════════════════════════

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    label_encoders    = PKG["label_encoders"]
    impute_values     = PKG.get("impute_values",     {})
    categorical_modes = PKG.get("categorical_modes", {})
    numeric_stats     = PKG.get("numeric_stats",     {})

    num_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns
                if c != "Churn"]
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if c != "Churn"]

    for col in num_cols:
        df[col] = df[col].fillna(impute_values.get(col, df[col].median()))
    for col in cat_cols:
        df[col] = df[col].fillna(categorical_modes.get(col, "Unknown"))

    # Feature engineering
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr", "5-6yr"],
        )
        df["tenure_group"] = df["tenure_group"].astype(object).map(TENURE_GROUP_MAP).fillna(-1).astype(int)
        df["tenure_to_max"] = df["tenure"] / 72.0
        if "MonthlyCharges" in df.columns:
            df["tenure_monthly_ratio"] = df["tenure"] / (df["MonthlyCharges"] + 1)

    if "MonthlyCharges" in df.columns:
        df["charge_group"] = pd.cut(
            df["MonthlyCharges"], bins=[0, 30, 60, 90, 120],
            labels=["low", "medium", "high", "very_high"],
        )
        df["charge_group"] = df["charge_group"].astype(object).map(CHARGE_GROUP_MAP).fillna(-1).astype(int)

    avail = [c for c in SERVICE_COLS if c in df.columns]
    if avail:
        active = (df[avail] != "No") & (df[avail] != "No internet service")
        df["total_services"] = active.sum(axis=1)

    if "InternetService" in df.columns:
        df["has_internet"] = df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    if "PhoneService"    in df.columns:
        df["has_phone"]    = (df["PhoneService"] == "Yes").astype(int)
    if "Contract"        in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    if "PaymentMethod"   in df.columns:
        df["is_automatic_payment"] = df["PaymentMethod"].str.contains("automatic", na=False).astype(int)
    if "SeniorCitizen"  in df.columns:
        df["is_senior"]      = (df["SeniorCitizen"] == 1).astype(int)
    if "Dependents"     in df.columns:
        df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)

    risk = []
    if "Contract"        in df.columns: risk.append((df["Contract"]        == "Month-to-month").astype(int))
    if "PaymentMethod"   in df.columns: risk.append((df["PaymentMethod"]   == "Electronic check").astype(int))
    if "InternetService" in df.columns: risk.append((df["InternetService"] == "Fiber optic").astype(int))
    if risk:
        df["churn_risk_score"] = sum(risk)

    for col, enc in label_encoders.items():
        if col in df.columns:
            mapping = {v: i for i, v in enumerate(enc.classes_)}
            df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)

    for col, stats in numeric_stats.items():
        if col in df.columns and stats["std"] > 0:
            df[f"{col}_zscore"] = (df[col] - stats["mean"]) / stats["std"]

    return df.fillna(0)


def _score(row_dict: dict):
    """Score a single customer dict → (label, probability)."""
    df    = pd.DataFrame([row_dict])
    X     = _preprocess(df)
    model = PKG["model"]
    feats = PKG["feature_names"]
    for f in set(feats) - set(X.columns):
        X[f] = 0
    X = X[feats]
    proba = float(model.predict_proba(X)[0][1])
    pred  = int(model.predict(X)[0])
    te    = PKG["metadata"].get("target_encoder")
    if te is not None:
        pred = te.inverse_transform([pred])[0]
    return pred, proba


def _score_batch(df: pd.DataFrame):
    """Score a DataFrame → (predictions array, probabilities array)."""
    X     = _preprocess(df)
    model = PKG["model"]
    feats = PKG["feature_names"]
    for f in set(feats) - set(X.columns):
        X[f] = 0
    X = X[feats]
    probas = model.predict_proba(X)[:, 1].astype(float)
    preds  = model.predict(X)
    te     = PKG["metadata"].get("target_encoder")
    if te is not None:
        preds = te.inverse_transform(preds)
    return preds, probas


# ══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════

def tier(p: float) -> str:
    if p < 0.30: return "LOW"
    if p < 0.50: return "MEDIUM"
    if p < 0.70: return "HIGH"
    return "CRITICAL"


def clv_estimate(monthly: float, tenure_months: int, contract: str) -> tuple[float, float]:
    """Returns (estimated_clv, months_remaining)."""
    if contract == "Two year":
        rem = max(6, 24 - tenure_months % 24)
    elif contract == "One year":
        rem = max(3, 12 - tenure_months % 12)
    else:
        rem = 6          # historical average for M2M before churn
    return monthly * rem, rem


def gauge_chart(p: float) -> go.Figure:
    t = tier(p)
    color, _, label = RISK_TIERS[t]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p * 100, 1),
        number={"suffix": "%", "font": {"size": 52, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#0d1117"},
                {"range": [30, 50], "color": "#0d1117"},
                {"range": [50, 70], "color": "#0d1117"},
                {"range": [70, 100],"color": "#0d1117"},
            ],
            "threshold": {
                "line": {"color": "#8b949e", "width": 2},
                "thickness": 0.85, "value": 50,
            },
        },
        title={"text": f"<b>{label}</b>", "font": {"color": color, "size": 16}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin={"t": 60, "b": 10, "l": 20, "r": 20},
    )
    return fig


def risk_drivers(row: dict) -> list[dict]:
    """Return a list of annotated risk / safe factors for a customer."""
    drivers = []

    def add(label, value, level, note=""):
        drivers.append({"label": label, "value": value, "level": level, "note": note})

    c, pm, it = row.get("Contract",""), row.get("PaymentMethod",""), row.get("InternetService","")
    t  = row.get("tenure", 0)
    mc = row.get("MonthlyCharges", 0)
    ts = row.get("TechSupport", "")
    os = row.get("OnlineSecurity", "")

    # Contract
    if c == "Month-to-month":
        add("Contract Type",    c,  "HIGH",   "No lock-in — easiest to leave")
    elif c == "One year":
        add("Contract Type",    c,  "MEDIUM", "Annual commitment moderates risk")
    else:
        add("Contract Type",    c,  "SAFE",   "2-year commitment — strong retention signal")

    # Tenure
    if t < 6:
        add("Tenure",  f"{t} months", "HIGH",   "Very new customer — highest churn window")
    elif t < 18:
        add("Tenure",  f"{t} months", "MEDIUM", "Still within the early churn zone")
    else:
        add("Tenure",  f"{t} months", "SAFE",   "Established customer — lower base risk")

    # Payment
    if pm == "Electronic check":
        add("Payment Method", pm, "HIGH",   "Electronic check shows 2× higher churn vs auto-pay")
    elif "automatic" in pm.lower():
        add("Payment Method", pm, "SAFE",   "Automatic payment correlates with retention")
    else:
        add("Payment Method", pm, "MEDIUM", "Manual payment — some friction with service")

    # Internet
    if it == "Fiber optic":
        add("Internet Service", it, "MEDIUM",
            "High-value service — unhappy fiber users churn fast")
    elif it == "No":
        add("Internet Service", "None", "MEDIUM", "Low engagement — at risk from competitor bundles")
    else:
        add("Internet Service", it, "SAFE", "DSL — stable middle-tier service")

    # Tech support
    if ts in ("No", "No internet service") and it != "No":
        add("Tech Support", "Not subscribed", "MEDIUM",
            "Unresolved technical issues are a top churn trigger")
    elif ts == "Yes":
        add("Tech Support", "Active", "SAFE", "Supported customers retain better")

    # Security
    if os in ("No", "No internet service") and it != "No":
        add("Online Security", "Not subscribed", "LOW", "Missing bundle — vulnerable to competitor offers")
    elif os == "Yes":
        add("Online Security", "Active", "SAFE", "Security bundle increases switching cost")

    # Charges vs tenure
    if mc > 80 and t < 12:
        add("High Charges / Short Tenure", f"${mc:.0f}/mo", "HIGH",
            "Paying premium without feeling loyalty yet")

    return drivers


def retention_playbook(row: dict, p: float) -> list[dict]:
    """Return prioritised retention actions."""
    actions = []
    c, pm, it = row.get("Contract",""), row.get("PaymentMethod",""), row.get("InternetService","")
    t  = row.get("tenure", 0)
    ts = row.get("TechSupport", "")
    os = row.get("OnlineSecurity", "")

    if p > 0.70:
        actions.append({"icon": "📞", "title": "Proactive Outreach — within 48 h",
                        "detail": "Assign a dedicated retention agent. High-risk customers respond best to personal contact.",
                        "priority": "CRITICAL", "color": "#da3633"})

    if c == "Month-to-month":
        actions.append({"icon": "📋", "title": "Offer Annual Contract Discount",
                        "detail": "A 15–20 % discount on a 1-year plan eliminates the month-to-month escape hatch.",
                        "priority": "HIGH", "color": "#bd561d"})

    if "automatic" not in pm.lower():
        actions.append({"icon": "💳", "title": "Enrol in Auto-Pay",
                        "detail": "Offer a $5/month credit for switching to automatic bank transfer or credit card.",
                        "priority": "HIGH", "color": "#bd561d"})

    if ts in ("No", "No internet service") and it != "No":
        actions.append({"icon": "🛠️", "title": "Bundle Tech Support (3 months free)",
                        "detail": "Removes friction from service issues — the #1 stated reason for cancellation.",
                        "priority": "MEDIUM", "color": "#9e6a03"})

    if os in ("No", "No internet service") and it != "No":
        actions.append({"icon": "🔒", "title": "Add Online Security Package",
                        "detail": "Increases perceived value and switching cost. Offer at $2/month off for 6 months.",
                        "priority": "MEDIUM", "color": "#9e6a03"})

    if t < 12:
        actions.append({"icon": "🎁", "title": "Loyalty Reward — Early Milestone",
                        "detail": "Send a 1-year anniversary offer 30 days early. Early tenure intervention has 3× ROI.",
                        "priority": "MEDIUM", "color": "#9e6a03"})

    if not actions:
        actions.append({"icon": "✅", "title": "Monitor & Maintain",
                        "detail": "Low risk — include in standard quarterly health check NPS survey.",
                        "priority": "LOW", "color": "#238636"})
    return actions


def retention_impact(row_dict: dict, base_p: float) -> pd.DataFrame:
    """
    Compute probability delta for each 'best-case' scenario.
    Returns a DataFrame sorted by impact descending.
    """
    scenarios = {
        "Upgrade to 2-year contract":         {"Contract": "Two year"},
        "Upgrade to 1-year contract":         {"Contract": "One year"},
        "Switch to auto bank transfer":       {"PaymentMethod": "Bank transfer (automatic)"},
        "Switch to credit card (auto)":       {"PaymentMethod": "Credit card (automatic)"},
        "Add Tech Support":                   {"TechSupport": "Yes"},
        "Add Online Security":                {"OnlineSecurity": "Yes"},
        "Add Online Backup":                  {"OnlineBackup": "Yes"},
        "Reduce monthly charges by 10 %":    {"MonthlyCharges": max(1, row_dict["MonthlyCharges"] * 0.9)},
    }
    rows = []
    for action, overrides in scenarios.items():
        modified = {**row_dict, **overrides}
        _, new_p = _score(modified)
        delta = base_p - new_p
        if abs(delta) > 0.005:         # skip negligible changes
            rows.append({"Action": action,
                         "Churn probability": round(new_p * 100, 1),
                         "Δ reduction (pp)":  round(delta * 100, 1)})
    return (pd.DataFrame(rows)
              .sort_values("Δ reduction (pp)", ascending=False)
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════

c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("## 🛡️ Churn AI")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.9rem;'>"
        "Intelligent Customer Retention Platform · XGBoost · Telecom</span>",
        unsafe_allow_html=True,
    )
with c2:
    pm = PKG.get("performance_metrics", {})
    auc = pm.get("test_auc") or PKG["metadata"].get("test_auc")
    if auc:
        st.markdown(
            f"<div class='cg-card' style='text-align:right'>"
            f"<div class='cg-card-title'>Model</div>"
            f"<div class='cg-card-value' style='font-size:0.85rem;'>{MODEL_NAME[:40]}</div>"
            f"<div class='cg-card-sub'>Test AUC: {auc:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(MODEL_NAME[:40])

st.markdown("<hr class='cg-divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════

tab_single, tab_batch, tab_whatif = st.tabs([
    "🔍 Single Prediction",
    "📁 Batch Scoring",
    "🔧 What-If Simulator",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Prediction
# ─────────────────────────────────────────────────────────────────────────
with tab_single:
    st.markdown("<div class='cg-section'>Customer Details</div>", unsafe_allow_html=True)
    with st.form("single_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            gender       = st.selectbox("Gender",          SELECTBOX_OPTIONS["gender"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            Partner      = st.selectbox("Partner",         SELECTBOX_OPTIONS["Partner"])
            Dependents   = st.selectbox("Dependents",      SELECTBOX_OPTIONS["Dependents"])
            tenure       = st.slider("Tenure (months)", 0, 72, 12)
            MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
            TotalCharges   = st.number_input("Total Charges ($)",   0.0, 10000.0,
                                             float(tenure * 70), step=10.0)

        with col_b:
            PhoneService    = st.selectbox("Phone Service",     SELECTBOX_OPTIONS["PhoneService"])
            MultipleLines   = st.selectbox("Multiple Lines",    SELECTBOX_OPTIONS["MultipleLines"])
            InternetService = st.selectbox("Internet Service",  SELECTBOX_OPTIONS["InternetService"])
            OnlineSecurity  = st.selectbox("Online Security",   SELECTBOX_OPTIONS["OnlineSecurity"])
            OnlineBackup    = st.selectbox("Online Backup",     SELECTBOX_OPTIONS["OnlineBackup"])
            DeviceProtection = st.selectbox("Device Protection", SELECTBOX_OPTIONS["DeviceProtection"])

        with col_c:
            TechSupport     = st.selectbox("Tech Support",     SELECTBOX_OPTIONS["TechSupport"])
            StreamingTV     = st.selectbox("Streaming TV",     SELECTBOX_OPTIONS["StreamingTV"])
            StreamingMovies = st.selectbox("Streaming Movies", SELECTBOX_OPTIONS["StreamingMovies"])
            Contract        = st.selectbox("Contract",         SELECTBOX_OPTIONS["Contract"])
            PaperlessBilling = st.selectbox("Paperless Billing", SELECTBOX_OPTIONS["PaperlessBilling"])
            PaymentMethod   = st.selectbox("Payment Method",  SELECTBOX_OPTIONS["PaymentMethod"])

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        row = dict(
            gender=gender, SeniorCitizen=SeniorCitizen, Partner=Partner,
            Dependents=Dependents, tenure=tenure, PhoneService=PhoneService,
            MultipleLines=MultipleLines, InternetService=InternetService,
            OnlineSecurity=OnlineSecurity, OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection, TechSupport=TechSupport,
            StreamingTV=StreamingTV, StreamingMovies=StreamingMovies,
            Contract=Contract, PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod, MonthlyCharges=MonthlyCharges,
            TotalCharges=TotalCharges,
        )
        st.session_state["last_row"]   = row          # shared with What-If tab
        st.session_state["last_proba"] = None

        with st.spinner("Scoring …"):
            label, p = _score(row)
        st.session_state["last_proba"] = p

        t_key = tier(p)
        t_color, t_icon, t_label = RISK_TIERS[t_key]
        clv, rem = clv_estimate(MonthlyCharges, tenure, Contract)

        st.markdown("<hr class='cg-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='cg-section'>Prediction Results</div>", unsafe_allow_html=True)

        # ── Row 1: gauge + metrics ──────────────────────────────────────
        col_g, col_m = st.columns([2, 3])

        with col_g:
            st.plotly_chart(gauge_chart(p), use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"<div style='text-align:center;margin-top:-10px;'>"
                f"<span class='risk-badge' style='background:{t_color};color:#fff;'>"
                f"{t_icon} {t_label}</span></div>",
                unsafe_allow_html=True,
            )

        with col_m:
            result_label = (
                '<b style="color:#da3633">Will Churn</b>'
                if p > 0.5 else
                '<b style="color:#238636">Will Stay</b>'
            )
            st.markdown(
                f"<div class='cg-card'>"
                f"<div class='cg-card-title'>Churn Probability</div>"
                f"<div class='cg-card-value' style='color:{t_color};'>{p*100:.1f}%</div>"
                f"<div class='cg-card-sub'>Threshold: 50% · Predicted: {result_label}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='cg-card'>"
                f"<div class='cg-card-title'>Customer Lifetime Value at Risk</div>"
                f"<div class='cg-card-value'>${clv * p:,.0f}</div>"
                f"<div class='cg-card-sub'>CLV (next {rem:.0f} mo): ${clv:,.0f} · "
                f"Expected loss if no action: ${clv*p:,.0f}</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='cg-card'>"
                f"<div class='cg-card-title'>Customer Snapshot</div>"
                f"<div class='cg-card-sub'>"
                f"Tenure: <b>{tenure} mo</b> · Contract: <b>{Contract}</b> · "
                f"Monthly: <b>${MonthlyCharges:.0f}</b> · Internet: <b>{InternetService}</b>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # ── Row 2: drivers + playbook ───────────────────────────────────
        col_d, col_r = st.columns(2)

        with col_d:
            st.markdown("<div class='cg-section'>Risk Driver Analysis</div>", unsafe_allow_html=True)
            level_color = {"HIGH": "#da3633", "MEDIUM": "#9e6a03",
                           "LOW": "#8b949e",  "SAFE": "#238636"}
            for d in risk_drivers(row):
                lc = level_color.get(d["level"], "#8b949e")
                st.markdown(
                    f"<div class='driver-row'>"
                    f"<span style='width:10px;height:10px;border-radius:50%;"
                    f"background:{lc};display:inline-block;flex-shrink:0;'></span>"
                    f"<span class='driver-label'>{d['label']}</span>"
                    f"<span class='driver-value'>{d['value']}</span>"
                    f"</div>"
                    f"<div style='font-size:0.75rem;color:#8b949e;margin:-2px 0 4px 22px;'>{d['note']}</div>",
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown("<div class='cg-section'>Retention Playbook</div>", unsafe_allow_html=True)
            for a in retention_playbook(row, p):
                ac, ai, at, ap, ad = a["color"], a["icon"], a["title"], a["priority"], a["detail"]
                st.markdown(
                    f"<div class='rec-card' style='border-left-color:{ac};'>"
                    f"<div class='rec-title'>{ai} {at}"
                    f"<span class='rec-badge' style='background:{ac};color:#fff;'>"
                    f"{ap}</span></div>"
                    f"<div class='rec-detail'>{ad}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Row 3: retention impact tornado ────────────────────────────
        st.markdown("<div class='cg-section'>Retention Impact — What If We Act?</div>",
                    unsafe_allow_html=True)
        with st.spinner("Computing scenario impacts …"):
            impact_df = retention_impact(row, p)

        if not impact_df.empty:
            fig_tornado = px.bar(
                impact_df.head(6),
                x="Δ reduction (pp)", y="Action",
                orientation="h",
                color="Δ reduction (pp)",
                color_continuous_scale=["#388bfd", "#238636"],
                text=impact_df.head(6)["Δ reduction (pp)"].apply(lambda v: f"−{v:.1f} pp"),
                title="Churn probability reduction by retention action",
            )
            fig_tornado.update_traces(textposition="outside")
            fig_tornado.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e6edf3"}, showlegend=False,
                coloraxis_showscale=False, height=280,
                title_font_color="#8b949e", title_font_size=13,
                margin={"t": 40, "b": 10, "l": 10, "r": 40},
                yaxis={"autorange": "reversed"},
            )
            fig_tornado.update_xaxes(showgrid=False, zeroline=False,
                                     tickcolor="#30363d", color="#8b949e")
            fig_tornado.update_yaxes(tickcolor="#30363d", color="#e6edf3")
            st.plotly_chart(fig_tornado, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No meaningful probability reduction found for standard interventions.")


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch Scoring
# ─────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("<div class='cg-section'>Batch Customer Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV with the same columns as the training data. "
        "The app will score every row and return an annotated file with "
        "churn probability, risk tier, and CLV at risk.",
    )

    # Template download
    template_df = pd.DataFrame([DEFAULT_ROW])
    st.download_button(
        "⬇️ Download CSV template",
        template_df.to_csv(index=False),
        "churnguard_template.csv", "text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(raw):,} customers loaded** — preview:")
        st.dataframe(raw.head(5), use_container_width=True)

        if st.button("🚀 Score all customers", use_container_width=True):
            with st.spinner(f"Scoring {len(raw):,} rows …"):
                preds, probas = _score_batch(raw)

            out = raw.copy()
            out["churn_probability"] = np.round(probas * 100, 2)
            out["predicted_churn"]   = preds
            out["risk_tier"]         = [tier(p) for p in probas]
            out["clv_at_risk"]       = out.apply(
                lambda r: clv_estimate(
                    r.get("MonthlyCharges", 0),
                    int(r.get("tenure", 0)),
                    r.get("Contract", "Month-to-month"),
                )[0] * (r["churn_probability"] / 100),
                axis=1,
            ).round(2)

            # ── Portfolio analytics ──────────────────────────────────
            st.markdown("<div class='cg-section'>Portfolio Analytics</div>",
                        unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            high_risk = out[out["churn_probability"] >= 50]
            m1.metric("Total Customers",   f"{len(out):,}")
            m2.metric("Predicted to Churn", f"{int((probas>=0.5).sum()):,}",
                      f"{(probas>=0.5).mean()*100:.1f}%")
            m3.metric("High/Critical Risk", f"{(out['risk_tier'].isin(['HIGH','CRITICAL'])).sum():,}")
            m4.metric("Portfolio CLV at Risk", f"${out['clv_at_risk'].sum():,.0f}")

            col_p, col_q = st.columns(2)

            with col_p:
                tier_counts = out["risk_tier"].value_counts().reset_index()
                tier_counts.columns = ["Risk Tier", "Count"]
                tier_order  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                tier_colors = {
                    "LOW": "#238636", "MEDIUM": "#9e6a03",
                    "HIGH": "#bd561d", "CRITICAL": "#da3633",
                }
                tier_counts["Risk Tier"] = pd.Categorical(
                    tier_counts["Risk Tier"], categories=tier_order, ordered=True
                )
                tier_counts = tier_counts.sort_values("Risk Tier")
                fig_pie = px.pie(
                    tier_counts, names="Risk Tier", values="Count",
                    color="Risk Tier",
                    color_discrete_map=tier_colors,
                    title="Risk Tier Distribution",
                    hole=0.45,
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e6edf3"}, title_font_color="#8b949e",
                    height=280, margin={"t": 40, "b": 10},
                    legend={"font": {"color": "#e6edf3"}},
                )
                st.plotly_chart(fig_pie, use_container_width=True,
                                config={"displayModeBar": False})

            with col_q:
                fig_hist = px.histogram(
                    out, x="churn_probability", nbins=30,
                    title="Churn Probability Distribution",
                    color_discrete_sequence=["#388bfd"],
                )
                fig_hist.add_vline(x=50, line_color="#da3633", line_dash="dash",
                                   annotation_text="50% threshold",
                                   annotation_font_color="#da3633")
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e6edf3"}, title_font_color="#8b949e",
                    height=280, margin={"t": 40, "b": 10},
                    xaxis_title="Churn Probability (%)", yaxis_title="Customers",
                )
                fig_hist.update_xaxes(tickcolor="#30363d", color="#8b949e", showgrid=False)
                fig_hist.update_yaxes(tickcolor="#30363d", color="#8b949e",
                                      gridcolor="#21262d")
                st.plotly_chart(fig_hist, use_container_width=True,
                                config={"displayModeBar": False})

            # Top at-risk customers
            st.markdown("<div class='cg-section'>Top 10 Highest-Risk Customers</div>",
                        unsafe_allow_html=True)
            top10 = (out.sort_values("churn_probability", ascending=False)
                        .head(10)
                        [["churn_probability", "risk_tier", "clv_at_risk",
                          "Contract", "tenure", "MonthlyCharges"]])
            st.dataframe(top10.style.background_gradient(
                subset=["churn_probability"], cmap="Reds"
            ), use_container_width=True)

            # Download
            csv_out = out.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download scored results",
                csv_out,
                "churnguard_scored.csv",
                "text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — What-If Simulator
# ─────────────────────────────────────────────────────────────────────────
with tab_whatif:
    st.markdown("<div class='cg-section'>What-If Simulator</div>", unsafe_allow_html=True)
    st.markdown(
        "Start from the customer scored in **Single Prediction** (or adjust below) "
        "and see how each change moves the churn needle in real-time."
    )

    base = st.session_state.get("last_row", DEFAULT_ROW.copy())

    with st.expander("✏️ Edit base customer profile", expanded=not st.session_state.get("last_row")):
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            w_gender   = st.selectbox("Gender",   SELECTBOX_OPTIONS["gender"],
                                      index=SELECTBOX_OPTIONS["gender"].index(base.get("gender","Male")),
                                      key="w_gender")
            w_senior   = st.selectbox("Senior Citizen", [0, 1],
                                      index=base.get("SeniorCitizen", 0),
                                      format_func=lambda x: "Yes" if x else "No",
                                      key="w_senior")
            w_partner  = st.selectbox("Partner",  SELECTBOX_OPTIONS["Partner"],
                                      index=SELECTBOX_OPTIONS["Partner"].index(base.get("Partner","No")),
                                      key="w_partner")
            w_dep      = st.selectbox("Dependents", SELECTBOX_OPTIONS["Dependents"],
                                      index=SELECTBOX_OPTIONS["Dependents"].index(base.get("Dependents","No")),
                                      key="w_dep")
            w_tenure   = st.slider("Tenure", 0, 72, int(base.get("tenure", 12)), key="w_tenure")
            w_mc       = st.number_input("Monthly Charges ($)", 0.0, 200.0,
                                         float(base.get("MonthlyCharges", 70.0)),
                                         step=1.0, key="w_mc")
            w_tc       = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                         float(base.get("TotalCharges", 840.0)),
                                         step=10.0, key="w_tc")
        with wc2:
            w_phone    = st.selectbox("Phone Service", SELECTBOX_OPTIONS["PhoneService"],
                                      index=SELECTBOX_OPTIONS["PhoneService"].index(base.get("PhoneService","Yes")),
                                      key="w_phone")
            w_multi    = st.selectbox("Multiple Lines", SELECTBOX_OPTIONS["MultipleLines"],
                                      index=SELECTBOX_OPTIONS["MultipleLines"].index(base.get("MultipleLines","No")),
                                      key="w_multi")
            w_inet     = st.selectbox("Internet Service", SELECTBOX_OPTIONS["InternetService"],
                                      index=SELECTBOX_OPTIONS["InternetService"].index(base.get("InternetService","Fiber optic")),
                                      key="w_inet")
            w_sec      = st.selectbox("Online Security", SELECTBOX_OPTIONS["OnlineSecurity"],
                                      index=SELECTBOX_OPTIONS["OnlineSecurity"].index(base.get("OnlineSecurity","No")),
                                      key="w_sec")
            w_bak      = st.selectbox("Online Backup", SELECTBOX_OPTIONS["OnlineBackup"],
                                      index=SELECTBOX_OPTIONS["OnlineBackup"].index(base.get("OnlineBackup","No")),
                                      key="w_bak")
            w_dev      = st.selectbox("Device Protection", SELECTBOX_OPTIONS["DeviceProtection"],
                                      index=SELECTBOX_OPTIONS["DeviceProtection"].index(base.get("DeviceProtection","No")),
                                      key="w_dev")
        with wc3:
            w_tech     = st.selectbox("Tech Support", SELECTBOX_OPTIONS["TechSupport"],
                                      index=SELECTBOX_OPTIONS["TechSupport"].index(base.get("TechSupport","No")),
                                      key="w_tech")
            w_tv       = st.selectbox("Streaming TV", SELECTBOX_OPTIONS["StreamingTV"],
                                      index=SELECTBOX_OPTIONS["StreamingTV"].index(base.get("StreamingTV","No")),
                                      key="w_tv")
            w_mov      = st.selectbox("Streaming Movies", SELECTBOX_OPTIONS["StreamingMovies"],
                                      index=SELECTBOX_OPTIONS["StreamingMovies"].index(base.get("StreamingMovies","No")),
                                      key="w_mov")
            w_contract = st.selectbox("Contract", SELECTBOX_OPTIONS["Contract"],
                                      index=SELECTBOX_OPTIONS["Contract"].index(base.get("Contract","Month-to-month")),
                                      key="w_contract")
            w_paper    = st.selectbox("Paperless Billing", SELECTBOX_OPTIONS["PaperlessBilling"],
                                      index=SELECTBOX_OPTIONS["PaperlessBilling"].index(base.get("PaperlessBilling","Yes")),
                                      key="w_paper")
            w_pay      = st.selectbox("Payment Method", SELECTBOX_OPTIONS["PaymentMethod"],
                                      index=SELECTBOX_OPTIONS["PaymentMethod"].index(base.get("PaymentMethod","Electronic check")),
                                      key="w_pay")

    base_row = dict(
        gender=w_gender, SeniorCitizen=w_senior, Partner=w_partner,
        Dependents=w_dep, tenure=w_tenure, PhoneService=w_phone,
        MultipleLines=w_multi, InternetService=w_inet, OnlineSecurity=w_sec,
        OnlineBackup=w_bak, DeviceProtection=w_dev, TechSupport=w_tech,
        StreamingTV=w_tv, StreamingMovies=w_mov, Contract=w_contract,
        PaperlessBilling=w_paper, PaymentMethod=w_pay,
        MonthlyCharges=w_mc, TotalCharges=w_tc,
    )

    _, base_p = _score(base_row)
    bt = tier(base_p)
    bt_color, bt_icon, bt_label = RISK_TIERS[bt]

    st.markdown(
        f"<div class='cg-card'>"
        f"<div class='cg-card-title'>Current baseline</div>"
        f"<div class='cg-card-value' style='color:{bt_color};'>{base_p*100:.1f}%</div>"
        f"<div class='cg-card-sub'>{bt_icon} {bt_label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='cg-section'>Sensitivity Analysis — Tornado Chart</div>",
                unsafe_allow_html=True)
    st.caption("Each bar shows how much the churn probability changes if that single variable is set to its optimal value.")

    with st.spinner("Running sensitivity analysis …"):
        impact = retention_impact(base_row, base_p)

    if not impact.empty:
        # Annotate with current vs new probability
        impact["label"] = impact.apply(
            lambda r: f"{r['Action']}  ({base_p*100:.1f}% → {r['Churn probability']:.1f}%)", axis=1
        )
        fig_t = px.bar(
            impact, x="Δ reduction (pp)", y="label",
            orientation="h",
            color="Δ reduction (pp)",
            color_continuous_scale=["#388bfd", "#3fb950"],
            text=impact["Δ reduction (pp)"].apply(lambda v: f"−{v:.1f} pp"),
        )
        fig_t.update_traces(textposition="outside")
        fig_t.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e6edf3"}, showlegend=False,
            coloraxis_showscale=False,
            height=max(250, len(impact) * 48),
            margin={"t": 20, "b": 10, "l": 10, "r": 60},
            yaxis={"autorange": "reversed", "tickfont": {"size": 12}},
            xaxis_title="Churn probability reduction (percentage points)",
        )
        fig_t.update_xaxes(showgrid=False, zeroline=True, zerolinecolor="#30363d",
                           tickcolor="#30363d", color="#8b949e")
        fig_t.update_yaxes(tickcolor="#30363d", color="#e6edf3")
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

        best = impact.iloc[0]
        st.success(
            f"🏆 **Highest-impact action:** {best['Action']} — "
            f"reduces churn probability from **{base_p*100:.1f}%** to "
            f"**{best['Churn probability']:.1f}%** (−{best['Δ reduction (pp)']:.1f} pp)"
        )
    else:
        st.info("This customer has no high-impact intervention scenarios — already at low risk.")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<hr class='cg-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#484f58;font-size:0.78rem;'>"
    "Churn AI · XGBoost · For internal retention team use only"
    "</div>",
    unsafe_allow_html=True,
)