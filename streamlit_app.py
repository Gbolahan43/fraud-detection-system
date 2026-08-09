"""Fraud Detection System — Streamlit App.

An interactive, multi-page Streamlit application for detecting fraudulent users
on a cryptocurrency trading platform using an ensemble of three machine-learning
models (Logistic Regression, Random Forest, XGBoost).

Pages
-----
    Dashboard         — Key metrics & interactive charts
    Fraud Detection    — Single-user analysis, existing-user lookup, batch CSV upload
    Analytics          — Trading patterns, user segmentation, deposit behaviour
    Model Comparison   — Real performance metrics + model strengths
    About              — Project documentation

Author: Abdulbasit Olanrewaju Gbolahan
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

# --------------------------------------------------------------------------- #
# Feature columns — order must match the one used during model training.
# The authoritative list ships with the models in feature_columns.pkl; the
# literal below is only a fallback for when that file is missing, so retraining
# with a different feature set can't silently desync the app.
# --------------------------------------------------------------------------- #
FALLBACK_FEATURE_COLUMNS = [
    "total_deposited", "deposit_count", "avg_deposit",
    "total_withdrawn", "withdrawal_count", "avg_withdrawal",
    "total_trade_volume_usd", "trade_count", "avg_trade_usd",
    "unique_pairs_traded", "deposit_withdrawal_ratio",
    "withdrawal_deposit_ratio", "hours_deposit_to_withdrawal",
    "trade_to_deposit_ratio", "total_unique_assets", "activity_frequency",
]


def _load_feature_columns() -> list[str]:
    """Read the trained feature order, falling back to the literal above."""
    try:
        with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
            cols = pickle.load(f)
        return list(cols) if cols else FALLBACK_FEATURE_COLUMNS
    except (FileNotFoundError, pickle.UnpicklingError):
        return FALLBACK_FEATURE_COLUMNS


FEATURE_COLUMNS = _load_feature_columns()

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
MODEL_SHORT = {"Logistic Regression": "lr", "Random Forest": "rf", "XGBoost": "xgb"}
MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "Random Forest": "#55A868",
    "XGBoost": "#C44E52",
}

# --------------------------------------------------------------------------- #
# Chart theme — kept in one place so every figure matches the CSS below
# --------------------------------------------------------------------------- #
CHART_BG = "#1e293b"
CHART_FG = "#e2e8f0"
CHART_MUTED = "#94a3b8"

# Horizontal legend pinned above the plot area.
_H_LEGEND = dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)

RISK_COLORS = {
    "Low": "#28a745",
    "Medium": "#ffc107",
    "High": "#fd7e14",
    "Extreme": "#dc3545",
}

# --------------------------------------------------------------------------- #
# Modern CSS theme
# --------------------------------------------------------------------------- #
st.markdown(
    r"""
    <style>
        /* ---- General ----
           Background/text now come from .streamlit/config.toml; this only
           targets the stable app-container test id as a belt-and-braces
           fallback if that config is ever absent. */
        [data-testid="stAppViewContainer"] { color: #e2e8f0; }

        /* ---- Header banner ---- */
        .main-header {
            font-size: 2.6rem; font-weight: 700;
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            color: #ffffff; text-align: center;
            padding: 1.4rem 0.5rem; border-radius: 14px;
            margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(30,58,138,.35);
        }

        h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }

        /* ---- Cards ---- */
        .metric-card {
            background: linear-gradient(145deg, #1e293b, #334155);
            padding: 1rem; border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,.25);
            border: 1px solid #475569;
        }
        .info-card {
            background: #1e293b; border-radius: 12px; padding: 1.1rem;
            border-left: 4px solid #38bdf8; margin: 1rem 0;
        }
        .success-card  { background:#142e26; border-left:4px solid #28a745; border-radius:10px; padding:1rem; margin:1rem 0; }
        .warning-card  { background:#32240f; border-left:4px solid #ffc107; border-radius:10px; padding:1rem; margin:1rem 0; }
        .danger-card   { background:#32151a; border-left:4px solid #dc3545; border-radius:10px; padding:1rem; margin:1rem 0; }
        .neutral-card  { background:#1c2536; border-left:4px solid #64748b; border-radius:10px; padding:1rem; margin:1rem 0; }

        /* ---- Risk badges ---- */
        .risk-badge {
            display:inline-block; padding:6px 16px; border-radius:999px;
            font-weight:700; text-transform:uppercase; letter-spacing:.5px;
            font-size:.85rem;
        }
        .risk-low    { background:#166534; color:#4ade80; }
        .risk-medium { background:#78350f; color:#fbbf24; }
        .risk-high   { background:#7f1d1d; color:#fca5a5; }
        .risk-extreme{ background:#450a0a; color:#fecaca; }

        /* ---- Expandable sections ---- */
        .streamlit-expanderSummary { background: transparent !important; }

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar { width:8px; }
        ::-webkit-scrollbar-thumb { background:#475569; border-radius:4px; }

        /* ---- Footer note ---- */
        .footer-note { text-align:center; color:#64748b; font-size:.85rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resource / data loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    """Load all three trained ML models, the scaler and feature columns."""
    try:
        lr_model = joblib.load(MODELS_DIR / "logistic_regression_model.pkl")
        rf_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
        xgb_model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        return {
            "lr": lr_model,
            "rf": rf_model,
            "xgb": xgb_model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }
    except FileNotFoundError as e:
        st.sidebar.error(f"Model file not found: {e}")
        return None


@st.cache_data
def load_data():
    """Load all CSV data artefacts used by the app."""
    data = {"models_ok": True}

    def _read(path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return None

    data["trades"] = _read(DATA_DIR / "trades.csv")
    data["user_activity"] = _read(DATA_DIR / "user_activitycsv.csv")
    data["features"] = _read(PROCESSED_DIR / "fraud_features.csv")
    data["volatility"] = _read(PROCESSED_DIR / "btcngn_volatility.csv")
    data["deposits_by_day"] = _read(PROCESSED_DIR / "deposits_by_day.csv")
    data["deposits_by_hour"] = _read(PROCESSED_DIR / "deposits_by_hour.csv")
    return data


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean the model feature columns, matching model_saver.py."""
    return df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 999).fillna(0)



def build_feature_vector(inputs: dict) -> tuple[pd.DataFrame, dict]:
    """Convert raw user inputs into a feature frame plus the derived metrics.

    Returns
    -------
    (X, derived)
        ``X`` is a one-row DataFrame ordered to match ``FEATURE_COLUMNS``.
        ``derived`` holds every computed metric shown in the Key Indicators
        section, so callers never have to recompute them.
    """
    # Compute derived metrics
    total_deposited = inputs["total_deposited"]
    total_withdrawn = inputs["total_withdrawn"]
    total_trade_volume_usd = inputs["total_trade_volume_usd"]
    deposit_count = inputs["deposit_count"]
    withdrawal_count = inputs["withdrawal_count"]
    trade_count = inputs["trade_count"]

    avg_deposit = total_deposited / deposit_count if deposit_count > 0 else 0
    avg_withdrawal = total_withdrawn / withdrawal_count if withdrawal_count > 0 else 0
    avg_trade_usd = total_trade_volume_usd / trade_count if trade_count > 0 else 0
    deposit_withdrawal_ratio = total_deposited / total_withdrawn if total_withdrawn > 0 else 999
    withdrawal_deposit_ratio = total_withdrawn / total_deposited if total_deposited > 0 else 0
    exchange_rate = 1500  # NGN per USD (used in original feature engineering)
    trade_to_deposit_ratio = (
        total_trade_volume_usd / (total_deposited / exchange_rate)
        if total_deposited > 0 else 0
    )
    total_unique_assets = (
        inputs["unique_deposit_assets"] + inputs["unique_withdrawal_assets"]
    )
    activity_frequency = deposit_count + withdrawal_count + trade_count

    feature_vec = {
        "total_deposited": total_deposited,
        "deposit_count": deposit_count,
        "avg_deposit": avg_deposit,
        "total_withdrawn": total_withdrawn,
        "withdrawal_count": withdrawal_count,
        "avg_withdrawal": avg_withdrawal,
        "total_trade_volume_usd": total_trade_volume_usd,
        "trade_count": trade_count,
        "avg_trade_usd": avg_trade_usd,
        "unique_pairs_traded": inputs["unique_pairs_traded"],
        "deposit_withdrawal_ratio": deposit_withdrawal_ratio,
        "withdrawal_deposit_ratio": withdrawal_deposit_ratio,
        "hours_deposit_to_withdrawal": inputs["hours_deposit_to_withdrawal"],
        "trade_to_deposit_ratio": trade_to_deposit_ratio,
        "total_unique_assets": total_unique_assets,
        "activity_frequency": activity_frequency,
    }

    X = prepare_features(pd.DataFrame([feature_vec]))
    return X, {
        "avg_deposit": avg_deposit,
        "avg_withdrawal": avg_withdrawal,
        "avg_trade_usd": avg_trade_usd,
        "total_unique_assets": total_unique_assets,
        "activity_frequency": activity_frequency,
        "deposit_withdrawal_ratio": deposit_withdrawal_ratio,
        "withdrawal_deposit_ratio": withdrawal_deposit_ratio,
        "trade_to_deposit_ratio": trade_to_deposit_ratio,
        "hours_deposit_to_withdrawal": inputs["hours_deposit_to_withdrawal"],
    }


def predict_user(inputs: dict, models: dict) -> dict:
    """Run an input through all three models and return a result dict."""
    X, derived = build_feature_vector(inputs)
    scaler = models["scaler"]
    X_scaled = scaler.transform(X)

    results = {}
    for name, short in MODEL_SHORT.items():
        model = models[short]
        proba = model.predict_proba(X_scaled)[0][1]
        pred = model.predict(X_scaled)[0]
        results[name] = {"prediction": int(pred), "probability": float(proba)}

    preds = [results[m]["prediction"] for m in MODEL_NAMES]
    probas = [results[m]["probability"] for m in MODEL_NAMES]
    suspicious_votes = sum(preds)
    avg_proba = np.mean(probas)

    # Consensus risk level
    if suspicious_votes >= 2 and avg_proba >= 0.7:
        risk = "extreme"
    elif suspicious_votes >= 2:
        risk = "high"
    elif suspicious_votes == 1 or (0.4 <= avg_proba < 0.7):
        risk = "medium"
    else:
        risk = "low"

    return {
        "models": results,
        "derived": derived,
        "suspicious_votes": suspicious_votes,
        "avg_probability": avg_proba,
        "risk_level": risk,
    }


def risk_level_label(level: str) -> str:
    return {
        "low": "Low Risk — Likely Legitimate",
        "medium": "Medium Risk — Review Recommended",
        "high": "High Risk — Likely Suspicious",
        "extreme": "Extreme Risk — Highly Suspicious",
    }.get(level, "Unknown")


# --------------------------------------------------------------------------- #
# Plotly helpers
# --------------------------------------------------------------------------- #
def _style_fig(fig: go.Figure, **overrides) -> go.Figure:
    """Apply the shared dark chart theme, then any per-chart overrides."""
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=CHART_FG),
        **overrides,
    )
    return fig


def make_probability_chart(results: dict) -> go.Figure:
    """Horizontal bar chart comparing the three models' fraud probabilities."""
    fig = go.Figure()
    for name in MODEL_NAMES:
        proba = results["models"][name]["probability"]
        fig.add_trace(go.Bar(
            y=[name],
            x=[proba * 100],
            name=name,
            marker_color=MODEL_COLORS[name],
            text=[f"{proba*100:.2f}%"],
            textposition="outside",
            orientation="h",
        ))
    return _style_fig(
        fig,
        title="Fraud Probability by Model",
        xaxis_title="Probability (%)",
        yaxis_title="",
        showlegend=False,
        height=220,
        margin=dict(l=0, r=20, t=40, b=20),
        xaxis=dict(range=[0, 105], color=CHART_MUTED),
        yaxis=dict(color=CHART_MUTED),
    )


def _download_button_csv(df: pd.DataFrame, filename: str, label: str):
    """Add a Streamlit download button for a DataFrame as CSV."""
    return st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(models: dict | None, data: dict):
    st.sidebar.title("🛡️ Fraud Detection System")
    st.sidebar.markdown("---")

    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["📊 Dashboard", "🔍 Fraud Detection", "📈 Analytics",
         "🔬 Model Comparison", "ℹ️ About"],
        help="Navigate between app sections",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    if models:
        st.sidebar.success("✅ All Models Loaded")
        st.sidebar.caption("🧠 Logistic Regression | 🌲 Random Forest | 🚀 XGBoost")
    else:
        st.sidebar.error("❌ Models Not Found")
        st.sidebar.caption("Run `python model_saver.py` to generate them")

    if data["features"] is not None:
        st.sidebar.success("✅ Data Loaded")
        st.sidebar.caption(f"👥 {len(data['features']):,} users in dataset")
    else:
        st.sidebar.error("❌ Data Not Found")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Fraud Detection System** — v2.0")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return app_mode


# --------------------------------------------------------------------------- #
# Dashboard page
# --------------------------------------------------------------------------- #
def render_dashboard(models, data):
    st.markdown(
        '<div class="main-header">📊 Fraud Detection Dashboard</div>',
        unsafe_allow_html=True,
    )

    features = data["features"]
    trades = data["trades"]
    user_activity = data["user_activity"]

    if features is None:
        st.error(
            "⚠️ Feature data not loaded. Please ensure "
            "`data/processed/fraud_features.csv` exists."
        )
        return

    st.markdown(
        '<div class="info-card">📍 Real-time monitoring of cryptocurrency trading '
        "platform activity. Hover over charts for details.</div>",
        unsafe_allow_html=True,
    )

    # ---- KPI row -------------------------------------------------------
    suspicious_count = (
        int(features["is_suspicious"].sum()) if "is_suspicious" in features else 0
    )
    total_users = len(features)
    fraud_rate = suspicious_count / total_users * 100 if total_users else 0
    total_trades = len(trades) if trades is not None else 0
    total_deposits = 0.0
    if user_activity is not None and "activity_type" in user_activity.columns:
        deposits = user_activity[user_activity["activity_type"] == "deposit"]
        total_deposits = (
            deposits["amount"].sum() if "amount" in deposits.columns else 0.0
        )
    total_trade_vol_usd = (
        features["total_trade_volume_usd"].sum()
        if "total_trade_volume_usd" in features else 0
    )
    avg_deposit = (
        features["avg_deposit"].mean() if "avg_deposit" in features else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", f"{total_users:,}", border=True)
    with col2:
        st.metric(
            "Suspicious Users",
            f"{suspicious_count:,}",
            delta=f"{fraud_rate:.2f}% flagged",
            border=True,
        )
    with col3:
        st.metric("Total Trades", f"{total_trades:,}", border=True)
    with col4:
        st.metric("Total Deposits", f"₦{total_deposits:,.0f}", border=True)

    st.markdown("---")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%", border=True)
    with col6:
        st.metric("Total Trade Vol (USD)", f"${total_trade_vol_usd:,.0f}", border=True)
    with col7:
        st.metric("Avg Deposit", f"₦{avg_deposit:,.0f}", border=True)
    with col8:
        low_vol = (
            int(features["low_volume_trader"].sum())
            if "low_volume_trader" in features else 0
        )
        st.metric("Low-Volume Traders", f"{low_vol:,}", border=True)

    # ---- Charts --------------------------------------------------------
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Fraud Distribution")
        if "is_suspicious" in features.columns:
            fraud_dist = features["is_suspicious"].value_counts()
            fig = px.pie(
                names=["Legitimate", "Suspicious"],
                values=[fraud_dist.get(0, 0), fraud_dist.get(1, 0)],
                color=["Legitimate", "Suspicious"],
                color_discrete_sequence=["#28a745", "#dc3545"],
                hole=0.45,
            )
            _style_fig(fig, height=300, showlegend=True, legend=_H_LEGEND)
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.subheader("User Activity Frequency")
        if "activity_frequency" in features.columns:
            fig = px.histogram(
                features, x="activity_frequency", nbins=40,
                title="Distribution of User Activity Frequency",
                color_discrete_sequence=["#38bdf8"],
            )
            _style_fig(fig, height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    chart_col3, chart_col4 = st.columns(2)

    # Top trading pairs
    with chart_col3:
        st.subheader("Top Trading Pairs by USD Volume")
        if trades is not None and "pair" in trades.columns and "amount" in trades.columns:
            pair_vol = (
                trades.groupby("pair")["amount"].sum().nlargest(8).sort_values()
            )
            fig = px.bar(
                pair_vol, orientation="h", color=pair_vol.values,
                color_continuous_scale="blues",
                labels={"x": "Volume", "y": "Pair"},
            )
            _style_fig(
                fig, height=350, showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trading data not available.")

    # BTCNGN volatility
    with chart_col4:
        st.subheader("BTCNGN Volatility Trend")
        vol = data.get("volatility")
        if vol is not None and "date" in vol.columns and "volatility" in vol.columns:
            fig = px.line(
                vol, x="date", y=["volatility", "rolling_avg_volatility"],
                labels={"value": "Volatility", "variable": "Series"},
                title="BTCNGN Volatility Over Time",
            )
            _style_fig(fig, height=350, legend=_H_LEGEND)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Volatility data not available.")

    # Suspicious users table
    st.markdown("---")
    st.subheader("Top Suspicious Users (Rules-Based)")
    if "is_suspicious" in features.columns and "user_id" in features.columns:
        susp = (
            features[features["is_suspicious"] == 1]
            .nlargest(10, "withdrawal_deposit_ratio")
        )
        display_cols = [
            "user_id", "total_deposited", "total_withdrawn",
            "withdrawal_deposit_ratio", "hours_deposit_to_withdrawal",
        ]
        display_cols = [c for c in display_cols if c in susp.columns]
        st.dataframe(susp[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No suspicious users flagged by rules-based logic.")


# --------------------------------------------------------------------------- #
# Fraud Detection page
# --------------------------------------------------------------------------- #
def render_fraud_detection(models, data):
    st.markdown(
        '<div class="main-header">🔍 Fraud Detection Tool</div>',
        unsafe_allow_html=True,
    )

    if not models:
        st.error("⚠️ Models not loaded. Please run `python model_saver.py` first.")
        return

    features = data["features"]

    # ---- Input mode selector ---------------------------------------------
    input_mode = st.segmented_control(
        "How would you like to check a user?",
        options=["📝 Manual Entry", "👤 Existing User Lookup", "📁 Batch Upload (CSV)"],
        default="📝 Manual Entry",
        help="Choose how to provide user data for analysis",
    )

    if input_mode == "📝 Manual Entry":
        _render_manual_form(models)
    elif input_mode == "👤 Existing User Lookup":
        _render_user_lookup(models, features)
    elif input_mode == "📁 Batch Upload (CSV)":
        _render_batch_upload(models, features)


def _render_manual_form(models):
    """Render the manual entry form with st.form for validation."""
    st.markdown("### Enter User Information")

    with st.form(key="manual_input_form", border=True):
        col1, col2 = st.columns(2, gap="large", vertical_alignment="top")

        with col1:
            st.markdown("#### Deposit Information")
            total_deposited = st.number_input(
                "Total Deposited (NGN)", min_value=0.0, value=100000.0,
                step=1000.0, key="m_dep",
            )
            deposit_count = st.number_input(
                "Number of Deposits", min_value=0, value=5, step=1, key="m_dcount",
            )
            avg_deposit = (
                total_deposited / deposit_count if deposit_count > 0 else 0
            )
            st.info(f"Average Deposit: ₦{avg_deposit:,.2f}")

            st.markdown("#### Withdrawal Information")
            total_withdrawn = st.number_input(
                "Total Withdrawn (NGN)", min_value=0.0, value=80000.0,
                step=1000.0, key="m_wd",
            )
            withdrawal_count = st.number_input(
                "Number of Withdrawals", min_value=0, value=3, step=1, key="m_wcount",
            )
            avg_withdrawal = (
                total_withdrawn / withdrawal_count if withdrawal_count > 0 else 0
            )
            st.info(f"Average Withdrawal: ₦{avg_withdrawal:,.2f}")

        with col2:
            st.markdown("#### Trading Information")
            total_trade_volume_usd = st.number_input(
                "Total Trade Volume (USD)", min_value=0.0, value=500.0,
                step=100.0, key="m_tv",
            )
            trade_count = st.number_input(
                "Number of Trades", min_value=0, value=2, step=1, key="m_tcount",
            )
            avg_trade_usd = (
                total_trade_volume_usd / trade_count if trade_count > 0 else 0
            )
            st.info(f"Average Trade: ${avg_trade_usd:,.2f}")

            unique_pairs_traded = st.number_input(
                "Unique Pairs Traded", min_value=0, value=2, step=1, key="m_pairs",
            )

            st.markdown("#### Timing Information")
            hours_deposit_to_withdrawal = st.number_input(
                "Hours: First Deposit → First Withdrawal",
                min_value=0.0, value=24.0, step=1.0, key="m_hours",
            )

        st.markdown("---")
        st.markdown("#### Additional Metrics")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            unique_deposit_assets = st.number_input(
                "Unique Deposit Assets", min_value=0, value=2, step=1, key="m_uda",
            )
            unique_withdrawal_assets = st.number_input(
                "Unique Withdrawal Assets", min_value=0, value=2, step=1, key="m_uwa",
            )
        with gcol2:
            total_unique_assets = unique_deposit_assets + unique_withdrawal_assets
            activity_frequency = deposit_count + withdrawal_count + trade_count
            st.info(f"Total Unique Assets: {total_unique_assets}")
            st.info(f"Activity Frequency: {activity_frequency}")

        submitted = st.form_submit_button(
            "🔍 Analyze User", type="primary", use_container_width=True
        )

    if submitted:
        inputs = {
            "total_deposited": total_deposited, "deposit_count": deposit_count,
            "total_withdrawn": total_withdrawn, "withdrawal_count": withdrawal_count,
            "total_trade_volume_usd": total_trade_volume_usd, "trade_count": trade_count,
            "unique_pairs_traded": unique_pairs_traded,
            "hours_deposit_to_withdrawal": hours_deposit_to_withdrawal,
            "unique_deposit_assets": unique_deposit_assets,
            "unique_withdrawal_assets": unique_withdrawal_assets,
        }
        result = predict_user(inputs, models)
        _render_prediction_results(result)

        # Download result
        st.markdown("---")
        res_df = pd.DataFrame([{
            "avg_proba": result["avg_probability"],
            "suspicious_votes": result["suspicious_votes"],
            "risk_level": result["risk_level"],
            "lr_proba": result["models"]["Logistic Regression"]["probability"],
            "rf_proba": result["models"]["Random Forest"]["probability"],
            "xgb_proba": result["models"]["XGBoost"]["probability"],
            **{k: v for k, v in result["derived"].items()},
            **inputs,
        }])
        _download_button_csv(res_df, "fraud_analysis_result.csv", "Download Analysis Result")


def _render_user_lookup(models, features):
    """Let the user pick an existing user from the dataset and analyse them."""
    st.markdown("### Select an Existing User")
    if features is None or "user_id" not in features.columns:
        st.warning("Feature data not available for user lookup.")
        return

    user_ids = features["user_id"].astype(str).tolist()
    selected = st.selectbox(
        "Select User ID", options=user_ids, format_func=lambda x: x
    )

    if selected:
        row = features[features["user_id"].astype(str) == selected].iloc[0]
        st.markdown(f"**User:** `{selected}`")

        # Display some raw info
        info_cols = {}
        for col in ["total_deposited", "total_withdrawn", "total_trade_volume_usd",
                     "activity_frequency", "is_suspicious"]:
            if col in row.index:
                info_cols[col] = row[col]
        if info_cols:
            info_df = pd.DataFrame([info_cols])
            st.dataframe(info_df.T, use_container_width=True, hide_index=True)

        if st.button(
            "🔍 Analyze This User", type="primary", use_container_width=True
        ):
            inputs = {
                "total_deposited": float(row.get("total_deposited", 0)),
                "deposit_count": int(row.get("deposit_count", 0)),
                "total_withdrawn": float(row.get("total_withdrawn", 0)),
                "withdrawal_count": int(row.get("withdrawal_count", 0)),
                "total_trade_volume_usd": float(row.get("total_trade_volume_usd", 0)),
                "trade_count": int(row.get("trade_count", 0)),
                "unique_pairs_traded": int(row.get("unique_pairs_traded", 0)),
                "hours_deposit_to_withdrawal": float(
                    row.get("hours_deposit_to_withdrawal", 0)
                ),
                "unique_deposit_assets": int(row.get("unique_deposit_assets", 0)),
                "unique_withdrawal_assets": int(row.get("unique_withdrawal_assets", 0)),
            }
            result = predict_user(inputs, models)
            _render_prediction_results(result)

            # Show actual label
            if "is_suspicious" in row.index:
                actual = row["is_suspicious"]
                label = "✅ Legitimate" if actual == 0 else "🚨 Suspicious"
                st.info(f"Actual label for this user: **{label}**")


def _render_batch_upload(models, features):
    """Allow uploading a CSV of users and run batch predictions."""
    st.markdown("### Batch Upload CSV")
    st.markdown(
        "Upload a CSV containing user data. The following columns are required:"
    )
    st.markdown(f"`{', '.join(FEATURE_COLUMNS)}`")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        with st.spinner("Analyzing batch..."):
            df = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                return

            X = prepare_features(df)
            X_scaled = models["scaler"].transform(X)

            for name, short in MODEL_SHORT.items():
                proba_col = f"{name}_proba"
                pred_col = f"{name}_pred"
                df[pred_col] = models[short].predict(X_scaled)
                df[proba_col] = models[short].predict_proba(X_scaled)[:, 1]

            df["suspicious_votes"] = df[[f"{m}_pred" for m in MODEL_NAMES]].sum(axis=1)
            df["avg_probability"] = df[
                [f"{m}_proba" for m in MODEL_NAMES]
            ].mean(axis=1)
            df["risk_level"] = pd.cut(
                df["avg_probability"],
                bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
                labels=["Low", "Medium", "High", "Extreme"],
            )

            st.success(f"✅ Analyzed **{len(df)}** users")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary chart — guard against categories with zero users
            counts = (
                df["risk_level"]
                .value_counts()
                .reindex(["Low", "Medium", "High", "Extreme"])
                .fillna(0)
            )
            counts_df = counts.rename("User Count").rename_axis("Risk Level").reset_index()
            fig = px.bar(
                counts_df, x="Risk Level", y="User Count", color="Risk Level",
                color_discrete_map=RISK_COLORS,
                title="Batch Risk Distribution",
            )
            _style_fig(fig, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            _download_button_csv(df, "batch_analysis_results.csv", "Download Batch Results")


def _render_prediction_results(result: dict):
    """Display the model prediction results with visual elements."""
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    risk = result["risk_level"]
    badge_class = f"risk-{risk}"
    badge_label = risk_level_label(risk)

    # Overall verdict
    st.markdown(
        f'<div style="text-align:center; padding:1rem 0;">'
        f'<span class="risk-badge {badge_class}">{badge_label}</span></div>',
        unsafe_allow_html=True,
    )

    # Model bar chart
    fig = make_probability_chart(result)
    st.plotly_chart(fig, use_container_width=True)

    # Per-model cards
    col1, col2, col3 = st.columns(3)
    for col, name in zip([col1, col2, col3], MODEL_NAMES):
        with col:
            info = result["models"][name]
            pred = info["prediction"]
            proba = info["probability"] * 100
            color_box = "danger-card" if pred == 1 else "success-card"
            status = "🚨 SUSPICIOUS" if pred == 1 else "✅ LEGITIMATE"
            st.markdown(f"#### {name}")
            st.markdown(
                f'<div class="{color_box}">{status}<br>Probability: {proba:.2f}%</div>',
                unsafe_allow_html=True,
            )

    # Consensus
    st.markdown("---")
    st.markdown("### 🎯 Model Consensus")
    votes = result["suspicious_votes"]
    avg_p = result["avg_probability"]

    if votes == 3:
        st.error("🚨 **ALL 3 MODELS AGREE: HIGHLY SUSPICIOUS USER**")
    elif votes == 2:
        st.warning("⚠️ **MAJORITY (2/3) VOTE: LIKELY SUSPICIOUS**")
    elif votes == 1:
        st.info("ℹ️ **SPLIT DECISION: MANUAL REVIEW RECOMMENDED**")
    else:
        st.success("✅ **ALL 3 MODELS AGREE: LEGITIMATE USER**")

    st.metric("Average Fraud Probability", f"{avg_p*100:.2f}%", border=True)

    # Key indicators
    st.markdown("---")
    st.markdown("### 🔎 Key Indicators")
    derived = result["derived"]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        wr = derived["withdrawal_deposit_ratio"]
        if wr > 0.8:
            st.error(f"⚠️ High Withdrawal Ratio: {wr:.2%}")
        else:
            st.success(f"✅ Normal Withdrawal Ratio: {wr:.2%}")
    with col_b:
        tr = derived["trade_to_deposit_ratio"]
        if tr < 0.1:
            st.error(f"⚠️ Low Trading Activity: {tr:.2%}")
        else:
            st.success(f"✅ Good Trading Activity: {tr:.2%}")
    with col_c:
        hrs = derived["hours_deposit_to_withdrawal"]
        if hrs < 48:
            st.warning(f"⚠️ Quick Withdrawal: {hrs:.1f}h")
        else:
            st.success(f"✅ Normal Timing: {hrs:.1f}h")


# --------------------------------------------------------------------------- #
# Analytics page
# --------------------------------------------------------------------------- #
def render_analytics(models, data):
    st.markdown(
        '<div class="main-header">📈 Advanced Analytics</div>',
        unsafe_allow_html=True,
    )

    features = data["features"]
    trades = data["trades"]
    user_activity = data["user_activity"]

    if features is None:
        st.error(
            "⚠️ Data not loaded. Please ensure "
            "`data/processed/fraud_features.csv` exists."
        )
        return

    st.markdown(
        '<div class="info-card">Explore market behaviour, user segments, '
        "and deposit patterns.</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📊 Trading Patterns", "👥 User Segments", "💰 Deposit Analysis"])

    with tabs[0]:
        st.subheader("Trading Volume Distribution")
        fig = px.box(
            features, y="total_trade_volume_usd", points="all",
            title="Trading Volume Distribution (USD)",
            color_discrete_sequence=["#38bdf8"],
        )
        _style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        if trades is not None and "pair" in trades.columns:
            st.markdown("---")
            st.subheader("Trading Pair Breakdown")
            tvol = (
                trades.groupby("pair")["amount"]
                .agg(["sum", "count"])
                .reset_index()
            )
            tvol.columns = ["Pair", "Volume", "Count"]
            tvol = tvol.sort_values("Volume", ascending=True)
            fig = px.bar(
                tvol, x="Volume", y="Pair", color="Count",
                title="Total Volume & Trade Count by Pair",
                color_continuous_scale="turbid",
            )
            _style_fig(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("User Segmentation by Activity Frequency")
        segments = pd.DataFrame({
            "Segment": [
                "High Activity (>20)", "Medium Activity (11-20)",
                "Low Activity (1-10)", "Inactive (0)",
            ],
            "Count": [
                int((features["activity_frequency"] > 20).sum()),
                int(((features["activity_frequency"] > 10) &
                     (features["activity_frequency"] <= 20)).sum()),
                int(((features["activity_frequency"] > 0) &
                     (features["activity_frequency"] <= 10)).sum()),
                int((features["activity_frequency"] == 0).sum()),
            ],
        })
        fig = px.bar(
            segments, x="Segment", y="Count", title="User Activity Segments",
            color="Count", color_continuous_scale="blues",
        )
        _style_fig(fig, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Campaign Targeting")
        low_vol = features[
            (features["total_trade_volume_usd"] < 500)
            & (features["trade_count"] >= 2)
            & (features["trade_count"] <= 10)
        ]
        if "kenya_trader" in features.columns:
            kes = features[features["kenya_trader"].astype(bool)]
        else:
            kes = pd.DataFrame()
        st.info(f"Potential campaign targets (low-volume traders): **{len(low_vol)}**")
        if not kes.empty:
            st.info(f"Kenya traders: **{len(kes)}**")

    with tabs[2]:
        st.subheader("Deposit Activity Patterns")
        # By day of week
        dep_day = data.get("deposits_by_day")
        if dep_day is not None and "day_of_week" in dep_day.columns:
            day_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            dep_day = (
                dep_day.set_index("day_of_week")
                .reindex(day_order)
                .reset_index()
            )
            fig = px.bar(
                dep_day, x="day_of_week", y="deposit_count",
                title="Deposits by Day of Week", color="deposit_count",
                color_continuous_scale="turbid",
            )
            _style_fig(
                fig, xaxis_title="Day", yaxis_title="Deposit Count",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # By hour
        dep_hr = data.get("deposits_by_hour")
        if dep_hr is not None and "hour" in dep_hr.columns:
            fig = px.bar(
                dep_hr, x="hour", y="deposit_count",
                title="Deposits by Hour of Day", color="total_amount",
                color_continuous_scale="turbid",
            )
            _style_fig(
                fig, xaxis_title="Hour (0-23)", yaxis_title="Deposit Count",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: deposit vs withdrawal
        if (
            "total_deposited" in features.columns
            and "total_withdrawn" in features.columns
        ):
            st.markdown("---")
            st.subheader("Deposit vs Withdrawal Scatter")
            scatter_df = features
            color_col = None
            if "is_suspicious" in features.columns:
                # Map to labels so Plotly treats this as categorical rather than
                # applying a continuous colour scale to the 0/1 flag.
                color_col = "Status"
                scatter_df = features.assign(
                    Status=np.where(
                        features["is_suspicious"] == 1, "Suspicious", "Legitimate"
                    )
                )
            fig = px.scatter(
                scatter_df, x="total_deposited", y="total_withdrawn",
                color=color_col,
                title="Total Deposited vs Total Withdrawn",
                color_discrete_map={"Legitimate": "#28a745", "Suspicious": "#dc3545"},
            )
            _style_fig(fig)
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Model Comparison page
# --------------------------------------------------------------------------- #
def render_model_comparison(models, data):
    st.markdown(
        '<div class="main-header">🔬 Model Performance Comparison</div>',
        unsafe_allow_html=True,
    )

    features = data["features"]

    st.markdown("""
        ### Three Models in Action

        This system uses an ensemble of three machine-learning models to detect fraudulent users:

        1. **Logistic Regression** — Fast, interpretable linear baseline
        2. **Random Forest** — Ensemble of decision trees capturing non-linear patterns
        3. **XGBoost** — Gradient boosting, typically the strongest performer

        Comparing their independent predictions gives a **higher-confidence consensus**.
    """)

    st.markdown("---")
    st.subheader("📊 Model Characteristics")
    comparison_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Type": ["Linear", "Ensemble (Bagging)", "Ensemble (Boosting)"],
        "Speed": ["⚡ Very Fast", "⚡⚡ Fast", "⚡⚡⚡ Medium"],
        "Interpretability": ["High", "Medium", "Medium"],
        "Handles Imbalance": ["Good", "Good", "Excellent"],
        "Best For": ["Linear patterns", "Non-linear patterns", "Complex patterns"],
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ---- Held-out model metrics ------------------------------------------
    st.markdown("---")
    st.subheader("📈 Held-Out Test Performance")
    if models and features is not None:
        with st.spinner("Computing model performance..."):
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
            )
            from sklearn.model_selection import train_test_split

            # Recreate the exact split model_saver.py trained with, so these
            # scores come from data the models never saw.
            X_all = prepare_features(features)
            y_all = features["is_suspicious"]
            _, X, _, y = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            X_scaled = models["scaler"].transform(X)

            st.caption(
                f"Scored on {len(X):,} held-out users (20% stratified test split, "
                "random_state=42) — the same split used during training."
            )

            metric_rows = []
            for name, short in MODEL_SHORT.items():
                m = models[short]
                y_pred = m.predict(X_scaled)
                metric_rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y, y_pred),
                    "Precision": precision_score(y, y_pred, zero_division=0),
                    "Recall": recall_score(y, y_pred, zero_division=0),
                    "F1-Score": f1_score(y, y_pred, zero_division=0),
                })
            metric_df = pd.DataFrame(metric_rows)
            st.dataframe(
                metric_df, use_container_width=True, hide_index=True,
                column_config={
                    "Accuracy": st.column_config.ProgressColumn(
                        "Accuracy", format="%.3f", min_value=0, max_value=1),
                    "Precision": st.column_config.ProgressColumn(
                        "Precision", format="%.3f", min_value=0, max_value=1),
                    "Recall": st.column_config.ProgressColumn(
                        "Recall", format="%.3f", min_value=0, max_value=1),
                    "F1-Score": st.column_config.ProgressColumn(
                        "F1-Score", format="%.3f", min_value=0, max_value=1),
                },
            )

            # Bar chart of metrics
            fig = go.Figure()
            for metric, color in [
                ("Accuracy", "#4C72B0"), ("Precision", "#55A868"),
                ("Recall", "#C44E52"), ("F1-Score", "#8172B3"),
            ]:
                fig.add_trace(go.Bar(
                    name=metric, x=metric_df["Model"], y=metric_df[metric],
                    marker_color=color,
                    text=[f"{v:.1%}" for v in metric_df[metric]],
                    textposition="outside",
                ))
            _style_fig(
                fig, barmode="group", title="Model Metrics Comparison",
                yaxis_title="Score", yaxis_range=[0, 1.15],
            )
            st.plotly_chart(fig, use_container_width=True)

            # Model agreement over the held-out split
            st.markdown("---")
            st.subheader("🔗 Model Agreement on Held-Out Users")
            for name, short in MODEL_SHORT.items():
                y_pred = models[short].predict(X_scaled)
                agree = (y_pred == y).mean()
                st.markdown(f"- **{name}**: {agree*100:.1f}% match with true labels")

            votes = np.column_stack(
                [models[s].predict(X_scaled) for s in ["lr", "rf", "xgb"]]
            )
            vote_sum = votes.sum(axis=1)
            agree_df = pd.DataFrame({
                "Consensus": ["0/3 agree", "1/3 agree", "2/3 agree", "3/3 agree"],
                "Count": [
                    int((vote_sum == 0).sum()),
                    int((vote_sum == 1).sum()),
                    int((vote_sum == 2).sum()),
                    int((vote_sum == 3).sum()),
                ],
            })
            fig = px.bar(
                agree_df, x="Consensus", y="Count", color="Count",
                color_continuous_scale="blues", title="Model Prediction Distribution",
            )
            _style_fig(fig, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "💡 **Tip**: When all three models agree, confidence is highest. "
                "Recall is prioritised over precision — it is costlier to miss fraudsters "
                "than to flag a few extra legitimate users for review."
            )
    else:
        st.warning("Models or data not available.")


# --------------------------------------------------------------------------- #
# About page
# --------------------------------------------------------------------------- #
def render_about():
    st.markdown('<div class="main-header">ℹ️ About This System</div>', unsafe_allow_html=True)

    st.markdown("""
        ### Fraud Detection System

        This interactive application analyses cryptocurrency trading-platform data to detect
        potentially fraudulent users based on behavioural patterns. It uses an ensemble of three
        machine-learning models whose consensus provides a higher-confidence risk assessment.

        #### 🎯 Features

        - **Real-time Fraud Detection** — Analyse user behaviour with 3 ML models
        - **Multiple Input Modes** — Manual entry, existing-user lookup, or batch CSV upload
        - **Model Consensus** — Higher confidence when models agree
        - **Comprehensive Analytics** — Trading patterns, user segments, deposit behaviour
        - **Interactive Dashboard** — Live metrics and KPIs with Plotly charts
        - **Export Results** — Download analysis as CSV for further review

        #### 🔍 Fraud Detection Criteria

        A user is flagged as **Suspicious** if they meet **ALL** of the following:

        1. Made at least one deposit and one withdrawal
        2. Minimal trading activity (< 10% of deposited amount)
        3. Quick withdrawal (< 48 hours from first deposit)
        4. High withdrawal ratio (> 80% of deposits)

        #### 📊 Models Used

        | Model | Type | Strengths |
        |-------|------|-----------|
        | Logistic Regression | Linear | Fast, interpretable, good baseline |
        | Random Forest | Ensemble (Bagging) | Handles non-linear interactions |
        | XGBoost | Ensemble (Boosting) | Best accuracy, handles class imbalance |

        #### 👨‍💻 Technical Stack

        - **Backend**: Python, scikit-learn, XGBoost, pandas, numpy
        - **Frontend**: Streamlit
        - **Visualisation**: Plotly

        ---

        **Developed by**: Abdulbasit Olanrewaju Gbolahan
        **Date**: December 2025 — Updated August 2026

        ---
    """)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main():
    # Load resources once
    models = load_models()
    data = load_data()

    app_mode = render_sidebar(models, data)

    if app_mode == "📊 Dashboard":
        render_dashboard(models, data)
    elif app_mode == "🔍 Fraud Detection":
        render_fraud_detection(models, data)
    elif app_mode == "📈 Analytics":
        render_analytics(models, data)
    elif app_mode == "🔬 Model Comparison":
        render_model_comparison(models, data)
    elif app_mode == "ℹ️ About":
        render_about()

    st.markdown(
        '<div class="footer-note">🛡️ Fraud Detection System — v2.0 | '
        "Ensemble of Logistic Regression, Random Forest & XGBoost</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
