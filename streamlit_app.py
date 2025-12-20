import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NUPAT Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return lr_model, rf_model, xgb_model, scaler, feature_columns
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please run model_saver.py first.")
        return None, None, None, None, None

@st.cache_data
def load_data():
    try:
        trades = pd.read_csv('data/trades.csv')
        user_activity = pd.read_csv('data/user_activitycsv.csv')
        features = pd.read_csv('data/processed/fraud_features.csv')
        return trades, user_activity, features
    except FileNotFoundError:
        return None, None, None

# Load resources
lr_model, rf_model, xgb_model, scaler, feature_columns = load_models()
trades, user_activity, features = load_data()

# Sidebar
st.sidebar.title("🛡️ Fraud Detection System")
st.sidebar.markdown("---")

app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["📊 Dashboard", "🔍 Fraud Detection", "📈 Analytics", "🔬 Model Comparison", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
if all([lr_model, rf_model, xgb_model]):
    st.sidebar.success("✅ All Models Loaded")
    st.sidebar.info("📊 3 Models: LR, RF, XGB")
else:
    st.sidebar.error("❌ Models Not Found")

if trades is not None:
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.error("❌ Data Not Found")

# Main content
if app_mode == "📊 Dashboard":
    st.markdown('<div class="main-header">🛡️ NUPAT Fraud Detection Dashboard</div>', unsafe_allow_html=True)

    if features is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Users",
                value=f"{len(features):,}",
                delta=None
            )

        with col2:
            suspicious_count = features['is_suspicious'].sum() if 'is_suspicious' in features.columns else 0
            st.metric(
                label="Suspicious Users",
                value=f"{suspicious_count}",
                delta=f"{(suspicious_count/len(features)*100):.2f}%"
            )

        with col3:
            total_trades = len(trades) if trades is not None else 0
            st.metric(
                label="Total Trades",
                value=f"{total_trades:,}",
                delta=None
            )

        with col4:
            total_deposits = user_activity[user_activity['activity_type']=='deposit']['amount'].sum() if user_activity is not None else 0
            st.metric(
                label="Total Deposits",
                value=f"₦{total_deposits:,.0f}",
                delta=None
            )

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud Distribution")
            if 'is_suspicious' in features.columns:
                fraud_dist = features['is_suspicious'].value_counts()
                fig = px.pie(
                    values=fraud_dist.values,
                    names=['Legitimate', 'Suspicious'],
                    color_discrete_sequence=['#28a745', '#dc3545']
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("User Activity Distribution")
            if 'activity_frequency' in features.columns:
                fig = px.histogram(
                    features,
                    x='activity_frequency',
                    nbins=50,
                    title="User Activity Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Data not loaded. Please ensure data files are in the correct location.")

elif app_mode == "🔍 Fraud Detection":
    st.markdown('<div class="main-header">🔍 Fraud Detection Tool</div>', unsafe_allow_html=True)

    st.markdown("### Enter User Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Deposit Information")
        total_deposited = st.number_input("Total Deposited (NGN)", min_value=0.0, value=100000.0, step=1000.0)
        deposit_count = st.number_input("Number of Deposits", min_value=0, value=5, step=1)
        avg_deposit = total_deposited / deposit_count if deposit_count > 0 else 0
        st.info(f"Average Deposit: ₦{avg_deposit:,.2f}")

        st.markdown("#### Withdrawal Information")
        total_withdrawn = st.number_input("Total Withdrawn (NGN)", min_value=0.0, value=80000.0, step=1000.0)
        withdrawal_count = st.number_input("Number of Withdrawals", min_value=0, value=3, step=1)
        avg_withdrawal = total_withdrawn / withdrawal_count if withdrawal_count > 0 else 0
        st.info(f"Average Withdrawal: ₦{avg_withdrawal:,.2f}")

    with col2:
        st.markdown("#### Trading Information")
        total_trade_volume_usd = st.number_input("Total Trade Volume (USD)", min_value=0.0, value=500.0, step=100.0)
        trade_count = st.number_input("Number of Trades", min_value=0, value=2, step=1)
        avg_trade_usd = total_trade_volume_usd / trade_count if trade_count > 0 else 0
        st.info(f"Average Trade: ${avg_trade_usd:,.2f}")

        unique_pairs_traded = st.number_input("Unique Pairs Traded", min_value=0, value=2, step=1)

        st.markdown("#### Timing Information")
        hours_deposit_to_withdrawal = st.number_input("Hours from First Deposit to First Withdrawal", min_value=0.0, value=24.0, step=1.0)

    st.markdown("#### Additional Metrics")
    col3, col4 = st.columns(2)
    with col3:
        unique_deposit_assets = st.number_input("Unique Deposit Assets", min_value=0, value=2, step=1)
        unique_withdrawal_assets = st.number_input("Unique Withdrawal Assets", min_value=0, value=2, step=1)
    with col4:
        total_unique_assets = unique_deposit_assets + unique_withdrawal_assets
        activity_frequency = deposit_count + withdrawal_count + trade_count
        st.info(f"Total Unique Assets: {total_unique_assets}")
        st.info(f"Activity Frequency: {activity_frequency}")

    # Calculate derived features
    deposit_withdrawal_ratio = total_deposited / total_withdrawn if total_withdrawn > 0 else 999
    withdrawal_deposit_ratio = total_withdrawn / total_deposited if total_deposited > 0 else 0
    trade_to_deposit_ratio = total_trade_volume_usd / (total_deposited/1500) if total_deposited > 0 else 0

    if st.button("🔍 Analyze User", type="primary"):
        if all([lr_model, rf_model, xgb_model, scaler, feature_columns]):
            # Prepare features
            user_features = {
                'total_deposited': total_deposited,
                'deposit_count': deposit_count,
                'avg_deposit': avg_deposit,
                'total_withdrawn': total_withdrawn,
                'withdrawal_count': withdrawal_count,
                'avg_withdrawal': avg_withdrawal,
                'total_trade_volume_usd': total_trade_volume_usd,
                'trade_count': trade_count,
                'avg_trade_usd': avg_trade_usd,
                'unique_pairs_traded': unique_pairs_traded,
                'deposit_withdrawal_ratio': deposit_withdrawal_ratio,
                'withdrawal_deposit_ratio': withdrawal_deposit_ratio,
                'hours_deposit_to_withdrawal': hours_deposit_to_withdrawal,
                'trade_to_deposit_ratio': trade_to_deposit_ratio,
                'total_unique_assets': total_unique_assets,
                'activity_frequency': activity_frequency
            }

            X = pd.DataFrame([user_features])[feature_columns]
            X = X.fillna(0).replace([np.inf, -np.inf], 999)
            X_scaled = scaler.transform(X)

            # Predictions from all models
            lr_pred = lr_model.predict(X_scaled)[0]
            lr_proba = lr_model.predict_proba(X_scaled)[0][1]

            rf_pred = rf_model.predict(X_scaled)[0]
            rf_proba = rf_model.predict_proba(X_scaled)[0][1]

            xgb_pred = xgb_model.predict(X_scaled)[0]
            xgb_proba = xgb_model.predict_proba(X_scaled)[0][1]

            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Logistic Regression")
                if lr_pred == 1:
                    st.markdown(f'<div class="warning-box">⚠️ <b>SUSPICIOUS</b><br>Probability: {lr_proba*100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box">✅ <b>LEGITIMATE</b><br>Probability: {lr_proba*100:.2f}%</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("#### Random Forest")
                if rf_pred == 1:
                    st.markdown(f'<div class="warning-box">⚠️ <b>SUSPICIOUS</b><br>Probability: {rf_proba*100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box">✅ <b>LEGITIMATE</b><br>Probability: {rf_proba*100:.2f}%</div>', unsafe_allow_html=True)

            with col3:
                st.markdown("#### XGBoost 🚀")
                if xgb_pred == 1:
                    st.markdown(f'<div class="danger-box">🚨 <b>SUSPICIOUS</b><br>Probability: {xgb_proba*100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box">✅ <b>LEGITIMATE</b><br>Probability: {xgb_proba*100:.2f}%</div>', unsafe_allow_html=True)

            # Consensus
            st.markdown("---")
            st.markdown("### 🎯 Model Consensus")

            predictions = [lr_pred, rf_pred, xgb_pred]
            suspicious_votes = sum(predictions)

            if suspicious_votes == 3:
                st.error("🚨 **ALL 3 MODELS AGREE: HIGHLY SUSPICIOUS USER**")
            elif suspicious_votes == 2:
                st.warning("⚠️ **MAJORITY (2/3) VOTE: LIKELY SUSPICIOUS**")
            elif suspicious_votes == 1:
                st.info("ℹ️ **SPLIT DECISION: REVIEW RECOMMENDED**")
            else:
                st.success("✅ **ALL 3 MODELS AGREE: LEGITIMATE USER**")

            # Average probability
            avg_proba = (lr_proba + rf_proba + xgb_proba) / 3
            st.metric("Average Fraud Probability", f"{avg_proba*100:.2f}%")

            st.markdown("---")
            st.markdown("### 🔎 Key Indicators")

            col1, col2, col3 = st.columns(3)

            with col1:
                if withdrawal_deposit_ratio > 0.8:
                    st.error(f"⚠️ High Withdrawal Ratio: {withdrawal_deposit_ratio:.2%}")
                else:
                    st.success(f"✅ Normal Withdrawal Ratio: {withdrawal_deposit_ratio:.2%}")

            with col2:
                if trade_to_deposit_ratio < 0.1:
                    st.error(f"⚠️ Low Trading Activity: {trade_to_deposit_ratio:.2%}")
                else:
                    st.success(f"✅ Good Trading Activity: {trade_to_deposit_ratio:.2%}")

            with col3:
                if hours_deposit_to_withdrawal < 48:
                    st.error(f"⚠️ Quick Withdrawal: {hours_deposit_to_withdrawal:.1f}h")
                else:
                    st.success(f"✅ Normal Timing: {hours_deposit_to_withdrawal:.1f}h")
        else:
            st.error("⚠️ Models not loaded. Please train and save models first.")

elif app_mode == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)

    if features is not None:
        tabs = st.tabs(["Trading Patterns", "User Segments", "Deposit Analysis"])

        with tabs[0]:
            st.subheader("Trading Volume Distribution")
            fig = px.box(
                features,
                y='total_trade_volume_usd',
                title="Trading Volume Distribution (USD)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.subheader("User Segmentation")
            segments = pd.DataFrame({
                'Segment': ['High Activity', 'Medium Activity', 'Low Activity', 'Inactive'],
                'Count': [
                    len(features[features['activity_frequency'] > 20]),
                    len(features[(features['activity_frequency'] > 10) & (features['activity_frequency'] <= 20)]),
                    len(features[(features['activity_frequency'] > 0) & (features['activity_frequency'] <= 10)]),
                    len(features[features['activity_frequency'] == 0])
                ]
            })
            fig = px.bar(segments, x='Segment', y='Count', title="User Activity Segments")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            if user_activity is not None:
                st.subheader("Deposit Patterns")
                deposits = user_activity[user_activity['activity_type'] == 'deposit']
                deposits['timestamp'] = pd.to_datetime(deposits['timestamp'])
                deposits['hour'] = deposits['timestamp'].dt.hour
                hourly_deposits = deposits.groupby('hour').size().reset_index()
                hourly_deposits.columns = ['hour', 'count']
                fig = px.bar(hourly_deposits, x='hour', y='count', title="Deposit Activity by Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Data not loaded.")

elif app_mode == "🔬 Model Comparison":
    st.markdown('<div class="main-header">🔬 Model Performance Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Three Models in Action

    This system uses an ensemble of three different machine learning models:

    1. **Logistic Regression** - Fast, interpretable baseline
    2. **Random Forest** - Ensemble of decision trees
    3. **XGBoost** - Gradient boosting, often the best performer

    Each model has different strengths, and comparing their predictions gives us higher confidence.
    """)

    if all([lr_model, rf_model, xgb_model]):
        st.markdown("---")
        st.subheader("📊 Model Characteristics")

        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Type': ['Linear', 'Ensemble (Bagging)', 'Ensemble (Boosting)'],
            'Speed': ['⚡ Very Fast', '⚡⚡ Fast', '⚡⚡⚡ Medium'],
            'Interpretability': ['High', 'Medium', 'Medium'],
            'Handles Imbalance': ['Good', 'Good', 'Excellent'],
            'Best For': ['Linear patterns', 'Non-linear patterns', 'Complex patterns']
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🎯 When Models Disagree")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **High Confidence Scenarios:**
            - ✅ All 3 models agree → Very reliable
            - ✅ 2 models agree → Reliable

            **Review Needed:**
            - ⚠️ Split decision (1 vs 2) → Manual review
            - ⚠️ Low probability scores → Borderline case
            """)

        with col2:
            st.markdown("""
            **Model Strengths:**
            - **Logistic Regression:** Good for simple fraud patterns
            - **Random Forest:** Handles complex interactions
            - **XGBoost:** Best overall performance, handles imbalance
            """)

        st.info("💡 **Pro Tip**: When all three models agree, confidence is highest. XGBoost tends to be most accurate for fraud detection.")
    else:
        st.error("⚠️ Models not loaded.")

elif app_mode == "ℹ️ About":
    st.markdown('<div class="main-header">ℹ️ About This System</div>', unsafe_allow_html=True)

    st.markdown("""
    ### NUPAT AI Fellowship - Fraud Detection System

    This system analyzes cryptocurrency trading platform data to detect potentially fraudulent users.

    #### 🎯 Features

    - **Real-time Fraud Detection**: Analyze user behavior patterns with 3 ML models
    - **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
    - **Model Consensus**: Higher confidence when models agree
    - **Comprehensive Analytics**: Visualize trading patterns and user segments
    - **Interactive Dashboard**: Monitor system metrics and KPIs

    #### 🔍 Fraud Detection Criteria

    A user is flagged as suspicious if they meet ALL of the following:

    1. Made at least one deposit and withdrawal
    2. Minimal trading activity (< 10% of deposited amount)
    3. Quick withdrawal (< 48 hours from first deposit)
    4. High withdrawal ratio (> 80% of deposits)

    #### 📊 Models Used

    - **Logistic Regression**: Fast, interpretable baseline model
    - **Random Forest**: Ensemble model capturing complex patterns
    - **XGBoost**: State-of-the-art gradient boosting (often best performer)

    #### 👨‍💻 Technical Stack

    - **Backend**: Python, scikit-learn, XGBoost, pandas, numpy
    - **Frontend**: Streamlit
    - **Visualization**: Plotly, matplotlib, seaborn

    ---

    **Developed for**: NUPAT AI Fellowship Stage Two Assessment  
    **Date**: December 2025
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**NUPAT AI Fellowship** | Stage Two Assessment")
st.sidebar.markdown(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")