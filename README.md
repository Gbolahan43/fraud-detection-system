# Fraud Detection System

**Live Demo**: [https://fraudml-app.streamlit.app/](https://fraudml-app.streamlit.app/)

> An interactive machine learning fraud detection application for cryptocurrency trading platforms.

---

A comprehensive fraud detection system for cryptocurrency trading platforms, built with machine learning and interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-006600.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Model Architecture](#model-architecture)
- [Streamlit Interface](#streamlit-interface)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Overview

This project provides tools to analyze cryptocurrency trading platform data and identify potentially fraudulent users based on behavioral patterns. Three models (Logistic Regression, Random Forest, XGBoost) vote on every user, and their consensus drives the risk rating.

### Key Objectives

1. **Exploratory Data Analysis**: Understand market dynamics and user behavior
2. **Fraud Detection**: Deploy an ensemble that flags suspicious users for review
3. **Strategic Insights**: Provide data-driven recommendations for business decisions
4. **Interactive Interface**: Use the Streamlit app for interactive analysis and predictions

---

## 🚀 Live Demo

Access the deployed application here:

**👉 [https://fraudml-app.streamlit.app/](https://fraudml-app.streamlit.app/)**

Try the fraud detection tool with sample data or explore the analytics dashboard!

---
## ✨ Features

- **📊 Comprehensive Dashboard**: 8 KPIs, fraud distribution, top trading pairs, BTCNGN volatility
- **🔍 Fraud Detection Engine**: 3-model ensemble (Logistic Regression + Random Forest + XGBoost) with consensus voting
- **🧾 Three Input Modes**: Manual entry, existing-user lookup, and batch CSV upload
- **🚦 Risk Levels**: Low / Medium / High / Extreme, derived from model votes and average probability
- **📈 Advanced Analytics**: Trading patterns, user segmentation, deposit analysis
- **🔬 Model Comparison**: Accuracy, precision, recall and F1 on the held-out test split
- **🎨 Interactive Visualizations**: Built with Plotly, on a consistent dark theme
- **📥 Export Results**: Download single or batch analyses as CSV

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── data/
│   ├── trades.csv                      # Trading transactions
│   ├── user_activitycsv.csv           # User deposits/withdrawals
│   └── processed/
│       ├── fraud_features.csv          # Engineered features
│       ├── btcngn_volatility.csv      # Volatility analysis
│       ├── deposits_by_day.csv        # Daily deposit patterns
│       └── deposits_by_hour.csv       # Hourly deposit patterns
│
├── models/
│   ├── logistic_regression_model.pkl  # Trained LR model
│   ├── random_forest_model.pkl        # Trained RF model
│   ├── xgboost_model.pkl              # Trained XGBoost model
│   ├── scaler.pkl                     # Feature scaler
│   └── feature_columns.pkl            # Feature names + training order
│
├── notebooks/
│   └── analysis.ipynb                 # Analysis notebook
│
├── .streamlit/
│   └── config.toml                    # Pins the dark theme the app is built for
│
├── .devcontainer/
│   └── devcontainer.json              # Codespaces / dev container setup
│
├── streamlit_app.py                   # Main Streamlit application
├── model_saver.py                     # Trains and saves the model artifacts
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules

``` 

---

## 🚀 Installation

### Prerequisites

- Python 3.11+ (developed on 3.14 — see `.python-version`)
- pip (Python package manager)
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Core Dependencies** (pinned in `requirements.txt`):
- `pandas==2.3.3` - Data manipulation
- `numpy==2.3.5` - Numerical computing
- `scikit-learn==1.8.0` - Machine learning
- `xgboost==3.1.2` - Gradient boosting (required — the app loads all three models)
- `streamlit==1.52.2` - Web interface
- `plotly==6.5.0` - Interactive visualizations
- `joblib==1.5.3` - Model persistence
- `matplotlib==3.10.8` / `seaborn==0.13.2` - Static plots used in the notebook

### Step 4: Set Up Data

Ensure your data files are in the correct location:

```bash
data/
├── trades.csv
└── user_activitycsv.csv
```

---

## 📊 Usage

### 1. (Optional) Run the analysis notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run cells in the notebook to reproduce feature engineering, model training, and exploratory analyses.

### 2. Save Trained Models

After running the notebook (or your training pipeline), save models for the Streamlit app:

```bash
python model_saver.py
```

This trains all three models on an 80/20 stratified split and writes the artifacts
to `models/`. The app reads the feature order back from `feature_columns.pkl`, so
retraining with a different feature set stays in sync automatically.

### 3. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

### 4. Using the Interface

Pick a section from the sidebar dropdown:

#### 📊 Dashboard
- 8 KPIs: users, suspicious users, trades, deposits, fraud rate, trade volume, avg deposit, low-volume traders
- Fraud distribution donut and user activity histogram
- Top trading pairs by USD volume and BTCNGN volatility trend
- Table of the top rules-flagged suspicious users

#### 🔍 Fraud Detection
- **Manual Entry** — type a user's figures into a form and analyse them
- **Existing User Lookup** — pick a `user_id` from the dataset and compare the prediction against the actual label
- **Batch Upload (CSV)** — score a whole file at once, with a risk-level breakdown
- All three modes show per-model verdicts, consensus, and key risk indicators
- Results download as CSV

#### 📈 Analytics
- Trading patterns, user segments, and deposit behaviour by day and hour
- Deposit vs withdrawal scatter, coloured by suspicious status

#### 🔬 Model Comparison
- Model characteristics table
- Accuracy / precision / recall / F1 on the held-out 20% test split
- Consensus distribution across the held-out users

#### ℹ️ About
- Project documentation, fraud criteria, and tech stack

---

## 📈 Data Description

### trades.csv

| Column    | Type   | Description                                      |
|-----------|--------|--------------------------------------------------|
| pair      | string | Trading pair (e.g., BTCNGN, ETHKES)             |
| amount    | float  | Trade value in quote currency                   |
| volume    | float  | Amount of base currency traded                  |
| side      | string | 'buy' or 'sell'                                 |
| timestamp | string | ISO timestamp of trade                          |
| user_id   | string | Unique user identifier                          |

### user_activitycsv.csv

| Column         | Type   | Description                           |
|----------------|--------|---------------------------------------|
| asset          | string | Crypto asset (BTC, ETH, USDT, etc.)  |
| amount         | float  | Transaction amount                    |
| activity_type  | string | 'deposit' or 'withdrawal'            |
| timestamp      | string | ISO timestamp of activity            |
| user_id        | string | Unique user identifier               |

---

## 🤖 Model Architecture

### Feature Engineering

The models take **16 features**, in this order (the authoritative list lives in
`models/feature_columns.pkl`):

1. **Deposit Metrics**
   - `total_deposited`, `deposit_count`, `avg_deposit`

2. **Withdrawal Metrics**
   - `total_withdrawn`, `withdrawal_count`, `avg_withdrawal`

3. **Trading Metrics**
   - `total_trade_volume_usd`, `trade_count`, `avg_trade_usd`, `unique_pairs_traded`

4. **Derived Features**
   - `deposit_withdrawal_ratio`, `withdrawal_deposit_ratio`
   - `hours_deposit_to_withdrawal`
   - `trade_to_deposit_ratio`
   - `total_unique_assets`, `activity_frequency`

Timestamps and per-asset breakdowns are used to *build* these features but are not
model inputs themselves. Infinities are clipped to `999` and nulls filled with `0`,
matching `model_saver.py`.

### Fraud Labeling Logic

A user is flagged as **SUSPICIOUS** if ALL criteria are met:

```python
suspicious = (
    total_deposited > 0 AND
    total_withdrawn > 0 AND
    trade_to_deposit_ratio < 0.1 AND
    hours_deposit_to_withdrawal < 48 AND
    withdrawal_deposit_ratio > 0.8
)
```

### ML Models

#### 1. Logistic Regression
- **Type**: Linear classifier
- **Advantages**: Fast, interpretable, good baseline
- **Configuration**: `class_weight='balanced'`, `max_iter=1000`

#### 2. Random Forest
- **Type**: Ensemble classifier (bagging)
- **Advantages**: Handles non-linear patterns, feature importance
- **Configuration**: `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`

#### 3. XGBoost
- **Type**: Ensemble classifier (gradient boosting)
- **Advantages**: Strongest overall performer, handles class imbalance well
- **Configuration**: `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `scale_pos_weight` set from the training class balance

### Consensus Logic

The app combines the three models rather than trusting any one of them. Given the
number of "suspicious" votes and the average predicted probability:

| Risk level | Condition |
|------------|-----------|
| **Extreme** | ≥2 votes **and** avg probability ≥ 0.70 |
| **High**    | ≥2 votes |
| **Medium**  | 1 vote, or avg probability in [0.40, 0.70) |
| **Low**     | otherwise |

### Model Evaluation

**Primary Metric**: **RECALL**

**Justification**: In fraud detection, missing fraudsters (False Negatives) is more costly than false alarms (False Positives). We prioritize catching suspicious users even if it means some legitimate users get flagged for manual review.

Metrics shown in the app are computed on the **held-out 20% test split** that
`model_saver.py` set aside during training (recreated with the same
`random_state=42`), not on the data the models were fitted to.

> **Note on class imbalance**: only 5 of 1,199 users are labelled suspicious
> (0.42%), so accuracy is close to meaningless here — a model that flags nobody
> scores 99.6%. Read recall and precision instead, and treat the small positive
> count as the main limitation of these numbers.

---

## 🖥️ Streamlit Interface

The interface is documented under [Using the Interface](#4-using-the-interface).

The app ships a dark theme pinned in `.streamlit/config.toml`; every Plotly figure
is styled through a single helper so charts and cards stay consistent.

---

## 📊 Results

### Part 1: Market Insights

**Top 3 Trading Pairs by USD Volume:**
1. **BTCNGN**: $136,215.49
2. **USDTNGN**: $59,180.57
3. **BTCUSDT**: $13,443.57

**Key Findings:**
- Peak deposit day: **Friday** (671 deposits)
- Peak deposit hour: **15:00** (162 deposits)
- BTCNGN shows high volatility with 7-day rolling average
- 2,324 trades across 50 pairs; NGN-quoted pairs dominate volume

### Part 2: Fraud Detection

- **Total Users Analyzed**: 1,199
- **Suspicious Users Identified**: 5
- **Fraud Rate**: 0.42%
- **Model Performance**: see the Model Comparison page — it computes accuracy,
  precision, recall and F1 live on the held-out test split rather than quoting
  fixed numbers here

### Part 3: Strategic Recommendation

**Kenya Low-Volume Trader Campaign Target:**

Target users with:
1. `total_trade_volume_usd < $500`
2. `trade_count >= 2 AND <= 10`
3. Traded KES pairs (e.g., ETHKES)

**Expected Impact**: 15-25% of user base

---

## 🛠️ Development

There is no automated test suite yet — see [Future Enhancements](#-future-enhancements).
Changes are currently verified by running the app and exercising each page.

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run the app and check the affected pages: `streamlit run streamlit_app.py`
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature/your-feature`
6. Create Pull Request

---

## 🐛 Troubleshooting

### Issue: Models not loading in Streamlit

**Solution**: Ensure you ran `model_saver.py` after training models in the notebook.

```bash
python model_saver.py
```

### Issue: Data files not found

**Solution**: Check that CSV files are in the correct directory:

```bash
data/
├── trades.csv
└── user_activitycsv.csv
```

### Issue: XGBoost installation fails

**Solution**: XGBoost is required — the app loads all three models and will report
missing artifacts without it.

On Windows:
```bash
pip install xgboost --no-cache-dir
```

On macOS with Apple Silicon:
```bash
brew install libomp
pip install xgboost
```

### Issue: Streamlit port already in use

**Solution**: Use a different port:

```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 📚 Documentation

### Notebooks
- **analysis.ipynb**: Complete data analysis with markdown explanations

### Code Comments
All major functions include docstrings explaining:
- Purpose
- Parameters
- Return values
- Example usage

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Write descriptive commit messages

---

## 👨‍💻 Author

**Abdulbasit Olanrewaju Gbolahan**
- Built December 2025, updated August 2026

---

## 🙏 Acknowledgments

- NUPAT AI Fellowship program for the opportunity
- scikit-learn community for excellent ML tools
- Streamlit team for the amazing framework
- Open-source contributors

---

## 📞 Contact

For questions or support:
- **Email**: gbolahanbasit43@gmail.com
- **GitHub**: [Olanrewaju Abdulbasit](https://github.com/Gbolahan43)
- **LinkedIn**: [Olanrewaju Abdulbasit](https://linkedin.com/in/abdulbasit-olanrewaju-gbolahan)

---

## 🔮 Future Enhancements

- [ ] Automated test suite (`pytest`) covering feature engineering and consensus logic
- [ ] Real-time data streaming integration
- [x] Ensemble with XGBoost
- [ ] Additional boosted models (LightGBM, CatBoost)
- [ ] Deep learning approaches (LSTM for time-series)
- [ ] Automated model retraining pipeline
- [ ] A/B testing framework
- [ ] Docker containerization
- [ ] REST API deployment
- [ ] Advanced feature engineering (behavioral embeddings)
- [ ] Explainable AI (SHAP values)
- [ ] Multi-language support

---

**⭐ If you found this project helpful, please give it a star!**

---

*Last Updated: August 9, 2026*
