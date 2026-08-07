# Fraud Detection System

**Live Demo**: [https://fraudml-app.streamlit.app/](https://fraudml-app.streamlit.app/)

> A machine learning-powered fraud detection system for cryptocurrency trading platforms, built for the NUPAT AI Fellowship Stage Two Assessment.

---

A comprehensive fraud detection system for cryptocurrency trading platforms, built with machine learning and interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

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
- [License](#license)

---

## 🎯 Overview

This project was developed as part of the NUPAT AI Fellowship Stage Two Assessment. It analyzes cryptocurrency trading platform data to identify potentially fraudulent users based on behavioral patterns.

### Key Objectives

1. **Exploratory Data Analysis**: Understand market dynamics and user behavior
2. **Fraud Detection**: Build ML models to identify suspicious users
3. **Strategic Insights**: Provide data-driven recommendations for business decisions
4. **Interactive Interface**: Deploy models via Streamlit for real-time analysis

---

## 🚀 Live Demo

Access the deployed application here:

**👉 [https://fraudml-app.streamlit.app/](https://fraudml-app.streamlit.app/)**

Try the fraud detection tool with sample data or explore the analytics dashboard!

---
## ✨ Features

- **📊 Comprehensive Dashboard**: Real-time metrics and KPIs
- **🔍 Fraud Detection Engine**: Multi-model approach (Logistic Regression + Random Forest)
- **📈 Advanced Analytics**: Trading patterns, user segmentation, deposit analysis
- **🎨 Interactive Visualizations**: Built with Plotly for dynamic exploration
- **⚡ Real-time Predictions**: Instant user risk assessment
- **📱 Responsive Design**: Works on desktop and mobile

---

## 📁 Project Structure

```
nupat-fraud-detection/
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
│   ├── scaler.pkl                     # Feature scaler
│   └── feature_columns.pkl            # Feature names
│
├── notebooks/
│   └── NUPAT_AI_Fellowship_Analysis.ipynb  # Main analysis notebook
│
├── streamlit_app.py                   # Main Streamlit application
├── model_saver.py                     # Script to save trained models
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
   
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nupat-fraud-detection.git
cd nupat-fraud-detection
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

**Core Dependencies:**
- `pandas==2.1.4` - Data manipulation
- `numpy==1.26.2` - Numerical computing
- `scikit-learn==1.3.2` - Machine learning
- `streamlit==1.29.0` - Web interface
- `plotly==5.18.0` - Interactive visualizations
- `matplotlib==3.8.2` - Static plots
- `seaborn==0.13.0` - Statistical visualizations
- `joblib==1.3.2` - Model persistence

**Optional:**
- `xgboost==2.0.3` - Gradient boosting (if available)

### Step 4: Set Up Data

Ensure your data files are in the correct location:

```bash
data/
├── trades.csv
└── user_activitycsv.csv
```

---

## 📊 Usage

### 1. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/NUPAT_AI_Fellowship_Analysis.ipynb
```

Execute all cells to:
- Perform exploratory data analysis
- Engineer features
- Train models
- Generate visualizations

### 2. Save Trained Models

After running the notebook, save models for Streamlit:

```bash
python model_saver.py
```

This creates:
- `models/logistic_regression_model.pkl`
- `models/random_forest_model.pkl`
- `models/scaler.pkl`
- `models/feature_columns.pkl`
- `xgboost_model.pkl`

### 3. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### 4. Using the Interface

#### Dashboard Tab
- View system metrics and KPIs
- Monitor fraud distribution
- Track trading activity trends

#### Fraud Detection Tab
- Enter user information manually
- Get real-time fraud predictions
- View key risk indicators

#### Analytics Tab
- Explore trading patterns
- Analyze user segments
- Study deposit behaviors

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

**16 engineered features** including:

1. **Deposit Metrics**
   - Total deposited amount
   - Deposit count
   - Average deposit size
   - First/last deposit timestamp
   - Unique deposit assets

2. **Withdrawal Metrics**
   - Total withdrawn amount
   - Withdrawal count
   - Average withdrawal size
   - First/last withdrawal timestamp
   - Unique withdrawal assets

3. **Trading Metrics**
   - Total trade volume (USD)
   - Trade count
   - Average trade size
   - Unique pairs traded

4. **Derived Features**
   - Deposit/withdrawal ratio
   - Withdrawal/deposit ratio
   - Hours from deposit to withdrawal
   - Trade-to-deposit ratio
   - Total unique assets
   - Activity frequency

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
- **Type**: Ensemble classifier
- **Advantages**: Handles non-linear patterns, feature importance
- **Configuration**: `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`

### Model Evaluation

**Primary Metric**: **RECALL**

**Justification**: In fraud detection, missing fraudsters (False Negatives) is more costly than false alarms (False Positives). We prioritize catching suspicious users even if it means some legitimate users get flagged for manual review.

---

## 🖥️ Streamlit Interface

### Dashboard View
- Total users, suspicious users, trades, and deposits
- Fraud distribution pie chart
- User activity histogram
- Daily trading volume trend

### Fraud Detection Tool
- Manual user data entry
- Real-time predictions from both models
- Fraud probability scores
- Key risk indicator analysis

### Analytics Section
- Trading volume distribution
- User segmentation
- Hourly deposit patterns
- Interactive filtering

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

### Part 2: Fraud Detection

- **Total Users Analyzed**: 1,199
- **Suspicious Users Identified**: 5
- **Fraud Rate**: 0.42%
- **Model Performance**: High precision and recall (test on your data)

### Part 3: Strategic Recommendation

**Kenya Low-Volume Trader Campaign Target:**

Target users with:
1. `total_trade_volume_usd < $500`
2. `trade_count >= 2 AND <= 10`
3. Traded KES pairs (e.g., ETHKES)

**Expected Impact**: 15-25% of user base

---

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
flake8 .
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run tests: `pytest`
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

**Solution**: XGBoost is optional. The app works with Logistic Regression and Random Forest.

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**NUPAT AI Fellowship Participant**  
Stage Two Assessment - December 2025

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

- [ ] Real-time data streaming integration
- [ ] Advanced ensemble models (XGBoost, LightGBM)
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

*Last Updated: December 19, 2025*
