import pandas as pd
import pickle
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

# Create models directory
Path("models").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("="*70)
print("LOADING DATA AND TRAINING MODELS")
print("="*70)

# Load features
print("\nLoading data...")
features = pd.read_csv('data/processed/fraud_features.csv')

# Feature columns
feature_cols = [
    'total_deposited', 'deposit_count', 'avg_deposit',
    'total_withdrawn', 'withdrawal_count', 'avg_withdrawal',
    'total_trade_volume_usd', 'trade_count', 'avg_trade_usd',
    'unique_pairs_traded', 'deposit_withdrawal_ratio',
    'withdrawal_deposit_ratio', 'hours_deposit_to_withdrawal',
    'trade_to_deposit_ratio', 'total_unique_assets', 'activity_frequency'
]

# Prepare data
X = features[feature_cols].replace([np.inf, -np.inf], 999).fillna(0)
y = features['is_suspicious']

print(f"Total samples: {len(X)}")
print(f"Suspicious samples: {y.sum()}")
print(f"Features: {len(feature_cols)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_score = lr.score(X_test_scaled, y_test)
print(f"   ✓ Trained - Test Accuracy: {lr_score:.4f}")

# 2. Random Forest
print("\n2. Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42, 
    max_depth=10
)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)
print(f"   ✓ Trained - Test Accuracy: {rf_score:.4f}")

# 3. XGBoost
print("\n3. Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)
xgb_score = xgb_model.score(X_test_scaled, y_test)
print(f"   ✓ Trained - Test Accuracy: {xgb_score:.4f}")

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

joblib.dump(lr, 'models/logistic_regression_model.pkl')
print("\n✓ Saved: models/logistic_regression_model.pkl")

joblib.dump(rf, 'models/random_forest_model.pkl')
print("✓ Saved: models/random_forest_model.pkl")

joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("✓ Saved: models/xgboost_model.pkl")

joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved: models/scaler.pkl")

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("✓ Saved: models/feature_columns.pkl")

print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)

print(f"\nLogistic Regression Accuracy: {lr_score:.4f}")
print(f"Random Forest Accuracy:       {rf_score:.4f}")
print(f"XGBoost Accuracy:             {xgb_score:.4f}")

best_model = max([
    ('Logistic Regression', lr_score),
    ('Random Forest', rf_score),
    ('XGBoost', xgb_score)
], key=lambda x: x[1])

print(f"\n🏆 Best Model: {best_model[0]} ({best_model[1]:.4f})")
