import pandas as pd
import pickle
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create models directory
Path("models").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("Loading data...")
features = pd.read_csv('fraud_features.csv')

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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                            random_state=42, max_depth=10)
rf.fit(X_train_scaled, y_train)

# Save models
print("Saving models...")
joblib.dump(lr, 'models/logistic_regression_model.pkl')
joblib.dump(rf, 'models/random_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("✓ All models saved successfully!")
print("\nSaved files:")
print("  - models/logistic_regression_model.pkl")
print("  - models/random_forest_model.pkl")
print("  - models/scaler.pkl")
print("  - models/feature_columns.pkl")