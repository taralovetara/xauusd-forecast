"""
Train ML models for XAUUSD 5-min direction prediction.
Models: XGBoost, Random Forest, Logistic Regression
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'xauusd_features.pkl')

def train():
    df = pd.read_pickle(DATA_PATH)
    print(f"Loaded: {len(df):,} rows")

    exclude_cols = ['datetime', 'target_direction', 'target_return', 'next_close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y = df['target_direction'].values

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # XGBoost
    print("\nTraining XGBoost...")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
        eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=20,
        min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_s, y_train)

    # Evaluate
    results = {}
    for name, model, scaled in [('XGBoost', xgb, False), ('RandomForest', rf, False), ('LogisticRegression', lr, True)]:
        Xt = X_test_s if scaled else X_test
        Xv = X_val_s if scaled else X_val
        val_acc = accuracy_score(y_val, model.predict(Xv))
        test_acc = accuracy_score(y_test, model.predict(Xt))
        proba = model.predict_proba(Xv)
        hc = np.max(proba, axis=1) > 0.6
        hc_acc = accuracy_score(y_val[hc], model.predict(Xv)[hc]) if hc.sum() > 0 else 0
        results[name] = {
            'val_accuracy': round(float(val_acc), 4),
            'test_accuracy': round(float(test_acc), 4),
            'val_hc60_accuracy': round(float(hc_acc), 4),
            'val_hc60_pct': round(float(hc.sum()/len(y_val)*100), 1),
        }
        print(f"\n{name}: Val={val_acc:.4f} Test={test_acc:.4f} HC60={hc_acc:.4f}")

    # Feature importance
    imp = xgb.feature_importances_
    fi = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)

    # Save
    with open(f'{MODEL_DIR}/xgb_model.pkl', 'wb') as f: pickle.dump(xgb, f)
    with open(f'{MODEL_DIR}/rf_model.pkl', 'wb') as f: pickle.dump(rf, f)
    with open(f'{MODEL_DIR}/lr_model.pkl', 'wb') as f: pickle.dump(lr, f)
    with open(f'{MODEL_DIR}/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open(f'{MODEL_DIR}/feature_cols.json', 'w') as f: json.dump(feature_cols, f, indent=2)
    with open(f'{MODEL_DIR}/training_results.json', 'w') as f: json.dump(results, f, indent=2)
    with open(f'{MODEL_DIR}/feature_importance.json', 'w') as f: json.dump({feat: float(v) for feat, v in fi}, f, indent=2)

    print(f"\nModels saved to {MODEL_DIR}/")
    return results

if __name__ == '__main__':
    train()
