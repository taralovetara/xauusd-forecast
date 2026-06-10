#!/usr/bin/env python3
"""
train_model_v2.py — Stacked Ensemble ML Model for XAUUSD 60-min Direction Prediction
======================================================================================
Architecture:
  Level 0 (Base Models): XGBoost, LightGBM, CatBoost
  Level 1 (Meta-Learner): Logistic Regression on base model probability outputs

Walk-forward validation (chronological 60/20/20 split) prevents data leakage.
"""

import sys
import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(DATA_DIR, "XAUUSD_M15_features_v2_enhanced.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "max_depth": 6,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

LGBM_PARAMS = {
    "num_leaves": 31,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

CB_PARAMS = {
    "depth": 6,
    "iterations": 500,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "verbose": 0,
    "random_seed": 42,
    "thread_count": -1,
}

META_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42,
}

MI_TOP_K = 80  # Select top 80 features by mutual information

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20


# =========================================================================
# Helper Functions
# =========================================================================

def load_data(path):
    """Load enhanced features CSV and create target column."""
    print("\n[1/7] Loading data...")
    t0 = time.time()
    df = pd.read_csv(path)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns in {time.time()-t0:.2f}s")
    return df


def prepare_features(df):
    """Remove OHLCV/time, filter NaN, fill NaN, select top-K features by MI."""
    print("\n[2/7] Preparing features...")

    # Create target: 1 if close is higher in 4 bars (60 min), else 0
    print("      Creating target (horizon=4 bars = 60 min)...")
    df["target"] = (df["close"].shift(-4) > df["close"]).astype(int)
    df = df.dropna(subset=["target"]).copy()
    print(f"      Rows after target creation: {len(df):,}")
    print(f"      Target distribution: Bull={df['target'].mean()*100:.1f}% / Bear={(1-df['target'].mean())*100:.1f}%")

    # Remove OHLCV and time columns
    drop_cols = ["time", "open", "high", "low", "close", "volume"]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["target"]]
    print(f"      Feature columns (before cleaning): {len(feature_cols)}")

    # Keep only numeric features
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"      Numeric feature columns: {len(numeric_cols)}")

    # Remove columns with >50% NaN
    nan_frac = df[numeric_cols].isna().mean()
    low_nan_cols = nan_frac[nan_frac <= 0.50].index.tolist()
    removed_high_nan = len(numeric_cols) - len(low_nan_cols)
    if removed_high_nan > 0:
        print(f"      Removed {removed_high_nan} columns with >50% NaN")
    numeric_cols = low_nan_cols

    # Fill remaining NaN with 0
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Replace infinities with 0
    inf_count = np.isinf(df[numeric_cols].values).sum()
    if inf_count > 0:
        print(f"      Replaced {inf_count} infinite values with 0")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

    X = df[numeric_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int32)

    print(f"      Feature matrix shape: {X.shape}")

    return X, y, numeric_cols


def select_features(X, y, feature_names, k=MI_TOP_K):
    """Select top-K features by mutual information with target."""
    print(f"\n[3/7] Feature selection (top-{k} by mutual information)...")
    t0 = time.time()

    if X.shape[1] <= k:
        print(f"      Only {X.shape[1]} features (< {k}), using all")
        return X, feature_names

    # Use a sample for MI computation if dataset is large (speed)
    sample_size = min(20000, X.shape[0])
    idx = np.random.RandomState(42).choice(X.shape[0], sample_size, replace=False)

    mi_scores = mutual_info_classif(
        X[idx], y[idx], random_state=42, n_neighbors=5
    )

    top_indices = np.argsort(mi_scores)[::-1][:k]
    selected_names = [feature_names[i] for i in top_indices]
    top_scores = mi_scores[top_indices]

    print(f"      MI computed on {sample_size:,} sample rows in {time.time()-t0:.2f}s")
    print(f"      Top 5 features by MI:")
    for i in range(min(5, k)):
        print(f"        {i+1}. {selected_names[i]:40s}  MI={top_scores[i]:.4f}")
    print(f"      Bottom selected: {selected_names[-1]:40s}  MI={top_scores[-1]:.4f}")

    return X[:, top_indices], selected_names


def walk_forward_split(X, y):
    """Chronological 60/20/20 split."""
    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n[4/7] Walk-forward split (chronological):")
    print(f"      Train: {len(X_train):,} rows ({TRAIN_FRAC*100:.0f}%)")
    print(f"      Val:   {len(X_val):,} rows ({VAL_FRAC*100:.0f}%)")
    print(f"      Test:  {len(X_test):,} rows ({TEST_FRAC*100:.0f}%)")
    print(f"      Train target dist: Bull={y_train.mean()*100:.1f}% / Bear={(1-y_train.mean())*100:.1f}%")
    print(f"      Val   target dist: Bull={y_val.mean()*100:.1f}% / Bear={(1-y_val.mean())*100:.1f}%")
    print(f"      Test  target dist: Bull={y_test.mean()*100:.1f}% / Bear={(1-y_test.mean())*100:.1f}%")

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(X_train, X_val, X_test):
    """StandardScaler fit on train, transform all."""
    print("\n      Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, scaler


def train_base_models(X_train, y_train, X_val, y_val):
    """Train XGBoost, LightGBM, CatBoost on training set."""
    print("\n[5/7] Training base models...")

    # --- XGBoost ---
    print("\n      Training XGBoost...")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_time = time.time() - t0
    xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_val_acc = accuracy_score(y_val, (xgb_val_proba >= 0.5).astype(int))
    xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)
    print(f"      XGBoost done in {xgb_time:.1f}s — Val Acc={xgb_val_acc:.4f}  AUC={xgb_val_auc:.4f}")

    # --- LightGBM ---
    print("\n      Training LightGBM...")
    t0 = time.time()
    lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgbm_time = time.time() - t0
    lgbm_val_proba = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_val_acc = accuracy_score(y_val, (lgbm_val_proba >= 0.5).astype(int))
    lgbm_val_auc = roc_auc_score(y_val, lgbm_val_proba)
    print(f"      LightGBM done in {lgbm_time:.1f}s — Val Acc={lgbm_val_acc:.4f}  AUC={lgbm_val_auc:.4f}")

    # --- CatBoost ---
    print("\n      Training CatBoost...")
    t0 = time.time()
    cb_model = CatBoostClassifier(**CB_PARAMS)
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0,
    )
    cb_time = time.time() - t0
    cb_val_proba = cb_model.predict_proba(X_val)[:, 1]
    cb_val_acc = accuracy_score(y_val, (cb_val_proba >= 0.5).astype(int))
    cb_val_auc = roc_auc_score(y_val, cb_val_proba)
    print(f"      CatBoost done in {cb_time:.1f}s — Val Acc={cb_val_acc:.4f}  AUC={cb_val_auc:.4f}")

    models = {
        "xgboost": xgb_model,
        "lightgbm": lgbm_model,
        "catboost": cb_model,
    }
    val_probas = {
        "xgboost": xgb_val_proba,
        "lightgbm": lgbm_val_proba,
        "catboost": cb_val_proba,
    }
    val_metrics = {
        "xgboost": {"accuracy": xgb_val_acc, "auc": xgb_val_auc, "time_s": xgb_time},
        "lightgbm": {"accuracy": lgbm_val_acc, "auc": lgbm_val_auc, "time_s": lgbm_time},
        "catboost": {"accuracy": cb_val_acc, "auc": cb_val_auc, "time_s": cb_time},
    }

    return models, val_probas, val_metrics


def train_meta_learner(val_probas, y_val):
    """Train Logistic Regression meta-learner on validation set meta-features."""
    print("\n[6/7] Training meta-learner (Logistic Regression)...")
    t0 = time.time()

    # Stack base model probabilities as meta-features
    meta_X_val = np.column_stack([val_probas[name] for name in ["xgboost", "lightgbm", "catboost"]])
    print(f"      Meta-features shape: {meta_X_val.shape}")

    meta_model = LogisticRegression(**META_PARAMS)
    meta_model.fit(meta_X_val, y_val)
    meta_time = time.time() - t0
    print(f"      Meta-learner trained in {meta_time:.2f}s")
    print(f"      Meta-learner coefficients: {meta_model.coef_[0]}")
    print(f"      Meta-learner intercept: {meta_model.intercept_[0]:.4f}")

    return meta_model


def evaluate_ensemble(models, meta_model, X_test, y_test):
    """Evaluate stacked ensemble on test set with confidence-based filtering."""
    print("\n[7/7] Evaluating on test set...")

    # Get base model test probabilities
    test_probas = {}
    for name, model in models.items():
        test_probas[name] = model.predict_proba(X_test)[:, 1]

    # Stack meta-features for test set
    meta_X_test = np.column_stack([test_probas[name] for name in ["xgboost", "lightgbm", "catboost"]])

    # Meta-learner predictions
    ensemble_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    # Overall metrics
    overall_acc = accuracy_score(y_test, ensemble_pred)
    overall_auc = roc_auc_score(y_test, ensemble_proba)

    # Individual base model test metrics
    base_test_metrics = {}
    for name, proba in test_probas.items():
        pred = (proba >= 0.5).astype(int)
        base_test_metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
        }

    # Confidence-based filtering
    confidence = np.abs(ensemble_proba - 0.5) * 2  # Scale to [0, 1]
    thresholds = [0.60, 0.65, 0.70, 0.80]
    conf_results = {}

    for thresh in thresholds:
        mask = confidence >= thresh
        n_filtered = mask.sum()
        if n_filtered > 0:
            filtered_acc = accuracy_score(y_test[mask], ensemble_pred[mask])
            filtered_auc_val = roc_auc_score(y_test[mask], ensemble_proba[mask]) if n_filtered > 10 else None
        else:
            filtered_acc = None
            filtered_auc_val = None
        conf_results[str(thresh)] = {
            "threshold": thresh,
            "n_samples": int(n_filtered),
            "pct_samples": float(n_filtered / len(y_test) * 100),
            "accuracy": filtered_acc,
            "auc": filtered_auc_val,
        }

    # Print results
    print("\n" + "=" * 70)
    print("STACKED ENSEMBLE — TEST SET RESULTS")
    print("=" * 70)

    print(f"\n  Overall Ensemble Performance:")
    print(f"    Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"    AUC-ROC:  {overall_auc:.4f}")

    print(f"\n  Base Model Performance (Test Set):")
    print(f"    {'Model':<15s} {'Accuracy':>10s} {'AUC-ROC':>10s}")
    print(f"    {'-'*15} {'-'*10} {'-'*10}")
    for name, metrics in base_test_metrics.items():
        print(f"    {name:<15s} {metrics['accuracy']:>10.4f} {metrics['auc']:>10.4f}")

    print(f"\n  Confidence-Based Filtering Results:")
    print(f"    {'Threshold':>12s} {'Samples':>10s} {'% Data':>10s} {'Accuracy':>10s} {'AUC-ROC':>10s}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for thresh in thresholds:
        r = conf_results[str(thresh)]
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"    {r['threshold']:>12.2f} {r['n_samples']:>10d} {r['pct_samples']:>9.1f}% {acc_str:>10s} {auc_str:>10s}")

    print(f"\n  Total test samples: {len(y_test):,}")
    print("=" * 70)

    return {
        "overall_accuracy": overall_acc,
        "overall_auc": overall_auc,
        "base_models": base_test_metrics,
        "confidence_filtering": conf_results,
        "test_samples": len(y_test),
    }


def save_models(models, meta_model, scaler, feature_names, results):
    """Save all models, scaler, feature list, and training results."""
    print("\n      Saving models...")

    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, "xgb_v2_model.json")
    models["xgboost"].save_model(xgb_path)
    print(f"        XGBoost → {xgb_path}")

    # LightGBM
    lgbm_path = os.path.join(MODEL_DIR, "lgbm_v2_model.txt")
    models["lightgbm"].booster_.save_model(lgbm_path)
    print(f"        LightGBM → {lgbm_path}")

    # CatBoost
    cb_path = os.path.join(MODEL_DIR, "cb_v2_model.cbm")
    models["catboost"].save_model(cb_path)
    print(f"        CatBoost → {cb_path}")

    # Meta-learner
    meta_path = os.path.join(MODEL_DIR, "meta_learner_v2.pkl")
    joblib.dump(meta_model, meta_path)
    print(f"        Meta-learner → {meta_path}")

    # Scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler_v2.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"        Scaler → {scaler_path}")

    # Feature list
    feature_data = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "selection_method": f"mutual_information_top_{MI_TOP_K}",
        "scaler": "StandardScaler",
        "created_at": datetime.now().isoformat(),
    }
    feat_path = os.path.join(MODEL_DIR, "feature_list_v2.json")
    with open(feat_path, "w") as f:
        json.dump(feature_data, f, indent=2)
    print(f"        Feature list → {feat_path}")

    # Training results
    training_data = {
        "model_type": "stacked_ensemble_v2",
        "base_models": ["xgboost", "lightgbm", "catboost"],
        "meta_learner": "logistic_regression",
        "split_method": "walk_forward_chronological",
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "xgb_params": {k: str(v) for k, v in XGB_PARAMS.items()},
        "lgbm_params": {k: str(v) for k, v in LGBM_PARAMS.items()},
        "cb_params": {k: str(v) for k, v in CB_PARAMS.items()},
        "meta_params": META_PARAMS,
        "feature_selection": {
            "method": "mutual_information",
            "top_k": MI_TOP_K,
            "n_features_selected": len(feature_names),
        },
        "results": results,
        "created_at": datetime.now().isoformat(),
    }
    results_path = os.path.join(MODEL_DIR, "training_results_v2.json")
    with open(results_path, "w") as f:
        json.dump(training_data, f, indent=2, default=str)
    print(f"        Training results → {results_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("XAUUSD Stacked Ensemble Model Training — v2")
    print("=" * 70)
    total_t0 = time.time()

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Prepare features
    X, y, feature_names = prepare_features(df)
    del df  # Free memory

    # 3. Feature selection
    X, feature_names = select_features(X, y, feature_names, k=MI_TOP_K)

    # 4. Walk-forward split
    X_train, y_train, X_val, y_val, X_test, y_test = walk_forward_split(X, y)

    # 5. Scale features
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # 6. Train base models
    models, val_probas, val_metrics = train_base_models(X_train_s, y_train, X_val_s, y_val)

    # 7. Train meta-learner
    meta_model = train_meta_learner(val_probas, y_val)

    # 8. Evaluate
    results = evaluate_ensemble(models, meta_model, X_test_s, y_test)

    # Include validation metrics in results
    results["validation_metrics"] = val_metrics

    # 9. Save models
    save_models(models, meta_model, scaler, feature_names, results)

    total_time = time.time() - total_t0
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nAll models saved to: {MODEL_DIR}/")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
