"""
Logistic Regression Training Script

Fits a logistic regression model to predict positive 10-minute forward returns
using CVD Z-Score and VWAP Distance % as predictors.

Data filter: 5m to 30m after event (300-1800 seconds), 2024/2025 data.

Usage:
    python analysis/fit_logistic_regression.py

Outputs saved to: analysis/logistic_regression_artifacts/
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import stats as scipy_stats

# Paths
DATA_PATH = "/Volumes/Extreme SSD/trading_data/stock/data/processed/earnings_0_to_120m_consolidated.csv"
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "logistic_regression_artifacts")


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load data and filter to 5m-30m window for 2024/2025."""
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df['acceptance_datetime_utc'] = pd.to_datetime(df['acceptance_datetime_utc'])
    
    if 'year' not in df.columns:
        df['year'] = df['acceptance_datetime_utc'].dt.year
    
    # Filter to 2024/2025
    df = df[df['year'].isin([2024, 2025])]
    print(f"After year filter (2024/2025): {len(df):,} rows")
    
    # Filter to 5m-30m window
    df = df[
        (df['seconds_since_event'] >= 300) & 
        (df['seconds_since_event'] <= 1800)
    ]
    print(f"After time filter (5m-30m): {len(df):,} rows")
    
    return df


def fit_logistic_regression(df: pd.DataFrame):
    """Fit logistic regression and compute all metrics."""
    
    required_cols = ['cvd_zscore', 'vwap_distance_pct', 'target_ret_600s']
    
    # Prepare data
    model_df = df[required_cols].dropna().copy()
    print(f"After dropping NaN: {len(model_df):,} rows")
    
    # Create binary target
    model_df['target_positive'] = (model_df['target_ret_600s'] > 0).astype(int)
    
    X = model_df[['cvd_zscore', 'vwap_distance_pct']]
    y = model_df['target_positive']
    
    print(f"Fitting logistic regression...")
    print(f"  - Positive returns: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  - Negative returns: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    
    # Fit model
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X, y)
    
    # Predictions
    y_pred = logreg.predict(X)
    y_prob = logreg.predict_proba(X)[:, 1]
    
    # Calculate standard errors and p-values (Wald test)
    n_samples = len(X)
    p = logreg.predict_proba(X)[:, 1]
    X_with_intercept = np.hstack([np.ones((n_samples, 1)), X.values])
    
    # Hessian approximation for variance calculation
    W = np.diag(p * (1 - p))
    try:
        cov_matrix = np.linalg.inv(X_with_intercept.T @ W @ X_with_intercept)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        print("Warning: Could not compute standard errors")
        se = np.array([np.nan, np.nan, np.nan])
    
    # Coefficients
    intercept = logreg.intercept_[0]
    coefs = logreg.coef_[0]
    all_coefs = np.concatenate([[intercept], coefs])
    
    # Z-scores and p-values
    z_scores = all_coefs / se
    p_values = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_scores)))
    
    # Odds ratios and confidence intervals
    odds_ratios = np.exp(all_coefs)
    ci_lower = np.exp(all_coefs - 1.96 * se)
    ci_upper = np.exp(all_coefs + 1.96 * se)
    
    # Model summary
    model_summary = pd.DataFrame({
        'Variable': ['Intercept', 'CVD Z-Score', 'VWAP Distance %'],
        'Coefficient': all_coefs,
        'Std_Error': se,
        'Z_Score': z_scores,
        'P_Value': p_values,
        'Odds_Ratio': odds_ratios,
        'CI_Lower_95': ci_lower,
        'CI_Upper_95': ci_upper
    })
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    confusion_data = {
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    })
    
    # Probability distribution data
    prob_dist = pd.DataFrame({
        'predicted_prob': y_prob,
        'actual': y.values
    })
    
    # Metadata
    n_events = df.groupby(['ticker', 'acceptance_datetime_utc']).ngroups
    metadata = {
        'total_observations': int(len(model_df)),
        'positive_returns': int(y.sum()),
        'negative_returns': int((y == 0).sum()),
        'positive_pct': float(y.mean() * 100),
        'n_events': int(n_events),
        'roc_auc': float(roc_auc),
        'data_filter': {
            'years': [2024, 2025],
            'time_window_seconds': [300, 1800],
            'time_window_minutes': '5m to 30m'
        },
        'predictors': ['cvd_zscore', 'vwap_distance_pct'],
        'target': 'target_ret_600s > 0'
    }
    
    return {
        'model_summary': model_summary,
        'confusion_data': confusion_data,
        'roc_data': roc_data,
        'roc_auc': roc_auc,
        'prob_dist': prob_dist,
        'metadata': metadata
    }


def save_artifacts(results: dict, output_dir: str):
    """Save all artifacts to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Model summary
    summary_path = os.path.join(output_dir, "model_summary.csv")
    results['model_summary'].to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    
    # Confusion matrix and metrics
    confusion_path = os.path.join(output_dir, "confusion_metrics.json")
    with open(confusion_path, 'w') as f:
        json.dump(results['confusion_data'], f, indent=2)
    print(f"Saved: {confusion_path}")
    
    # ROC data
    roc_path = os.path.join(output_dir, "roc_curve.csv")
    results['roc_data'].to_csv(roc_path, index=False)
    print(f"Saved: {roc_path}")
    
    # Probability distribution
    prob_path = os.path.join(output_dir, "probability_distribution.csv")
    results['prob_dist'].to_csv(prob_path, index=False)
    print(f"Saved: {prob_path}")
    
    # Metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(results['metadata'], f, indent=2)
    print(f"Saved: {meta_path}")
    
    print(f"\nAll artifacts saved to: {output_dir}")


def main():
    print("=" * 60)
    print("Logistic Regression Training")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data(DATA_PATH)
    
    # Fit model
    results = fit_logistic_regression(df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(results['model_summary'].to_string(index=False))
    
    print(f"\nROC AUC: {results['roc_auc']:.4f}")
    print(f"Accuracy: {results['confusion_data']['accuracy']:.4f}")
    print(f"F1 Score: {results['confusion_data']['f1_score']:.4f}")
    
    # Save artifacts
    print("\n" + "=" * 60)
    print("Saving Artifacts")
    print("=" * 60)
    save_artifacts(results, ARTIFACTS_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
