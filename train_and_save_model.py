# train_and_save_model.py
"""
Usage:
    python train_and_save_model.py --data-path creditcard.csv --out-dir artifacts

This script trains an XGBoost classifier with a small pipeline and SMOTE balancing.
It saves model artifacts to `artifacts/` including:
 - model.joblib (the trained pipeline)
 - feature_columns.json (ordered list of features)

"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from utils import load_data, eval_metrics
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def main(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(data_path)
    print("Data shape:", df.shape)

    # Basic EDA printouts (you can expand)
    print("Fraud ratio:")
    print(df['Class'].value_counts(normalize=True))

    # Features and label
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Keep feature column order for later inference
    feature_columns = X.columns.tolist()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Build pipeline with imputation, scaling (optional), SMOTE and XGB
    pipeline = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    param_dist = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 6, 9],
        "clf__learning_rate": [0.01, 0.05, 0.1]
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    rs = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=8,
        scoring='average_precision', cv=cv, n_jobs=-1, verbose=1, random_state=42
    )

    print("Starting hyperparameter search/training...")
    rs.fit(X_train, y_train)

    print("Best params:", rs.best_params_)
    print("Best score (PR-AUC CV):", rs.best_score_)

    best_model = rs.best_estimator_

    # Evaluate on test
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics_default = eval_metrics(y_test, y_prob, threshold=0.5)
    print("Test metrics at 0.5 threshold:")
    print(metrics_default)

    # Save artifacts
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(best_model, model_path)
    print("Saved model to", model_path)

    feat_path = os.path.join(out_dir, "feature_columns.json")
    with open(feat_path, 'w') as f:
        json.dump(feature_columns, f)
    print("Saved feature column order to", feat_path)

    # Save a small evaluation report
    report = {
        "test_metrics_threshold_0.5": metrics_default,
        "best_params": rs.best_params_,
        "cv_best_score": rs.best_score_,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0])
    }
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # SHAP explanation on a small sample (be mindful of speed)
    print("Computing SHAP values (this may take a little while)...")
    # Use tree explainer on the XGB model inside the pipeline
    # We need to pass transformed features to the model. We'll extract the trained steps.

    # Build a helper to get the transformed X for SHAP (pipeline excluding SMOTE and classifier)
    from sklearn.compose import ColumnTransformer

    # Create a transform-only pipeline: imputer + scaler
    from sklearn.pipeline import Pipeline as SkPipeline
    transform_pipeline = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_trans = transform_pipeline.fit_transform(X_train)
    X_test_trans = transform_pipeline.transform(X_test)

    # Extract the trained XGBoost model
    xgb_model = best_model.named_steps['clf']

    explainer = shap.TreeExplainer(xgb_model)
    # Use a small sample for SHAP to save time
    sample_idx = np.random.choice(X_test_trans.shape[0], size=min(1000, X_test_trans.shape[0]), replace=False)
    shap_values = explainer.shap_values(X_test_trans[sample_idx])

    # Plot summary
    plt.figure(figsize=(8,6))
    try:
        shap.summary_plot(shap_values, X_test.iloc[sample_idx], show=False)
        plt.tight_layout()
        shap_path = os.path.join(out_dir, 'shap_summary.png')
        plt.savefig(shap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved SHAP summary to", shap_path)
    except Exception as e:
        print("Could not produce SHAP plot in this environment:", e)

    print("All done. Artifacts in:", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='creditcard.csv')
    parser.add_argument('--out-dir', type=str, default='artifacts')
    args = parser.parse_args()

    main(args.data_path, args.out_dir)