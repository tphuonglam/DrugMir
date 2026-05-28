import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer # <-- NEW: Imported QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    make_scorer, confusion_matrix
)
import shap
import json
import matplotlib.pyplot as plt
import argparse
import joblib

from ml_model_utils import get_model_and_transform
from viz_utils import *


def calc_g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return np.sqrt(specificity * sensitivity)
    return 0.0

g_mean_scorer = make_scorer(calc_g_mean)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models on pre-processed miRNA and Gene data.")
    
    # Path Arguments
    parser.add_argument("--root", type=str, default="data", help="Root directory containing pre-processed drug folders")
    parser.add_argument("--outdir", type=str, default="outputs_sep/12-19-2025_new", help="Output directory for results")
    parser.add_argument("--split_file", type=str, default=None, help="Path to custom split CSV file (patient_id, split)")
    
    # Execution Arguments
    parser.add_argument("--seed", type=int, default=0, help="Random state seed")
    parser.add_argument("--models", nargs="+", default=["rf", "logreg", "gbm", "ada", "svm"], help="List of models to run")
    parser.add_argument("--settings", nargs="+", default=["DEmiRs", "DEGs", "DEGs + DEmiRs"], help="List of data settings")
    parser.add_argument("--drugs", nargs="+", default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"], help="List of drugs to process")
    
    # Feature Engineering Arguments
    parser.add_argument("--threshold_metric", type=str, choices=["youden", "gmean"], default="youden", 
                        help="Metric used to find the optimal ROC threshold (youden or gmean)")
    
    # NEW: Option to choose the scaler
    parser.add_argument("--scaler", type=str, choices=["standard", "quantile", "none"], default="standard",
                        help="Scaling method to apply to features before training")
    
    return parser.parse_args()


def train_model(model_type, df_feats, df_labels, n_splits=5, random_state=42, param_grid=None, threshold_metric="youden", scaler_type="standard"):
    # --- 1. Align Features (X) and Labels (y) ---
    common_idx = df_feats.index.intersection(df_labels.index)
    X = df_feats.loc[common_idx].values
    y = df_labels.loc[common_idx].iloc[:, 0].values 
    sample_idx = common_idx

    model_class, model_kwargs, tf_class = get_model_and_transform(model_type)
    
    if model_type == "logreg":
        model_kwargs['max_iter'] = 2000

    # --- NEW: Configure the Scaler ---
    if scaler_type == "quantile":
        # output_distribution='normal' maps the data to a Gaussian distribution, which is usually best for ML.
        n_quantiles = 100
        scaler_inst = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=random_state)
    elif scaler_type == "standard":
        scaler_inst = tf_class() if tf_class else None
    else:
        scaler_inst = None

    # --- 2. Build Base Pipeline ---
    steps = []
    if scaler_inst is not None:
        steps.append(('scaler', scaler_inst))
    steps.append(('model', model_class(**model_kwargs)))
    pipe = Pipeline(steps)

    # --- 3. Handle Hyperparameter Tuning ---
    if param_grid:
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        base_estimator = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    else:
        base_estimator = pipe

    # --- 4. Handle Threshold Tuning ---
    if threshold_metric == "gmean":
        scoring_metric = g_mean_scorer
    else:
        scoring_metric = "balanced_accuracy"

    cv_thresh = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    tuned_model = TunedThresholdClassifierCV(
        estimator=base_estimator,
        scoring=scoring_metric,
        cv=cv_thresh,
        n_jobs=-1
    )

    # --- 5. Final Fit ---
    tuned_model.fit(X, y)

    optimal_threshold = tuned_model.best_threshold_
    best_params = {}
    if param_grid:
        best_params = tuned_model.estimator_.best_params_

    # --- 6. Generate Training Metrics ---
    all_preds = tuned_model.predict(X)
    all_probs = tuned_model.predict_proba(X)[:, 1]

    pooled_metrics = {
        "Accuracy": accuracy_score(y, all_preds),
        "Precision": precision_score(y, all_preds, zero_division=0),
        "Recall": recall_score(y, all_preds, zero_division=0),
        "F1": f1_score(y, all_preds, zero_division=0),
        "AUC": roc_auc_score(y, all_probs),
        "MCC": matthews_corrcoef(y, all_preds),
        "Optimal_Threshold": optimal_threshold,
        "Threshold_Metric": threshold_metric,
        "Used_Scaler": scaler_type # Log which scaler was used
    }
    
    all_preds_series = pd.Series(data=all_preds, index=sample_idx, dtype=int)
    all_probs_series = pd.Series(data=all_probs, index=sample_idx, dtype=float)

    return pooled_metrics, all_preds_series, all_probs_series, tuned_model, X, y, best_params


if __name__ == "__main__":
    args = parse_args()
    
    for folder in Path(args.root).iterdir():
        if not folder.is_dir(): 
            continue
        drug = folder.stem
        if drug not in args.drugs:
            continue
        print(f"\n[INFO] Processing Drug: {drug}")

        df_mirna = pd.read_csv(folder / f"{drug}_allmiRNA.csv", index_col=0) 
        df_gene = pd.read_csv(folder / f"{drug}_gene.csv", index_col=0)
        df_labels = pd.read_csv(folder / f"{drug}_label.csv", index_col=0)
        
        common_patients = df_mirna.index.intersection(df_gene.index).intersection(df_labels.index)
        
        if args.split_file:
            splits_df = pd.read_csv(args.split_file, index_col=0)
            train_patients = splits_df[splits_df['split'].str.lower() == 'train'].index
            common_patients = common_patients.intersection(train_patients)
            print(f"    [INFO] Training on {len(common_patients)} 'train' patients. (Metric: {args.threshold_metric}, Scaler: {args.scaler})")
        else:
            print(f"    [INFO] Training on all {len(common_patients)} common patients. (Metric: {args.threshold_metric}, Scaler: {args.scaler})")
            
        df_mirna = df_mirna.loc[common_patients]
        df_gene = df_gene.loc[common_patients]
        df_labels = df_labels.loc[common_patients]
        
        for model_arg in args.models:
            print(f"[INFO] Model: {model_arg}")
            model_results_list = []
            roc_data_collection = {} 
            
            for setting_arg in args.settings:
                df_X = None
                run_shap = False
                name_display = setting_arg
                
                if setting_arg == "DEmiRs":
                    df_X = df_mirna
                elif setting_arg == "DEGs":
                    df_X = df_gene
                elif setting_arg == "DEGs + DEmiRs":
                    df_X = pd.concat([df_gene, df_mirna], axis=1)
                else:
                    print(f"[WARNING] Skipping unknown setting: {setting_arg}")
                    continue

                param_grid = None
                if model_arg == "logreg":
                    print(f"[INFO] Running Nested GridSearchCV for {model_arg} on {setting_arg}...")
                    param_grid = [
                        {'model__solver': ['liblinear'], 'model__penalty': ['l1', 'l2'], 'model__C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                        {'model__solver': ['lbfgs'], 'model__penalty': ['l2'], 'model__C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                        {'model__solver': ['lbfgs'], 'model__penalty': [None]}
                    ]                
                
                # Pass the scaler argument to the training function
                pooled_metrics, preds, probs, tuned_model, X_raw, y_aligned, best_params = train_model(
                    model_arg, df_X, df_labels, 
                    random_state=args.seed, 
                    param_grid=param_grid, 
                    threshold_metric=args.threshold_metric,
                    scaler_type=args.scaler
                )
                
                roc_data_collection[name_display] = (y_aligned, probs.values)
                run_type = setting_arg.replace(' ', '_').replace('+', '_')
                out_root = Path(args.outdir) / drug / f"{model_arg}_{args.scaler}_{args.threshold_metric}" / "training"
                out_dir = out_root / run_type
                out_dir.mkdir(parents=True, exist_ok=True)
                
                metrics_entry = {**pooled_metrics, "Setting": name_display}
                model_results_list.append(metrics_entry)
                
                with open(out_dir / "metrics.json", 'w') as f:
                    json.dump(metrics_entry, f, indent=4)
                
                if best_params:
                    with open(out_dir / "best_params.json", 'w') as f:
                        json.dump(best_params, f, indent=4)

                joblib.dump(tuned_model, out_dir / "model.joblib")

                preds.to_csv(out_dir / "predictions.csv", header=["prediction"], index_label="patient_id")
                probs.to_csv(out_dir / "probabilities.csv", header=["probability"], index_label="patient_id")
                
                current_feature_names = df_X.columns.tolist()
                
                if run_shap and tuned_model: 
                    est = tuned_model.estimator_
                    pipe = est.best_estimator_ if isinstance(est, GridSearchCV) else est
                    scaler = pipe.named_steps.get('scaler', None)
                    base_model = pipe.named_steps['model']

                    X_explain = scaler.transform(X_raw) if scaler else X_raw
                    
                    if model_arg == "logreg":
                        explainer = shap.LinearExplainer(base_model, X_explain)
                        shap_values = explainer.shap_values(X_explain)
                        X_shap_for_plot = X_explain
                    elif model_arg in ["rf", "gbm"]:
                        explainer = shap.TreeExplainer(base_model)
                        shap_values = explainer.shap_values(X_explain)
                        if isinstance(shap_values, list): shap_values = shap_values[1]
                        X_shap_for_plot = X_explain
                    else: 
                        background = X_explain[np.random.choice(X_explain.shape[0], min(50, len(X_explain)), replace=False)]
                        explainer = shap.KernelExplainer(base_model.predict_proba, background)
                        X_shap_sample = X_explain[:min(50, len(X_explain))]
                        shap_values = explainer.shap_values(X_shap_sample)[1]
                        X_shap_for_plot = X_shap_sample 

                    plot_shap_beeswarm(shap_values, X_shap_for_plot, current_feature_names, out_dir / "shap_beeswarm.png")
                    
                    shap_importance = pd.DataFrame({
                        "feature": current_feature_names,
                        "importance": np.abs(shap_values).mean(0)
                    }).sort_values("importance", ascending=False)
                    plot_shap_importance(shap_importance, out_dir / "shap_bar.png")

            if model_results_list:
                df_results = pd.DataFrame(model_results_list)
                plot_dir = out_root / "comparison"
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]
                plot_metrics_line_chart(df_results, metrics, plot_dir / "line.png")
                plot_metrics_bar_chart(df_results, metrics, plot_dir / "bar.png")
                if roc_data_collection:
                    plot_roc_comparison(roc_data_collection, plot_dir / "roc_curves.png")