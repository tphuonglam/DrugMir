import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
import shap
import json
import matplotlib.pyplot as plt
import argparse
import joblib

from ml_model_utils import get_model_and_transform
from viz_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models on pre-processed miRNA and Gene data.")
    
    # Path Arguments
    parser.add_argument("--root", type=str, default="data", help="Root directory containing pre-processed drug folders")
    parser.add_argument("--outdir", type=str, default="outputs_sep/12-19-2025_new", help="Output directory for results")
    parser.add_argument("--split_file", type=str, default=None, help="Path to custom split CSV file (patient_id, split)")
    
    # Execution Arguments
    parser.add_argument("--seed", type=int, default=0, help="Random state seed")
    parser.add_argument("--models", nargs="+", default=["rf", "logreg", "gbm", "ada", "svm"], 
                        help="List of models to run (e.g., rf logreg gbm)")
    parser.add_argument("--settings", nargs="+", 
                        default=["DEmiRs", "DEGs", "DEGs + DEmiRs"],
                        help="List of data settings (e.g., 'DEmiRs' 'DEGs + DEmiRs')")
    parser.add_argument("--drugs", nargs="+", 
                        default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"],
                        help="List of drugs to process")
    
    return parser.parse_args()


def train_model(model_type, df_feats, df_labels, n_splits=5, random_state=42, param_grid=None):
    # --- 1. Align Features (X) and Labels (y) ---
    common_idx = df_feats.index.intersection(df_labels.index)
    X = df_feats.loc[common_idx].values
    y = df_labels.loc[common_idx].iloc[:, 0].values 
    sample_idx = common_idx

    all_preds = np.zeros(X.shape[0], dtype=int)
    all_probs = np.zeros(X.shape[0], dtype=float) 
    
    model_class, model_kwargs, tf_class = get_model_and_transform(model_type)
    
    # Prevent ConvergenceWarnings
    if model_type == "logreg":
        model_kwargs['max_iter'] = 2000

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_per_fold = []
    
    last_model = None
    X_last_val = None
    scaler = None
    best_params_per_fold = [] 
    
    for train_idx, val_idx in kf.split(X):
        # OUTER LOOP: Split data completely
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if param_grid:
            # --- NESTED CV FOR GRID SEARCH ---
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
            
            steps = []
            if tf_class:
                steps.append(('scaler', tf_class()))
            steps.append(('model', model_class(**model_kwargs)))
            pipe = Pipeline(steps)
            
            gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            best_pipe = gs.best_estimator_
            best_params_per_fold.append(gs.best_params_)
            
            if tf_class:
                scaler = best_pipe.named_steps['scaler']
            last_model = best_pipe.named_steps['model']
            
            try:
                y_probs_fold = best_pipe.predict_proba(X_val)[:, 1] 
            except AttributeError:
                y_preds_fold = best_pipe.predict(X_val).astype(int)
                y_probs_fold = y_preds_fold.astype(float)
                
            X_last_val = X_val 
            
        else:
            # --- STANDARD TRAINING (No Grid Search) ---
            if tf_class:
                scaler = tf_class()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            last_model = model
            X_last_val = X_val

            try:
                y_probs_fold = model.predict_proba(X_val)[:, 1] 
            except AttributeError:
                y_preds_fold = model.predict(X_val).astype(int)
                y_probs_fold = y_preds_fold.astype(float)
            
        y_preds_fold = (y_probs_fold > 0.5).astype(int)
        all_preds[val_idx] = y_preds_fold
        all_probs[val_idx] = y_probs_fold

        metrics = {
            "Accuracy": accuracy_score(y_val, y_preds_fold),
            "Precision": precision_score(y_val, y_preds_fold, zero_division=0),
            "Recall": recall_score(y_val, y_preds_fold, zero_division=0),
            "F1": f1_score(y_val, y_preds_fold, zero_division=0),
            "AUC": roc_auc_score(y_val, y_probs_fold),
            "MCC": matthews_corrcoef(y_val, y_preds_fold),
        }
        metrics_per_fold.append(metrics)

    keys = metrics_per_fold[0].keys() if metrics_per_fold else []
    avg_metrics = {k: np.mean([m[k] for m in metrics_per_fold]) for k in keys}
    std_metrics = {k: np.std([m[k] for m in metrics_per_fold]) for k in keys}
    
    all_preds_series = pd.Series(data=all_preds, index=sample_idx, dtype=int)
    all_probs_series = pd.Series(data=all_probs, index=sample_idx, dtype=float)

    return avg_metrics, std_metrics, metrics_per_fold, all_preds_series, all_probs_series, last_model, X_last_val, scaler, y, best_params_per_fold


if __name__ == "__main__":
    args = parse_args()
    
    for folder in Path(args.root).iterdir():
        if not folder.is_dir(): 
            continue
        drug = folder.stem
        if drug not in args.drugs:
            continue
        print(f"\n[INFO] Processing Drug: {drug}")

        # Load clean, pre-processed features and labels
        df_mirna = pd.read_csv(folder / f"{drug}_allmiRNA.csv", index_col=0) 
        df_gene = pd.read_csv(folder / f"{drug}_gene.csv", index_col=0)
        df_labels = pd.read_csv(folder / f"{drug}_label.csv", index_col=0)
        
        # 1. Base alignment: Find patients present in ALL three files
        common_patients = df_mirna.index.intersection(df_gene.index).intersection(df_labels.index)
        
        # 2. Split alignment: If a split file is provided, keep ONLY the "train" patients
        if args.split_file:
            splits_df = pd.read_csv(args.split_file, index_col=0)
            train_patients = splits_df[splits_df['split'].str.lower() == 'train'].index
            common_patients = common_patients.intersection(train_patients)
            print(f"    [INFO] Applying KFold CV on {len(common_patients)} 'train' patients.")
        else:
            print(f"    [INFO] Applying KFold CV on all {len(common_patients)} common patients.")
            
        # 3. Filter dataframes to strictly the aligned subset
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
                
                # Safe Feature Selection (Data is perfectly aligned, concat won't create NaNs)
                if setting_arg == "DEmiRs":
                    df_X = df_mirna
                elif setting_arg == "DEGs":
                    df_X = df_gene
                elif setting_arg == "DEGs + DEmiRs":
                    df_X = pd.concat([df_gene, df_mirna], axis=1)
                else:
                    print(f"[WARNING] Skipping unknown setting: {setting_arg}")
                    continue

                # Nested Grid Search logic
                param_grid = None
                if model_arg == "logreg":
                    print(f"[INFO] Running Nested GridSearchCV for {model_arg} on {setting_arg}...")
                    param_grid = [
                        {'model__solver': ['liblinear'], 'model__penalty': ['l1', 'l2'], 'model__C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                        {'model__solver': ['lbfgs'], 'model__penalty': ['l2'], 'model__C': [0.01, 0.1, 1.0, 10.0, 100.0]},
                        {'model__solver': ['lbfgs'], 'model__penalty': [None]} # Separated to prevent warnings
                    ]                
                
                avg_metrics, std_metrics, metrics_per_fold, preds, probs, last_model, X_last_val, scaler, y_aligned, best_params_per_fold = train_model(
                    model_arg, df_X, df_labels, random_state=args.seed, 
                    param_grid=param_grid
                )
                
                roc_data_collection[name_display] = (y_aligned, probs.values)
                run_type = setting_arg.replace(' ', '_').replace('+', '_')
                out_root = Path(args.outdir) / drug / model_arg / "training" 
                out_dir = out_root / run_type
                out_dir.mkdir(parents=True, exist_ok=True)
                
                metrics_entry = {**avg_metrics, "Setting": name_display}
                model_results_list.append(metrics_entry)
                
                # Save Jsons
                for filename, data in [("metrics.json", avg_metrics), ("metrics_std.json", std_metrics), ("metrics_folds.json", metrics_per_fold)]:
                    with open(out_dir / filename, 'w') as f:
                        json.dump(data, f, indent=4)
                
                if best_params_per_fold:
                    with open(out_dir / "best_params_per_fold.json", 'w') as f:
                        json.dump(best_params_per_fold, f, indent=4)

                # Save the actual model and scaler for evaluate.py
                joblib.dump(last_model, out_dir / "model.joblib")
                if scaler:
                    joblib.dump(scaler, out_dir / "scaler.joblib")

                preds.to_csv(out_dir / "predictions.csv", header=["prediction"], index_label="patient_id")
                probs.to_csv(out_dir / "probabilities.csv", header=["probability"], index_label="patient_id")
                
                current_feature_names = df_X.columns.tolist()
                
                if run_shap and last_model: 
                    X_explain = scaler.transform(X_last_val) if scaler else X_last_val
                    
                    if model_arg == "logreg":
                        explainer = shap.LinearExplainer(last_model, X_explain)
                        shap_values = explainer.shap_values(X_explain)
                        X_shap_for_plot = X_explain
                    elif model_arg in ["rf", "gbm"]:
                        explainer = shap.TreeExplainer(last_model)
                        shap_values = explainer.shap_values(X_explain)
                        if isinstance(shap_values, list): shap_values = shap_values[1]
                        X_shap_for_plot = X_explain
                    else: 
                        background = X_explain[np.random.choice(X_explain.shape[0], min(50, len(X_explain)), replace=False)]
                        explainer = shap.KernelExplainer(last_model.predict_proba, background)
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