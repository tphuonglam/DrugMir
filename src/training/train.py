import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GridSearchCV, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 3, 4, 7], help="List of random state seeds to run and average")
    parser.add_argument("--models", nargs="+", default=["rf", "logreg", "gbm", "ada", "svm"], help="List of models to run")
    parser.add_argument("--settings", nargs="+", default=["DEmiRs", "Target Genes", "Integration"], help="List of data settings")
    parser.add_argument("--drugs", nargs="+", default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"], help="List of drugs to process")
    
    # Feature Engineering Arguments
    parser.add_argument("--threshold_metric", type=str, choices=["youden", "gmean"], default="youden", 
                        help="Metric used to find the optimal ROC threshold (youden or gmean)")
    
    # Option to choose the scaler
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
    
    # --- Configure the Scaler ---
    if scaler_type == "quantile":
        n_quantiles = 20
        scaler_inst = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=random_state)
    elif scaler_type == "standard":
        scaler_inst = tf_class() if tf_class else None
    else:
        scaler_inst = None

    # --- 2. Build Base Pipeline ---
    steps = []
    if scaler_inst is not None:
        steps.append(('scaler', scaler_inst))
        
    # --- FIX: Safely inject the random state into kwargs to avoid duplicate arguments ---
    if 'random_state' in model_class().get_params():
        model_kwargs['random_state'] = random_state
        
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
        "Used_Scaler": scaler_type
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
            print(f"[INFO] Skipping Drug: {drug} (not in specified drug list)")
            continue
        print(f"\n[INFO] Processing Drug: {drug}")

        df_mirna = pd.read_csv(folder / f"{drug}_allmiRNA.csv", index_col=0)
        try:
            df_gene = pd.read_csv(folder / f"{drug}_gene.csv", index_col=0)
        except FileNotFoundError:
            print(f"[WARNING] Gene expression data not found for {drug}, set to None")
            df_gene = None
        df_labels = pd.read_csv(folder / f"{drug}_label.csv", index_col=0)
        
        common_patients = df_mirna.index.intersection(df_labels.index)
        if df_gene is not None:
            common_patients = common_patients.intersection(df_gene.index)
        
        if args.split_file:
            split_file = Path(args.root) / drug / args.split_file 
            splits_df = pd.read_csv(split_file, index_col=0)
            train_patients = splits_df[splits_df['split'].str.lower() == 'train'].index
            test_patients = splits_df[splits_df['split'].str.lower() == 'test'].index
            
            train_idx = common_patients.intersection(train_patients)
            test_idx = common_patients.intersection(test_patients)
            print(f"[INFO] Data Split -> Train: {len(train_idx)} | Valid/Test Folds: {len(test_idx)} (Metric: {args.threshold_metric}, Scaler: {args.scaler})")
        else:
            train_idx = common_patients
            test_idx = pd.Index([])
            print(f"[INFO] No split file. Training on all {len(common_patients)} common patients. (Metric: {args.threshold_metric}, Scaler: {args.scaler})")
            
        all_valid_patients = train_idx.union(test_idx)
        df_mirna = df_mirna.loc[all_valid_patients]
        if df_gene is not None:
            df_gene = df_gene.loc[all_valid_patients]
        df_labels = df_labels.loc[all_valid_patients]
        
        for model_arg in args.models:
            print(f"\n[INFO] Model: {model_arg}")
            model_results_list = []
            roc_data_collection = {} 
            
            for setting_arg in args.settings:
                df_X = None
                run_shap = False
                name_display = setting_arg
                
                if setting_arg == "DEmiRs":
                    df_X = df_mirna
                elif setting_arg == "Target Genes":
                    df_X = df_gene
                elif setting_arg == "Integration":
                    df_X = pd.concat([df_gene, df_mirna], axis=1)
                else:
                    print(f"[WARNING] Skipping unknown setting: {setting_arg}")
                    continue

                param_grid = None
                if model_arg == "logreg":
                    param_grid = [
                        {'model__solver': ['liblinear'], 'model__penalty': ['l1', 'l2'], 'model__C': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]},
                        {'model__solver': ['lbfgs'], 'model__penalty': ['l2'], 'model__C': [0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0]},
                        {'model__solver': ['lbfgs'], 'model__penalty': [None]}
                    ]
                
                df_X_train = df_X.loc[train_idx]
                df_labels_train = df_labels.loc[train_idx]
                
                run_type = setting_arg.replace(' ', '_').replace('+', '_')
                out_root = Path(args.outdir) / drug / f"{model_arg}_{args.scaler}_{args.threshold_metric}"
                
                train_dir = out_root / "training" / run_type
                train_dir.mkdir(parents=True, exist_ok=True)
                
                seed_metrics_list = []
                train_probs_list = []
                
                # --- Loop over multiple seeds ---
                for seed in args.seeds:
                    print(f"  -> Running {setting_arg} | Seed {seed}...")
                    
                    pooled_metrics, preds, probs, tuned_model, X_train_raw, y_train_aligned, best_params = train_model(
                        model_arg, df_X_train, df_labels_train, 
                        random_state=seed, 
                        param_grid=param_grid, 
                        threshold_metric=args.threshold_metric,
                        scaler_type=args.scaler
                    )
                    
                    train_probs_list.append(probs.values)
                    metrics_entry = {**pooled_metrics, "Setting": name_display, "Seed": seed}
                    seed_metrics_list.append(metrics_entry)
                    
                    # Save seed-specific outputs to avoid overwriting
                    with open(train_dir / f"metrics_seed{seed}.json", 'w') as f:
                        json.dump(metrics_entry, f, indent=4)
                    
                    if best_params:
                        with open(train_dir / f"best_params_seed{seed}.json", 'w') as f:
                            json.dump(best_params, f, indent=4)

                    joblib.dump(tuned_model, train_dir / f"model_seed{seed}.joblib")

                    preds.to_csv(train_dir / f"predictions_train_seed{seed}.csv", header=["prediction"], index_label="patient_id")
                    probs.to_csv(train_dir / f"probabilities_train_seed{seed}.csv", header=["probability"], index_label="patient_id")
                    
                    current_feature_names = df_X.columns.tolist()
                    
                    # --- Process SHAP plots on ALL DATA (as clarified) ---
                    if run_shap and tuned_model: 
                        est = tuned_model.estimator_
                        pipe = est.best_estimator_ if isinstance(est, GridSearchCV) else est
                        scaler = pipe.named_steps.get('scaler', None)
                        base_model = pipe.named_steps['model']

                        # We use df_X.values here to capture the entirety of the data the script sees
                        X_all_raw = df_X.values
                        X_all_explain = scaler.transform(X_all_raw) if scaler else X_all_raw
                        
                        if model_arg == "logreg":
                            explainer = shap.LinearExplainer(base_model, X_all_explain)
                            shap_values_all = explainer.shap_values(X_all_explain)
                            X_shap_for_plot = X_all_explain
                        elif model_arg in ["rf", "gbm"]:
                            explainer = shap.TreeExplainer(base_model)
                            shap_values_all = explainer.shap_values(X_all_explain)
                            if isinstance(shap_values_all, list): shap_values_all = shap_values_all[1]
                            X_shap_for_plot = X_all_explain
                        else: 
                            background = X_all_explain[np.random.choice(X_all_explain.shape[0], min(50, len(X_all_explain)), replace=False)]
                            explainer = shap.KernelExplainer(base_model.predict_proba, background)
                            X_shap_sample = X_all_explain[:min(50, len(X_all_explain))]
                            shap_values_all = explainer.shap_values(X_shap_sample)[1]
                            X_shap_for_plot = X_shap_sample 

                        plot_shap_beeswarm(shap_values_all, X_shap_for_plot, current_feature_names, train_dir / f"shap_beeswarm_alldata_seed{seed}.png")
                        
                        shap_importance = pd.DataFrame({
                            "feature": current_feature_names,
                            "importance": np.abs(shap_values_all).mean(0)
                        }).sort_values("importance", ascending=False)
                        plot_shap_importance(shap_importance, train_dir / f"shap_bar_alldata_seed{seed}.png")

                # --- Calculate Average Metrics Across All Seeds ---
                metric_keys = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC", "Optimal_Threshold"]
                avg_metrics = {
                    "Setting": name_display,
                    "Threshold_Metric": args.threshold_metric,
                    "Used_Scaler": args.scaler,
                    "Total_Seeds_Averaged": len(args.seeds)
                }
                for m_key in metric_keys:
                    avg_metrics[m_key] = np.mean([m[m_key] for m in seed_metrics_list])
                    avg_metrics[f"{m_key}_std"] = np.std([m[m_key] for m in seed_metrics_list])
                
                with open(train_dir / "metrics_averaged.json", 'w') as f:
                    json.dump(avg_metrics, f, indent=4)
                    
                model_results_list.append(avg_metrics)
                
                # Ensemble ROC: average the predicted probabilities across seeds
                mean_train_probs = np.mean(train_probs_list, axis=0)
                roc_data_collection[name_display] = (y_train_aligned, mean_train_probs)

            # Plot comparison charts using the Averaged Metrics
            if model_results_list:
                df_results = pd.DataFrame(model_results_list)
                plot_dir = out_root / "training" / "comparison"
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]
                plot_metrics_line_chart(df_results, metrics, plot_dir / "line.png")
                plot_metrics_bar_chart(df_results, metrics, plot_dir / "bar.png")
                if roc_data_collection:
                    plot_roc_comparison(roc_data_collection, plot_dir / "roc_curves.png")