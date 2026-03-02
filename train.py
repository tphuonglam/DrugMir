import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
import shap
import json
import matplotlib.pyplot as plt
import argparse

from ml_model_utils import get_model_and_transform
from viz_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML models on miRNA and Gene data.")
    
    # Path Arguments
    parser.add_argument("--root", type=str, default="data", help="Root directory for raw data")
    parser.add_argument("--root_filter", type=str, default="data_drug_response", help="Directory for filtered drug response data")
    parser.add_argument("--outdir", type=str, default="outputs_sep/12-19-2025_new", help="Output directory for results")
    
    # Execution Arguments
    parser.add_argument("--seed", type=int, default=0, help="Random state seed")
    parser.add_argument("--models", nargs="+", default=["logreg"], 
                        help="List of models to run (e.g., rf logreg gbm)")
    parser.add_argument("--settings", nargs="+", 
                        default=["DEmiRs", "DEGs", "DEGs + DEmiRs"],
                        help="List of data settings (e.g., 'DEmiRs' 'DEGs + DEmiRs')")
    
    return parser.parse_args()

def train_model(model_type, df_feats, df_labels, n_splits=5, random_state=42):
    # --- 1. Align Features (X) and Labels (y) ---
    common_idx = df_feats.index.intersection(df_labels.index)
    X = df_feats.loc[common_idx].values
    y = df_labels.loc[common_idx].iloc[:, 0].values 
    sample_idx = common_idx

    all_preds = np.zeros(X.shape[0], dtype=int)
    all_probs = np.zeros(X.shape[0], dtype=float) 
    
    model_class, model_kwargs, tf_class = get_model_and_transform(model_type)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_per_fold = []
    
    last_model = None
    X_last_val = None
    scaler = None
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if tf_class:
            scaler = tf_class()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

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
        last_model = model
        X_last_val = X_val

    keys = metrics_per_fold[0].keys() if metrics_per_fold else []
    avg_metrics = {k: np.mean([m[k] for m in metrics_per_fold]) for k in keys}
    std_metrics = {k: np.std([m[k] for m in metrics_per_fold]) for k in keys}
    
    all_preds_series = pd.Series(data=all_preds, index=sample_idx, dtype=int)
    all_probs_series = pd.Series(data=all_probs, index=sample_idx, dtype=float)

    return avg_metrics, std_metrics, metrics_per_fold, all_preds_series, all_probs_series, last_model, X_last_val, scaler, y

def get_idx(drug, root_filter):
    path = Path(root_filter) / drug
    csv_path = list(path.glob("DEmiRNA*"))[0]
    df = pd.read_csv(csv_path)["miRNA"]
    return df.to_list()

def join(drug, root_filter):
    path_drug = Path(root_filter) / drug
    path1, path2 = list(path_drug.glob("*pairs*"))
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    all_targets = pd.concat([df1['target_symbol'], df2['target_symbol']], ignore_index=True)
    return all_targets.unique().tolist()

def get_mapping(drug, root_filter):
    path_drug = Path(root_filter) / drug
    path_mrna = list(path_drug.glob("DEmRNA*"))[0]
    df = pd.read_csv(path_mrna)
    return dict(zip(df['mRNA'], df['hgnc_symbol']))

def process_gene_features(df_gene: pd.DataFrame, filtered_idx: list, mapping: dict) -> pd.DataFrame:
    columns_to_keep = [col for col in df_gene.columns if col in mapping]
    df_filtered = df_gene[columns_to_keep]
    df_renamed = df_filtered.rename(columns=mapping)
    
    if filtered_idx:
        final_cols = [col for col in df_renamed.columns if col in filtered_idx]
        return df_renamed[final_cols]
    return df_renamed

if __name__ == "__main__":
    args = parse_args()
    
    for folder in Path(args.root).iterdir():
        if not folder.is_dir(): continue
        drug = folder.stem
        print(f"[INFO] Processing Drug: {drug}")

        # Load miRNA features
        csv_path_mirna = folder / (drug + "_miRNA.csv")
        df_mirna = pd.read_csv(csv_path_mirna, index_col=0) 
        
        # Load gene features (DEG)
        filtered_idx = join(drug, args.root_filter)
        mapping = get_mapping(drug, args.root_filter)
        csv_path_gene = folder / (drug + "_gene.csv")
        df_gene = pd.read_csv(csv_path_gene, index_col=0)
        
        df_gene_intersect = process_gene_features(df_gene, filtered_idx, mapping)
        df_gene_full = process_gene_features(df_gene, None, mapping)

        csv_path_label = folder / (drug + "_label.csv")
        df_labels = pd.read_csv(csv_path_label, index_col=0)
        df_labels['Response'] = df_labels['Response'].map({"Responder": 1, "NonResponder": 0})
        
        for model_arg in args.models:
            print(f"[INFO] Model: {model_arg}")
            model_results_list = []
            roc_data_collection = {} 
            
            for setting_arg in args.settings:
                df_X = None
                current_feature_names = []
                run_shap = False
                name_display = setting_arg
                
                if setting_arg == "DEmiRs":
                    idx = get_idx(drug, args.root_filter)
                    df_X = df_mirna[idx]
                    run_shap = True
                elif setting_arg == "Intersect DEGs":
                    df_X = df_gene_intersect
                elif setting_arg == "Combine DEmiRs":
                    df_X = df_mirna
                elif setting_arg == "Intersect DEGs + DEmiRs":
                    idx = get_idx(drug, args.root_filter)
                    df_X = pd.concat([df_gene_intersect, df_mirna[idx]], axis=1)
                    run_shap = True
                elif setting_arg == "DEGs":
                    df_X = df_gene_full
                    run_shap = True
                elif setting_arg == "DEGs + DEmiRs":
                    idx = get_idx(drug, args.root_filter)
                    df_X = pd.concat([df_gene_full, df_mirna[idx]], axis=1)
                    run_shap = True
                
                current_feature_names = df_X.columns.tolist()
                
                avg_metrics, std_metrics, metrics_per_fold, preds, probs, last_model, X_last_val, scaler, y_aligned = train_model(
                    model_arg, df_X, df_labels, random_state=args.seed
                )
                
                roc_data_collection[name_display] = (y_aligned, probs.values)
                run_type = setting_arg.replace(' ', '_').replace('+', '_')
                out_dir = Path(args.outdir) / drug / model_arg / run_type
                out_dir.mkdir(parents=True, exist_ok=True)
                
                metrics_entry = {**avg_metrics, "Setting": name_display}
                model_results_list.append(metrics_entry)
                
                # Save Jsons
                for filename, data in [("metrics.json", avg_metrics), ("metrics_std.json", std_metrics), ("metrics_folds.json", metrics_per_fold)]:
                    with open(out_dir / filename, 'w') as f:
                        json.dump(data, f, indent=4)

                preds.to_csv(out_dir / "predictions.csv", header=["prediction"], index_label="patient_id")
                probs.to_csv(out_dir / "probabilities.csv", header=["probability"], index_label="patient_id")
                
                if run_shap and last_model: 
                    X_explain = scaler.transform(X_last_val) if scaler else X_last_val
                    X_shap_for_plot = X_explain 
                    
                    if model_arg in ["rf", "gbm"]:
                        explainer = shap.TreeExplainer(last_model)
                        shap_values = explainer.shap_values(X_explain)
                        if isinstance(shap_values, list): shap_values = shap_values[1]
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
                plot_dir = Path(args.outdir) / drug / model_arg / "comparison"
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]
                plot_metrics_line_chart(df_results, metrics, plot_dir / "line.png")
                plot_metrics_bar_chart(df_results, metrics, plot_dir / "bar.png")
                if roc_data_collection:
                    plot_roc_comparison(roc_data_collection, plot_dir / "roc_curves.png")