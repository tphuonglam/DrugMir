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

from ml_model_utils import get_model_and_transform
from viz_utils import *

ROOT = "data"
ROOT_FILTER =  "data_drug_response"
OUTDIR = "outputs_sep/12-19-2025_new"
RANDOM_STATE = 0

MODELS = [
    # "rf", 
    "logreg", 
    # "gbm", 
    # "ada", 
    # "svm",
]   # List of models for experimenting
SETTINGS = [
    "DEmiRs", 
    # "Combine DEmiRs",
    # "Intersect DEGs", 
    "DEGs", 
    # "Intersect DEGs + DEmiRs", 
    "DEGs + DEmiRs", 
]   # List of data settings for experimenting


def train_model(model_type, df_feats, df_labels, n_splits=5, random_state=42):
    
    # --- 1. Align Features (X) and Labels (y) ---
    common_idx = df_feats.index.intersection(df_labels.index)
    
    X = df_feats.loc[common_idx].values
    y = df_labels.loc[common_idx].iloc[:, 0].values 
    
    sample_idx = common_idx

    all_preds = np.zeros(X.shape[0], dtype=int)
    all_probs = np.zeros(X.shape[0], dtype=float) # Store probabilities for ROC
    
    # --- 2. Get Model Class and Transformation Class ---
    model_class, model_kwargs, tf_class = get_model_and_transform(model_type)

    # --- 3. K-Fold Setup ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_per_fold = []
    
    # Variables to capture last model and scaler for SHAP
    last_model = None
    X_last_val = None
    scaler = None
    
    # --- 4. Cross-Validation Loop ---
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

    # --- 5. Calculate Average and STD Metrics ---        
    if not metrics_per_fold:
        avg_metrics = {}
        std_metrics = {}
    else:
        keys = metrics_per_fold[0].keys()
        avg_metrics = {k: np.mean([m[k] for m in metrics_per_fold]) for k in keys}
        std_metrics = {k: np.std([m[k] for m in metrics_per_fold]) for k in keys}
    
    all_preds_series = pd.Series(
        data=all_preds,
        index=sample_idx,
        dtype=int
    )
    
    all_probs_series = pd.Series(
        data=all_probs,
        index=sample_idx,
        dtype=float
    )

    # Return model/data needed for SHAP outside of K-Fold
    # Also return 'y' (aligned true labels) for ROC plotting later
    return avg_metrics, std_metrics, metrics_per_fold, all_preds_series, all_probs_series, last_model, X_last_val, scaler, y


def get_idx(drug):
    path = Path(ROOT_FILTER) / drug
    csv_path = list(path.glob("DEmiRNA*"))[0]
    df = pd.read_csv(csv_path)["miRNA"]
    df = df.to_list()
    return df


# =========================================================================
# === PROCESS GENE
# =========================================================================
def join(drug):
    path_drug = Path(ROOT_FILTER) / drug

    path1, path2 = list(path_drug.glob("*pairs*"))

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    targets1 = df1['target_symbol']
    targets2 = df2['target_symbol']
    all_targets = pd.concat([targets1, targets2], ignore_index=True)
    unique_targets = all_targets.unique().tolist()

    return unique_targets


def get_mapping(drug):
    path_drug = Path(ROOT_FILTER) / drug
    path_mrna = list(path_drug.glob("DEmRNA*"))[0]
    df = pd.read_csv(path_mrna)
    mapping_dict = dict(zip(df['mRNA'], df['hgnc_symbol']))
    return mapping_dict


def process_gene_features(df_gene: pd.DataFrame, filtered_idx: list, mapping: dict) -> pd.DataFrame:
    columns_to_keep = [col for col in df_gene.columns if col in mapping]
    df_filtered = df_gene[columns_to_keep]
    df_renamed = df_filtered.rename(columns=mapping)
    
    if filtered_idx:
        final_cols = [col for col in df_renamed.columns if col in filtered_idx]
        df_final_ml_ready = df_renamed[final_cols]
        return df_final_ml_ready
    return df_renamed


if __name__ == "__main__":
    
    for folder in Path(ROOT).iterdir():
        drug = folder.stem
        print(f"[INFO] Processing Drug: {drug}")

        # 1. Load Data Once Per Drug (Features and Labels)

        # Load miRNA features
        csv_path_mirna = folder / (drug + "_miRNA.csv")
        df_mirna = pd.read_csv(csv_path_mirna, index_col=0) 
        
        # Load gene features (DEG)
        filtered_idx = join(drug)               # list of string, sth like ['PREX1', 'VAV3', 'TBC1D9', 'TRPS1', 'CXCL10', 'AR', 'BMP2', 'C4BPB', 'AKR1B10', 'VSIG1']
        mapping = get_mapping(drug)             # a dict from sth to filtered_idx values
        csv_path_gene = folder / (drug + "_gene.csv")
        df_gene = pd.read_csv(csv_path_gene, index_col=0)
        df_gene_intersect = process_gene_features(
            df_gene=df_gene, 
            filtered_idx=filtered_idx, 
            mapping=mapping
        )
        df_gene_full = process_gene_features(
            df_gene=df_gene, 
            filtered_idx=None, 
            mapping=mapping
        )

        csv_path_label = folder / (drug + "_label.csv")
        df_labels = pd.read_csv(csv_path_label, index_col=0)
        mapping_lbl = {"Responder": 1, "NonResponder": 0}
        df_labels['Response'] = df_labels['Response'].map(mapping_lbl)
        
        # --- Loop over Models ---
        for model_arg in MODELS:
            print(f"[INFO] Model: {model_arg}")
            
            model_results_list = []
            roc_data_collection = {} # Store data for ROC comparison
            
            # --- Loop over Settings ---
            for setting_arg in SETTINGS:
                
                df_X = None
                current_feature_names = []
                run_shap = False
                name_display = ""
                
                # --- Prepare Feature Data for the current setting ---
                ### [1]
                if setting_arg == "DEmiRs":
                    idx = get_idx(drug)
                    df_X = df_mirna[idx]
                    current_feature_names = df_mirna[idx].columns.tolist()
                    name_display = "DEmiRs"
                    run_shap = True
                
                ### [2]
                elif setting_arg == "Intersect DEGs":
                    df_X = df_gene_intersect
                    current_feature_names = df_gene_intersect.columns.tolist()
                    name_display = "Intersect DEGs"
                    run_shap = False
                
                ### [3]
                elif setting_arg == "Combine DEmiRs":
                    df_X = df_mirna
                    current_feature_names = df_mirna.columns.tolist()
                    name_display = "Combine DEmiRs"
                    run_shap = False

                ### [4]
                elif setting_arg == "Intersect DEGs + DEmiRs":
                    idx = get_idx(drug)
                    assert df_mirna.shape[0] == df_gene_intersect.shape[0]
                    df_X = pd.concat([df_gene_intersect, df_mirna[idx]], axis=1)
                    current_feature_names = df_gene_intersect.columns.tolist() + df_mirna[idx].columns.tolist()
                    name_display = "Intersect DEGs + DEmiRs"
                    run_shap = True

                ### [5]
                elif setting_arg == "DEGs":
                    assert df_mirna.shape[0] == df_gene_full.shape[0]
                    df_X = df_gene_full
                    current_feature_names = df_gene_full.columns.tolist()
                    name_display = "DEGs"
                    run_shap = True

                ### [6]
                elif setting_arg == "DEGs + DEmiRs":
                    idx = get_idx(drug)
                    assert df_mirna.shape[0] == df_gene_full.shape[0]
                    df_X = pd.concat([df_gene_full, df_mirna[idx]], axis=1)
                    current_feature_names = df_gene_full.columns.tolist() + df_mirna[idx].columns.tolist()
                    name_display = "DEGs + DEmiRs"
                    run_shap = True
                
                # 1. Run training
                # Now returns std_metrics and all_probs and y_aligned
                avg_metrics, std_metrics, metrics_per_fold, preds, probs, last_model, X_last_val, scaler, y_aligned = train_model(
                    model_arg, df_X, df_labels, random_state=RANDOM_STATE
                )
                
                # 2. Collect ROC Data
                roc_data_collection[name_display] = (y_aligned, probs.values)

                # 3. Prepare output directory
                run_type = setting_arg.replace(' ', '_').replace('+', '_')
                out_dir = Path(OUTDIR) / drug / model_arg / run_type
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # 4. Store metrics
                metrics_entry = avg_metrics.copy()
                metrics_entry["Setting"] = name_display
                model_results_list.append(metrics_entry)
                
                # 5. Save results
                with open(out_dir / "metrics.json", 'w') as f:
                    json.dump(avg_metrics, f, indent=4)
                
                # Save standard deviations as well
                with open(out_dir / "metrics_std.json", 'w') as f:
                    json.dump(std_metrics, f, indent=4)

                # Save Per-Fold Results
                with open(out_dir / "metrics_folds.json", 'w') as f:
                    json.dump(metrics_per_fold, f, indent=4)


                preds.to_csv(out_dir / "predictions.csv", header=["prediction"], index_label="patient_id")
                probs.to_csv(out_dir / "probabilities.csv", header=["probability"], index_label="patient_id")
                
                # 6. Run SHAP analysis
                if run_shap and last_model: 
                    if not (
                        (out_dir / "shap_beeswarm.png").exists() and 
                        (out_dir / "shap_bar.png").exists()
                    ):
                        X_explain = scaler.transform(X_last_val) if scaler else X_last_val
                        X_shap_for_plot = X_explain # Default for TreeExplainer
                        
                        # Determine explainer type
                        if model_arg in ["rf", "gbm"]:
                            explainer = shap.TreeExplainer(last_model)
                            shap_values = explainer.shap_values(X_explain)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1] # Class 1
                        else: 
                            # KernelExplainer
                            background = X_explain[np.random.choice(X_explain.shape[0], min(50, len(X_explain)), replace=False)]
                            explainer = shap.KernelExplainer(last_model.predict_proba, background)
                            
                            X_shap_sample = X_explain[:min(50, len(X_explain))]
                            shap_values = explainer.shap_values(X_shap_sample) 
                            shap_values = shap_values[1] # Class 1
                            X_shap_for_plot = X_shap_sample # Use the sample for plotting

                        # A. Beeswarm Plot (New)
                        if not (out_dir / "shap_beeswarm.png").exists():
                            plot_shap_beeswarm(
                                shap_values, 
                                X_shap_for_plot, 
                                feature_names=current_feature_names, 
                                save_path=out_dir / "shap_beeswarm.png"
                            )

                        # B. Bar Plot (Existing - aggregated)
                        if not (out_dir / "shap_bar.png").exists():
                            shap_values_mean = np.abs(shap_values).mean(0)
                            shap_importance = pd.DataFrame({
                                "feature": current_feature_names,
                                "importance": shap_values_mean
                            }).sort_values("importance", ascending=False)
                            
                            plot_shap_importance(shap_importance, out_dir / "shap_bar.png")

            # --- 7. Plotting Comparisons (Per Model, across Settings) ---
            if model_results_list:
                df_results = pd.DataFrame(model_results_list)
                metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]
                
                plot_dir = Path(OUTDIR) / drug / model_arg / "comparison"
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                plot_metrics_line_chart(df_results, metrics, plot_dir / "line.png")
                plot_metrics_bar_chart(df_results, metrics, plot_dir / "bar.png")
                
                # Plot ROC Comparison (New)
                if roc_data_collection:
                    plot_roc_comparison(roc_data_collection, plot_dir / "roc_curves.png")
                
                print(f"[INFO] Comparison charts for {model_arg} saved to {plot_dir}")