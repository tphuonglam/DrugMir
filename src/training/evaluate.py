import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)
import joblib
import json
import argparse
import shap

from viz_utils import (
    plot_metrics_bar_chart, plot_roc_comparison, plot_metrics_line_chart,
    plot_shap_beeswarm, plot_shap_importance
)


# --- joblib needs this function to exist in this file to successfully load the model ---
def calc_g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return np.sqrt(specificity * sensitivity)
    return 0.0
# -------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ML models on a test or external dataset.")
    
    # Path Arguments
    parser.add_argument("--data_root", type=str, required=True, help="Directory containing the pre-processed drug folders for evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where trained models are saved (e.g., outputs_sep/12-19-2025_new)")
    parser.add_argument("--outdir", type=str, default="eval_outputs", help="Output directory for evaluation results")
    parser.add_argument("--split_file", type=str, default=None, help="Path to custom split CSV file. If provided, evaluates ONLY on 'test' patients.")
    
    # Execution Arguments
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="List of random state seeds to evaluate")
    parser.add_argument("--models", nargs="+", default=["rf", "logreg", "gbm", "ada", "svm"], help="List of models to evaluate")
    parser.add_argument("--settings", nargs="+", default=["DEmiRs", "Target Genes", "Integration"], help="List of data settings")
    parser.add_argument("--drugs", nargs="+", default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"], help="Drugs to evaluate")

    # Feature Engineering Arguments
    parser.add_argument("--threshold_metric", type=str, choices=["youden", "gmean"], default="youden", 
                        help="Metric used to find the optimal ROC threshold (youden or gmean)")
    
    # Option to choose the scaler
    parser.add_argument("--scaler", type=str, choices=["standard", "quantile", "none"], default="standard",
                        help="Scaling method to apply to features before training")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    for folder in Path(args.data_root).iterdir():
        if not folder.is_dir(): 
            continue
        drug = folder.stem
        if drug not in args.drugs:
            continue
            
        print(f"\n[INFO] Evaluating Drug: {drug}")

        # 1. Load clean, pre-processed features and labels
        df_mirna = pd.read_csv(folder / f"{drug}_allmiRNA.csv", index_col=0) 
        df_labels = pd.read_csv(folder / f"{drug}_label.csv", index_col=0)
        try:
            df_gene = pd.read_csv(folder / f"{drug}_gene.csv", index_col=0)
        except:
            df_gene = None
        
        # 2. Base alignment: Find patients present in ALL three files
        common_patients = df_mirna.index.intersection(df_labels.index)
        if df_gene is not None:
            common_patients = common_patients.intersection(df_gene.index)
        
        # 3. Split alignment: If a split file is provided, keep ONLY the "test" patients
        if args.split_file:
            split_file = folder / args.split_file
            
            splits_df = pd.read_csv(split_file, index_col=0)
            test_patients = splits_df[splits_df['split'].str.lower() == 'test'].index
            common_patients = common_patients.intersection(test_patients)
            print(f"    [INFO] Evaluating on {len(common_patients)} 'test' patients.")
        else:
            print(f"    [INFO] Evaluating on all {len(common_patients)} common patients (External dataset mode).")
            
        # If no patients match, skip this drug
        if len(common_patients) == 0:
            print(f"    [WARNING] No matching patients found for evaluation. Skipping {drug}.")
            continue
            
        # 4. Filter dataframes strictly to the aligned subset
        df_mirna = df_mirna.loc[common_patients]
        if df_gene is not None:
            df_gene = df_gene.loc[common_patients]
        df_labels = df_labels.loc[common_patients]
        y_test = df_labels.iloc[:, 0].values
        
        for model_arg in args.models:
            print(f"  -> Model: {model_arg}")
            model_results_list = []
            roc_data_collection = {} 
            
            for setting_arg in args.settings:
                run_type = setting_arg.replace(' ', '_').replace('+', '_')
                
                # Notice we point to the exact subfolder structure created by train.py
                saved_model_dir = Path(args.model_dir) / drug / f"{model_arg}_{args.scaler}_{args.threshold_metric}" / "training" / run_type
                out_root = Path(args.outdir) / drug / f"{model_arg}_{args.scaler}_{args.threshold_metric}" / "evaluation"
                out_dir = out_root / run_type
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # 5. Safe Feature Selection
                if setting_arg == "DEmiRs":
                    df_X = df_mirna
                elif setting_arg == "Target Genes":
                    df_X = df_gene
                elif setting_arg == "Integration":
                    df_X = pd.concat([df_gene, df_mirna], axis=1)
                else:
                    continue
                
                X_test_raw = df_X.values
                current_feature_names = df_X.columns.tolist()
                
                seed_metrics_list = []
                eval_probs_list = []
                
                # 6. Loop over seeds to load and evaluate each model
                for seed in args.seeds:
                    model_path = saved_model_dir / f"model_seed{seed}.joblib"
                    metrics_path = saved_model_dir / f"metrics_seed{seed}.json"
                    
                    if not model_path.exists():
                        print(f"    [SKIP] No saved model found at {model_path}")
                        continue
                        
                    model = joblib.load(model_path)
                    
                    optimal_threshold = 0.5
                    if metrics_path.exists():
                        with open(metrics_path, 'r') as f:
                            train_metrics = json.load(f)
                            optimal_threshold = train_metrics.get("Optimal_Threshold", 0.5)
                    
                    # 7. Predict (Pipeline automatically handles scaling internally)
                    y_preds = model.predict(X_test_raw).astype(int)
                    
                    if hasattr(model, "predict_proba"):
                        y_probs = model.predict_proba(X_test_raw)[:, 1] 
                    elif hasattr(model, "decision_function"):
                        y_probs = model.decision_function(X_test_raw)
                    else:
                        y_probs = y_preds.astype(float)

                    # 8. Calculate Metrics
                    metrics = {
                        "Setting": setting_arg,
                        "Seed": seed,
                        "Accuracy": accuracy_score(y_test, y_preds),
                        "Precision": precision_score(y_test, y_preds, zero_division=0),
                        "Recall": recall_score(y_test, y_preds, zero_division=0),
                        "F1": f1_score(y_test, y_preds, zero_division=0),
                        "AUC": roc_auc_score(y_test, y_probs),
                        "MCC": matthews_corrcoef(y_test, y_preds),
                        "Used_Threshold": optimal_threshold
                    }
                    seed_metrics_list.append(metrics)
                    eval_probs_list.append(y_probs)
                    
                    # 9. Save Individual Run Outputs
                    with open(out_dir / f"eval_metrics_seed{seed}.json", 'w') as f:
                        json.dump(metrics, f, indent=4)
                        
                    preds_series = pd.Series(data=y_preds, index=common_patients, name="prediction")
                    probs_series = pd.Series(data=y_probs, index=common_patients, name="probability")
                    preds_series.to_csv(out_dir / f"eval_predictions_seed{seed}.csv", index_label="patient_id")
                    probs_series.to_csv(out_dir / f"eval_probabilities_seed{seed}.csv", index_label="patient_id")

                    # ---------------------------------------------------------
                    # 10. Compute and Save SHAP Explanations for Evaluation Set
                    # ---------------------------------------------------------
                    run_shap = False
                    if run_shap:
                        est = model.estimator_ if hasattr(model, "estimator_") else model
                        pipe = est.best_estimator_ if isinstance(est, GridSearchCV) else est
                        scaler = pipe.named_steps.get('scaler', None)
                        base_model = pipe.named_steps['model']

                        # Manually apply the scaler so SHAP explains the scaled features (if applicable)
                        X_test_explain = scaler.transform(X_test_raw) if scaler else X_test_raw

                        
                        if model_arg == "logreg":
                            explainer = shap.LinearExplainer(base_model, X_test_explain)
                            shap_values_eval = explainer.shap_values(X_test_explain)
                            X_shap_for_plot = X_test_explain
                        elif model_arg in ["rf", "gbm"]:
                            explainer = shap.TreeExplainer(base_model)
                            shap_values_eval = explainer.shap_values(X_test_explain)
                            if isinstance(shap_values_eval, list): 
                                shap_values_eval = shap_values_eval[1]
                            X_shap_for_plot = X_test_explain
                        else:
                            # For SVM or other models, use KernelExplainer with a background sample of the test set
                            background = X_test_explain[np.random.choice(X_test_explain.shape[0], min(50, len(X_test_explain)), replace=False)]
                            explainer = shap.KernelExplainer(base_model.predict_proba, background)
                            X_shap_sample_eval = X_test_explain[:min(50, len(X_test_explain))]
                            shap_values_eval = explainer.shap_values(X_shap_sample_eval)[1]
                            X_shap_for_plot = X_shap_sample_eval

                        # Save SHAP Plots
                        plot_shap_beeswarm(shap_values_eval, X_shap_for_plot, current_feature_names, out_dir / f"shap_beeswarm_eval_seed{seed}.png")
                        
                        shap_importance_eval = pd.DataFrame({
                            "feature": current_feature_names,
                            "importance": np.abs(shap_values_eval).mean(0)
                        }).sort_values("importance", ascending=False)
                        plot_shap_importance(shap_importance_eval, out_dir / f"shap_bar_eval_seed{seed}.png")

                # 11. Calculate Average Metrics Across All Evaluated Seeds
                if seed_metrics_list:
                    metric_keys = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC", "Used_Threshold"]
                    avg_metrics = {
                        "Setting": setting_arg,
                        "Total_Seeds_Evaluated": len(seed_metrics_list)
                    }
                    for m_key in metric_keys:
                        avg_metrics[m_key] = np.mean([m[m_key] for m in seed_metrics_list])
                        avg_metrics[f"{m_key}_std"] = np.std([m[m_key] for m in seed_metrics_list])
                    
                    with open(out_dir / "eval_metrics_averaged.json", 'w') as f:
                        json.dump(avg_metrics, f, indent=4)
                        
                    model_results_list.append(avg_metrics)
                    
                    # Ensemble ROC: average the predicted probabilities across seeds
                    mean_eval_probs = np.mean(eval_probs_list, axis=0)
                    roc_data_collection[setting_arg] = (y_test, mean_eval_probs)

            # 12. Plot Comparisons using Averaged Metrics
            if model_results_list:
                df_results = pd.DataFrame(model_results_list)
                plot_dir = out_root / "comparison"
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                metrics_list = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"]
                plot_metrics_bar_chart(df_results, metrics_list, plot_dir / "eval_bar.png")
                plot_metrics_line_chart(df_results, metrics_list, plot_dir / "eval_line.png")
                if roc_data_collection:
                    plot_roc_comparison(roc_data_collection, plot_dir / "eval_roc_curves.png")

    print("\n[SUCCESS] Evaluation complete. Check the output directory.")