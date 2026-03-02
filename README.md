# DrugMir

DrugMiR is an explainable multi-omics framework designed to predict drug response and uncover the underlying regulatory mechanisms across five major chemotherapies. This framework utilizes interpretable machine learning (ML) models combined with a targeted feature selection strategy. By prioritizing DEmiRs as primary features, we demonstrate that these regulatory molecules contain sufficient biological information to achieve high predictive accuracy in clinical cohorts. Beyond statistical prediction, DrugMiR provides a functional layer of analysis to bridge the gap between computational outcomes and clinical reality. 

Overview of the study workflow. (A) data collection and preprocessing: harmonization of clinical drug response labels with multi-omics expression profiles from the TCGA; (B) feature selection and integration: DEA is utilized to identify drug-specific gene and miRNA signatures; (C) drug response prediction: a comparative ML framework to evaluate the predictive power of single and multi-omics feature sets; (D) functional interpretation and biological significance: a multi-faceted approach, including SHAP interpretability, survival analysis, and regulatory network construction, to evaluate the clinical and biological significance of the identified markers

---


## *Installation*
1. **Clone the repository:**
   ```bash
   git clone https://github.com/tphuonglam/DrugMir.git
   cd DrugMir
2. **Install dependencies:**

    We used `python 3.8` in our experiments. To install other dependencies, run:

    ```bash
    pip install -r requirements.txt
    ```

## *Training & Evaluation*
The core logic is contained in `train.py`. It supports various machine learning models (e.g., Random Forest, GBM, Logistic Regression) across multiple genomic data settings.

### Quick Start
Use the provided shell script to run the pipeline:
```bash
bash run_train.sh
```

### Output Structure

Results are organized hierarchically: `[Output Directory] / [Drug] / [Model] / [Setting]`.

1. Performance Metrics

    For every run, the following files are generated to evaluate model performance:

    1. `metrics.json`: Average performance across all cross-validation folds.

    2. `metrics_std.json`: Standard deviation of metrics across folds.

    3. `metrics_folds.json`: Raw results for each individual fold.

    4. `predictions.csv`: Model predictions for each patient.

    5. `probabilities.csv`: Raw probability scores for each patient.

2. Interpretability (SHAP)

    We generates SHAP (SHapley Additive exPlanations) visualizations:

    1. `shap_beeswarm.png`: Distribution of feature impacts on model output.

    2. `shap_bar.png`: Global feature importance ranking.

3. Comparison & Visualizations

    Found in the `comparison/` subdirectory:

    1. `line.png` & `bar.png`: Performance trends across different data settings.
    2. `roc_curves.png`: Combined ROC curves for setting comparison.
