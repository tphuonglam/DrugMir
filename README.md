# DrugMir

[Insert Project Purpose/Description Here]

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