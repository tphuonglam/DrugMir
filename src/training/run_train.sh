#!/bin/bash

DATASET=data_0616/GEO
# SEED=0

ROOT_TRAIN=/mnt/HDD3/khanh/tplam/something/all_data/processed/$DATASET
# OUT_DIR=./results/${DATASET}/seed_${SEED}     # Single-seed version
OUT_DIR=./results/${DATASET}                    # Multi-seed version

### MODEL ARGS
SCALER=standard             # quantile, standard, none
THRESHOLD_METRIC=gmean      # gmean, youden

# The name of the split file inside each drug folder
SPLIT_FILE="patient_splits.csv"

# SETTINGS=("DEmiRs" "Target Genes" "Integration")
SETTINGS=("DEmiRs")

# MODELS=("rf" "logreg" "gbm" "ada" "svm")
MODELS=("logreg")

DRUGS=("cisplatin" "carboplatin" "fluorouracil" "gemcitabine" "paclitaxel")

echo "=========================================================================="
echo "[RUNNING] Starting Multi-Drug Training Pipeline..."
echo "=========================================================================="

# 1. Train all drugs and settings in one go
python train.py \
    --root "$ROOT_TRAIN" \
    --split_file "$SPLIT_FILE" \
    --models "${MODELS[@]}" \
    --settings "${SETTINGS[@]}" \
    --outdir "$OUT_DIR" \
    --drugs "${DRUGS[@]}" \
    --threshold_metric $THRESHOLD_METRIC \
    --scaler $SCALER

# 2. Evaluate all drugs and settings in one go
python evaluate.py \
    --data_root "$ROOT_TRAIN" \
    --model_dir "$OUT_DIR" \
    --models "${MODELS[@]}" \
    --settings "${SETTINGS[@]}" \
    --split_file "$SPLIT_FILE" \
    --drugs "${DRUGS[@]}" \
    --outdir "$OUT_DIR" \
    --threshold_metric $THRESHOLD_METRIC \
    --scaler $SCALER