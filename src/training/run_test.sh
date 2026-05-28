#!/bin/bash

DATASET=gse_55_20260528
MODEL_DIR=results_tcga_55_20260528_aggregate
# SPLIT_FILE=/mnt/HDD3/khanh/tplam/something/all_data/processed/tcga_20260527/data/cisplatin/patient_splits.csv


ROOT_TEST=/mnt/HDD3/khanh/tplam/something/all_data/processed/$DATASET/data
OUT_DIR=$MODEL_DIR/test_on_$DATASET


python evaluate.py --data_root $ROOT_TEST \
    --model_dir $MODEL_DIR \
    --models logreg \
    --outdir "$OUT_DIR/" \
    # --split_file "$SPLIT_FILE" \