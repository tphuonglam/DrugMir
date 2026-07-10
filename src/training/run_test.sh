#!/bin/bash

TEST_DATASET=ccle_20260607
TRAIN_DATASET=tcga_ccle_20260607
MODEL_DIR=results/${TRAIN_DATASET}_seed_1

### MODEL ARGS
SCALER=quantile             # quantile, standard, none
THRESHOLD_METRIC=gmean      # gmean, youden


ROOT_TEST=/mnt/HDD3/khanh/tplam/something/all_data/processed/$TEST_DATASET/data
OUT_DIR=$MODEL_DIR/test_on_$TEST_DATASET


python evaluate.py --data_root $ROOT_TEST \
    --model_dir $MODEL_DIR \
    --models logreg \
    --outdir "$OUT_DIR/" \
    --threshold_metric $THRESHOLD_METRIC \
    --scaler $SCALER \
    # --split_file "$SPLIT_FILE" \