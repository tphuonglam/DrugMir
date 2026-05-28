#!/bin/bash

DATASET=tcga_20260527
ROOT_TRAIN=/mnt/HDD3/khanh/tplam/something/all_data/processed/$DATASET/data
OUT_DIR=./results_${DATASET}_aggregate
SPLIT_FILE=/mnt/HDD3/khanh/tplam/something/all_data/processed/$DATASET/data/cisplatin/patient_splits.csv

### MODEL ARGS
SCALER=standard             # quantile, standard, none
THRESHOLD_METRIC=gmean      # gmean, youden

python train_aggregate.py --seed 4 --root $ROOT_TRAIN --split_file $SPLIT_FILE \
    --models logreg \
    --settings "DEmiRs" \
    --outdir $OUT_DIR \
    --drug "cisplatin" \
    --threshold_metric $THRESHOLD_METRIC \
    --scaler $SCALER \
    # --mirna "hsa-mir-99a" "hsa-mir-508" "hsa-mir-218-2" "hsa-mir-181c" \

python evaluate.py --data_root $ROOT_TRAIN \
    --model_dir $OUT_DIR \
    --models logreg \
    --threshold_metric $THRESHOLD_METRIC \
    --scaler $SCALER \
    --split_file "$SPLIT_FILE" \
    --outdir "$OUT_DIR/"

# python train_old.py --seed 4 \
#     --root data --root_filter data_drug_response \
#     --models logreg \
#     --settings "DEmiRs" "DEGs" "DEGs + DEmiRs" \
#     --outdir ./exp_4demir_0325_linearexplainer \
#     # --drug "paclitaxel" \
#     # --mirna "hsa-mir-99a" "hsa-mir-508" "hsa-mir-218-2" "hsa-mir-181c" \