#!/bin/bash

python train.py --seed 42 \
    --models rf gbm \
    --settings "DEmiRs" "DEGs + DEmiRs" \
    --outdir ./experimental_results