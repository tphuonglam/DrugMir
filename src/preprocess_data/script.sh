#!/bin/bash

RAW_DIR="all_data/processed/tcga_20260607/data"
OUT_DIR="all_data/processed/tcga_ccle_20260607"
MAP_DIR="all_data/processed/ccle_20260607/"

DRUGS=("cisplatin" "carboplatin" "fluorouracil" "gemcitabine" "paclitaxel")

echo "=========================================================================="
echo "[RUNNING] Starting Data Preprocessing Pipeline..."
echo "=========================================================================="

# 2. Loop through each drug and run the preprocessing script
for drug in "${DRUGS[@]}"; do
    echo " -> Preprocessing data for: ${drug}"
    
    python src/preprocess_data/filter_mirna.py \
        --drug="$drug" \
        --raw_dir="$RAW_DIR" \
        --output_dir="$OUT_DIR" \
        --mapping_dir="$MAP_DIR"
    
    python src/preprocess_data/filter_gene.py \
        --drug="$drug" \
        --raw_dir="$RAW_DIR" \
        --output_dir="$OUT_DIR" \
        --mapping_dir="$MAP_DIR"
done

 

# ########## TCGA ##########

# ### Keep only listed MiRNAs
# python src/preprocess_data/filter_mirna.py  --drug="cisplatin" --raw_dir="all_data/raw/tcga_20260527" --output_dir="all_data/processed/tcga_58_20260529" --mapping_dir="all_data/processed/gse_58_20260529/"

# ### Convert response labels to binary
# python src/preprocess_data/process_label.py --drug="cisplatin" --raw_dir="all_data/raw/tcga_20260527" --output_dir="all_data/processed/tcga_58_20260529"

# ### Rename gene columns from mRNA to hgnc_symbol    
# python src/preprocess_data/rename_gene.py   --drug="cisplatin" --raw_dir="all_data/raw/tcga_20260527" --output_dir="all_data/processed/tcga_58_20260529"


######### CCLE ##########

# ### Keep only listed MiRNAs
# python src/preprocess_data/filter_mirna.py  --drug="cisplatin" --raw_dir="all_data/raw/ccle_20260527" --output_dir="all_data/processed/ccle_40_20260528" --mapping_dir="all_data/processed/ccle_40_20260528/"

# ### Convert response labels to binary
# python src/preprocess_data/process_label.py --drug="cisplatin" --raw_dir="all_data/raw/ccle_20260527" --output_dir="all_data/processed/ccle_40_20260528"

# ### Rename gene columns from mRNA to hgnc_symbol    
# python src/preprocess_data/rename_gene.py   --drug="cisplatin" --raw_dir="all_data/raw/ccle_20260527" --output_dir="all_data/processed/ccle_40_20260528"


######### GSE ##########

# ### Keep only listed MiRNAs
# python src/preprocess_data/filter_mirna.py  --drug="cisplatin" --raw_dir="all_data/raw/gse_20260528" --output_dir="all_data/processed/gse_55_20260528" --mapping_dir="all_data/processed/gse_55_20260528/"

# ### Convert response labels to binary
# python src/preprocess_data/process_label.py --drug="cisplatin" --raw_dir="all_data/raw/gse_20260528" --output_dir="all_data/processed/gse_55_20260528"

# ### Rename gene columns from mRNA to hgnc_symbol    
# python src/preprocess_data/rename_gene.py   --drug="cisplatin" --raw_dir="all_data/raw/gse_20260528" --output_dir="all_data/processed/gse_55_20260528"