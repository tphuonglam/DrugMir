#!/bin/bash


# ########## TCGA ##########

# ### Keep only listed MiRNAs
# python src/preprocess_data/filter_mirna.py  --drug="cisplatin" --raw_dir="all_data/raw/tcga_20260527" --output_dir="all_data/processed/tcga_55_20260528" --mapping_dir="all_data/processed/gse_20260528/"

# ### Convert response labels to binary
# python src/preprocess_data/process_label.py --drug="cisplatin" --raw_dir="all_data/raw/tcga_20260527" --output_dir="all_data/processed/tcga_55_20260528"

# ### Rename gene columns from mRNA to hgnc_symbol    
# python src/preprocess_data/rename_gene.py   --drug="cisplatin" --raw_dir="all_data/raw/tcga_20260527" --output_dir="all_data/processed/tcga_55_20260528"


######### CCLE ##########

# ### Keep only listed MiRNAs
# python src/preprocess_data/filter_mirna.py  --drug="cisplatin" --raw_dir="all_data/raw/ccle_20260527" --output_dir="all_data/processed/ccle_40_20260528" --mapping_dir="all_data/processed/ccle_40_20260528/"

# ### Convert response labels to binary
# python src/preprocess_data/process_label.py --drug="cisplatin" --raw_dir="all_data/raw/ccle_20260527" --output_dir="all_data/processed/ccle_40_20260528"

# ### Rename gene columns from mRNA to hgnc_symbol    
# python src/preprocess_data/rename_gene.py   --drug="cisplatin" --raw_dir="all_data/raw/ccle_20260527" --output_dir="all_data/processed/ccle_40_20260528"


######### GSE ##########

### Keep only listed MiRNAs
python src/preprocess_data/filter_mirna.py  --drug="cisplatin" --raw_dir="all_data/raw/gse_20260528" --output_dir="all_data/processed/gse_55_20260528" --mapping_dir="all_data/processed/gse_55_20260528/"

### Convert response labels to binary
python src/preprocess_data/process_label.py --drug="cisplatin" --raw_dir="all_data/raw/gse_20260528" --output_dir="all_data/processed/gse_55_20260528"

### Rename gene columns from mRNA to hgnc_symbol    
python src/preprocess_data/rename_gene.py   --drug="cisplatin" --raw_dir="all_data/raw/gse_20260528" --output_dir="all_data/processed/gse_55_20260528"