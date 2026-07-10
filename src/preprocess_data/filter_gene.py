import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--raw_dir", type=str, default="data", help="Root directory for raw data")
parser.add_argument("--mapping_dir", type=str, default="data", help="Root directory for mapping data")
parser.add_argument("--output_dir", type=str, default="data", help="Root directory for processed data")
parser.add_argument("--drugs", nargs="+", 
                    default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"],
                    help="List of drugs to process")
args = parser.parse_args()

for drug in args.drugs:
    # Update paths for gene data
    all_gene_path = Path(args.raw_dir) / drug / f"{drug}_gene.csv"
    used_gene_path = Path(args.mapping_dir) / "mappings" / drug / "DEgene.csv"
    out_path = Path(args.output_dir) / "data" / drug / f"{drug}_gene.csv"

    if not Path(all_gene_path).exists():
        print(f"[WARNING] {all_gene_path} does not exist. Skipping {drug}.")
        continue

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load the datasets
    df_de = pd.read_csv(used_gene_path)
    df_all = pd.read_csv(all_gene_path)

    # Rename the first column (Cell_Line_ID) to an empty string
    df_all = df_all.rename(columns={df_all.columns[0]: ''})

    # Extract DE genes and convert to lowercase for case-insensitive matching
    # Ensure they are strings to avoid errors with purely numeric gene IDs
    de_genes = set(df_de['Gene'].astype(str).str.lower().dropna())
    all_cols = df_all.columns.tolist()

    matched_cols = []
    # Keep the (now renamed) first column
    matched_cols.append(all_cols[0])

    # Filter columns
    for col in all_cols:
        if col == "":  # Skip the empty column name if already added
            continue
            
        col_lower = str(col).lower()
        
        # For genes, exact case-insensitive match is usually sufficient
        if col_lower in de_genes:
            matched_cols.append(col)

    # Create the filtered DataFrame and save it
    df_filtered = df_all[matched_cols]
    
    print(f"--- Processing {drug} ---")
    print("Before dropping NaNs:", df_filtered.shape)
    df_filtered = df_filtered.dropna()
    print("After dropping NaNs:", df_filtered.shape)
    df_filtered.to_csv(out_path, index=False)

    print("Original raw data shape:", df_all.shape)
    print("Final filtered shape:", df_filtered.shape)
    print("\n")