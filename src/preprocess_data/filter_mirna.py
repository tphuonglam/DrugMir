import argparse
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("--raw_dir", type=str, default="data", help="Root directory for raw data")
parser.add_argument("--mapping_dir", type=str, default="data", help="Root directory for processed data")
parser.add_argument("--output_dir", type=str, default="data", help="Root directory for processed data")
parser.add_argument("--drugs", nargs="+", 
                    default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"],
                    help="List of drugs to process")
args = parser.parse_args()


for drug in args.drugs:
    all_mirna_path = Path(args.raw_dir) / drug / f"{drug}_allmiRNA.csv"
    used_mirna_path = Path(args.mapping_dir) / "mappings" / drug / "DEmiRNA.csv"
    out_path = Path(args.output_dir) / "data" / drug / f"{drug}_allmiRNA.csv"

    if not Path(all_mirna_path).exists():
        print(f"[WARNING] {all_mirna_path} does not exist. Skipping {drug}.")
        continue

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load the datasets
    df_de = pd.read_csv(used_mirna_path)
    df_all = pd.read_csv(all_mirna_path)

    # Rename the first column to an empty string
    df_all = df_all.rename(columns={df_all.columns[0]: ''})

    # Extract DE miRNAs and convert to lowercase
    de_mirnas = set(df_de['miRNA'].str.lower().dropna())
    all_cols = df_all.columns.tolist()

    matched_cols = []
    # Keep the (now renamed) first column
    matched_cols.append(all_cols[0])

    # Safely filter columns
    for col in all_cols:
        if col == "": # Skip the empty column name if already added
            continue
            
        col_lower = col.lower()
        
        for target in de_mirnas:
            # Check for exact match OR a version with a suffix (e.g., -3p, -5p)
            if col_lower == target or col_lower.startswith(target + '-'):
                matched_cols.append(col)
                break

    # Create the filtered DataFrame and save it
    df_filtered = df_all[matched_cols]
    print("Before dropping NaNs:", df_filtered.shape)
    df_filtered = df_filtered.dropna()
    print("After dropping NaNs:", df_filtered.shape)
    df_filtered.to_csv(out_path, index=False)