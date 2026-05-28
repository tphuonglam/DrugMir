import argparse
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("--raw_dir", type=str, default="data", help="Root directory for raw data")
parser.add_argument("--output_dir", type=str, default="data", help="Root directory for processed data")
parser.add_argument("--drugs", nargs="+", 
                    default=["carboplatin", "cisplatin", "fluorouracil", "gemcitabine", "paclitaxel"],
                    help="List of drugs to process")
args = parser.parse_args()


for drug in args.drugs:
    mirna_path = f"{args.raw_dir}/{drug}/{drug}_gene.csv"
    mapping_path = f"{args.output_dir}/mappings/{drug}/DEmRNA.csv"
    out_path = f"{args.output_dir}/data/{drug}/{drug}_gene.csv"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Load the datasets
    df_og = pd.read_csv(mirna_path)

    # This file is a mapping from "mRNA" to "hgnc_symbol"
    df_mapping = pd.read_csv(mapping_path)

    # Rename the first column to an empty string
    df_og = df_og.rename(columns={df_og.columns[0]: ''})
    # Current columns have format "mRNA" and we want to rename them to "hgnc_symbol"
    
    mapping_dict = dict(zip(df_mapping['mRNA'], df_mapping['hgnc_symbol']))
    df_filtered = df_og.rename(columns=mapping_dict)
    df_filtered.to_csv(out_path, index=False)


    assert df_og.shape == df_filtered.shape