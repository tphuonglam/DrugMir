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
    label_path = f"{args.raw_dir}/{drug}/{drug}_label.csv"
    out_path = f"{args.output_dir}/data/{drug}/{drug}_label.csv"

    if not Path(label_path).exists():
        print(f"[WARNING] {label_path} does not exist. Skipping {drug}.")
        continue

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dataset_name = args.output_dir.split('/')[-1]
    
    df_labels = pd.read_csv(label_path)

    if "tcga" in dataset_name:
        df_new = df_labels.copy()
        df_new['Response'] = df_new['Response'].map({"Responder": 1, "NonResponder": 0})
    elif "ccle" in dataset_name:
        df_new = df_labels[["Cell_Line_ID", "CISPLATIN"]]
        # First, rename the column "CISPLATIN" to "Response"
        df_new = df_new.rename(columns={"CISPLATIN": "Response"})
        # Drop all rows where "Response" is not in ["Resistant", "Sensitive"]
        df_new = df_new[df_new["Response"].isin(["Resistant", "Sensitive"])]
        # Then, map the values in the "Response" column to 1 and 0
        df_new['Response'] = df_new['Response'].map({"Sensitive": 1, "Resistant": 0})
    elif "gse" in dataset_name:
        df_new = df_labels.copy()[["Sample_ID", "Label"]]
        df_new = df_new.rename(columns={"Label": "Response"})
        
    df_new.to_csv(out_path, index=False)