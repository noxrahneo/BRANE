#!/usr/bin/env python3

"""01_create_bigboss.py: building the BigBoss inventory from metadata (SampleStats.txt + Table EV4) and raw file names."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the BigBoss inventory from metadata (SampleStats.txt + Table EV4) and raw file names."
    )
    parser.add_argument(
        "--data-dir", default="devil_fruit/GSE161529_RAW", help="Path to raw data folder with matrix/barcodes files."
    )
    parser.add_argument(
        "--sample-stats", default="data/HumanBreast10X-main/Tables/SampleStats.txt", help="Path to SampleStats.txt from the paper repo."
    )
    parser.add_argument(
        "--table-ev4", default="data/embj2020107333-sup-0006-tableev4.csv", help="Path to the Table EV4."
    )
    parser.add_argument(
        "--output", default="results/00_data_ingestion/bigboss.csv", help="Output CSV path"
    )
    return parser.parse_args()

def load_and_validate_metadata(sample_stats_path: Path, table_ev4_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    #load both sources
    sample_stats = pd.read_csv(sample_stats_path, sep="\t")
    table_ev4 = pd.read_csv(table_ev4_path, skiprows=2)  #skip first two rows
    table_ev4.columns = table_ev4.columns.str.strip()
    table_ev4 = table_ev4.dropna(how="all")
    table_ev4 = table_ev4[table_ev4["Sample Name"].notna()].copy()
    table_ev4["Sample Name"] = table_ev4["Sample Name"].astype(str).str.strip()
    sample_stats["SampleName"] = sample_stats["SampleName"].astype(str).str.strip()
    table_ev4 = table_ev4[~table_ev4["Sample Name"].str.contains(r"\s", regex=True)].copy()
    
    #validate sample names match in order
    print("Validating sample names")
    stats_names = sample_stats["SampleName"].tolist()
    ev4_names = table_ev4["Sample Name"].tolist()
    
    if len(stats_names) != len(ev4_names):
        raise ValueError(f"Sample name count mismatch: {len(stats_names)} in SampleStats vs {len(ev4_names)} in Table EV4")
    
    #check for each match
    mismatches = []
    for i, (s, e) in enumerate(zip(stats_names, ev4_names)):
        if s != e:
            mismatches.append((i, s, e))
    
    if mismatches:
        preview = "\n".join(
            [f"  Row {i}: SampleStats='{s}' vs TableEV4='{e}'" for i, s, e in mismatches[:5]]
        )
        more = ""
        if len(mismatches) > 5:
            more = f"\n  ... and {len(mismatches) - 5} more"
        raise ValueError(f"Found {len(mismatches)} sample name mismatches:\n{preview}{more}")

    print("All sample names match between SampleStats and Table EV4.")
    
    return sample_stats, table_ev4
    
def merge_metadata(sample_stats: pd.DataFrame, table_ev4: pd.DataFrame) -> pd.DataFrame:
    #merge on SampleName (avoid duplicating cell/gene counts)
    #drop duplicate columns from table_ev4
    
    ev4_clean = table_ev4.drop(
        columns=["Number of Cells", "Number of Cells After Filtering", "Number of Genes Detected"],
        errors="ignore",
    )
    merged = pd.merge(sample_stats, ev4_clean, left_on="SampleName", right_on="Sample Name", how="left")
    
    return merged
     
def add_file_paths(bigboss: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    #find matrix/barcode files by GEO_ID
    rows_with_files = []
    for _, row in bigboss.iterrows():
        geo_id = row["GEO_ID"]
        
        mtx_files = sorted(data_dir.glob(f"{geo_id}_*-matrix.mtx.gz"))
        bc_files = sorted(data_dir.glob(f"{geo_id}_*-barcodes.tsv.gz"))

        if not mtx_files or not bc_files:
            continue
        
        #add a row for each matching file pair
        for mtx, bc in zip(mtx_files, bc_files):
            new_row = row.copy()
            new_row["MatrixFile"] = mtx.name
            new_row["BarcodesFile"] = bc.name
            rows_with_files.append(new_row)
    
    return pd.DataFrame(rows_with_files)

def format_output_columns(bigboss: pd.DataFrame) -> pd.DataFrame:
    #remove duplicate and unneeded columns
    bigboss = bigboss.drop(columns=["Title", "Sample Name", "Patient"], errors="ignore")

    #final readable order
    final_order = [
        "MatrixFile",
        "BarcodesFile",
        "GEO_ID",
        "SampleName",
        "Description",
        "Source",
        "Condition",
        "Gender",
        "Parity",
        "Menopause",
        "CellNum",
        "CellNumAfter",
        "Mito",
        "GeneLower",
        "GeneUpper",
        "LibSize",
        "GenesDetected",
    ]

    #ensure missing columns still exist (as empty) so output schema is stable
    for col in final_order:
        if col not in bigboss.columns:
            bigboss[col] = pd.NA

    #normalize missing values in key phenotype columns
    for col in ["Parity", "Menopause"]:
        bigboss[col] = bigboss[col].replace(r"^\s*$", pd.NA, regex=True).fillna("NA")

    return bigboss[final_order]
    
def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = (repo_root / args.data_dir).resolve()
    sample_stats_path = (repo_root / args.sample_stats).resolve()
    table_ev4_path = (repo_root / args.table_ev4).resolve()
    out_path = (repo_root / args.output).resolve()
    
    #validate file exists
    for path, name in [
        (sample_stats_path, "Sample Stats.txt"),
        (table_ev4_path, "Table EV4"),
        (data_dir, "Data Directory"),
    ]:
        if not path.exists():
            print(f"Error: {name} not found at {path}")
            return 1
    
    print("Loading metadata from both sources")
    sample_stats, table_ev4 = load_and_validate_metadata(sample_stats_path, table_ev4_path)

    print("Merging metadata")
    bigboss = merge_metadata(sample_stats, table_ev4)
    
    print("Adding file paths from raw data")
    bigboss = add_file_paths(bigboss, data_dir)

    if bigboss.empty:
        print("Error: no matching matrix/barcodes files found for metadata rows")
        return 1

    print("Formatting output columns")
    bigboss = format_output_columns(bigboss)
    
    #save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bigboss.to_csv(out_path, index=False)
    
    print(f"Saved: {len(bigboss)} entries to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())