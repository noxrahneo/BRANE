#!/usr/bin/env python3
"""Compute condition-level log-fold-change and directional category (UP/DOWN/STABLE) for network genes."""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]
TAGGING_DIR = REPO_ROOT / "results" / "20_node_annotation"
INPUT_TAGGED_DIR = TAGGING_DIR / "02_output"
OUTPUT_WITH_LFC_DIR = TAGGING_DIR / "03_output_with_lfc"

#define all condition pairs and their corresponding h5ad file mappings
#using stage 03 integrated h5ad files (cell-level data with annotations)
STAGE03_INTEGRATION_DIR = REPO_ROOT / "results" / "03_integration" / "integrated"

CONDITION_PAIRS = [
    #(pair_name, test_condition, ref_condition, test_h5ad, ref_h5ad)
    (
        "ER_tumor__vs__Normal",
        "ER_tumor",
        "Normal",
        STAGE03_INTEGRATION_DIR / "ER_tumor" / "ER_tumor_integrated.h5ad",
        STAGE03_INTEGRATION_DIR / "Normal" / "Normal_integrated.h5ad",
    ),
    (
        "HER2_tumor__vs__Normal",
        "HER2_tumor",
        "Normal",
        STAGE03_INTEGRATION_DIR / "HER2_tumor" / "HER2_tumor_integrated.h5ad",
        STAGE03_INTEGRATION_DIR / "Normal" / "Normal_integrated.h5ad",
    ),
    (
        "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
        "Normal_BRCA1_-_pre-neoplastic",
        "Normal",
        STAGE03_INTEGRATION_DIR / "Normal_BRCA1_-_pre-neoplastic" / "Normal_BRCA1_-_pre-neoplastic_integrated.h5ad",
        STAGE03_INTEGRATION_DIR / "Normal" / "Normal_integrated.h5ad",
    ),
    (
        "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
        "Triple_negative_BRCA1_tumor",
        "Normal_BRCA1_-_pre-neoplastic",
        STAGE03_INTEGRATION_DIR / "Triple_negative_BRCA1_tumor" / "Triple_negative_BRCA1_tumor_integrated.h5ad",
        STAGE03_INTEGRATION_DIR / "Normal_BRCA1_-_pre-neoplastic" / "Normal_BRCA1_-_pre-neoplastic_integrated.h5ad",
    ),
    (
        "Triple_negative_BRCA1_tumor__vs__Normal",
        "Triple_negative_BRCA1_tumor",
        "Normal",
        STAGE03_INTEGRATION_DIR / "Triple_negative_BRCA1_tumor" / "Triple_negative_BRCA1_tumor_integrated.h5ad",
        STAGE03_INTEGRATION_DIR / "Normal" / "Normal_integrated.h5ad",
    ),
    (
        "Triple_negative_tumor__vs__Normal",
        "Triple_negative_tumor",
        "Normal",
        STAGE03_INTEGRATION_DIR / "Triple_negative_tumor" / "Triple_negative_tumor_integrated.h5ad",
        STAGE03_INTEGRATION_DIR / "Normal" / "Normal_integrated.h5ad",
    ),
]

#known genes for QC sanity checks
#format: (gene_name, expected_direction_in_tumor, biological_role)
QC_GENES = [
    ("ESR1", "UP", "ER positive tumour marker; upregulated in ER_tumor"),
    ("ERBB2", "UP", "HER2 tumour marker; upregulated in HER2_tumor"),
    ("TP53", "DOWN", "Tumour suppressor; often lost or downregulated in cancer"),
    ("MKI67", "UP", "Proliferation marker; highly expressed in tumours"),
    ("ACTB", "STABLE", "Housekeeping control; should be stable across conditions"),
]

PSEUDOCOUNT = 1.0  #standard for log-normalized count data
LFC_UP_THRESHOLD = 0.5  # ≥ ~1.4-fold increase
LFC_DOWN_THRESHOLD = -0.5  # ≤ ~0.71-fold decrease


def _read_h5ad_safe(path: Path) -> sc.AnnData:
    #read h5ad, working around uns/log1p version incompatibility in older files
    try:
        return sc.read_h5ad(path)
    except Exception:
        tmp_path = Path(tempfile.mktemp(suffix=".h5ad"))
        try:
            shutil.copy2(path, tmp_path)
            with h5py.File(tmp_path, "a") as f:
                if "uns/log1p" in f:
                    del f["uns/log1p"]
            return sc.read_h5ad(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


@dataclass
class PreflightResult:
    pair_name: str
    tagged_csv_exists: bool
    test_h5ad_exists: bool
    ref_h5ad_exists: bool
    tagged_genes: int | None
    test_cells: int | None
    ref_cells: int | None
    all_exists: bool


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add condition-level LFC to tagged gene tables"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight diagnostics only, then exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def run_preflight() -> list[PreflightResult]:
    #validate all input files and report structure
    logging.info("PREFLIGHT: Validating input files and data structure")

    results = []
    for pair_name, _, _, test_h5ad, ref_h5ad in CONDITION_PAIRS:
        tagged_csv = INPUT_TAGGED_DIR / f"{pair_name}_tagged.csv"

        tagged_exists = tagged_csv.exists()
        test_h5ad_exists = test_h5ad.exists()
        ref_h5ad_exists = ref_h5ad.exists()

        tagged_genes = None
        test_cells = None
        ref_cells = None

        if tagged_exists:
            df = pd.read_csv(tagged_csv, nrows=1)
            tagged_genes = len(pd.read_csv(tagged_csv))

        if test_h5ad_exists:
            adata_test = _read_h5ad_safe(test_h5ad)
            test_cells = adata_test.n_obs

        if ref_h5ad_exists:
            adata_ref = _read_h5ad_safe(ref_h5ad)
            ref_cells = adata_ref.n_obs

        all_exists = tagged_exists and test_h5ad_exists and ref_h5ad_exists
        result = PreflightResult(
            pair_name=pair_name,
            tagged_csv_exists=tagged_exists,
            test_h5ad_exists=test_h5ad_exists,
            ref_h5ad_exists=ref_h5ad_exists,
            tagged_genes=tagged_genes,
            test_cells=test_cells,
            ref_cells=ref_cells,
            all_exists=all_exists,
        )
        results.append(result)

        status = "ok" if all_exists else "fail"
        logging.info(
            f"{status} {pair_name}: "
            f"CSV={tagged_genes} genes, "
            f"test={test_cells} cells, "
            f"ref={ref_cells} cells"
        )

    return results


def compute_lfc_for_pair(
    pair_name: str,
    test_condition: str,
    ref_condition: str,
    test_h5ad: Path,
    ref_h5ad: Path,
    output_dir: Path,
) -> dict[str, Any]:
    #compute LFC and direction for a single condition pair
    logging.info(f"\n--- Processing pair: {pair_name} ---")

    #load input tagged CSV
    tagged_csv = INPUT_TAGGED_DIR / f"{pair_name}_tagged.csv"
    df_tagged = pd.read_csv(tagged_csv)
    logging.info(f"Loaded {len(df_tagged)} genes from {tagged_csv.name}")

    #load h5ad files
    logging.info(f"Loading test condition h5ad: {test_h5ad.name}")
    adata_test = _read_h5ad_safe(test_h5ad)
    logging.info(
        f"  {adata_test.n_obs} cells × {adata_test.n_vars} genes, "
        f"layer: {adata_test.layers.keys() if adata_test.layers else 'none'}"
    )

    logging.info(f"Loading ref condition h5ad: {ref_h5ad.name}")
    adata_ref = _read_h5ad_safe(ref_h5ad)
    logging.info(
        f"  {adata_ref.n_obs} cells × {adata_ref.n_vars} genes, "
        f"layer: {adata_ref.layers.keys() if adata_ref.layers else 'none'}"
    )

    #extract gene names from tagged table
    network_genes = df_tagged["gene"].values
    logging.info(f"Extracting expression for {len(network_genes)} network genes")

    #find intersection of network genes with h5ad gene names
    test_genes_idx = [i for i, g in enumerate(adata_test.var_names) if g in network_genes]
    ref_genes_idx = [i for i, g in enumerate(adata_ref.var_names) if g in network_genes]

    genes_in_test = set(adata_test.var_names[test_genes_idx])
    genes_in_ref = set(adata_ref.var_names[ref_genes_idx])
    genes_common = genes_in_test & genes_in_ref

    logging.info(
        f"Gene overlap: test={len(genes_in_test)}, ref={len(genes_in_ref)}, "
        f"common={len(genes_common)}"
    )

    #extract expression matrices from raw counts if available, otherwise use X
    #this ensures we compute LFC on raw counts, not pre-normalized log data
    if 'counts' in adata_test.layers:
        X_test = adata_test.layers['counts']
        logging.info("Using 'counts' layer from test condition h5ad")
    else:
        X_test = adata_test.X
        logging.info("Using default X layer from test condition h5ad (note: may be pre-normalized)")

    if 'counts' in adata_ref.layers:
        X_ref = adata_ref.layers['counts']
        logging.info("Using 'counts' layer from ref condition h5ad")
    else:
        X_ref = adata_ref.X
        logging.info("Using default X layer from ref condition h5ad (note: may be pre-normalized)")

    #convert sparse to dense if needed (safer for mean computation)
    if sparse.issparse(X_test):
        X_test = X_test.toarray()
    if sparse.issparse(X_ref):
        X_ref = X_ref.toarray()

    #compute condition-level means for all network genes
    lfc_values = []
    direction_values = []
    genes_missing = []

    for gene in network_genes:
        gene_found = False

        #find gene in test condition
        try:
            test_idx = list(adata_test.var_names).index(gene)
            mean_test = X_test[:, test_idx].mean()
            gene_found = True
        except (ValueError, IndexError):
            mean_test = np.nan
            genes_missing.append((gene, "not_in_test"))

        #find gene in ref condition
        try:
            ref_idx = list(adata_ref.var_names).index(gene)
            mean_ref = X_ref[:, ref_idx].mean()
            gene_found = True
        except (ValueError, IndexError):
            mean_ref = np.nan
            if gene not in genes_missing:
                genes_missing.append((gene, "not_in_ref"))

        #compute LFC with pseudocount
        if np.isnan(mean_test) or np.isnan(mean_ref):
            lfc = np.nan
            direction = "Unknown"
        else:
            lfc = np.log2((mean_test + PSEUDOCOUNT) / (mean_ref + PSEUDOCOUNT))
            if lfc >= LFC_UP_THRESHOLD:
                direction = "UP"
            elif lfc <= LFC_DOWN_THRESHOLD:
                direction = "DOWN"
            else:
                direction = "STABLE"

        lfc_values.append(lfc)
        direction_values.append(direction)

    #add new columns to tagged dataframe
    df_tagged["mean_expr_test"] = np.nan  #placeholder; can expand if needed
    df_tagged["mean_expr_ref"] = np.nan  #placeholder; can expand if needed
    df_tagged["lfc"] = lfc_values
    df_tagged["direction"] = direction_values

    #write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{pair_name}_tagged_with_lfc.csv"
    df_tagged.to_csv(output_csv, index=False)
    logging.info(f"wrote {len(df_tagged)} genes to {output_csv.name}")

    #summary stats
    direction_counts = pd.Series(direction_values).value_counts().to_dict()
    logging.info(f"Direction breakdown: {direction_counts}")

    return {
        "success": True,
        "pair_name": pair_name,
        "genes_processed": len(network_genes),
        "genes_missing": len(genes_missing),
        "direction_counts": direction_counts,
    }


def validate_known_genes(output_dir: Path) -> None:
    #qC check: validate LFC of known genes against expected biology
    logging.info("QC VALIDATION: Checking known genes against expected biology")

    validation_rows = []

    for pair_name, test_cond, ref_cond, _, _ in CONDITION_PAIRS:
        input_csv = output_dir / f"{pair_name}_tagged_with_lfc.csv"

        if not input_csv.exists():
            logging.warning(f"Skipping {pair_name}: output CSV not found")
            continue

        df = pd.read_csv(input_csv)
        logging.info(f"\n--- Pair: {pair_name} (test={test_cond} vs ref={ref_cond}) ---")

        for gene_name, expected_dir, description in QC_GENES:
            #find gene in this pair's output
            gene_rows = df[df["gene"] == gene_name]

            if len(gene_rows) == 0:
                logging.info(f"  {gene_name:8s}: NOT IN NETWORK for this pair")
                validation_rows.append({
                    "pair": pair_name,
                    "gene": gene_name,
                    "description": description,
                    "expected_direction": expected_dir,
                    "observed_direction": "N/A",
                    "observed_lfc": np.nan,
                    "status": "not_in_network",
                })
                continue

            lfc = gene_rows.iloc[0]["lfc"]
            direction = gene_rows.iloc[0]["direction"]
            match = direction == expected_dir

            status = "ok" if match else "fail"
            logging.info(
                f"  {status} {gene_name:8s}: "
                f"LFC={lfc:+7.3f} ({direction:7s}) "
                f"[expected {expected_dir}]"
            )

            validation_rows.append({
                "pair": pair_name,
                "gene": gene_name,
                "description": description,
                "expected_direction": expected_dir,
                "observed_direction": direction,
                "observed_lfc": lfc,
                "status": "match" if match else "mismatch",
            })

    #write validation report
    df_validation = pd.DataFrame(validation_rows)
    report_csv = output_dir / "lfc_validation_report.csv"
    df_validation.to_csv(report_csv, index=False)
    logging.info(f"\n✓ Wrote validation report to {report_csv.name}")

    #summary
    match_count = (df_validation["status"] == "match").sum()
    mismatch_count = (df_validation["status"] == "mismatch").sum()
    na_count = (df_validation["status"] == "not_in_network").sum()
    logging.info(
        f"Validation summary: {match_count} matches, "
        f"{mismatch_count} mismatches, {na_count} not_in_network"
    )


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    logging.info("computing condition-level lfc for network genes")
    logging.info(f"pseudocount={PSEUDOCOUNT}, up_threshold={LFC_UP_THRESHOLD}, down_threshold={LFC_DOWN_THRESHOLD}")

    #preflight validation
    preflight_results = run_preflight()
    all_ok = all(r.all_exists for r in preflight_results)

    if not all_ok:
        logging.error("preflight validation found missing files, aborting")
        for r in preflight_results:
            if not r.all_exists:
                logging.error(f"  {r.pair_name}: CSV={r.tagged_csv_exists}, test={r.test_h5ad_exists}, ref={r.ref_h5ad_exists}")
        return 1

    if args.preflight_only:
        logging.info("preflight passed, exiting (--preflight-only)")
        return 0

    #compute lfc for all condition pairs
    output_with_lfc_dir = OUTPUT_WITH_LFC_DIR
    results = []
    for pair_name, test_cond, ref_cond, test_h5ad, ref_h5ad in CONDITION_PAIRS:
        results.append(compute_lfc_for_pair(pair_name, test_cond, ref_cond, test_h5ad, ref_h5ad, output_with_lfc_dir))

    #qc validation against known genes
    validate_known_genes(output_with_lfc_dir)

    logging.info(f"done. {len(results)} pair tables written to {output_with_lfc_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
