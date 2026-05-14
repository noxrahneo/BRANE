#!/usr/bin/env python3
"""Prepare a clean pre-correlation input pack for network analysis.

Outputs a harmonized set of files:
- combined_logcpm.csv
- metadata_with_qc.csv
- flagged_mask.csv
- include_mask_all.csv
- include_mask_exclude_flagged.csv
- pack_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pre-correlation network input pack"
    )
    parser.add_argument(
        "--pseudobulk-dir",
        default="results/06_pseudobulk",
        help="Directory containing all_conditions pseudobulk exports",
    )
    parser.add_argument(
        "--qc-dir",
        default="results/07_pseudobulk_qc",
        help="Directory containing qc_profile_metrics and qc_flagged_profiles",
    )
    parser.add_argument(
        "--output-dir",
        default="results/08_pre_correlation",
        help="Output directory for pre-correlation pack",
    )
    return parser.parse_args()


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    pb_dir = _resolve(args.pseudobulk_dir)
    qc_dir = _resolve(args.qc_dir)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logcpm_file = pb_dir / "all_conditions_pseudobulk_logcpm.csv"
    meta_file = pb_dir / "all_conditions_pseudobulk_metadata.csv"
    qc_metrics_file = qc_dir / "qc_profile_metrics.csv"
    qc_flagged_file = qc_dir / "qc_flagged_profiles.csv"

    if not logcpm_file.exists():
        raise FileNotFoundError(
            f"Missing all-conditions logCPM matrix: {logcpm_file}"
        )

    logcpm = pd.read_csv(logcpm_file, index_col=0)
    meta = _load_required_csv(meta_file)
    qc_metrics = _load_required_csv(qc_metrics_file)
    qc_flagged = _load_required_csv(qc_flagged_file)

    if "pseudobulk_id" not in meta.columns:
        raise ValueError("Expected pseudobulk_id in metadata file")
    if "pseudobulk_id" not in qc_metrics.columns:
        raise ValueError("Expected pseudobulk_id in qc_profile_metrics.csv")

    ids = [
        pid
        for pid in meta["pseudobulk_id"].tolist()
        if pid in logcpm.columns
    ]
    if not ids:
        raise ValueError(
            "No overlapping pseudobulk IDs between metadata and logCPM"
        )

    aligned_logcpm = logcpm[ids].copy()
    meta = meta.set_index("pseudobulk_id").loc[ids].reset_index()

    qc_cols = [
        "pseudobulk_id",
        "library_size",
        "detected_genes",
        "counts_match_meta",
        "low_n_cells_flag",
        "profile_outlier_flag",
    ]
    qc_cols = [c for c in qc_cols if c in qc_metrics.columns]
    qc_small = qc_metrics[qc_cols].drop_duplicates(
        subset=["pseudobulk_id"]
    )

    meta_qc = meta.merge(qc_small, on="pseudobulk_id", how="left")

    flagged_ids = set()
    if "pseudobulk_id" in qc_flagged.columns:
        flagged_ids = set(qc_flagged["pseudobulk_id"].astype(str).tolist())

    flagged_mask = pd.DataFrame(
        {
            "pseudobulk_id": ids,
            "is_flagged": [pid in flagged_ids for pid in ids],
        }
    )

    include_all = flagged_mask.rename(columns={"is_flagged": "_flag"}).copy()
    include_all["include_in_network"] = True
    include_all = include_all[["pseudobulk_id", "include_in_network"]]

    include_excl = flagged_mask.rename(columns={"is_flagged": "_flag"}).copy()
    include_excl["include_in_network"] = ~include_excl["_flag"]
    include_excl = include_excl[["pseudobulk_id", "include_in_network"]]

    summary = pd.DataFrame(
        [
            {
                "n_genes": int(aligned_logcpm.shape[0]),
                "n_profiles": int(aligned_logcpm.shape[1]),
                "n_flagged_profiles": int(flagged_mask["is_flagged"].sum()),
                "n_unflagged_profiles": int(
                    (~flagged_mask["is_flagged"]).sum()
                ),
                "n_conditions": int(meta_qc["condition"].astype(str).nunique())
                if "condition" in meta_qc.columns
                else None,
            }
        ]
    )

    values = aligned_logcpm.to_numpy(dtype=float)
    sparsity_diag = pd.DataFrame(
        [
            {
                "n_genes": int(aligned_logcpm.shape[0]),
                "n_profiles": int(aligned_logcpm.shape[1]),
                "matrix_size": int(values.size),
                "n_nonfinite": int(values.size - np.isfinite(values).sum()),
                "n_negative": int(np.sum(values < 0)),
                "zero_fraction": float(np.mean(np.abs(values) <= 1e-12)),
                "lt_0p01_fraction": float(np.mean(values < 0.01)),
                "lt_0p1_fraction": float(np.mean(values < 0.1)),
            }
        ]
    )

    aligned_logcpm.to_csv(out_dir / "combined_logcpm.csv")
    meta_qc.to_csv(out_dir / "metadata_with_qc.csv", index=False)
    flagged_mask.to_csv(out_dir / "flagged_mask.csv", index=False)
    include_all.to_csv(out_dir / "include_mask_all.csv", index=False)
    include_excl.to_csv(
        out_dir / "include_mask_exclude_flagged.csv",
        index=False,
    )
    summary.to_csv(out_dir / "pack_summary.csv", index=False)
    sparsity_diag.to_csv(
        out_dir / "combined_logcpm_sparsity_diagnostics.csv",
        index=False,
    )

    print(f"Pre-correlation pack ready: {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
