#!/usr/bin/env python3
"""Create per-condition logCPM-normalized h5ad files for network inputs.

For each condition folder in pseudobulk output:
- loads *_pseudobulk_counts.csv (genes x pseudobulk profiles)
- loads *_pseudobulk_metadata.csv
- computes log1p(CPM) using target_sum
- writes h5ad with:
    X = logCPM
    layers['counts'] = raw counts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare per-condition normalized h5ad files"
    )
    parser.add_argument(
        "--pseudobulk-dir",
        default="results/06_pseudobulk",
        help="Directory containing per-condition pseudobulk CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/08_pre_correlation/per_condition",
        help="Directory to write normalized per-condition h5ad files",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1_000_000.0,
        help="Target library size for CPM normalization",
    )
    parser.add_argument(
        "--include-mask-csv",
        default="",
        help=(
            "Optional CSV with pseudobulk_id and include_in_network columns; "
            "if set, only included profiles are kept"
        ),
    )
    parser.add_argument(
        "--include-col",
        default="include_in_network",
        help=(
            "Column name in include-mask CSV that indicates "
            "profile inclusion"
        ),
    )
    return parser.parse_args()


def resolve(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def normalize_logcpm(counts: pd.DataFrame, target_sum: float) -> pd.DataFrame:
    lib = counts.sum(axis=0).replace(0.0, np.nan)
    cpm = counts.divide(lib, axis=1).fillna(0.0) * float(target_sum)
    return np.log1p(cpm)


def list_condition_dirs(pb_root: Path) -> list[Path]:
    if not pb_root.exists():
        return []
    return sorted([p for p in pb_root.iterdir() if p.is_dir()])


def load_include_mask(mask_csv: str, include_col: str) -> set[str] | None:
    if not str(mask_csv).strip():
        return None

    mask_path = resolve(mask_csv)
    if not mask_path.exists():
        raise FileNotFoundError(f"Include-mask CSV not found: {mask_path}")

    mask_df = pd.read_csv(mask_path)
    if "pseudobulk_id" not in mask_df.columns:
        raise ValueError("Include-mask CSV must contain 'pseudobulk_id'")
    if include_col not in mask_df.columns:
        raise ValueError(
            f"Include-mask CSV must contain include column '{include_col}'"
        )

    include_vals = mask_df[include_col]
    if include_vals.dtype == bool:
        include_bool = include_vals
    else:
        include_bool = include_vals.astype(str).str.strip().str.lower().isin(
            ["1", "true", "t", "yes", "y"]
        )

    keep_ids = set(
        mask_df.loc[include_bool, "pseudobulk_id"].astype(str).tolist()
    )
    return keep_ids


def main() -> None:
    args = parse_args()
    pb_root = resolve(args.pseudobulk_dir)
    out_root = resolve(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    include_ids = load_include_mask(args.include_mask_csv, args.include_col)

    condition_dirs = list_condition_dirs(pb_root)
    if not condition_dirs:
        raise FileNotFoundError(
            f"No condition directories found in: {pb_root}"
        )

    wrote = 0
    for cdir in condition_dirs:
        condition = cdir.name
        counts_file = cdir / f"{condition}_pseudobulk_counts.csv"
        meta_file = cdir / f"{condition}_pseudobulk_metadata.csv"

        if not counts_file.exists() or not meta_file.exists():
            continue

        counts = pd.read_csv(counts_file, index_col=0)
        meta = pd.read_csv(meta_file)
        if "pseudobulk_id" not in meta.columns:
            raise ValueError(f"Missing pseudobulk_id in {meta_file}")

        profile_ids = [
            pid
            for pid in meta["pseudobulk_id"].tolist()
            if pid in counts.columns
        ]
        if include_ids is not None:
            profile_ids = [pid for pid in profile_ids if pid in include_ids]
        if not profile_ids:
            continue

        counts = counts[profile_ids]
        meta = meta.set_index("pseudobulk_id").loc[profile_ids]
        logcpm = normalize_logcpm(counts, args.target_sum)

        adata = ad.AnnData(
            X=logcpm.T.values,
            obs=meta.copy(),
            var=pd.DataFrame(index=logcpm.index.astype(str)),
        )
        adata.obs_names = logcpm.columns.astype(str)
        adata.var_names = logcpm.index.astype(str)
        adata.layers["counts"] = counts.T.values

        adata.uns["normalization"] = {
            "method": "log1p_cpm",
            "target_sum": float(args.target_sum),
        }

        out_file = out_root / f"{condition}_pseudobulk_logcpm.h5ad"
        adata.write_h5ad(out_file)
        wrote += 1
        print(f"[{condition}] wrote {out_file}")

    print(
        "Done. Wrote "
        f"{wrote} normalized condition h5ad files to: {out_root}"
    )


if __name__ == "__main__":
    main()
