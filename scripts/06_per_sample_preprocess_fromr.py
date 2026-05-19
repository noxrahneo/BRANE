#!/usr/bin/env python3
"""Per-sample preprocessing before integration (Seurat-style).

R_TRANSLATED: yes

R mapping (NormTotal.R concept):
- NormalizeData
- FindVariableFeatures
- ScaleData
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-sample preprocessing before integration"
    )
    parser.add_argument(
        "--condition",
        default="Normal",
        help="Condition folder name, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition folders and exit",
    )
    parser.add_argument(
        "--input-dir",
        default="results/02_preprocess/01_filtered",
        help="Root directory with filtered sample .h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/02_preprocess/02_preprocessed",
        help="Root directory for preprocessed sample .h5ad files",
    )
    parser.add_argument(
        "--n-top-genes", type=int, default=3000, help="Top HVGs per sample"
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Normalization target sum",
    )
    parser.add_argument(
        "--scale-max-value",
        type=float,
        default=10.0,
        help="Maximum value used by scaling",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "unknown"


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]

    mapped = safe_name(requested)
    if mapped in available:
        return [mapped]

    near = difflib.get_close_matches(requested, available, n=5, cutoff=0.45)
    hint = [mapped, *near] if mapped != requested else near
    msg = [
        f"Condition not found: '{requested}'",
        "Available condition folders:",
        *[f"- {x}" for x in available],
    ]
    if hint:
        msg += ["Closest matches:", *[f"- {x}" for x in dict.fromkeys(hint)]]
    raise ValueError("\n".join(msg))


def sample_name(path: Path) -> str:
    if path.name.endswith("_filtered.h5ad"):
        return path.name[:-14]
    return path.stem


def preprocess(
    adata: sc.AnnData,
    n_top_genes: int,
    target_sum: float,
    scale_max_value: float,
) -> sc.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
        subset=False,
        inplace=True,
    )
    sc.pp.scale(adata, max_value=scale_max_value)
    return adata


def run_condition(
    condition: str,
    input_base: Path,
    output_base: Path,
    args: argparse.Namespace,
) -> list[dict]:
    in_dir = input_base / condition
    out_dir = output_base / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    #find all filtered h5ad files for this condition
    files = sorted(in_dir.glob("*_filtered.h5ad"))
    if not files:
        print(f"Skipping {condition}: no *_filtered.h5ad files found")
        return []

    print(f"\nCondition: {condition} | Samples: {len(files)}")
    rows: list[dict] = []

    #preprocess each sample and save to output dir
    for idx, fpath in enumerate(files, start=1):
        sname = sample_name(fpath)
        print(f"[{idx}/{len(files)}] Processing {sname} ...")

        adata = sc.read_h5ad(fpath)
        n_cells, n_genes = adata.n_obs, adata.n_vars
        adata = preprocess(adata, args.n_top_genes, args.target_sum, args.scale_max_value)
        n_hvg = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var.columns else 0

        out_file = out_dir / f"{sname}_preprocessed.h5ad"
        adata.write_h5ad(out_file)

        rows.append(
            {
                "Condition": condition,
                "SampleName": sname,
                "InputFile": str(fpath),
                "OutputFile": str(out_file),
                "Cells": n_cells,
                "Genes": n_genes,
                "HVGs": n_hvg,
                "TargetSum": args.target_sum,
                "NTopGenes": args.n_top_genes,
                "ScaleMaxValue": args.scale_max_value,
            }
        )

    return rows


def main() -> int:
    args = parse_args()

    #resolve input and output roots
    input_base = resolve_base(args.input_dir)
    output_base = resolve_base(args.output_dir)

    #list available conditions and exit if requested
    if args.list_conditions:
        conditions = list_conditions(input_base)
        if not conditions:
            print(f"No condition folders found in: {input_base}")
            return 1
        print("Available condition folders:")
        for name in conditions:
            print(f"- {name}")
        return 0

    #resolve target conditions with fuzzy matching on failure
    try:
        targets = resolve_conditions(input_base, args.condition)
    except ValueError as exc:
        print(exc)
        return 1

    if not targets:
        print(f"No condition folders found in: {input_base}")
        return 1

    #run preprocessing per condition and collect summary rows
    rows: list[dict] = []
    for condition in targets:
        rows.extend(run_condition(condition, input_base, output_base, args))

    if not rows:
        print(f"No filtered files processed from: {input_base}")
        return 1

    #write summary csv alongside processed files
    summary_root = output_base if len(targets) > 1 else output_base / targets[0]
    summary_path = summary_root / "preprocess_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Samples processed: {len(rows)}")
    print(f"Preprocessed root: {output_base}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
