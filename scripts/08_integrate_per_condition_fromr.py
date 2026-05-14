#!/usr/bin/env python3
"""Integrate preprocessed samples per condition.

R_TRANSLATED: yes

R mapping (NormTotal.R concept):
- Create per-sample objects
- FindIntegrationAnchors / IntegrateData (Seurat)
- Run joint PCA/UMAP/clustering on integrated object

Python equivalent in this script:
- Concatenate preprocessed AnnData objects by condition
- Rebuild joint normalized matrix from counts
- Select HVGs on merged data
- Optional batch correction with ComBat (batch = sample)
- Joint PCA / neighbors / UMAP / Leiden
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate preprocessed samples per condition"
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
        default="results/02_preprocess/02_preprocessed",
        help="Root directory with preprocessed sample .h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/03_integration/integrated",
        help="Root directory for integrated condition outputs",
    )
    parser.add_argument(
        "--integration-method",
        choices=["combat", "none"],
        default="combat",
        help="Batch integration method applied after merge",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=3000,
        help="Top HVGs for joint analysis",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=30,
        help="Number of PCs",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Neighbors for graph construction",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.6,
        help="Leiden clustering resolution",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.5,
        help="UMAP min_dist",
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
    hints = [mapped, *near] if mapped != requested else near
    msg = [
        f"Condition not found: '{requested}'",
        "Available condition folders:",
        *[f"- {name}" for name in available],
    ]
    if hints:
        msg += [
            "Closest matches:",
            *[f"- {name}" for name in dict.fromkeys(hints)],
        ]
    raise ValueError("\n".join(msg))


def sample_name(path: Path) -> str:
    if path.name.endswith("_preprocessed.h5ad"):
        return path.name[:-18]
    return path.stem


def load_condition_samples(condition_dir: Path) -> list[ad.AnnData]:
    files = sorted(condition_dir.glob("*_preprocessed.h5ad"))
    samples: list[ad.AnnData] = []
    for fpath in files:
        sample = sample_name(fpath)
        adata = sc.read_h5ad(fpath)
        adata.obs["SampleName"] = sample
        adata.obs["Condition"] = condition_dir.name
        adata.obs["SampleCondition"] = f"{sample}__{condition_dir.name}"
        samples.append(adata)
    return samples


def build_joint_object(
    samples: list[ad.AnnData],
    n_top_genes: int,
) -> ad.AnnData:
    merged = ad.concat(
        samples,
        join="inner",
        label="batch_file",
        keys=[a.obs["SampleName"].iloc[0] for a in samples],
        index_unique="-",
        merge="same",
    )

    if "counts" in merged.layers:
        merged.X = merged.layers["counts"].copy()

    sc.pp.normalize_total(merged, target_sum=1e4)
    sc.pp.log1p(merged)

    try:
        sc.pp.highly_variable_genes(
            merged,
            n_top_genes=n_top_genes,
            flavor="seurat",
            batch_key="SampleName",
            subset=False,
            inplace=True,
        )
    except Exception:
        sc.pp.highly_variable_genes(
            merged,
            n_top_genes=n_top_genes,
            flavor="seurat",
            subset=False,
            inplace=True,
        )

    if "highly_variable" in merged.var.columns:
        merged = merged[:, merged.var["highly_variable"].values].copy()

    return merged


def run_joint_analysis(
    adata: ad.AnnData,
    integration_method: str,
    n_pcs: int,
    n_neighbors: int,
    resolution: float,
    umap_min_dist: float,
) -> ad.AnnData:
    if integration_method == "combat":
        sc.pp.combat(adata, key="SampleName")

    sc.pp.scale(adata, max_value=10.0)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata, min_dist=umap_min_dist)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
    return adata


def integrate_condition(
    condition: str,
    args: argparse.Namespace,
) -> dict | None:
    input_base = resolve_base(args.input_dir)
    output_base = resolve_base(args.output_dir)

    condition_dir = input_base / condition
    samples = load_condition_samples(condition_dir)
    if not samples:
        print(f"Skipping {condition}: no *_preprocessed.h5ad files found")
        return None

    print(f"\nCondition: {condition} | Samples: {len(samples)}")
    merged = build_joint_object(samples, n_top_genes=args.n_top_genes)
    merged = run_joint_analysis(
        merged,
        integration_method=args.integration_method,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        resolution=args.resolution,
        umap_min_dist=args.umap_min_dist,
    )

    out_dir = output_base / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    integrated_file = out_dir / f"{condition}_integrated.h5ad"
    merged.write_h5ad(integrated_file)

    obs_cols = [
        col
        for col in ["SampleName", "Condition", "leiden"]
        if col in merged.obs.columns
    ]
    obs_path = out_dir / f"{condition}_cell_metadata.csv"
    merged.obs[obs_cols].to_csv(obs_path)

    umap_path = out_dir / f"{condition}_umap.csv"
    if "X_umap" in merged.obsm:
        umap_df = pd.DataFrame(
            merged.obsm["X_umap"],
            index=merged.obs_names,
            columns=["UMAP1", "UMAP2"],
        )
        umap_df.to_csv(umap_path)

    if "leiden" in merged.obs.columns:
        n_clusters = int(merged.obs["leiden"].nunique())
    else:
        n_clusters = 0
    return {
        "Condition": condition,
        "Samples": len(samples),
        "Cells": int(merged.n_obs),
        "GenesAfterHVG": int(merged.n_vars),
        "Clusters": n_clusters,
        "IntegrationMethod": args.integration_method,
        "IntegratedFile": str(integrated_file),
        "CellMetadata": str(obs_path),
        "UmapFile": str(umap_path),
    }


def main() -> int:
    args = parse_args()
    input_base = resolve_base(args.input_dir)
    output_base = resolve_base(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    if args.list_conditions:
        conditions = list_conditions(input_base)
        if not conditions:
            print(f"No condition folders found in: {input_base}")
            return 1
        print("Available condition folders:")
        for name in conditions:
            print(f"- {name}")
        return 0

    try:
        targets = resolve_conditions(input_base, args.condition)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not targets:
        print(f"ERROR: No condition folders found in {input_base}")
        return 1

    rows = []
    for condition in targets:
        row = integrate_condition(condition, args)
        if row is not None:
            rows.append(row)

    if not rows:
        print("ERROR: No conditions were integrated.")
        return 1

    summary_name = (
        "integration_summary.csv"
        if len(targets) > 1
        else f"integration_summary_{targets[0]}.csv"
    )
    summary_path = output_base / summary_name
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("\nIntegration complete")
    print(f"Conditions integrated: {len(rows)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
