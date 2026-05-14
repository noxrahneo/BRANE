#!/usr/bin/env python3
"""Freeze manual cluster-to-cell-type curation for thesis outputs.

Workflow:
1) Export editable curation tables from predicted mappings.
2) Apply curated mappings onto integrated (non-fixed) h5ad objects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTEGRATED_ROOT = "results/03_integration/integrated"
DEFAULT_ANNOTATION_ROOT = "results/04_annotation"
DEFAULT_CURATED_COL = "curated_cell_type"
DEFAULT_CLUSTER_COL = "leiden"


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze manual cluster-to-cell-type mappings"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export curated templates")
    add_common_args(export_parser)
    export_parser.add_argument(
        "--curated-label-col",
        default=DEFAULT_CURATED_COL,
        help="Column created for manual curation",
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing curated mapping files",
    )

    apply_parser = sub.add_parser("apply", help="Apply curated templates")
    add_common_args(apply_parser)
    apply_parser.add_argument(
        "--cluster-col",
        default=DEFAULT_CLUSTER_COL,
        help="Cluster column in adata.obs",
    )
    apply_parser.add_argument(
        "--curated-label-col",
        default=DEFAULT_CURATED_COL,
        help="Curated label column in curated mapping files",
    )
    apply_parser.add_argument(
        "--freeze-tag",
        default="thesis_v1",
        help="Version tag stored in outputs and summaries",
    )

    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition folder name, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available conditions and exit",
    )
    parser.add_argument(
        "--integration-dir",
        default=DEFAULT_INTEGRATED_ROOT,
        help="Integrated root (integrated_fixed is blocked)",
    )
    parser.add_argument(
        "--annotation-dir",
        default=DEFAULT_ANNOTATION_ROOT,
        help="Annotation root with per-condition mapping files",
    )


def assert_thesis_integrated_root(path: Path) -> None:
    norm = str(path).replace("\\", "/")
    if "integrated_fixed" in norm:
        raise ValueError(
            "integration-dir points to integrated_fixed, which is excluded "
            "for thesis freeze runs"
        )
    if not norm.endswith("/03_integration/integrated"):
        print(
            "WARNING: integration-dir is not the default integrated root. "
            "Proceeding because integrated_fixed is not used."
        )


def integrated_file_for(condition_dir: Path) -> Path:
    preferred = condition_dir / f"{condition_dir.name}_integrated.h5ad"
    if preferred.exists():
        return preferred
    hits = sorted(condition_dir.glob("*_integrated.h5ad"))
    if not hits:
        raise FileNotFoundError(
            f"No integrated h5ad found in: {condition_dir}"
        )
    return hits[0]


def list_conditions(
    integration_root: Path,
    annotation_root: Path,
) -> list[str]:
    if not integration_root.exists() or not annotation_root.exists():
        return []

    names: list[str] = []
    for cond_dir in sorted(
        p for p in integration_root.iterdir() if p.is_dir()
    ):
        condition = cond_dir.name
        try:
            _ = integrated_file_for(cond_dir)
        except FileNotFoundError:
            continue

        map_file = (
            annotation_root
            / condition
            / f"{condition}_cluster_to_celltype_mapping.csv"
        )
        if map_file.exists():
            names.append(condition)
    return names


def resolve_or_list_conditions(
    integration_root: Path,
    annotation_root: Path,
    requested: str,
    list_only: bool,
) -> list[str]:
    available = list_conditions(integration_root, annotation_root)
    if list_only:
        if not available:
            print("No conditions found")
            return []
        print("Available conditions:")
        for name in available:
            print(f"- {name}")
        return []
    if not available:
        raise FileNotFoundError(
            "No conditions found with both integrated data and mapping file"
        )
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise FileNotFoundError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def export_curated_table(
    condition: str,
    annotation_root: Path,
    curated_label_col: str,
    overwrite: bool,
) -> Path:
    cond_dir = annotation_root / condition
    in_file = cond_dir / f"{condition}_cluster_to_celltype_mapping.csv"
    out_file = (
        cond_dir / f"{condition}_cluster_to_celltype_mapping_curated.csv"
    )

    if out_file.exists() and not overwrite:
        print(f"[{condition}] curated file exists, skip: {out_file}")
        return out_file

    df = pd.read_csv(in_file)
    if "predicted_cell_type" not in df.columns:
        raise ValueError(
            f"predicted_cell_type missing in mapping file: {in_file}"
        )

    df = df.copy()
    df[curated_label_col] = df["predicted_cell_type"].astype(str)
    if "curation_note" not in df.columns:
        df["curation_note"] = ""

    first_cols = [
        c
        for c in [
            "leiden",
            "predicted_cell_type",
            curated_label_col,
            "curation_note",
        ]
        if c in df.columns
    ]
    rest_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + rest_cols]

    df.to_csv(out_file, index=False)
    print(f"[{condition}] wrote curated template: {out_file}")
    return out_file


def validate_curated_mapping(
    df: pd.DataFrame,
    condition: str,
    cluster_col: str,
    curated_label_col: str,
) -> pd.DataFrame:
    need = {cluster_col, curated_label_col}
    if not need.issubset(df.columns):
        raise ValueError(
            "[{}] curated mapping missing columns: {}".format(
                condition,
                need - set(df.columns),
            )
        )

    out = df.copy()
    out[cluster_col] = out[cluster_col].astype(str)
    out[curated_label_col] = out[curated_label_col].astype(str).str.strip()

    if out[cluster_col].duplicated().any():
        dup = out.loc[out[cluster_col].duplicated(), cluster_col].tolist()
        raise ValueError(
            f"[{condition}] duplicate clusters in curated mapping: {dup}"
        )

    bad = out[curated_label_col].str.lower().isin(["", "nan", "none"])
    if bad.any():
        raise ValueError(
            f"[{condition}] curated labels contain empty/invalid values"
        )

    return out


def apply_freeze_one_condition(
    condition: str,
    integration_root: Path,
    annotation_root: Path,
    cluster_col: str,
    curated_label_col: str,
    freeze_tag: str,
) -> dict[str, object]:
    cond_integrated_dir = integration_root / condition
    cond_annot_dir = annotation_root / condition
    cond_annot_dir.mkdir(parents=True, exist_ok=True)

    integrated_h5ad = integrated_file_for(cond_integrated_dir)

    curated_file = (
        cond_annot_dir
        / f"{condition}_cluster_to_celltype_mapping_curated.csv"
    )
    if not curated_file.exists():
        raise FileNotFoundError(
            f"[{condition}] curated mapping missing: {curated_file}"
        )

    curated_df = pd.read_csv(curated_file)
    curated_df = validate_curated_mapping(
        df=curated_df,
        condition=condition,
        cluster_col=cluster_col,
        curated_label_col=curated_label_col,
    )

    adata = sc.read_h5ad(integrated_h5ad)
    if cluster_col not in adata.obs.columns:
        raise ValueError(
            "[{}] cluster column '{}' missing in adata.obs".format(
                condition,
                cluster_col,
            )
        )

    clusters_present = set(
        adata.obs[cluster_col].astype(str).unique().tolist()
    )
    clusters_mapped = set(
        curated_df[cluster_col].astype(str).unique().tolist()
    )

    missing = sorted(clusters_present - clusters_mapped)
    if missing:
        raise ValueError(
            "[{}] curated mapping misses clusters present in data: {}".format(
                condition,
                missing,
            )
        )

    mapping = dict(
        zip(
            curated_df[cluster_col].astype(str),
            curated_df[curated_label_col].astype(str),
        )
    )

    frozen_labels = adata.obs[cluster_col].astype(str).map(mapping)
    if frozen_labels.isna().any():
        raise ValueError(
            "[{}] unresolved labels after mapping; "
            "check curated file".format(condition)
        )

    adata.obs["cell_type_annot"] = frozen_labels.values
    adata.obs["cell_type_annot_frozen"] = frozen_labels.values

    freeze_meta = {
        "freeze_tag": freeze_tag,
        "freeze_date_utc": utc_now_iso(),
        "source_integrated_h5ad": str(integrated_h5ad),
        "source_mapping_curated": str(curated_file),
        "cluster_col": cluster_col,
        "curated_label_col": curated_label_col,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_clusters": int(adata.obs[cluster_col].nunique()),
        "n_cell_types_frozen": int(
            adata.obs["cell_type_annot_frozen"].nunique()
        ),
    }
    adata.uns["annotation_freeze"] = freeze_meta

    frozen_h5ad = cond_annot_dir / f"{condition}_annotated_frozen.h5ad"
    adata.write_h5ad(frozen_h5ad)

    frozen_map = curated_df.copy()
    frozen_map["freeze_tag"] = freeze_tag
    frozen_map["freeze_date_utc"] = freeze_meta["freeze_date_utc"]
    frozen_map_file = (
        cond_annot_dir / f"{condition}_cluster_to_celltype_mapping_frozen.csv"
    )
    frozen_map.to_csv(frozen_map_file, index=False)

    summary = {
        "condition": condition,
        "freeze_tag": freeze_tag,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "clusters": int(adata.obs[cluster_col].nunique()),
        "cell_types_frozen": int(
            adata.obs["cell_type_annot_frozen"].nunique()
        ),
        "integrated_input": str(integrated_h5ad),
        "curated_mapping_input": str(curated_file),
        "frozen_mapping_output": str(frozen_map_file),
        "frozen_h5ad_output": str(frozen_h5ad),
    }

    pd.DataFrame([summary]).to_csv(
        cond_annot_dir / f"{condition}_annotation_freeze_summary.csv",
        index=False,
    )

    print(f"[{condition}] frozen mapping: {frozen_map_file}")
    print(f"[{condition}] frozen h5ad: {frozen_h5ad}")
    return summary


def run_export(
    args: argparse.Namespace,
    integration_root: Path,
    annotation_root: Path,
) -> int:
    conditions = resolve_or_list_conditions(
        integration_root=integration_root,
        annotation_root=annotation_root,
        requested=args.condition,
        list_only=args.list_conditions,
    )
    if args.list_conditions:
        return 0

    rows = []
    for condition in conditions:
        out_file = export_curated_table(
            condition=condition,
            annotation_root=annotation_root,
            curated_label_col=args.curated_label_col,
            overwrite=args.overwrite,
        )
        rows.append({"condition": condition, "curated_file": str(out_file)})

    out_index = annotation_root / "annotation_curation_templates.csv"
    pd.DataFrame(rows).to_csv(out_index, index=False)
    print(f"Template index: {out_index}")
    return 0


def run_apply(
    args: argparse.Namespace,
    integration_root: Path,
    annotation_root: Path,
) -> int:
    conditions = resolve_or_list_conditions(
        integration_root=integration_root,
        annotation_root=annotation_root,
        requested=args.condition,
        list_only=args.list_conditions,
    )
    if args.list_conditions:
        return 0

    arg_hash = params_hash(vars(args))
    now = utc_now_iso()
    rows: list[dict[str, object]] = []
    warehouse_rows: list[WarehouseRecord] = []

    for condition in conditions:
        row = apply_freeze_one_condition(
            condition=condition,
            integration_root=integration_root,
            annotation_root=annotation_root,
            cluster_col=args.cluster_col,
            curated_label_col=args.curated_label_col,
            freeze_tag=args.freeze_tag,
        )
        rows.append(row)
        warehouse_rows.append(
            WarehouseRecord(
                input_file=str(row["integrated_input"]),
                output_file=str(row["frozen_h5ad_output"]),
                script="scripts/13_freeze_annotation.py",
                date_utc=now,
                params_hash=arg_hash,
                condition=str(row["condition"]),
                stage="annotation_freeze",
            )
        )

    summary_file = annotation_root / "annotation_freeze_summary.csv"
    pd.DataFrame(rows).to_csv(summary_file, index=False)
    print(f"Freeze summary: {summary_file}")

    warehouse_file = append_warehouse(annotation_root, warehouse_rows)
    print(f"Warehouse log: {warehouse_file}")
    return 0


def main() -> int:
    args = parse_args()
    integration_root = resolve_base(args.integration_dir)
    annotation_root = resolve_base(args.annotation_dir)

    assert_thesis_integrated_root(integration_root)

    if args.command == "export":
        return run_export(args, integration_root, annotation_root)
    if args.command == "apply":
        return run_apply(args, integration_root, annotation_root)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
