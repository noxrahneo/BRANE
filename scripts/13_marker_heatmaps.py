#!/usr/bin/env python3
"""Create marker heatmaps from annotation outputs.

R -> Python mapping used here:
- R: pheatmap(marker_matrix)
  Python: mean-expression matrix + row z-score + matplotlib heatmap.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build marker heatmaps for one condition or all conditions"
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition folder under results/04_annotation, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List condition folders in annotation output and exit",
    )
    parser.add_argument(
        "--input-dir",
        default="results/04_annotation",
        help="Root annotation output directory",
    )
    parser.add_argument(
        "--group-col",
        default="cell_type_annot",
        help="obs column to aggregate heatmap columns",
    )
    parser.add_argument(
        "--gene-set-file",
        default="",
        help=(
            "Optional TSV/CSV/TXT file with genes for heatmap rows. "
            "If set, marker-based selection is skipped"
        ),
    )
    parser.add_argument(
        "--gene-col",
        default="Gene",
        help=(
            "Column name to read genes from gene-set file. "
            "If missing, first column is used"
        ),
    )
    parser.add_argument(
        "--gene-set-name",
        default="",
        help=(
            "Short label used in output filenames and plot titles "
            "when --gene-set-file is provided"
        ),
    )
    parser.add_argument(
        "--top-per-cluster",
        type=int,
        default=10,
        help="Top markers per cluster to include",
    )
    parser.add_argument(
        "--min-padj",
        type=float,
        default=0.05,
        help="Adjusted p-value threshold for marker selection",
    )
    parser.add_argument(
        "--fig-dir-name",
        default="figures",
        help="Figure subdirectory name inside each condition output",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    names: list[str] = []
    for p in sorted(x for x in root.iterdir() if x.is_dir()):
        condition = p.name
        if (
            (p / f"{condition}_annotated.h5ad").exists()
            and (p / f"{condition}_cluster_markers_top.csv").exists()
        ):
            names.append(condition)
    return names


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if not available:
        raise FileNotFoundError(f"No condition folders found in: {root}")
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise FileNotFoundError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    vals = df.to_numpy(dtype=float)
    mean = vals.mean(axis=1, keepdims=True)
    std = vals.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    out = (vals - mean) / std
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def read_genes_from_file(gene_set_file: Path, gene_col: str) -> list[str]:
    suffix = gene_set_file.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(gene_set_file, sep="\t")
    else:
        df = pd.read_csv(gene_set_file)

    if df.empty:
        return []

    if gene_col in df.columns:
        raw = df[gene_col]
    else:
        raw = df.iloc[:, 0]

    genes = [g.strip() for g in raw.astype(str).tolist()]
    genes = [g for g in genes if g and g.lower() != "nan"]
    return list(dict.fromkeys(genes))


def read_marker_genes(
    marker_file: Path,
    top_per_cluster: int,
    min_padj: float,
) -> list[str]:
    # Marker selection mirrors the R pattern:
    # 1) keep positive, significant markers,
    # 2) keep top genes per cluster,
    # 3) de-duplicate for a compact heatmap signature panel.
    mk = pd.read_csv(marker_file)
    need = {"cluster", "names"}
    if not need.issubset(set(mk.columns)):
        raise ValueError(
            f"Marker file missing required columns: {marker_file}"
        )

    if "pvals_adj" in mk.columns:
        mk = mk[mk["pvals_adj"].fillna(1.0) <= min_padj]
    if "logfoldchanges" in mk.columns:
        mk = mk[mk["logfoldchanges"].fillna(0.0) > 0]

    if mk.empty:
        return []

    mk = mk.sort_values(["cluster", "pvals_adj", "logfoldchanges"])
    top = mk.groupby("cluster", as_index=False).head(top_per_cluster)
    genes = [g for g in top["names"].astype(str).tolist() if g]
    return list(dict.fromkeys(genes))


def lognorm_from_counts(adata: sc.AnnData) -> sc.AnnData:
    # In R scripts, heatmaps are drawn from log-CPM-like values.
    # Here we rebuild log-normalized expression from raw counts
    # (if available) before computing group means.
    ad = adata.copy()
    if "counts" in ad.layers:
        ad.X = ad.layers["counts"].copy()
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    return ad


def mean_matrix_by_group(
    adata: sc.AnnData,
    genes: list[str],
    group_col: str,
) -> pd.DataFrame:
    if group_col not in adata.obs.columns:
        raise ValueError(f"obs column '{group_col}' not found")
    keep_genes = [g for g in genes if g in set(map(str, adata.var_names))]
    if not keep_genes:
        return pd.DataFrame()

    ad = adata[:, keep_genes].copy()
    mat = ad.to_df()
    mat[group_col] = ad.obs[group_col].astype(str).values
    mean_df = mat.groupby(group_col, observed=False).mean().T
    return mean_df


def render_heatmap(
    matrix_z: pd.DataFrame,
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    if matrix_z.empty:
        return
    n_rows, n_cols = matrix_z.shape
    fig_w = max(7.0, 0.22 * n_cols + 4.0)
    fig_h = max(7.0, 0.18 * n_rows + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        matrix_z.to_numpy(),
        aspect="auto",
        cmap="bwr",
        interpolation="nearest",
        vmin=-2.5,
        vmax=2.5,
    )
    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix_z.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix_z.index)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Row z-score")
    fig.tight_layout()
    fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def process_condition(
    root: Path,
    condition: str,
    group_col: str,
    top_per_cluster: int,
    min_padj: float,
    fig_dir_name: str,
    dpi: int,
    gene_set_file: Path | None,
    gene_col: str,
    gene_set_name: str,
) -> dict[str, object]:
    # R equivalent flow:
    # FindMarkers output -> marker matrix by group -> pheatmap.
    cond_dir = root / condition
    h5ad = cond_dir / f"{condition}_annotated.h5ad"
    marker_file = cond_dir / f"{condition}_cluster_markers_top.csv"
    if not h5ad.exists() or not marker_file.exists():
        raise FileNotFoundError(
            "Missing required files for "
            f"{condition}: {h5ad.name}, {marker_file.name}"
        )

    if gene_set_file is not None:
        genes = read_genes_from_file(gene_set_file, gene_col=gene_col)
        heatmap_type = (
            gene_set_name.strip()
            if gene_set_name.strip()
            else gene_set_file.stem
        )
        file_tag = heatmap_type
    else:
        genes = read_marker_genes(
            marker_file=marker_file,
            top_per_cluster=top_per_cluster,
            min_padj=min_padj,
        )
        heatmap_type = "marker"
        file_tag = "marker"

    if not genes:
        raise ValueError(
            f"No genes selected for condition: {condition}"
        )

    adata = sc.read_h5ad(h5ad)
    adata = lognorm_from_counts(adata)
    mean_df = mean_matrix_by_group(
        adata=adata,
        genes=genes,
        group_col=group_col,
    )
    if mean_df.empty:
        raise ValueError(
            f"No marker genes present in data for condition: {condition}"
        )
    z_df = zscore_rows(mean_df)

    fig_dir = cond_dir / fig_dir_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    expr_file = cond_dir / f"{condition}_{file_tag}_heatmap_expression.csv"
    z_file = cond_dir / f"{condition}_{file_tag}_heatmap_zscore.csv"
    fig_file = fig_dir / f"{condition}_{file_tag}_heatmap_{group_col}.png"

    mean_df.to_csv(expr_file)
    z_df.to_csv(z_file)
    render_heatmap(
        matrix_z=z_df,
        title=f"{condition} | {heatmap_type} heatmap by {group_col}",
        out_file=fig_file,
        dpi=dpi,
    )

    print(f"Marker heatmap complete: {condition}")
    return {
        "condition": condition,
        "heatmap_type": heatmap_type,
        "input_file": str(h5ad),
        "group_col": group_col,
        "marker_genes": int(z_df.shape[0]),
        "groups": int(z_df.shape[1]),
        "expression_file": str(expr_file),
        "zscore_file": str(z_file),
        "figure_file": str(fig_file),
    }


def main() -> int:
    args = parse_args()
    root = resolve_base(args.input_dir)
    available = list_conditions(root)
    if args.list_conditions:
        if not available:
            print("No conditions found")
            return 0
        print("Available conditions:")
        for name in available:
            print(f"- {name}")
        return 0

    conditions = resolve_conditions(root, args.condition)
    gene_set_file: Path | None = None
    if args.gene_set_file.strip():
        gene_set_file = resolve_base(args.gene_set_file)
        if not gene_set_file.exists():
            raise FileNotFoundError(
                f"Gene set file not found: {gene_set_file}"
            )

    arg_hash = params_hash(vars(args))
    now = utc_now_iso()
    rows: list[dict[str, object]] = []
    warehouse_rows: list[WarehouseRecord] = []
    for condition in conditions:
        row = process_condition(
            root=root,
            condition=condition,
            group_col=args.group_col,
            top_per_cluster=args.top_per_cluster,
            min_padj=args.min_padj,
            fig_dir_name=args.fig_dir_name,
            dpi=args.dpi,
            gene_set_file=gene_set_file,
            gene_col=args.gene_col,
            gene_set_name=args.gene_set_name,
        )
        rows.append(row)
        warehouse_rows.append(
            WarehouseRecord(
                input_file=str(row["input_file"]),
                output_file=str(row["figure_file"]),
                script="scripts/13_marker_heatmaps.py",
                date_utc=now,
                params_hash=arg_hash,
                condition=str(row["condition"]),
                stage="marker_heatmaps",
            )
        )
        gc.collect()

    if len(rows) > 1:
        summary_file = root / "marker_heatmap_summary.csv"
        pd.DataFrame(rows).to_csv(summary_file, index=False)
        print(f"Summary: {summary_file}")

    warehouse_file = append_warehouse(root, warehouse_rows)
    print(f"Warehouse log: {warehouse_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
