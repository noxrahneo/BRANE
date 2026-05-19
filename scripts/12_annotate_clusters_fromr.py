#!/usr/bin/env python3
"""Annotate integrated clusters for one or all conditions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]

from utils.annotation_signature_utils import build_signatures  # noqa: E402
from utils.h5ad_compat import read_h5ad_compat  # noqa: E402
from utils.warehouse import (  # noqa: E402
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)
COARSE_EPITHELIAL_LABELS = {
    "Epithelial",
    "Basal_Epi",
    "Luminal_Epi",
    "Basal",
    "LP",
    "ML",
}

LINEAGE_LABEL_MAP = {
    "Basal": "Basal",
    "Basal_Epi": "Basal",
    "LP": "LP",
    "ML": "ML",
    "Luminal_Epi": "ML",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate integrated clusters for one condition or all"
    )
    parser.add_argument(
        "--condition",
        default="Normal",
        help=(
            "Condition under results/03_integration/integrated, "
            "or 'all'"
        ),
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition folders and exit",
    )
    parser.add_argument(
        "--input-dir",
        default="results/03_integration/integrated",
        help="Root directory with integrated condition outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/04_annotation",
        help="Directory to write annotation outputs",
    )
    parser.add_argument(
        "--cluster-col",
        default="leiden",
        help="Cluster column",
    )
    parser.add_argument(
        "--n-top-markers",
        type=int,
        default=25,
        help="Top DE markers per cluster to export",
    )
    parser.add_argument(
        "--signature-file",
        default="data/HumanBreast10X-main/Signatures/ImmuneMarkers2.txt",
        help="Optional TSV with columns CellType and Signatures",
    )
    parser.add_argument(
        "--lineage-rdata",
        default="data/HumanBreast10X-main/Signatures/Human-PosSigGenes.RData",
        help="RData with lineage signatures (Basal/LP/ML/Str)",
    )
    parser.add_argument(
        "--disable-lineage-rdata",
        action="store_true",
        help="Disable loading lineage signatures from RData",
    )
    parser.add_argument(
        "--pam50-file",
        default="data/HumanBreast10X-main/Signatures/PAM50.txt",
        help="Optional PAM50 TSV with columns Gene and Subtype",
    )
    parser.add_argument(
        "--include-pam50",
        action="store_true",
        help="Include PAM50 subtype signatures in scoring",
    )
    parser.add_argument(
        "--disable-alias-collapse",
        action="store_true",
        help="Keep raw signature names without alias collapsing",
    )
    parser.add_argument(
        "--disable-redundant-pruning",
        action="store_true",
        help="Keep overlapping signatures without deterministic pruning",
    )
    parser.add_argument(
        "--score-layer",
        choices=["X", "counts"],
        default="X",
        help="Matrix source for score_genes",
    )
    parser.add_argument(
        "--compute-tsne",
        action="store_true",
        help="Compute and export t-SNE plots for annotated outputs",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="Learning rate for t-SNE",
    )
    parser.add_argument(
        "--tsne-n-pcs",
        type=int,
        default=30,
        help="n_pcs fallback if PCA embedding is missing",
    )
    parser.add_argument(
        "--tsne-seed",
        type=int,
        default=0,
        help="Random seed for t-SNE",
    )
    parser.add_argument(
        "--previous-label-col",
        default="",
        help="Optional existing label column to compare with annotation",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional custom plot title prefix",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    parser.add_argument(
        "--uncertain-margin-threshold",
        type=float,
        default=0.10,
        help=(
            "Mark coarse compartment as Uncertain if score margin "
            "is below this"
        ),
    )
    parser.add_argument(
        "--uncertain-min-overlap",
        type=int,
        default=1,
        help=(
            "Mark coarse compartment as Uncertain if marker overlap "
            "is below this"
        ),
    )
    parser.add_argument(
        "--external-marker-files",
        nargs="*",
        default=[],
        help=(
            "Optional external marker DB files (csv/tsv/txt) with gene and "
            "cell_type-like columns"
        ),
    )
    parser.add_argument(
        "--external-marker-tissue-hint",
        default="",
        help="Optional tissue substring filter for external marker rows",
    )
    parser.add_argument(
        "--external-marker-min-score",
        type=float,
        default=0.0,
        help="Minimum external marker confidence/score to include",
    )
    parser.add_argument(
        "--external-min-overlap",
        type=int,
        default=2,
        help="Minimum marker overlaps to accept external cluster label",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


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


def list_conditions(in_root: Path) -> list[str]:
    if not in_root.exists():
        return []
    names: list[str] = []
    for child in sorted(p for p in in_root.iterdir() if p.is_dir()):
        try:
            _ = integrated_file_for(child)
        except FileNotFoundError:
            continue
        names.append(child.name)
    return names


def resolve_conditions(in_root: Path, requested: str) -> list[str]:
    available = list_conditions(in_root)
    if not available:
        raise FileNotFoundError(
            f"No condition folders with integrated files found in: {in_root}"
        )
    if requested.strip().lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise FileNotFoundError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def ensure_annotation_inputs(adata: sc.AnnData, cluster_col: str) -> None:
    if cluster_col not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in obs")
    if "X_umap" not in adata.obsm:
        raise ValueError(
            "X_umap missing. Run integration plotting step first."
        )


def maybe_write_confusion(
    adata: sc.AnnData,
    previous_label_col: str,
    out_file: Path,
) -> None:
    if not previous_label_col or previous_label_col not in adata.obs.columns:
        return
    confusion = pd.crosstab(
        adata.obs[previous_label_col].astype(str),
        adata.obs["cell_type_annot"].astype(str),
        dropna=False,
    )
    confusion.to_csv(out_file)


def normalize_gene_symbol(gene: str) -> str:
    return str(gene).strip().upper()


def pick_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    col_map = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def load_external_marker_mappings(
    marker_files: list[Path],
    tissue_hint: str,
    min_score: float,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    gene_to_celltypes: dict[str, set[str]] = {}
    audit_rows: list[dict[str, object]] = []
    tissue_hint_norm = tissue_hint.strip().lower()

    for marker_file in marker_files:
        if not marker_file.exists():
            audit_rows.append(
                {
                    "file": str(marker_file),
                    "status": "missing",
                    "rows_loaded": 0,
                    "rows_kept": 0,
                    "genes_kept": 0,
                    "cell_types_kept": 0,
                }
            )
            continue

        sep = "\t" if marker_file.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            df = pd.read_csv(marker_file, sep=sep)
        except Exception:
            audit_rows.append(
                {
                    "file": str(marker_file),
                    "status": "parse_error",
                    "rows_loaded": 0,
                    "rows_kept": 0,
                    "genes_kept": 0,
                    "cell_types_kept": 0,
                }
            )
            continue

        if df.empty:
            audit_rows.append(
                {
                    "file": str(marker_file),
                    "status": "empty",
                    "rows_loaded": 0,
                    "rows_kept": 0,
                    "genes_kept": 0,
                    "cell_types_kept": 0,
                }
            )
            continue

        gene_col = pick_first_existing_column(
            df,
            [
                "gene",
                "gene_symbol",
                "symbol",
                "marker",
                "signatures",
                "markers",
            ],
        )
        ct_col = pick_first_existing_column(
            df,
            ["cell_type", "celltype", "cell type", "CellType", "cell_name"],
        )
        score_col = pick_first_existing_column(
            df,
            ["score", "specificity", "confidence", "evidence", "marker_score"],
        )
        tissue_col = pick_first_existing_column(
            df,
            ["tissue", "organ", "source_tissue", "tumor_type"],
        )

        if gene_col is None or ct_col is None:
            audit_rows.append(
                {
                    "file": str(marker_file),
                    "status": "missing_required_columns",
                    "rows_loaded": int(len(df)),
                    "rows_kept": 0,
                    "genes_kept": 0,
                    "cell_types_kept": 0,
                }
            )
            continue

        work = df.copy()
        if tissue_hint_norm and tissue_col is not None:
            tissue_vals = work[tissue_col].fillna("").astype(str).str.lower()
            work = work.loc[
                tissue_vals.str.contains(tissue_hint_norm, na=False)
            ].copy()

        if score_col is not None:
            scores = pd.to_numeric(
                work[score_col], errors="coerce"
            ).fillna(0.0)
        else:
            scores = pd.Series(
                np.ones(len(work), dtype=float),
                index=work.index,
            )

        rows_kept = 0
        genes_local: set[str] = set()
        cell_types_local: set[str] = set()
        for idx, row in work.iterrows():
            gene = row.get(gene_col)
            cell_type = row.get(ct_col)
            if pd.isna(gene) or pd.isna(cell_type):
                continue
            score = float(scores.loc[idx])
            if score < float(min_score):
                continue
            norm_gene = normalize_gene_symbol(str(gene))
            ct = str(cell_type).strip()
            if not norm_gene or not ct:
                continue
            gene_to_celltypes.setdefault(norm_gene, set()).add(ct)
            genes_local.add(norm_gene)
            cell_types_local.add(ct)
            rows_kept += 1

        audit_rows.append(
            {
                "file": str(marker_file),
                "status": "ok",
                "rows_loaded": int(len(df)),
                "rows_kept": int(rows_kept),
                "genes_kept": int(len(genes_local)),
                "cell_types_kept": int(len(cell_types_local)),
            }
        )

    flat_map = {
        gene: sorted(list(cell_types))
        for gene, cell_types in gene_to_celltypes.items()
    }
    audit_df = pd.DataFrame(audit_rows)
    return flat_map, audit_df


def infer_external_cluster_labels(
    marker_df: pd.DataFrame,
    cluster_col: str,
    external_gene_to_celltypes: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if marker_df.empty or not external_gene_to_celltypes:
        return pd.DataFrame(
            columns=[
                cluster_col,
                "external_predicted_cell_type",
                "external_marker_overlap_n",
                "external_marker_overlap_genes",
            ]
        )

    for cluster, cluster_df in marker_df.groupby("cluster", observed=False):
        votes: dict[str, set[str]] = {}
        for gene in cluster_df["names"].astype(str).tolist():
            norm_gene = normalize_gene_symbol(gene)
            cell_types = external_gene_to_celltypes.get(norm_gene, [])
            for cell_type in cell_types:
                votes.setdefault(cell_type, set()).add(gene)

        if not votes:
            rows.append(
                {
                    cluster_col: str(cluster),
                    "external_predicted_cell_type": "",
                    "external_marker_overlap_n": 0,
                    "external_marker_overlap_genes": "",
                }
            )
            continue

        best_cell_type = ""
        best_genes: set[str] = set()
        best_count = -1
        for cell_type, genes in votes.items():
            count = len(genes)
            if count > best_count:
                best_count = count
                best_cell_type = cell_type
                best_genes = genes

        rows.append(
            {
                cluster_col: str(cluster),
                "external_predicted_cell_type": best_cell_type,
                "external_marker_overlap_n": int(max(0, best_count)),
                "external_marker_overlap_genes": ";".join(sorted(best_genes)),
            }
        )

    return pd.DataFrame(rows)


def invert_signatures(
    signatures: dict[str, list[str]],
) -> dict[str, list[str]]:
    gene_to_labels: dict[str, set[str]] = {}
    for label, genes in signatures.items():
        for gene in genes:
            norm_gene = normalize_gene_symbol(str(gene))
            if not norm_gene:
                continue
            gene_to_labels.setdefault(norm_gene, set()).add(str(label))
    return {
        gene: sorted(list(labels))
        for gene, labels in gene_to_labels.items()
    }


def build_gene_source_provenance(
    marker_df: pd.DataFrame,
    internal_gene_to_labels: dict[str, list[str]],
    external_gene_to_celltypes: dict[str, list[str]],
) -> pd.DataFrame:
    if marker_df.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "gene_normalized",
                "clusters",
                "n_clusters_marker_top",
                "internal_hit",
                "internal_labels",
                "external_hit",
                "external_cell_types",
                "source_class",
            ]
        )

    grouped = marker_df.groupby("names", observed=False)["cluster"].apply(
        lambda values: sorted({str(v) for v in values})
    )

    rows: list[dict[str, object]] = []
    for gene, clusters in grouped.items():
        gene_str = str(gene)
        norm_gene = normalize_gene_symbol(gene_str)
        internal_labels = internal_gene_to_labels.get(norm_gene, [])
        external_labels = external_gene_to_celltypes.get(norm_gene, [])
        internal_hit = len(internal_labels) > 0
        external_hit = len(external_labels) > 0

        if internal_hit and external_hit:
            source_class = "both"
        elif internal_hit:
            source_class = "internal_only"
        elif external_hit:
            source_class = "external_only"
        else:
            source_class = "none"

        rows.append(
            {
                "gene": gene_str,
                "gene_normalized": norm_gene,
                "clusters": ";".join(clusters),
                "n_clusters_marker_top": int(len(clusters)),
                "internal_hit": bool(internal_hit),
                "internal_labels": ";".join(internal_labels),
                "external_hit": bool(external_hit),
                "external_cell_types": ";".join(external_labels),
                "source_class": source_class,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["source_class", "n_clusters_marker_top", "gene"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def run_markers(
    adata: sc.AnnData,
    cluster_col: str,
    n_top_markers: int,
) -> pd.DataFrame:
    adata_de = adata.copy()
    if "counts" in adata_de.layers:
        adata_de.X = adata_de.layers["counts"].copy()
        sc.pp.normalize_total(adata_de, target_sum=1e4)
        sc.pp.log1p(adata_de)

    sc.tl.rank_genes_groups(
        adata_de,
        groupby=cluster_col,
        method="wilcoxon",
        use_raw=False,
        n_genes=max(50, n_top_markers),
    )
    df = sc.get.rank_genes_groups_df(adata_de, group=None)
    df = df.rename(columns={"group": "cluster"})
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["cluster", "pvals_adj", "logfoldchanges"])
    return df.groupby(
        "cluster",
        as_index=False,
        observed=False,
    ).head(n_top_markers)


def score_signatures(
    adata: sc.AnnData,
    signatures: dict[str, list[str]],
    score_layer: str,
) -> tuple[list[str], dict[str, int]]:
    use_layer = None
    if score_layer == "counts" and "counts" in adata.layers:
        adata.layers["_score_tmp"] = adata.layers["counts"].copy()
        use_layer = "_score_tmp"

    score_cols: list[str] = []
    genes_used: dict[str, int] = {}
    gene_pool = set(map(str, adata.var_names))

    for label, genes in signatures.items():
        present = [g for g in genes if g in gene_pool]
        genes_used[label] = len(present)
        if not present:
            continue
        score_col = f"sig_{label}"
        sc.tl.score_genes(
            adata,
            gene_list=present,
            score_name=score_col,
            use_raw=False,
            layer=use_layer,
        )
        score_cols.append(score_col)

    if "_score_tmp" in adata.layers:
        del adata.layers["_score_tmp"]

    return score_cols, genes_used


def cluster_signature_table(
    adata: sc.AnnData,
    cluster_col: str,
    score_cols: list[str],
) -> pd.DataFrame:
    if not score_cols:
        return pd.DataFrame(columns=[cluster_col])
    return (
        adata.obs[[cluster_col, *score_cols]]
        .groupby(cluster_col, observed=False)
        .mean()
        .reset_index()
    )


def infer_cluster_labels(
    score_table: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    score_cols = [c for c in score_table.columns if c.startswith("sig_")]
    if not score_cols:
        return pd.DataFrame(columns=[cluster_col, "predicted_cell_type"])

    rows: list[dict[str, object]] = []
    for _, row in score_table.iterrows():
        vals = row[score_cols].sort_values(ascending=False)
        top1 = vals.index[0]
        top2 = vals.index[1] if len(vals) > 1 else top1
        rows.append(
            {
                cluster_col: str(row[cluster_col]),
                "predicted_cell_type": top1.replace("sig_", ""),
                "score_top1": float(vals.iloc[0]),
                "score_top2": float(vals.iloc[1]) if len(vals) > 1 else np.nan,
                "score_margin": float(vals.iloc[0] - vals.iloc[1])
                if len(vals) > 1
                else np.nan,
                "top1_signature": top1,
                "top2_signature": top2,
            }
        )
    return pd.DataFrame(rows)


def marker_sanity(
    mapping_df: pd.DataFrame,
    marker_df: pd.DataFrame,
    signatures: dict[str, list[str]],
    cluster_col: str,
) -> pd.DataFrame:
    top_markers = marker_df.groupby(
        "cluster", observed=False
    )["names"].apply(list).to_dict()
    rows: list[dict[str, object]] = []
    for _, row in mapping_df.iterrows():
        cl = str(row[cluster_col])
        label = str(row["predicted_cell_type"])
        overlap = sorted(
            set(top_markers.get(cl, [])).intersection(
                signatures.get(label, [])
            )
        )
        rows.append(
            {
                cluster_col: cl,
                "predicted_cell_type": label,
                "marker_overlap_n": len(overlap),
                "marker_overlap_genes": ";".join(overlap),
            }
        )
    return pd.DataFrame(rows)


def map_major_compartment(label: str) -> str:
    if label in COARSE_EPITHELIAL_LABELS or label.startswith("PAM50_"):
        return "Epithelial"
    if label in {"Unknown", "Uncertain", ""}:
        return "Uncertain"
    return "Stromal_like"


def apply_coarse_compartment(
    mapping_df: pd.DataFrame,
    margin_threshold: float,
    min_overlap: int,
) -> pd.DataFrame:
    out = mapping_df.copy()
    out["major_compartment"] = out["predicted_cell_type"].map(
        lambda value: map_major_compartment(str(value))
    )

    low_margin = out["score_margin"].fillna(-np.inf) < margin_threshold
    low_overlap = out["marker_overlap_n"].fillna(0).astype(int) < min_overlap
    uncertain_mask = low_margin | low_overlap
    out["major_compartment"] = out["major_compartment"].where(
        ~uncertain_mask,
        "Uncertain",
    )
    out["coarse_uncertain_reason"] = np.select(
        [low_margin & low_overlap, low_margin, low_overlap],
        [
            "low_margin_and_low_overlap",
            "low_margin",
            "low_overlap",
        ],
        default="",
    )
    return out


def _label_from_signature(sig_name: object) -> str:
    if not isinstance(sig_name, str):
        return ""
    return sig_name.replace("sig_", "")


def refine_epithelial_lineage(mapping_df: pd.DataFrame) -> pd.DataFrame:
    out = mapping_df.copy()
    lineage: list[str] = []

    for _, row in out.iterrows():
        major = str(row.get("major_compartment", ""))
        if major != "Epithelial":
            lineage.append("Non_epithelial")
            continue

        candidates = [
            str(row.get("predicted_cell_type", "")),
            _label_from_signature(row.get("top1_signature", "")),
            _label_from_signature(row.get("top2_signature", "")),
        ]

        assigned = "Uncertain"
        for candidate in candidates:
            if candidate in LINEAGE_LABEL_MAP:
                assigned = LINEAGE_LABEL_MAP[candidate]
                break
        lineage.append(assigned)

    out["epithelial_lineage"] = lineage
    return out


def make_umap_plot(
    adata: sc.AnnData,
    color: str,
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    legend_fontsize = 8
    fig = sc.pl.umap(
        adata,
        color=color,
        show=False,
        frameon=False,
        return_fig=True,
        title=title,
        legend_loc="none",
        legend_fontsize=legend_fontsize,
    )
    if color in {"cell_type_annot", "SampleName"} and color in adata.obs.columns:
        categories = [
            str(v) for v in adata.obs[color].astype("category").cat.categories
        ]
        palette = adata.uns.get(f"{color}_colors", [])
        handles = []
        for idx, label in enumerate(categories):
            marker_color = palette[idx] if idx < len(palette) else "#666666"
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=marker_color,
                    markeredgecolor=marker_color,
                    markersize=4,
                    label=label,
                )
            )
        if handles:
            legend_cols = 4
            legend_rows = math.ceil(len(handles) / legend_cols)
            fig.subplots_adjust(bottom=min(0.30, 0.11 + 0.03 * legend_rows))
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=legend_cols,
                frameon=False,
                fontsize=max(6, legend_fontsize - 1),
                bbox_to_anchor=(0.5, 0.01),
            )
    save_fn = getattr(fig, "savefig", None)
    if callable(save_fn):
        save_fn(out_file, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        plt.close("all")


def make_tsne_plot(
    adata: sc.AnnData,
    color: str,
    title: str,
    out_file: Path,
    dpi: int,
) -> None:
    legend_fontsize = 8
    fig = sc.pl.tsne(
        adata,
        color=color,
        show=False,
        frameon=False,
        return_fig=True,
        title=title,
        legend_loc="none",
        legend_fontsize=legend_fontsize,
    )
    if color in {"cell_type_annot", "SampleName"} and color in adata.obs.columns:
        categories = [
            str(v) for v in adata.obs[color].astype("category").cat.categories
        ]
        palette = adata.uns.get(f"{color}_colors", [])
        handles = []
        for idx, label in enumerate(categories):
            marker_color = palette[idx] if idx < len(palette) else "#666666"
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=marker_color,
                    markeredgecolor=marker_color,
                    markersize=4,
                    label=label,
                )
            )
        if handles:
            legend_cols = 4
            legend_rows = math.ceil(len(handles) / legend_cols)
            fig.subplots_adjust(bottom=min(0.30, 0.11 + 0.03 * legend_rows))
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=legend_cols,
                frameon=False,
                fontsize=max(6, legend_fontsize - 1),
                bbox_to_anchor=(0.5, 0.01),
            )
    save_fn = getattr(fig, "savefig", None)
    if callable(save_fn):
        save_fn(out_file, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        plt.close("all")


def maybe_compute_tsne(adata: sc.AnnData, args: argparse.Namespace) -> None:
    if not args.compute_tsne:
        return
    tsne_kwargs: dict[str, Any] = {
        "perplexity": args.tsne_perplexity,
        "learning_rate": args.tsne_learning_rate,
        "random_state": args.tsne_seed,
    }
    if "X_pca" in adata.obsm:
        tsne_kwargs["use_rep"] = "X_pca"
    else:
        tsne_kwargs["n_pcs"] = args.tsne_n_pcs
    sc.tl.tsne(adata, **tsne_kwargs)


def annotate_one_condition(
    args: argparse.Namespace,
    in_root: Path,
    out_root: Path,
    sig_path: Path,
    condition: str,
) -> dict[str, object]:
    #load integrated h5ad and create output dirs
    h5ad = integrated_file_for(in_root / condition)
    out_dir = out_root / condition
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata = read_h5ad_compat(h5ad)
    ensure_annotation_inputs(adata, args.cluster_col)
    maybe_compute_tsne(adata, args)

    #run wilcoxon de markers per cluster
    marker_df = run_markers(adata, args.cluster_col, args.n_top_markers)
    marker_df.to_csv(out_dir / f"{condition}_cluster_markers_top.csv", index=False)

    #build internal cell-type signatures from file sources
    signatures = build_signatures(
        sig_path=sig_path,
        lineage_rdata_path=resolve_base(args.lineage_rdata),
        use_lineage_rdata=not args.disable_lineage_rdata,
        pam50_path=resolve_base(args.pam50_file),
        include_pam50=args.include_pam50,
        collapse_aliases=not args.disable_alias_collapse,
        prune_redundant=not args.disable_redundant_pruning,
    )
    internal_gene_to_labels = invert_signatures(signatures)

    #load optional external marker databases
    external_marker_paths = [resolve_base(p) for p in args.external_marker_files]
    external_gene_to_celltypes, external_audit_df = load_external_marker_mappings(
        marker_files=external_marker_paths,
        tissue_hint=args.external_marker_tissue_hint,
        min_score=float(args.external_marker_min_score),
    )
    if not external_audit_df.empty:
        external_audit_df.to_csv(out_dir / f"{condition}_external_marker_import_audit.csv", index=False)

    #build gene source provenance table
    provenance_df = build_gene_source_provenance(
        marker_df=marker_df,
        internal_gene_to_labels=internal_gene_to_labels,
        external_gene_to_celltypes=external_gene_to_celltypes,
    )
    provenance_df.to_csv(out_dir / f"{condition}_marker_gene_source_provenance.csv", index=False)

    #score each signature per cell and aggregate to cluster means
    score_cols, genes_used = score_signatures(adata, signatures, args.score_layer)
    score_table = cluster_signature_table(adata, args.cluster_col, score_cols)
    score_table.to_csv(out_dir / f"{condition}_cluster_signature_scores.csv", index=False)

    #infer primary cell-type label per cluster from top signature score
    mapping_df = infer_cluster_labels(score_table, args.cluster_col)
    if mapping_df.empty:
        raise ValueError("No signature scores were computed for annotation.")

    #validate labels with marker overlap sanity check
    sanity_df = marker_sanity(mapping_df, marker_df, signatures, args.cluster_col)
    mapping_df = mapping_df.merge(sanity_df, on=[args.cluster_col, "predicted_cell_type"], how="left")

    #merge external cluster label predictions
    external_cluster_df = infer_external_cluster_labels(
        marker_df=marker_df,
        cluster_col=args.cluster_col,
        external_gene_to_celltypes=external_gene_to_celltypes,
    )
    mapping_df = mapping_df.merge(external_cluster_df, on=[args.cluster_col], how="left")
    mapping_df["external_marker_overlap_n"] = mapping_df["external_marker_overlap_n"].fillna(0).astype(int)
    mapping_df["external_predicted_cell_type"] = mapping_df["external_predicted_cell_type"].fillna("").astype(str)
    mapping_df["external_label_used"] = False

    #fall back to external label for clusters with low margin or low marker overlap
    low_margin = mapping_df["score_margin"].fillna(-np.inf) < float(args.uncertain_margin_threshold)
    low_overlap = mapping_df["marker_overlap_n"].fillna(0).astype(int) < int(args.uncertain_min_overlap)
    external_good = mapping_df["external_marker_overlap_n"] >= int(args.external_min_overlap)
    has_external = mapping_df["external_predicted_cell_type"].str.len() > 0
    use_external = (low_margin | low_overlap) & external_good & has_external

    mapping_df.loc[use_external, "predicted_cell_type"] = mapping_df.loc[use_external, "external_predicted_cell_type"]
    mapping_df.loc[use_external, "external_label_used"] = True

    #apply coarse compartment and epithelial lineage refinement
    mapping_df = apply_coarse_compartment(mapping_df, margin_threshold=args.uncertain_margin_threshold, min_overlap=args.uncertain_min_overlap)
    mapping_df = refine_epithelial_lineage(mapping_df)

    #save mapping tables
    mapping_df.to_csv(out_dir / f"{condition}_cluster_to_celltype_mapping.csv", index=False)
    mapping_df[[args.cluster_col, "predicted_cell_type", "major_compartment", "coarse_uncertain_reason", "score_margin", "marker_overlap_n"]].to_csv(
        out_dir / f"{condition}_cluster_to_major_compartment_mapping.csv", index=False)
    mapping_df[[args.cluster_col, "major_compartment", "epithelial_lineage", "coarse_uncertain_reason", "score_margin", "marker_overlap_n"]].to_csv(
        out_dir / f"{condition}_cluster_to_epithelial_lineage_mapping.csv", index=False)

    #transfer cluster-level labels back to cell obs
    cluster_to_label = dict(zip(mapping_df[args.cluster_col].astype(str), mapping_df["predicted_cell_type"]))
    adata.obs["cell_type_annot"] = adata.obs[args.cluster_col].astype(str).map(cluster_to_label).fillna("Unknown")

    cluster_to_major = dict(zip(mapping_df[args.cluster_col].astype(str), mapping_df["major_compartment"]))
    adata.obs["major_compartment"] = adata.obs[args.cluster_col].astype(str).map(cluster_to_major).fillna("Uncertain")

    cluster_to_lineage = dict(zip(mapping_df[args.cluster_col].astype(str), mapping_df["epithelial_lineage"]))
    adata.obs["epithelial_lineage"] = adata.obs[args.cluster_col].astype(str).map(cluster_to_lineage).fillna("Uncertain")

    #save signature gene coverage csv
    pd.DataFrame({"signature": list(genes_used.keys()), "genes_present_n": list(genes_used.values())}).sort_values(
        ["genes_present_n", "signature"], ascending=[False, True]).to_csv(out_dir / f"{condition}_signature_gene_coverage.csv", index=False)

    #generate umap plots coloured by annotation layers
    title_prefix = args.title_prefix.strip() or condition
    make_umap_plot(adata, color="cell_type_annot", title=f"{title_prefix} | UMAP: cell_type_annot", out_file=fig_dir / f"{condition}_umap_cell_type_annot.png", dpi=args.dpi)
    make_umap_plot(adata, color="major_compartment", title=f"{title_prefix} | UMAP: major_compartment", out_file=fig_dir / f"{condition}_umap_major_compartment.png", dpi=args.dpi)
    make_umap_plot(adata, color="epithelial_lineage", title=f"{title_prefix} | UMAP: epithelial_lineage", out_file=fig_dir / f"{condition}_umap_epithelial_lineage.png", dpi=args.dpi)
    make_umap_plot(adata, color=args.cluster_col, title=f"{title_prefix} | UMAP: {args.cluster_col}", out_file=fig_dir / f"{condition}_umap_{args.cluster_col}.png", dpi=args.dpi)
    if "SampleName" in adata.obs.columns:
        make_umap_plot(adata, color="SampleName", title=f"{title_prefix} | UMAP: SampleName", out_file=fig_dir / f"{condition}_umap_sample.png", dpi=args.dpi)

    #optional tsne plots if embedding was computed
    if args.compute_tsne and "X_tsne" in adata.obsm:
        make_tsne_plot(adata, color="cell_type_annot", title=f"{title_prefix} | t-SNE: cell_type_annot", out_file=fig_dir / f"{condition}_tsne_cell_type_annot.png", dpi=args.dpi)
        make_tsne_plot(adata, color="major_compartment", title=f"{title_prefix} | t-SNE: major_compartment", out_file=fig_dir / f"{condition}_tsne_major_compartment.png", dpi=args.dpi)
        make_tsne_plot(adata, color="epithelial_lineage", title=f"{title_prefix} | t-SNE: epithelial_lineage", out_file=fig_dir / f"{condition}_tsne_epithelial_lineage.png", dpi=args.dpi)
        make_tsne_plot(adata, color=args.cluster_col, title=f"{title_prefix} | t-SNE: {args.cluster_col}", out_file=fig_dir / f"{condition}_tsne_{args.cluster_col}.png", dpi=args.dpi)
        if "SampleName" in adata.obs.columns:
            make_tsne_plot(adata, color="SampleName", title=f"{title_prefix} | t-SNE: SampleName", out_file=fig_dir / f"{condition}_tsne_sample.png", dpi=args.dpi)

    #optional confusion matrix against prior label column
    maybe_write_confusion(adata, previous_label_col=args.previous_label_col, out_file=out_dir / f"{condition}_confusion_previous_vs_annot.csv")

    #fix log1p base if stored as non-float (scanpy write compatibility)
    log1p = adata.uns.get("log1p")
    if isinstance(log1p, dict) and "base" in log1p:
        try:
            log1p["base"] = float(log1p["base"])
        except (TypeError, ValueError):
            log1p["base"] = float(np.e)

    #save annotated h5ad and per-condition summary csv
    annotated_file = out_dir / f"{condition}_annotated.h5ad"
    adata.write_h5ad(annotated_file)

    summary_row = {
        "condition": condition,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "clusters": int(adata.obs[args.cluster_col].nunique()),
        "annotated_cell_types": int(adata.obs["cell_type_annot"].nunique()),
        "major_compartments": int(adata.obs["major_compartment"].nunique()),
        "epithelial_lineages": int(adata.obs["epithelial_lineage"].nunique()),
        "integrated_input": str(h5ad),
        "annotated_output": str(annotated_file),
    }
    pd.DataFrame([summary_row]).to_csv(out_dir / f"{condition}_annotation_summary.csv", index=False)

    print(f"Annotation complete: {condition}")
    print(f"Output directory: {out_dir}")
    print(f"Annotated object: {annotated_file}")
    return summary_row


def main() -> int:
    args = parse_args()

    #resolve paths
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    sig_path = resolve_base(args.signature_file)

    #list available conditions and exit if requested
    if args.list_conditions:
        available = list_conditions(in_root)
        if not available:
            print("No conditions found")
            return 0
        print("Available conditions:")
        for name in available:
            print(f"- {name}")
        return 0

    #resolve target conditions and annotate each one
    conditions = resolve_conditions(in_root, args.condition)
    arg_hash = params_hash(vars(args))
    now = utc_now_iso()
    rows: list[dict[str, object]] = []
    warehouse_rows: list[WarehouseRecord] = []

    for condition in conditions:
        row = annotate_one_condition(args, in_root, out_root, sig_path, condition)
        rows.append(row)
        warehouse_rows.append(
            WarehouseRecord(
                input_file=str(row["integrated_input"]),
                output_file=str(row["annotated_output"]),
                script="scripts/12_annotate_clusters_fromr.py",
                date_utc=now,
                params_hash=arg_hash,
                condition=str(row["condition"]),
                stage="annotation",
            )
        )

    #write combined summary csv when multiple conditions were processed
    if len(rows) > 1:
        summary_file = out_root / "annotation_summary.csv"
        pd.DataFrame(rows).to_csv(summary_file, index=False)
        print(f"Combined summary: {summary_file}")

    #append provenance to warehouse log
    warehouse_file = append_warehouse(out_root, warehouse_rows)
    print(f"Warehouse log: {warehouse_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
