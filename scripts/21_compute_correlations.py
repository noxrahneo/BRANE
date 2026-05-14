#!/usr/bin/env python3
"""Compute per-condition gene-gene correlations from normalized pseudobulk.

Inputs are expected from:
`results/08_pre_correlation/per_condition/*_pseudobulk_logcpm.h5ad`

For each condition, this script writes:
- compressed correlation matrix (`*_pearson_corr.npz`)
- condition gene and profile index files
- summary statistics (`*_corr_summary.csv`)
- top absolute-correlation pairs (`*_top_pairs.csv`)
- thresholded edge list (`*_edges_abs_ge_*.csv`)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-condition Pearson correlation matrices"
    )
    parser.add_argument(
        "--input-dir",
        default="results/08_pre_correlation/per_condition",
        help="Directory containing *_pseudobulk_logcpm.h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/09_correlation/pearson",
        help="Output directory for correlation artifacts",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name to process, or 'all'",
    )
    parser.add_argument(
        "--method",
        choices=["pearson"],
        default="pearson",
        help="Correlation method (currently only pearson)",
    )
    parser.add_argument(
        "--min-abs-r",
        type=float,
        default=0.5,
        help="Absolute correlation threshold for edge export",
    )
    parser.add_argument(
        "--top-n-pairs",
        type=int,
        default=2000,
        help="Number of top absolute-correlation gene pairs to export",
    )
    parser.add_argument(
        "--drop-zero-variance-genes",
        action="store_true",
        help=(
            "Drop genes with zero variance across profiles "
            "before correlation"
        ),
    )
    parser.add_argument(
        "--max-export-edges",
        type=int,
        default=2_000_000,
        help="Maximum edges to write in thresholded edge file",
    )
    parser.add_argument(
        "--de-filter-dir",
        default=(
            "results/10_deg_ttest/"
            "zzz_de_gene_filters/per_condition"
        ),
        help="Directory containing <condition>_de_gene_allowlist.csv files",
    )
    parser.add_argument(
        "--use-de-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter genes to DE allowlist before correlation",
    )
    parser.add_argument(
        "--require-de-filter",
        action="store_true",
        help=(
            "Fail if DE allowlist file is missing for a condition "
            "when DE filtering is enabled"
        ),
    )
    parser.add_argument(
        "--de-min-hit-count",
        type=int,
        default=1,
        help="Minimum hit_count in DE allowlist file to keep a gene",
    )
    parser.add_argument(
        "--de-min-abs-log2fc",
        type=float,
        default=1.0,
        help="Minimum max_abs_log2fc in DE allowlist file to keep a gene",
    )
    parser.add_argument(
        "--de-max-padj",
        type=float,
        default=0.05,
        help="Maximum min_padj in DE allowlist file to keep a gene",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def condition_name_from_file(h5ad_file: Path) -> str:
    suffix = "_pseudobulk_logcpm.h5ad"
    name = h5ad_file.name
    return name[:-len(suffix)] if name.endswith(suffix) else h5ad_file.stem


def list_condition_files(in_root: Path) -> list[Path]:
    if not in_root.exists():
        return []
    return sorted(in_root.glob("*_pseudobulk_logcpm.h5ad"))


def resolve_requested_files(in_root: Path, requested: str) -> list[Path]:
    files = list_condition_files(in_root)
    if not files:
        raise FileNotFoundError(
            f"No *_pseudobulk_logcpm.h5ad files in {in_root}"
        )

    if requested.strip().lower() == "all":
        return files

    target = requested.strip()
    matched = [f for f in files if condition_name_from_file(f) == target]
    if not matched:
        available = [condition_name_from_file(f) for f in files]
        raise ValueError(
            f"Condition '{target}' not found. Available: {available}"
        )
    return matched


def compute_pearson_corr(x: np.ndarray) -> np.ndarray:
    """Compute a Pearson correlation matrix between columns of ``x``.

    Notes
    -----
    Uses SciPy correlation distance to derive Pearson correlation:
    ``corr = 1 - correlation_distance``.
    Variables are columns (genes) and rows are
    observations (pseudobulk profiles).

    We cast to float32 for storage efficiency and sanitize NaN/Inf values
    (e.g., from zero-variance genes) to 0.0, then enforce a 1.0 diagonal.
    """
    # scipy.spatial.distance.pdist(..., metric="correlation") returns
    # 1 - Pearson correlation between column vectors after centering.
    dist = pdist(x.T, metric="correlation")
    corr = 1.0 - squareform(dist)
    corr = np.asarray(corr, dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def summarize_corr(corr: np.ndarray) -> dict[str, float]:
    tri = np.triu_indices(corr.shape[0], k=1)
    vals = corr[tri]
    abs_vals = np.abs(vals)
    if vals.size == 0:
        return {
            "offdiag_mean": float("nan"),
            "offdiag_median": float("nan"),
            "offdiag_abs_q90": float("nan"),
            "offdiag_abs_q95": float("nan"),
            "offdiag_abs_max": float("nan"),
        }
    return {
        "offdiag_mean": float(np.mean(vals)),
        "offdiag_median": float(np.median(vals)),
        "offdiag_abs_q90": float(np.quantile(abs_vals, 0.90)),
        "offdiag_abs_q95": float(np.quantile(abs_vals, 0.95)),
        "offdiag_abs_max": float(np.max(abs_vals)),
    }


def build_pair_exports(
    corr: np.ndarray,
    genes: list[str],
    top_n: int,
    min_abs_r: float,
    max_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    vals = corr[tri_i, tri_j]
    abs_vals = np.abs(vals)

    if vals.size == 0:
        empty = pd.DataFrame(columns=["gene_a", "gene_b", "r", "abs_r"])
        return empty, empty

    top_n = max(0, min(int(top_n), vals.size))
    if top_n > 0:
        top_idx = np.argpartition(abs_vals, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(abs_vals[top_idx])[::-1]]
        top_df = pd.DataFrame(
            {
                "gene_a": [genes[tri_i[k]] for k in top_idx],
                "gene_b": [genes[tri_j[k]] for k in top_idx],
                "r": vals[top_idx],
                "abs_r": abs_vals[top_idx],
            }
        )
    else:
        top_df = pd.DataFrame(columns=["gene_a", "gene_b", "r", "abs_r"])

    edge_idx = np.where(abs_vals >= float(min_abs_r))[0]
    if edge_idx.size > int(max_edges):
        edge_idx = edge_idx[
            np.argsort(abs_vals[edge_idx])[::-1][: int(max_edges)]
        ]
    edges_df = pd.DataFrame(
        {
            "gene_a": [genes[tri_i[k]] for k in edge_idx],
            "gene_b": [genes[tri_j[k]] for k in edge_idx],
            "r": vals[edge_idx],
            "abs_r": abs_vals[edge_idx],
        }
    )
    if not edges_df.empty:
        edges_df = edges_df.sort_values(
            "abs_r", ascending=False
        ).reset_index(drop=True)

    return top_df, edges_df


def sparsity_fraction(arr: np.ndarray, eps: float = 1e-12) -> float:
    flat = np.asarray(arr)
    if flat.size == 0:
        return float("nan")
    return float(np.mean(np.abs(flat) <= float(eps)))


def load_de_allowlist(
    de_filter_root: Path,
    condition: str,
    min_hit_count: int,
    min_abs_log2fc: float,
    max_padj: float,
) -> tuple[set[str], Path | None]:
    allow_file = de_filter_root / f"{condition}_de_gene_allowlist.csv"
    if not allow_file.exists():
        return set(), None

    df = pd.read_csv(allow_file)
    if df.empty or "gene" not in df.columns:
        return set(), allow_file

    work = df.copy()
    if "hit_count" in work.columns:
        work["hit_count"] = pd.to_numeric(work["hit_count"], errors="coerce")
        work = work[work["hit_count"].ge(float(min_hit_count))]
    if "max_abs_log2fc" in work.columns:
        work["max_abs_log2fc"] = pd.to_numeric(
            work["max_abs_log2fc"],
            errors="coerce",
        )
        work = work[work["max_abs_log2fc"].ge(float(min_abs_log2fc))]
    if "min_padj" in work.columns:
        work["min_padj"] = pd.to_numeric(work["min_padj"], errors="coerce")
        work = work[work["min_padj"].le(float(max_padj))]

    genes = set(work["gene"].astype(str).tolist())
    return genes, allow_file


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    de_filter_root = resolve_base(args.de_filter_dir)

    print(
        "Method: Pearson correlation "
        "(SciPy correlation distance -> r = 1 - d)"
    )

    files = resolve_requested_files(in_root, args.condition)
    records: list[WarehouseRecord] = []

    for h5ad_file in files:
        condition = condition_name_from_file(h5ad_file)
        cond_dir = out_root / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        pdata = ad.read_h5ad(h5ad_file)
        x = np.asarray(pdata.X, dtype=np.float64)

        n_profiles_in, n_genes_in = x.shape
        genes = pdata.var_names.astype(str).tolist()
        profiles = pdata.obs_names.astype(str).tolist()

        input_nonfinite = int(np.size(x) - np.isfinite(x).sum())
        input_neg = int(np.sum(x < 0))
        input_zero_frac = sparsity_fraction(x)

        de_filter_file_used = ""
        n_de_allowlist_total = 0
        n_de_genes_kept = 0
        if args.use_de_filter:
            allow_genes, allow_file = load_de_allowlist(
                de_filter_root=de_filter_root,
                condition=condition,
                min_hit_count=int(args.de_min_hit_count),
                min_abs_log2fc=float(args.de_min_abs_log2fc),
                max_padj=float(args.de_max_padj),
            )
            if allow_file is None:
                if args.require_de_filter:
                    raise FileNotFoundError(
                        f"{condition}: missing DE allowlist "
                        f"file in {de_filter_root}"
                    )
            else:
                de_filter_file_used = str(allow_file)
                n_de_allowlist_total = int(len(allow_genes))
                if allow_genes:
                    keep = np.array(
                        [g in allow_genes for g in genes],
                        dtype=bool,
                    )
                    n_de_genes_kept = int(keep.sum())
                    x = x[:, keep]
                    genes = [g for g, k in zip(genes, keep) if k]
                elif args.require_de_filter:
                    raise ValueError(
                        f"{condition}: DE allowlist file exists "
                        "but yielded 0 genes after thresholds"
                    )

        zero_var_removed = 0
        if args.drop_zero_variance_genes:
            std = x.std(axis=0)
            keep = std > 0
            zero_var_removed = int((~keep).sum())
            x = x[:, keep]
            genes = [g for g, k in zip(genes, keep) if k]

        if x.shape[1] < 2:
            raise ValueError(
                f"{condition}: need >=2 genes after filtering, "
                f"got {x.shape[1]}"
            )
        if x.shape[0] < 3:
            raise ValueError(
                f"{condition}: need >=3 profiles for stable "
                f"correlation, got {x.shape[0]}"
            )

        corr = compute_pearson_corr(x)
        corr_offdiag = corr[np.triu_indices(corr.shape[0], k=1)]
        corr_offdiag_abs = np.abs(corr_offdiag)
        top_df, edges_df = build_pair_exports(
            corr=corr,
            genes=genes,
            top_n=args.top_n_pairs,
            min_abs_r=args.min_abs_r,
            max_edges=args.max_export_edges,
        )
        stats = summarize_corr(corr)

        np.savez_compressed(
            cond_dir / f"{condition}_pearson_corr.npz",
            corr=corr,
            genes=np.array(genes, dtype=object),
            profiles=np.array(profiles, dtype=object),
        )
        pd.DataFrame({"gene": genes}).to_csv(
            cond_dir / f"{condition}_genes.csv", index=False
        )
        pd.DataFrame({"profile_id": profiles}).to_csv(
            cond_dir / f"{condition}_profiles.csv", index=False
        )
        top_df.to_csv(cond_dir / f"{condition}_top_pairs.csv", index=False)
        edges_df.to_csv(
            cond_dir / f"{condition}_edges_abs_ge_{args.min_abs_r:.2f}.csv",
            index=False,
        )

        summary_df = pd.DataFrame(
            [
                {
                    "condition": condition,
                    "method": args.method,
                    "n_profiles": int(x.shape[0]),
                    "n_genes_input": int(n_genes_in),
                    "n_genes_used": int(x.shape[1]),
                    "de_filter_enabled": bool(args.use_de_filter),
                    "de_filter_file": de_filter_file_used,
                    "de_allowlist_genes": int(n_de_allowlist_total),
                    "n_genes_kept_by_de_filter": int(n_de_genes_kept),
                    "n_genes_zero_var_removed": int(zero_var_removed),
                    "input_nonfinite_values": int(input_nonfinite),
                    "input_negative_values": int(input_neg),
                    "input_zero_fraction": float(input_zero_frac),
                    "corr_abs_lt_0p1": (
                        float(np.mean(corr_offdiag_abs < 0.1))
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                    "corr_abs_lt_0p3": (
                        float(np.mean(corr_offdiag_abs < 0.3))
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                    "corr_abs_lt_0p5": (
                        float(np.mean(corr_offdiag_abs < 0.5))
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                    "min_abs_r_threshold": float(args.min_abs_r),
                    "n_edges_exported": int(edges_df.shape[0]),
                    "n_top_pairs_exported": int(top_df.shape[0]),
                    **stats,
                }
            ]
        )
        summary_file = cond_dir / f"{condition}_corr_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        diagnostics_file = (
            cond_dir / f"{condition}_correlation_sparsity_diagnostics.csv"
        )
        pd.DataFrame(
            [
                {
                    "condition": condition,
                    "n_profiles": int(x.shape[0]),
                    "n_genes_used": int(x.shape[1]),
                    "input_zero_fraction": float(input_zero_frac),
                    "corr_abs_lt_0p1": (
                        float(np.mean(corr_offdiag_abs < 0.1))
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                    "corr_abs_lt_0p3": (
                        float(np.mean(corr_offdiag_abs < 0.3))
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                    "corr_abs_lt_0p5": (
                        float(np.mean(corr_offdiag_abs < 0.5))
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                    "corr_abs_ge_export_threshold": (
                        float(
                            np.mean(
                                corr_offdiag_abs
                                >= float(args.min_abs_r)
                            )
                        )
                        if corr_offdiag_abs.size
                        else float("nan")
                    ),
                }
            ]
        ).to_csv(diagnostics_file, index=False)

        records.append(
            WarehouseRecord(
                input_file=str(h5ad_file),
                output_file=str(summary_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=condition,
                stage="08_correlation_pearson",
            )
        )

        print(
            f"[{condition}] profiles={x.shape[0]}, genes={x.shape[1]}, "
            f"edges(|r|>={args.min_abs_r})={edges_df.shape[0]}"
        )

    append_warehouse(out_root, records)
    print(f"Done. Correlation outputs: {out_root}")


if __name__ == "__main__":
    main()
