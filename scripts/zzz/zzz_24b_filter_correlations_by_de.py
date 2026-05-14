#!/usr/bin/env python3
"""Filter already-computed correlation matrices using DE allowlists.

This script is intentionally post-correlation:
1) reads `*_pearson_corr.npz` per condition,
2) filters genes to DE allowlist intersection,
3) writes a separate filtered correlation root for downstream
    WGCNA/network prep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter per-condition correlation matrices "
            "using DE allowlists"
        )
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition correlation outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/09_correlation/"
            "zzz_pearson_de_filtered"
        ),
        help="Output root for DE-filtered correlation outputs",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name to process, or 'all'",
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
        "--require-de-filter",
        action="store_true",
        help="Fail if a DE allowlist is missing for a requested condition",
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
        "--max-export-edges",
        type=int,
        default=2_000_000,
        help="Maximum edges to write in thresholded edge file",
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


def list_condition_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def resolve_conditions(root: Path, requested: str) -> list[Path]:
    dirs = list_condition_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No condition directories in {root}")
    if requested.strip().lower() == "all":
        return dirs
    match = [d for d in dirs if d.name == requested]
    if not match:
        raise ValueError(
            f"Condition '{requested}' not found; "
            f"available={[d.name for d in dirs]}"
        )
    return match


def find_corr_npz(cond_dir: Path) -> Path:
    matches = sorted(cond_dir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cond_dir}")
    return matches[0]


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


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    de_filter_root = resolve_base(args.de_filter_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cond_dirs = resolve_conditions(in_root, args.condition)
    records: list[WarehouseRecord] = []

    for cdir in cond_dirs:
        condition = cdir.name
        npz_file = find_corr_npz(cdir)

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
                    f"{condition}: missing DE allowlist file "
                    f"in {de_filter_root}"
                )
            print(f"[{condition}] skip (no DE allowlist file)")
            continue

        payload = np.load(npz_file, allow_pickle=True)
        corr = np.asarray(payload["corr"], dtype=np.float32)
        genes = payload["genes"].astype(str)
        profiles = payload["profiles"].astype(str)

        keep = np.array([g in allow_genes for g in genes], dtype=bool)
        kept_genes = genes[keep]
        n_input = int(len(genes))
        n_kept = int(keep.sum())

        if n_kept < 2:
            print(f"[{condition}] skip (<2 genes kept after DE filter)")
            continue

        corr_f = corr[np.ix_(keep, keep)]
        np.fill_diagonal(corr_f, 1.0)

        cond_out = out_root / condition
        cond_out.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            cond_out / f"{condition}_pearson_corr.npz",
            corr=np.asarray(corr_f, dtype=np.float32),
            genes=np.asarray(kept_genes, dtype=object),
            profiles=np.asarray(profiles, dtype=object),
        )

        gene_list = kept_genes.tolist()
        top_df, edges_df = build_pair_exports(
            corr=corr_f,
            genes=gene_list,
            top_n=args.top_n_pairs,
            min_abs_r=args.min_abs_r,
            max_edges=args.max_export_edges,
        )
        stats = summarize_corr(corr_f)
        corr_offdiag = corr_f[np.triu_indices(corr_f.shape[0], k=1)]
        corr_offdiag_abs = np.abs(corr_offdiag)

        pd.DataFrame({"gene": gene_list}).to_csv(
            cond_out / f"{condition}_genes.csv", index=False
        )
        pd.DataFrame({"profile_id": profiles}).to_csv(
            cond_out / f"{condition}_profiles.csv", index=False
        )
        top_df.to_csv(cond_out / f"{condition}_top_pairs.csv", index=False)
        edges_df.to_csv(
            cond_out / f"{condition}_edges_abs_ge_{args.min_abs_r:.2f}.csv",
            index=False,
        )

        summary_file = cond_out / f"{condition}_corr_summary.csv"
        pd.DataFrame(
            [
                {
                    "condition": condition,
                    "method": "pearson",
                    "n_profiles": int(len(profiles)),
                    "n_genes_input": n_input,
                    "n_genes_used": n_kept,
                    "de_filter_enabled": True,
                    "de_filter_file": str(allow_file),
                    "de_allowlist_genes": int(len(allow_genes)),
                    "n_genes_kept_by_de_filter": n_kept,
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
        ).to_csv(summary_file, index=False)

        records.append(
            WarehouseRecord(
                input_file=str(npz_file),
                output_file=str(summary_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=condition,
                stage="08b_postcorr_de_filter",
            )
        )

        print(
            f"[{condition}] genes: {n_input} -> {n_kept} | "
            f"edges(|r|>={args.min_abs_r}): {edges_df.shape[0]}"
        )

    append_warehouse(out_root, records)
    print(f"Done. Post-correlation DE-filtered outputs: {out_root}")


if __name__ == "__main__":
    main()
