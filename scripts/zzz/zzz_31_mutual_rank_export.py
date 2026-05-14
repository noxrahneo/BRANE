#!/usr/bin/env python3
"""Export mutual-rank (MR) co-expression tables from Pearson correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-condition mutual-rank exports from *_pearson_corr.npz"
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root containing per-condition correlation outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_network/zzz_13_mutual_rank",
        help="Root output directory for mutual-rank exports",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--max-mr",
        type=float,
        default=50.0,
        help="MR threshold for network export (keep MR <= threshold)",
    )
    parser.add_argument(
        "--min-abs-r",
        type=float,
        default=0.0,
        help="Optional |r| filter for network export",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=50000,
        help="Maximum edges to keep in thresholded MR network export",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=200000,
        help="Maximum rows for full MR pair export (sorted by MR asc)",
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
        raise FileNotFoundError(f"No condition directories found in {root}")
    if requested.strip().lower() == "all":
        return dirs
    match = [d for d in dirs if d.name == requested]
    if not match:
        available = [d.name for d in dirs]
        raise ValueError(f"Condition '{requested}' not found. Available: {available}")
    return match


def resolve_corr_file(cond_dir: Path) -> Path:
    files = sorted(cond_dir.glob("*_pearson_corr.npz"))
    if not files:
        raise FileNotFoundError(f"No *_pearson_corr.npz found in {cond_dir}")
    return files[-1]


def rank_matrix_from_corr(corr: np.ndarray) -> np.ndarray:
    n = corr.shape[0]
    work = np.asarray(corr, dtype=np.float64).copy()
    np.fill_diagonal(work, -np.inf)

    order = np.argsort(-work, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)

    row_idx = np.arange(n, dtype=np.int32)[:, None]
    col_rank = np.arange(1, n + 1, dtype=np.int32)[None, :]
    ranks[row_idx, order] = col_rank
    np.fill_diagonal(ranks, 0)
    return ranks


def build_pair_tables(
    corr: np.ndarray,
    genes: np.ndarray,
    ranks: np.ndarray,
    top_pairs: int,
    max_mr: float,
    min_abs_r: float,
    max_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    r_vals = corr[tri_i, tri_j].astype(np.float64)
    abs_r = np.abs(r_vals)

    rank_ab = ranks[tri_i, tri_j].astype(np.float64)
    rank_ba = ranks[tri_j, tri_i].astype(np.float64)
    mr = np.sqrt(rank_ab * rank_ba)

    df = pd.DataFrame(
        {
            "gene_a": genes[tri_i].astype(str),
            "gene_b": genes[tri_j].astype(str),
            "r": r_vals,
            "abs_r": abs_r,
            "rank_a_to_b": rank_ab.astype(np.int32),
            "rank_b_to_a": rank_ba.astype(np.int32),
            "mr": mr,
        }
    )
    df = df.sort_values(["mr", "abs_r"], ascending=[True, False]).reset_index(drop=True)

    top_n = max(0, min(int(top_pairs), int(df.shape[0])))
    top_df = df.head(top_n).copy()

    network_df = df[(df["mr"] <= float(max_mr)) & (df["abs_r"] >= float(min_abs_r))].copy()
    if network_df.shape[0] > int(max_edges):
        network_df = network_df.head(int(max_edges)).copy()

    stats = {
        "n_pairs_total": float(df.shape[0]),
        "n_pairs_exported": float(top_df.shape[0]),
        "n_network_edges": float(network_df.shape[0]),
        "mr_min": float(np.nanmin(mr)) if mr.size else float("nan"),
        "mr_q10": float(np.nanquantile(mr, 0.10)) if mr.size else float("nan"),
        "mr_q50": float(np.nanquantile(mr, 0.50)) if mr.size else float("nan"),
        "mr_q90": float(np.nanquantile(mr, 0.90)) if mr.size else float("nan"),
        "mr_max": float(np.nanmax(mr)) if mr.size else float("nan"),
        "r_abs_q90": float(np.nanquantile(abs_r, 0.90)) if abs_r.size else float("nan"),
        "r_abs_q95": float(np.nanquantile(abs_r, 0.95)) if abs_r.size else float("nan"),
    }
    return top_df, network_df, stats


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cond_dirs = resolve_conditions(in_root, args.condition)
    records: list[WarehouseRecord] = []

    for cond_dir in cond_dirs:
        condition = cond_dir.name
        cond_out = out_root / condition
        cond_out.mkdir(parents=True, exist_ok=True)

        corr_file = resolve_corr_file(cond_dir)
        payload = np.load(corr_file, allow_pickle=True)
        corr = np.asarray(payload["corr"], dtype=np.float64)
        genes = payload["genes"].astype(str)

        if corr.shape[0] != corr.shape[1]:
            raise ValueError(f"{condition}: correlation matrix must be square")
        if corr.shape[0] != genes.shape[0]:
            raise ValueError(f"{condition}: gene count and corr dimension mismatch")

        ranks = rank_matrix_from_corr(corr)
        top_df, network_df, stats = build_pair_tables(
            corr=corr,
            genes=genes,
            ranks=ranks,
            top_pairs=int(args.top_pairs),
            max_mr=float(args.max_mr),
            min_abs_r=float(args.min_abs_r),
            max_edges=int(args.max_edges),
        )

        top_file = cond_out / f"{condition}_mr_pairs.csv"
        net_file = cond_out / (
            f"{condition}_mr_network_mr_le_{args.max_mr:g}_absr_ge_{args.min_abs_r:g}.csv"
        )
        summary_file = cond_out / f"{condition}_mr_summary.csv"

        top_df.to_csv(top_file, index=False)
        network_df.to_csv(net_file, index=False)

        pd.DataFrame(
            [
                {
                    "condition": condition,
                    "input_corr_file": str(corr_file),
                    "n_genes": int(genes.shape[0]),
                    "top_pairs": int(args.top_pairs),
                    "max_mr": float(args.max_mr),
                    "min_abs_r": float(args.min_abs_r),
                    "max_edges": int(args.max_edges),
                    **stats,
                    "mr_pairs_file": str(top_file),
                    "mr_network_file": str(net_file),
                }
            ]
        ).to_csv(summary_file, index=False)

        records.append(
            WarehouseRecord(
                input_file=str(corr_file),
                output_file=str(summary_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=condition,
                stage="08i_mutual_rank_export",
            )
        )

        print(
            f"[{condition}] genes={genes.shape[0]}, "
            f"mr_pairs={top_df.shape[0]}, mr_network_edges={network_df.shape[0]}"
        )

    append_warehouse(out_root, records)
    print(f"Done. MR outputs: {out_root}")


if __name__ == "__main__":
    main()
