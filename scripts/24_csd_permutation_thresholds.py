#!/usr/bin/env python3
# flake8: noqa: E501
"""Compute pair-specific permutation thresholds for C/S/D edge filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from utils.h5ad_compat import read_h5ad_compat
from utils.network_utils import resolve_base, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run permutation threshold estimation for CSD scoring")
    parser.add_argument(
        "--expr-dir",
        default="results/08_pre_correlation/per_condition",
        help="Root with <condition>_pseudobulk_logcpm.h5ad",
    )
    parser.add_argument(
        "--shared-dir",
        default="results/12_csd_scoring",
        help="CSD scoring output directory from 23_csd_scoring.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/13_csd_thresholds",
        help="Output directory for permutation thresholds",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Pair folder name case__vs__control. Repeat for multiple",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Number of random permutations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=50000,
        help="Random seed",
    )
    return parser.parse_args()


def read_expr_for_genes(expr_file: Path, genes_keep: np.ndarray, fill_missing_expr: float = 0.0) -> tuple[np.ndarray, int]:
    adata = read_h5ad_compat(expr_file)
    var = adata.var_names.astype(str)
    idx_map = {g: i for i, g in enumerate(var)}

    x = adata.X
    if sparse.issparse(x):
        x = sparse.csr_matrix(x).toarray()
    x = np.asarray(x, dtype=np.float64)

    out = np.full((x.shape[0], genes_keep.size), float(fill_missing_expr), dtype=np.float64)
    present_pairs: list[tuple[int, int]] = []
    for j, g in enumerate(genes_keep):
        src = idx_map.get(str(g))
        if src is not None:
            present_pairs.append((j, int(src)))
    if present_pairs:
        dst_idx = np.array([d for d, _ in present_pairs], dtype=int)
        src_idx = np.array([s for _, s in present_pairs], dtype=int)
        out[:, dst_idx] = x[:, src_idx]

    return out, int(len(present_pairs))


def permute_gene_wise_samples(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.array(x, copy=True)
    n_samples, n_genes = out.shape
    for j in range(n_genes):
        perm = rng.permutation(n_samples)
        out[:, j] = out[perm, j]
    return out


def corr_from_expr(x: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def plot_threshold_convergence(maxima_df: pd.DataFrame, out_png: Path, pair_name: str) -> None:
    run = maxima_df["permutation"].to_numpy(dtype=int)
    c_run = maxima_df["max_C"].expanding().mean().to_numpy(dtype=float)
    s_run = maxima_df["max_S"].expanding().mean().to_numpy(dtype=float)
    d_run = maxima_df["max_D"].expanding().mean().to_numpy(dtype=float)

    #colours match edge colours used in 26_csd_visualization.py
    CSD_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.plot(run, c_run, label="C", linewidth=1.8, color=CSD_COLORS["C"])
    ax.plot(run, s_run, label="S", linewidth=1.8, color=CSD_COLORS["S"])
    ax.plot(run, d_run, label="D", linewidth=1.8, color=CSD_COLORS["D"])
    ax.set_xlabel("Permutation")
    ax.set_ylabel("Running mean of permutation maxima")
    ax.set_title(f"{pair_name}: threshold convergence")
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def run_pair(pair_dir: Path, expr_root: Path, out_root: Path, n_permutations: int, seed: int) -> dict[str, object]:
    pair_name = pair_dir.name
    if "__vs__" not in pair_name:
        raise ValueError(f"Invalid pair folder name: {pair_name}")
    case, ctrl = pair_name.split("__vs__", 1)

    #load genes to use for this pair
    genes_keep_file = pair_dir / f"{pair_name}_genes_keep.txt"
    if not genes_keep_file.exists():
        raise FileNotFoundError(f"Missing genes_keep file: {genes_keep_file}")

    genes_keep = np.array(
        [line.strip() for line in genes_keep_file.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=str,
    )
    if genes_keep.size < 2:
        raise ValueError(f"{pair_name}: too few genes for correlation/permutation")

    #load expression matrices for case and control
    case_expr_file = expr_root / f"{case}_pseudobulk_logcpm.h5ad"
    ctrl_expr_file = expr_root / f"{ctrl}_pseudobulk_logcpm.h5ad"
    if not case_expr_file.exists() or not ctrl_expr_file.exists():
        raise FileNotFoundError(f"Missing h5ad files for {pair_name}")

    x_case, n_present_case = read_expr_for_genes(case_expr_file, genes_keep)
    x_ctrl, n_present_ctrl = read_expr_for_genes(ctrl_expr_file, genes_keep)

    ii, jj = np.triu_indices(genes_keep.size, k=1)
    rng = np.random.default_rng(int(seed))

    rows: list[dict[str, object]] = []
    running_c: list[float] = []
    running_s: list[float] = []
    running_d: list[float] = []
    c_seen: list[float] = []
    s_seen: list[float] = []
    d_seen: list[float] = []

    import time as _time
    _t0 = _time.perf_counter()
    _print_every = max(1, int(n_permutations) // 10)

    #run permutations: shuffle expression per gene, recompute correlations, record maxima
    for i in range(1, int(n_permutations) + 1):
        x_case_p = permute_gene_wise_samples(x_case, rng)
        x_ctrl_p = permute_gene_wise_samples(x_ctrl, rng)

        corr_case = corr_from_expr(x_case_p)
        corr_ctrl = corr_from_expr(x_ctrl_p)

        r1 = corr_case[ii, jj]
        r2 = corr_ctrl[ii, jj]

        c_val = np.abs(r1 + r2)
        s_val = np.abs(np.abs(r1) - np.abs(r2))
        d_val = np.abs(r1) + np.abs(r2) - np.abs(r1 + r2)

        c_max = float(np.max(c_val)) if c_val.size else float("nan")
        s_max = float(np.max(s_val)) if s_val.size else float("nan")
        d_max = float(np.max(d_val)) if d_val.size else float("nan")

        c_seen.append(c_max)
        s_seen.append(s_max)
        d_seen.append(d_max)

        rows.append({"pair": pair_name, "permutation": i, "max_C": c_max, "max_S": s_max, "max_D": d_max})
        running_c.append(float(np.mean(c_seen)))
        running_s.append(float(np.mean(s_seen)))
        running_d.append(float(np.mean(d_seen)))

        if i % _print_every == 0 or i == int(n_permutations):
            elapsed = _time.perf_counter() - _t0
            rate = i / elapsed
            eta = (int(n_permutations) - i) / rate if rate > 0 else float("inf")
            print(
                f"  [{pair_name}] perm {i}/{n_permutations}  "
                f"C={running_c[-1]:.4f} S={running_s[-1]:.4f} D={running_d[-1]:.4f}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                flush=True,
            )

    thresholds = {
        "pair": pair_name,
        "case": case,
        "control": ctrl,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "threshold_C": float(np.mean(c_seen)),
        "threshold_S": float(np.mean(s_seen)),
        "threshold_D": float(np.mean(d_seen)),
        "running_mean_C": running_c,
        "running_mean_S": running_s,
        "running_mean_D": running_d,
        "genes_keep_file": str(genes_keep_file),
        "n_genes_present_in_case_expr": int(n_present_case),
        "n_genes_present_in_control_expr": int(n_present_ctrl),
        "n_genes_missing_in_case_expr": int(genes_keep.size - n_present_case),
        "n_genes_missing_in_control_expr": int(genes_keep.size - n_present_ctrl),
    }

    #save maxima csv, convergence plot, and threshold json
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    maxima_file = pair_out / f"{pair_name}_permutation_maxima.csv"
    maxima_df = pd.DataFrame(rows)
    maxima_df.to_csv(maxima_file, index=False)

    convergence_png = pair_out / f"{pair_name}_permutation_threshold_convergence.png"
    plot_threshold_convergence(maxima_df=maxima_df, out_png=convergence_png, pair_name=pair_name)

    threshold_file = pair_out / f"{pair_name}_permutation_thresholds.json"
    save_json(threshold_file, thresholds)

    print(
        f"[{pair_name}] perms={n_permutations} "
        f"tC={thresholds['threshold_C']:.4f} tS={thresholds['threshold_S']:.4f} tD={thresholds['threshold_D']:.4f}"
    )

    return {
        "pair": pair_name,
        "n_permutations": int(n_permutations),
        "threshold_C": float(thresholds["threshold_C"]),
        "threshold_S": float(thresholds["threshold_S"]),
        "threshold_D": float(thresholds["threshold_D"]),
        "maxima_file": str(maxima_file),
        "convergence_plot": str(convergence_png),
        "threshold_file": str(threshold_file),
    }


def main() -> int:
    args = parse_args()

    #resolve paths and create output dir
    expr_root = resolve_base(args.expr_dir)
    shared_root = resolve_base(args.shared_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    #discover pair dirs and optionally filter to requested subset
    pair_dirs = sorted([p for p in shared_root.iterdir() if p.is_dir() and "__vs__" in p.name])
    if args.pair:
        keep = set(args.pair)
        pair_dirs = [p for p in pair_dirs if p.name in keep]

    if not pair_dirs:
        raise ValueError("No pair directories found to process")

    #run permutation thresholds for each pair
    summaries: list[dict[str, object]] = []
    for idx, pair_dir in enumerate(pair_dirs):
        summaries.append(run_pair(
            pair_dir=pair_dir, expr_root=expr_root, out_root=out_root,
            n_permutations=int(args.n_permutations), seed=int(args.seed) + idx,
        ))

    #write combined summary csv
    pd.DataFrame(summaries).sort_values("pair").to_csv(out_root / "differential_permutation_thresholds_summary.csv", index=False)
    print(f"Done. Permutation thresholds: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
