#!/usr/bin/env python3
"""Generate QC/diagnostic plots for per-condition correlation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
        description="Create correlation QC plots from Pearson outputs"
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root directory with per-condition correlation outputs",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition to process, or 'all'",
    )
    parser.add_argument(
        "--max-sample-pairs",
        type=int,
        default=500_000,
        help="Max off-diagonal pairs to sample for distribution plots",
    )
    parser.add_argument(
        "--heatmap-top-genes",
        type=int,
        default=60,
        help="Top genes by absolute connectivity to include in heatmap",
    )
    parser.add_argument(
        "--heatmap-top-list",
        default="",
        help=(
            "Comma-separated list of top-N heatmaps to generate "
            "(e.g. '10,20,50,100'). If set, overrides --heatmap-top-genes."
        ),
    )
    parser.add_argument(
        "--heatmap-label-fontsize",
        type=float,
        default=14.0,
        help="Base font size for heatmap gene labels",
    )
    parser.add_argument(
        "--heatmap-title-fontsize",
        type=float,
        default=20.0,
        help="Font size for heatmap title",
    )
    parser.add_argument(
        "--normalized-h5ad-dir",
        default="results/08_pre_correlation/per_condition",
        help="Directory with per-condition *_pseudobulk_logcpm.h5ad files",
    )
    parser.add_argument(
        "--random-heatmap-size",
        type=int,
        default=100,
        help="Number of random genes for unbiased random heatmap",
    )
    parser.add_argument(
        "--variance-heatmap-size",
        type=int,
        default=100,
        help="Number of top-variance genes for variance-based heatmap",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Random seed for random-gene heatmap selection",
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
        raise ValueError(
            f"Condition '{requested}' not found; available={[d.name for d in dirs]}"
        )
    return match


def sample_offdiag(corr: np.ndarray, max_pairs: int) -> np.ndarray:
    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    vals = corr[tri_i, tri_j]
    if vals.size <= max_pairs:
        return vals
    rng = np.random.default_rng(7)
    idx = rng.choice(vals.size, size=max_pairs, replace=False)
    return vals[idx]


def parse_top_list(text: str, fallback_top: int) -> list[int]:
    raw = text.strip()
    if not raw:
        return [int(fallback_top)]
    values: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        n = int(p)
        if n > 1:
            values.append(n)
    if not values:
        return [int(fallback_top)]
    return sorted(set(values))


def plot_distribution(vals: np.ndarray, out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(vals, bins=80, color="#4C78A8", alpha=0.85)
    axes[0].set_title("Pearson r distribution (off-diagonal)")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("Number of gene pairs")

    abs_vals = np.abs(vals)
    axes[1].hist(abs_vals, bins=80, color="#F58518", alpha=0.85)
    axes[1].set_title("|r| distribution (off-diagonal)")
    axes[1].set_xlabel("|r|")
    axes[1].set_ylabel("Number of gene pairs")

    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    corr: np.ndarray,
    genes: list[str],
    top_n: int,
    out_file: Path,
    label_fontsize: float,
    title_fontsize: float,
) -> None:
    n = min(int(top_n), corr.shape[0])
    if n < 2:
        return
    abs_conn = np.mean(np.abs(corr - np.eye(corr.shape[0], dtype=corr.dtype)), axis=1)
    idx = np.argsort(abs_conn)[-n:]
    sub = corr[np.ix_(idx, idx)]
    sub_genes = [genes[i] for i in idx]

    # Dynamic canvas sizing for readability on dense gene labels.
    side = float(np.clip(0.30 * n + 14.0, 24.0, 56.0))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"Top-{n} genes by mean |r|", fontsize=title_fontsize)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = float(np.clip(label_fontsize - 0.02 * n, 10.0, label_fontsize))
    ax.set_xticklabels(sub_genes, fontsize=fs, rotation=90)
    ax.set_yticklabels(sub_genes, fontsize=fs)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    cb.ax.tick_params(labelsize=max(10, fs))
    fig.tight_layout()
    fig.savefig(out_file, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_from_indices(
    corr: np.ndarray,
    genes: list[str],
    idx: np.ndarray,
    out_file: Path,
    title: str,
    label_fontsize: float,
    title_fontsize: float,
) -> None:
    if idx.size < 2:
        return
    sub = corr[np.ix_(idx, idx)]
    sub_genes = [genes[i] for i in idx]
    n = len(sub_genes)

    side = float(np.clip(0.30 * n + 14.0, 24.0, 56.0))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = float(np.clip(label_fontsize - 0.02 * n, 10.0, label_fontsize))
    ax.set_xticklabels(sub_genes, fontsize=fs, rotation=90)
    ax.set_yticklabels(sub_genes, fontsize=fs)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    cb.ax.tick_params(labelsize=max(10, fs))
    fig.tight_layout()
    fig.savefig(out_file, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_degree_from_edges(edges_df: pd.DataFrame, out_file: Path) -> None:
    if edges_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No edges at threshold", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    deg = pd.concat([edges_df["gene_a"], edges_df["gene_b"]]).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(deg.values, bins=60, color="#54A24B", alpha=0.85)
    ax.set_title("Degree distribution from thresholded edges")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of genes")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_panel(summary: pd.Series, out_file: Path) -> None:
    lines = [
        f"Condition: {summary.get('condition', '')}",
        f"Profiles: {int(summary.get('n_profiles', np.nan))}",
        f"Genes used: {int(summary.get('n_genes_used', np.nan))}",
        f"Edges exported: {int(summary.get('n_edges_exported', np.nan))}",
        f"offdiag mean: {summary.get('offdiag_mean', np.nan):.4f}",
        f"offdiag median: {summary.get('offdiag_median', np.nan):.4f}",
        f"|r| q90: {summary.get('offdiag_abs_q90', np.nan):.4f}",
        f"|r| q95: {summary.get('offdiag_abs_q95', np.nan):.4f}",
        f"|r| max: {summary.get('offdiag_abs_max', np.nan):.4f}",
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cross_condition_summary(summary_df: pd.DataFrame, out_file: Path) -> None:
    if summary_df.empty:
        return
    df = summary_df.copy()
    df = df.sort_values("offdiag_abs_q95", ascending=False)
    x = np.arange(df.shape[0])
    labels = df["condition"].astype(str).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    axes[0].bar(x, df["offdiag_abs_q95"].values, color="#E45756", alpha=0.85, width=0.7, align="center")
    axes[0].set_title("|r| q95 by condition")
    axes[0].set_ylabel("|r| q95")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="center")

    axes[1].bar(x, df["n_edges_exported"].values, color="#72B7B2", alpha=0.85, width=0.7, align="center")
    axes[1].set_title("Edges (|r| threshold) by condition")
    axes[1].set_ylabel("# edges")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="center")

    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    cond_dirs = resolve_conditions(in_root, args.condition)
    top_list = parse_top_list(args.heatmap_top_list, args.heatmap_top_genes)
    norm_root = resolve_base(args.normalized_h5ad_dir)

    records: list[WarehouseRecord] = []
    all_summary: list[pd.Series] = []

    for cond_ix, cdir in enumerate(cond_dirs):
        condition = cdir.name
        summary_file = cdir / f"{condition}_corr_summary.csv"
        corr_file = cdir / f"{condition}_pearson_corr.npz"

        if not summary_file.exists() or not corr_file.exists():
            raise FileNotFoundError(
                f"Missing required files for {condition}: {summary_file} or {corr_file}"
            )

        summary = pd.read_csv(summary_file).iloc[0]
        all_summary.append(summary)

        data = np.load(corr_file, allow_pickle=True)
        corr = data["corr"]
        genes = [str(g) for g in data["genes"].tolist()]

        edges_glob = sorted(cdir.glob(f"{condition}_edges_abs_ge_*.csv"))
        edges_df = pd.read_csv(edges_glob[0]) if edges_glob else pd.DataFrame()

        vals = sample_offdiag(corr, args.max_sample_pairs)

        fig_dir = cdir / "figures"
        heat_dir = fig_dir / "heatmaps"
        fig_dir.mkdir(parents=True, exist_ok=True)
        heat_dir.mkdir(parents=True, exist_ok=True)
        dist_file = fig_dir / f"{condition}_corr_distribution.png"
        deg_file = fig_dir / f"{condition}_edge_degree_distribution.png"
        panel_file = fig_dir / f"{condition}_corr_qc_panel.png"

        plot_distribution(vals, dist_file)
        for top_n in top_list:
            heat_file = heat_dir / f"{condition}_corr_top{top_n}_heatmap.png"
            plot_heatmap(
                corr,
                genes,
                top_n,
                heat_file,
                label_fontsize=args.heatmap_label_fontsize,
                title_fontsize=args.heatmap_title_fontsize,
            )

        # Unbiased random-gene heatmap.
        random_n = min(int(args.random_heatmap_size), corr.shape[0])
        if random_n >= 2:
            rng = np.random.default_rng(args.random_seed + cond_ix)
            idx_rand = np.sort(rng.choice(corr.shape[0], size=random_n, replace=False))
            rand_file = heat_dir / f"{condition}_corr_random{random_n}_heatmap.png"
            plot_heatmap_from_indices(
                corr=corr,
                genes=genes,
                idx=idx_rand,
                out_file=rand_file,
                title=f"Random {random_n} genes (unbiased view)",
                label_fontsize=args.heatmap_label_fontsize,
                title_fontsize=args.heatmap_title_fontsize,
            )

        # Top-variance gene heatmap (from normalized per-condition matrix).
        var_n = int(args.variance_heatmap_size)
        h5ad_file = norm_root / f"{condition}_pseudobulk_logcpm.h5ad"
        if h5ad_file.exists() and var_n >= 2:
            try:
                import anndata as ad

                pdata = ad.read_h5ad(h5ad_file)
                x = np.asarray(pdata.X, dtype=np.float64)
                var_genes = pdata.var_names.astype(str).tolist()
                if x.shape[1] == corr.shape[0] and var_genes == genes:
                    v = np.var(x, axis=0)
                    idx_var = np.argsort(v)[-min(var_n, v.shape[0]) :]
                    var_file = heat_dir / f"{condition}_corr_topvar{len(idx_var)}_heatmap.png"
                    plot_heatmap_from_indices(
                        corr=corr,
                        genes=genes,
                        idx=idx_var,
                        out_file=var_file,
                        title=f"Top {len(idx_var)} genes by variance",
                        label_fontsize=args.heatmap_label_fontsize,
                        title_fontsize=args.heatmap_title_fontsize,
                    )
                else:
                    print(
                        f"[warn] {condition}: variance heatmap skipped due to gene order/shape mismatch"
                    )
            except Exception as exc:
                print(f"[warn] {condition}: variance heatmap skipped ({exc})")
        plot_degree_from_edges(edges_df, deg_file)
        plot_summary_panel(summary, panel_file)

        records.append(
            WarehouseRecord(
                input_file=str(corr_file),
                output_file=str(panel_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=condition,
                stage="08_correlation_qc",
            )
        )

        print(f"[{condition}] wrote correlation QC figures to {fig_dir}")

    if all_summary:
        cross_df = pd.DataFrame(all_summary)
        cross_file = in_root / "correlation_qc_cross_condition.png"
        plot_cross_condition_summary(cross_df, cross_file)
        print(f"[all] wrote cross-condition summary: {cross_file}")

    append_warehouse(in_root, records)
    print(f"Done. Correlation QC outputs in {in_root}")


if __name__ == "__main__":
    main()
