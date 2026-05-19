#!/usr/bin/env python3
"""Sanity checks for pseudobulk outputs from 15_pseudobulk.py."""

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
        description="Run pseudobulk sanity checks"
    )
    parser.add_argument(
        "--input-dir",
        default="results/06_pseudobulk",
        help="Pseudobulk directory from 15_pseudobulk.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_pseudobulk_qc",
        help="Directory to write sanity-check outputs",
    )
    parser.add_argument(
        "--sample-col",
        default="SampleName",
        help="Sample column in pseudobulk metadata",
    )
    parser.add_argument(
        "--group-col",
        default="cell_type_annot",
        help="Cell type/group column in pseudobulk metadata",
    )
    parser.add_argument(
        "--condition-col",
        default="condition",
        help="Condition column in pseudobulk metadata",
    )
    parser.add_argument(
        "--min-cells-warning",
        type=int,
        default=20,
        help="Warning threshold for profile-level n_cells",
    )
    parser.add_argument(
        "--min-profiles-per-group",
        type=int,
        default=5,
        help="Warning threshold for replicates per condition x cell type",
    )
    parser.add_argument(
        "--min-cells-per-type",
        type=int,
        default=50,
        help="Minimum total cells per condition x cell type to trust pseudobulk correlations",
    )
    parser.add_argument(
        "--max-sample-dominance",
        type=float,
        default=0.4,
        help="Flag if a single sample contributes more than this fraction of cells for a condition x cell type",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return repo_candidate


def discover_condition_dirs(pb_root: Path) -> list[Path]:
    if not pb_root.exists():
        return []

    condition_dirs: list[Path] = []
    for cdir in sorted([p for p in pb_root.iterdir() if p.is_dir()]):
        counts = cdir / f"{cdir.name}_pseudobulk_counts.csv"
        meta = cdir / f"{cdir.name}_pseudobulk_metadata.csv"
        if counts.exists() and meta.exists():
            condition_dirs.append(cdir)
    return condition_dirs


def ensure_pseudobulk_id_column(
    meta: pd.DataFrame,
    source: Path,
) -> pd.DataFrame:
    out = meta.copy()

    if "pseudobulk_id" in out.columns:
        return out

    rename_candidates = ["index", "Unnamed: 0"]
    for cand in rename_candidates:
        if cand in out.columns:
            out = out.rename(columns={cand: "pseudobulk_id"})
            return out

    if out.index.name == "pseudobulk_id":
        return out.reset_index()

    raise ValueError(
        f"Missing pseudobulk_id in {source}; columns={out.columns.tolist()}"
    )


def load_all_pseudobulk(
    pb_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_dirs = discover_condition_dirs(pb_root)
    if not condition_dirs:
        raise FileNotFoundError(
            "No condition-level pseudobulk files found. "
            "Run scripts/15_pseudobulk.py first."
        )

    counts_all: list[pd.DataFrame] = []
    meta_all: list[pd.DataFrame] = []

    for cdir in condition_dirs:
        condition = cdir.name
        counts_file = cdir / f"{condition}_pseudobulk_counts.csv"
        meta_file = cdir / f"{condition}_pseudobulk_metadata.csv"

        counts = pd.read_csv(counts_file, index_col=0)
        meta = ensure_pseudobulk_id_column(pd.read_csv(meta_file), meta_file)

        missing_cols = [
            col
            for col in meta["pseudobulk_id"].tolist()
            if col not in counts.columns
        ]
        if missing_cols:
            raise ValueError(
                f"{condition}: metadata IDs not present in counts: "
                f"{missing_cols[:5]}"
            )

        keep = [c for c in counts.columns if c in set(meta["pseudobulk_id"])]
        counts = counts[keep]
        meta = meta.set_index("pseudobulk_id").loc[keep].reset_index()
        meta = ensure_pseudobulk_id_column(meta, meta_file)

        counts_all.append(counts)
        meta_all.append(meta)

    combined_counts = pd.concat(counts_all, axis=1, join="inner")
    combined_meta = pd.concat(meta_all, axis=0, ignore_index=True)
    combined_meta = combined_meta.set_index("pseudobulk_id").loc[
        combined_counts.columns
    ].reset_index()
    combined_meta = ensure_pseudobulk_id_column(
        combined_meta,
        pb_root / "all_conditions_pseudobulk_metadata.csv",
    )

    return combined_counts, combined_meta


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(series.shape[0]), index=series.index)
    return 0.6745 * (series - med) / mad


def compute_profile_metrics(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    sample_col: str,
    group_col: str,
    condition_col: str,
    min_cells_warning: int,
) -> pd.DataFrame:
    metrics = ensure_pseudobulk_id_column(meta.copy(), Path("<in_memory_meta>"))

    lib_size = counts.sum(axis=0)
    detected = (counts > 0).sum(axis=0)

    metrics["library_size"] = metrics["pseudobulk_id"].map(lib_size)
    metrics["detected_genes"] = metrics["pseudobulk_id"].map(detected)

    if "total_counts" in metrics.columns:
        metrics["counts_match_meta"] = np.isclose(
            metrics["library_size"].astype(float),
            metrics["total_counts"].astype(float),
            atol=1e-6,
        )
    else:
        metrics["counts_match_meta"] = True

    if "n_cells" in metrics.columns:
        metrics["low_n_cells_flag"] = (
            metrics["n_cells"].astype(float) < float(min_cells_warning)
        )
    else:
        metrics["low_n_cells_flag"] = False

    log_lib = np.log10(metrics["library_size"].clip(lower=1.0))
    log_det = np.log10(metrics["detected_genes"].clip(lower=1.0))
    metrics["libsize_rz"] = robust_z(log_lib)
    metrics["detected_rz"] = robust_z(log_det)
    metrics["profile_outlier_flag"] = (
        metrics["libsize_rz"].abs() > 3.5
    ) | (
        metrics["detected_rz"].abs() > 3.5
    )

    req = [sample_col, group_col, condition_col]
    missing = [c for c in req if c not in metrics.columns]
    if missing:
        raise ValueError(
            "Missing required metadata columns for sanity check: "
            f"{missing}"
        )

    return metrics


def logcpm_matrix(
    counts: pd.DataFrame,
    target_sum: float = 1_000_000.0,
) -> pd.DataFrame:
    lib = counts.sum(axis=0).replace(0.0, np.nan)
    norm = counts.divide(lib, axis=1).fillna(0.0) * float(target_sum)
    return np.log1p(norm)


def pca_scores(
    x: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]
    var = (s**2) / max(1, x.shape[0] - 1)
    var_ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    return scores, var_ratio[:n_components]


def save_plots(
    metrics: pd.DataFrame,
    logcpm: pd.DataFrame,
    fig_dir: Path,
    sample_col: str,
    group_col: str,
    condition_col: str,
    min_cells_warning: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_genes = logcpm.shape[0]

    if n_genes > 0:
        metrics["detected_gene_fraction"] = (
            metrics["detected_genes"].astype(float) / float(n_genes)
        )

    #library size and detected genes distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(
        np.log10(metrics["library_size"].clip(lower=1.0)),
        bins=35,
        color="#4C78A8",
        alpha=0.85,
    )
    axes[0].set_title("log10 library size")
    axes[0].set_xlabel("log10(total counts per pseudobulk profile)")
    axes[0].set_ylabel("Number of pseudobulk profiles")

    axes[1].hist(
        np.log10(metrics["detected_genes"].clip(lower=1.0)),
        bins=35,
        color="#F58518",
        alpha=0.85,
    )
    axes[1].set_title("log10 detected genes")
    axes[1].set_xlabel("log10(detected genes per pseudobulk profile)")
    axes[1].set_ylabel("Number of pseudobulk profiles")

    fig.tight_layout()
    fig.savefig(fig_dir / "qc_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    #condition-wise detectable-gene fraction boxplot
    cond_levels = sorted(metrics[condition_col].astype(str).unique())
    median_by_cond = {
        cond: metrics.loc[
            metrics[condition_col].astype(str) == cond,
            "detected_gene_fraction",
        ].median()
        for cond in cond_levels
    }
    cond_levels = sorted(
        cond_levels,
        key=lambda c: (median_by_cond[c], c),
        reverse=True,
    )

    box_data = [
        metrics.loc[
            metrics[condition_col].astype(str) == cond,
            "detected_gene_fraction",
        ].dropna().values
        for cond in cond_levels
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    rng = np.random.default_rng(7)
    y_positions = np.arange(1, len(cond_levels) + 1)

    for y, vals in zip(y_positions, box_data):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            vals,
            np.full(len(vals), y) + jitter,
            s=14,
            alpha=0.35,
            color="#4C78A8",
            linewidths=0,
            zorder=1,
        )

    bp = ax.boxplot(
        box_data,
        tick_labels=cond_levels,
        patch_artist=True,
        showfliers=False,
        vert=False,
        widths=0.55,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#A0CBE8")
        patch.set_alpha(0.55)
        patch.set_zorder(2)

    all_vals = np.concatenate([v for v in box_data if len(v) > 0])
    q01, q99 = np.quantile(all_vals, [0.01, 0.99])
    xmin = max(0.0, float(q01) - 0.05)
    xmax = min(1.0, float(q99) + 0.02)
    if xmax - xmin < 0.08:
        mid = 0.5 * (xmin + xmax)
        xmin = max(0.0, mid - 0.04)
        xmax = min(1.0, mid + 0.04)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Detected-gene fraction per profile")
    ax.set_ylabel("Condition")
    ax.set_title(
        "Detected-gene fraction by condition\n"
        f"(detected genes / {n_genes} shared genes)"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "qc_detected_gene_fraction_by_condition.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    #n_cells vs library size scatter coloured by condition
    if "n_cells" in metrics.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        conds = sorted(metrics[condition_col].astype(str).unique())
        cmap = plt.cm.get_cmap("tab10", max(1, len(conds)))
        color_map = {c: cmap(i) for i, c in enumerate(conds)}
        for cond in conds:
            sub = metrics[metrics[condition_col].astype(str) == cond]
            ax.scatter(
                sub["n_cells"],
                sub["library_size"],
                s=24,
                alpha=0.8,
                label=cond,
                color=color_map[cond],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_cells per pseudobulk")
        ax.set_ylabel("library size (sum counts)")

        low_library_threshold = float(
            np.quantile(metrics["library_size"].astype(float), 0.10)
        )
        ax.axvline(
            float(min_cells_warning),
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"n_cells warning = {int(min_cells_warning)}",
        )
        ax.axhline(
            low_library_threshold,
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"low library (10th pct) = {int(low_library_threshold):,}",
        )

        ax.set_title("Pseudobulk size sanity")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "qc_ncells_vs_library.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    #pca on pseudobulk logcpm coloured by condition and cell type
    x = logcpm.T.values
    if x.shape[0] >= 3 and x.shape[1] >= 2:
        scores, var_ratio = pca_scores(x, n_components=2)
        pca_df = pd.DataFrame(
            {
                "PC1": scores[:, 0],
                "PC2": scores[:, 1],
                "pseudobulk_id": logcpm.columns,
            }
        )
        pca_df = pca_df.merge(metrics, on="pseudobulk_id", how="left")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        conds = sorted(pca_df[condition_col].astype(str).unique())
        cmap_c = plt.cm.get_cmap("tab10", max(1, len(conds)))
        c_map = {c: cmap_c(i) for i, c in enumerate(conds)}
        for cond in conds:
            sub = pca_df[pca_df[condition_col].astype(str) == cond]
            axes[0].scatter(
                sub["PC1"],
                sub["PC2"],
                s=25,
                alpha=0.85,
                label=cond,
                color=c_map[cond],
            )
        axes[0].set_title(
            "PCA by condition\n"
            f"PC1 {var_ratio[0]*100:.1f}%, PC2 {var_ratio[1]*100:.1f}%"
        )
        axes[0].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)")
        axes[0].set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)")
        axes[0].legend(loc="best", fontsize=8)

        groups = sorted(pca_df[group_col].astype(str).unique())
        cmap_g = plt.cm.get_cmap("tab20", max(1, len(groups)))
        g_map = {g: cmap_g(i) for i, g in enumerate(groups)}
        for grp in groups:
            sub = pca_df[pca_df[group_col].astype(str) == grp]
            axes[1].scatter(
                sub["PC1"],
                sub["PC2"],
                s=22,
                alpha=0.85,
                label=grp,
                color=g_map[grp],
            )
        axes[1].set_title("PCA by cell type")
        axes[1].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% variance)")
        axes[1].set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% variance)")
        axes[1].legend(loc="best", fontsize=7)

        fig.tight_layout()
        fig.savefig(fig_dir / "qc_pca.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    #pearson correlation heatmap ordered by condition -> cell type -> sample
    corr = logcpm.corr(method="pearson")
    order_meta = ensure_pseudobulk_id_column(
        metrics.copy(),
        Path("<in_memory_metrics>"),
    )
    order_meta = order_meta[["pseudobulk_id", condition_col, group_col, sample_col]].copy()
    order_meta = order_meta.drop_duplicates(subset=["pseudobulk_id"])
    order_meta = order_meta.set_index("pseudobulk_id").loc[corr.index].reset_index()
    order_meta = ensure_pseudobulk_id_column(
        order_meta,
        Path("<in_memory_order_meta_after_reset>"),
    )
    order_meta = order_meta.sort_values(
        by=[condition_col, group_col, sample_col],
        kind="mergesort",
    )
    ordered_ids = order_meta["pseudobulk_id"].tolist()
    corr = corr.loc[ordered_ids, ordered_ids]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        corr.values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    ax.set_title(
        "Pseudobulk profile correlation (Pearson, logCPM)\n"
        "Rows/columns are the same pseudobulk profiles (ordered by condition | cell type)"
    )
    ax.set_xlabel("Pseudobulk profiles")
    ax.set_ylabel("Pseudobulk profiles")

    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "qc_correlation_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def compute_celltype_balance(
    meta: pd.DataFrame,
    sample_col: str,
    group_col: str,
    condition_col: str,
    min_cells_per_type: int,
) -> pd.DataFrame:
    #total cells per condition x cell type; flags groups below min_cells_per_type
    if "n_cells" not in meta.columns:
        return pd.DataFrame()

    balance = (
        meta.groupby([condition_col, group_col], as_index=False)
        .agg(
            total_cells=("n_cells", "sum"),
            n_profiles=("pseudobulk_id", "count"),
            n_samples=(sample_col, lambda s: s.astype(str).nunique()),
        )
    )
    balance["below_min_cells_flag"] = (
        balance["total_cells"].astype(int) < int(min_cells_per_type)
    )
    balance = balance.sort_values(
        [condition_col, "total_cells"], ascending=[True, True]
    )
    return balance


def compute_sample_dominance(
    meta: pd.DataFrame,
    sample_col: str,
    group_col: str,
    condition_col: str,
    max_sample_dominance: float,
) -> pd.DataFrame:
    #per-sample cell fraction per condition x cell type; flags dominant samples
    if "n_cells" not in meta.columns:
        return pd.DataFrame()

    group_total = (
        meta.groupby([condition_col, group_col])["n_cells"]
        .sum()
        .rename("group_total_cells")
        .reset_index()
    )
    dom = meta.merge(group_total, on=[condition_col, group_col], how="left")
    dom["sample_fraction"] = (
        dom["n_cells"].astype(float) / dom["group_total_cells"].astype(float)
    )

    #aggregate to per-sample max fraction within each condition x cell type group
    dom_summary = (
        dom.groupby([condition_col, group_col, sample_col], as_index=False)
        .agg(
            sample_cells=("n_cells", "sum"),
            group_total_cells=("group_total_cells", "first"),
        )
    )
    dom_summary["sample_fraction"] = (
        dom_summary["sample_cells"].astype(float)
        / dom_summary["group_total_cells"].astype(float)
    )
    dom_summary["dominant_sample_flag"] = (
        dom_summary["sample_fraction"] > float(max_sample_dominance)
    )
    dom_summary = dom_summary.sort_values(
        "sample_fraction", ascending=False
    )
    return dom_summary


def build_summary_tables(
    metrics: pd.DataFrame,
    sample_col: str,
    group_col: str,
    condition_col: str,
    min_profiles_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_condition = (
        metrics.groupby(condition_col, as_index=False)
        .agg(
            n_profiles=("pseudobulk_id", "count"),
            n_samples=(sample_col, lambda s: s.astype(str).nunique()),
            n_celltypes=(group_col, lambda s: s.astype(str).nunique()),
            median_n_cells=("n_cells", "median"),
            median_library_size=("library_size", "median"),
            n_low_cells=("low_n_cells_flag", "sum"),
            n_profile_outliers=("profile_outlier_flag", "sum"),
        )
    )

    by_group = (
        metrics.groupby([condition_col, group_col], as_index=False)
        .agg(
            n_profiles=("pseudobulk_id", "count"),
            min_n_cells=("n_cells", "min"),
            median_n_cells=("n_cells", "median"),
            min_library_size=("library_size", "min"),
            median_library_size=("library_size", "median"),
        )
    )
    by_group["low_replicates_flag"] = (
        by_group["n_profiles"].astype(int) < int(min_profiles_per_group)
    )

    flagged = metrics[
        metrics["low_n_cells_flag"] | metrics["profile_outlier_flag"]
    ].copy()

    return by_condition, by_group, flagged


def main() -> int:
    args = parse_args()

    #resolve paths and create output dirs
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    #load all pseudobulk counts and metadata
    counts, meta = load_all_pseudobulk(in_root)

    #compute per-profile qc metrics and logcpm matrix
    metrics = compute_profile_metrics(
        counts=counts, meta=meta, sample_col=args.sample_col,
        group_col=args.group_col, condition_col=args.condition_col,
        min_cells_warning=args.min_cells_warning,
    )
    logcpm = logcpm_matrix(counts)

    #generate qc plots
    save_plots(
        metrics=metrics, logcpm=logcpm, fig_dir=fig_dir,
        sample_col=args.sample_col, group_col=args.group_col,
        condition_col=args.condition_col, min_cells_warning=args.min_cells_warning,
    )

    #build condition and group summary tables
    by_condition, by_group, flagged = build_summary_tables(
        metrics=metrics, sample_col=args.sample_col, group_col=args.group_col,
        condition_col=args.condition_col, min_profiles_per_group=args.min_profiles_per_group,
    )

    #compute cell-type balance and sample dominance tables
    celltype_balance = compute_celltype_balance(
        meta=meta, sample_col=args.sample_col, group_col=args.group_col,
        condition_col=args.condition_col, min_cells_per_type=args.min_cells_per_type,
    )
    sample_dominance = compute_sample_dominance(
        meta=meta, sample_col=args.sample_col, group_col=args.group_col,
        condition_col=args.condition_col, max_sample_dominance=args.max_sample_dominance,
    )

    #save csv outputs
    metrics.to_csv(out_root / "qc_profile_metrics.csv", index=False)
    by_condition.to_csv(out_root / "qc_summary_by_condition.csv", index=False)
    by_group.to_csv(out_root / "qc_summary_by_condition_celltype.csv", index=False)
    flagged.to_csv(out_root / "qc_flagged_profiles.csv", index=False)

    if not celltype_balance.empty:
        celltype_balance.to_csv(out_root / "qc_celltype_balance.csv", index=False)
        n_flagged_balance = int(celltype_balance["below_min_cells_flag"].sum())
        if n_flagged_balance:
            print(f"  WARNING: {n_flagged_balance} condition x cell-type groups below {args.min_cells_per_type} total cells — correlations may be unreliable")

    if not sample_dominance.empty:
        sample_dominance.to_csv(out_root / "qc_sample_dominance.csv", index=False)
        n_dominant = int(sample_dominance["dominant_sample_flag"].sum())
        if n_dominant:
            print(f"  WARNING: {n_dominant} entries exceed {args.max_sample_dominance:.0%} dominance — pseudobulk may reflect a single sample")

    #write warehouse provenance record
    record = WarehouseRecord(
        input_file=str(in_root),
        output_file=str(out_root / "qc_summary_by_condition.csv"),
        script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        date_utc=utc_now_iso(),
        params_hash=params_hash(vars(args)),
        condition="all",
        stage="pseudobulk_qc",
    )
    append_warehouse(out_root.parent, [record])

    print("Pseudobulk sanity check complete.")
    print(f"Profiles: {metrics.shape[0]} | Genes: {counts.shape[0]}")
    print(f"Outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
