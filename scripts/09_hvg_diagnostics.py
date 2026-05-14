#!/usr/bin/env python3
"""Generate HVG and PCA diagnostics per condition.

Outputs per condition:
- hvg_gene_stats.csv (mean/variance/HVG flag)
- hvg_diagnostic_panel.png:
  A) mean vs variance with binned trend line
  B) mean vs normalized dispersion with HVGs highlighted
  C) histogram of gene variance (all genes vs selected HVGs)
- hvg_summary.json
- pca_scree_with_elbow.png (if integrated PCA exists)
- pca_variance_ratio.csv (if integrated PCA exists)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from utils.h5ad_compat import read_h5ad_compat

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HVG diagnostics and PCA elbow plots per condition"
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition folder name, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition folders and exit",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="results/02_preprocess/02_preprocessed",
        help="Root with per-condition *_preprocessed.h5ad files",
    )
    parser.add_argument(
        "--integration-dir",
        default="results/03_integration/integrated",
        help="Root with per-condition integrated h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/03_integration/hvg_diagnostics",
        help="Output root for HVG diagnostics",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=3000,
        help="Target number of HVGs",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Target sum used before HVG selection",
    )
    parser.add_argument(
        "--hvg-flavor",
        choices=["seurat", "cell_ranger", "seurat_v3"],
        default="seurat",
        help="Scanpy HVG flavor",
    )
    parser.add_argument(
        "--batch-key",
        default="SampleName",
        help="Batch key used in HVG computation when available",
    )
    parser.add_argument(
        "--meanvar-bins",
        type=int,
        default=30,
        help="Number of bins for mean-variance trend line",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=190,
        help="Output figure DPI",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "unknown"


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    for cdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if list(cdir.glob("*_preprocessed.h5ad")):
            out.append(cdir.name)
    return out


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]

    mapped = safe_name(requested)
    if mapped in available:
        return [mapped]

    raise ValueError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def sample_name(path: Path) -> str:
    if path.name.endswith("_preprocessed.h5ad"):
        return path.name[:-18]
    return path.stem


def load_merged_condition(
    pre_dir: Path,
    condition: str,
    target_sum: float,
) -> ad.AnnData:
    cdir = pre_dir / condition
    files = sorted(cdir.glob("*_preprocessed.h5ad"))
    if not files:
        raise FileNotFoundError(f"No *_preprocessed.h5ad files found in {cdir}")

    samples: list[ad.AnnData] = []
    for fpath in files:
        adata = read_h5ad_compat(fpath)
        adata.obs["SampleName"] = sample_name(fpath)
        adata.obs["Condition"] = condition
        samples.append(adata)

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

    sc.pp.normalize_total(merged, target_sum=float(target_sum))
    sc.pp.log1p(merged)
    return merged


def compute_mean_var(x) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(x):
        arr = x.astype(np.float64)
        mean = np.asarray(arr.mean(axis=0)).ravel()
        sq_mean = np.asarray(arr.power(2).mean(axis=0)).ravel()
        var = sq_mean - np.square(mean)
        var = np.clip(var, 0.0, None)
        return mean, var

    arr = np.asarray(x, dtype=np.float64)
    mean = arr.mean(axis=0)
    var = arr.var(axis=0)
    return mean, var


def binned_trend(mean: np.ndarray, var: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(mean) & np.isfinite(var) & (mean > 0) & (var >= 0)
    m = mean[valid]
    v = var[valid]
    if m.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    bins = max(5, int(n_bins))
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(m, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Ensure full range is represented even when quantile edges collapse.
    edges[0] = np.min(m)
    edges[-1] = np.max(m)
    x_line: list[float] = []
    y_line: list[float] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (m >= lo) & (m <= hi)
        else:
            mask = (m >= lo) & (m < hi)
        if np.sum(mask) < 3:
            continue
        x_line.append(float(np.median(m[mask])))
        y_line.append(float(np.median(v[mask])))
    return np.asarray(x_line), np.asarray(y_line)


def elbow_from_variance(var_ratio: np.ndarray) -> int:
    """Quantitative elbow detection following the HBC scRNA-seq training approach.

    Two criteria are computed and the minimum (most conservative) is returned:
      co1 — first PC where cumulative variance > 90% AND individual PC < 5%
      co2 — last PC where the drop between consecutive PCs exceeds 0.1%

    Reference: https://github.com/hbctraining/scRNA-seq/blob/master/lessons/elbow_plot_metric.md
    """
    pct = np.asarray(var_ratio, dtype=float) * 100.0
    if pct.size <= 2:
        return int(max(1, pct.size))

    cumulative = np.cumsum(pct)

    # co1: first PC where cumulative > 90% and individual contribution < 5%
    co1_candidates = np.where((cumulative > 90.0) & (pct < 5.0))[0]
    co1 = int(co1_candidates[0] + 1) if co1_candidates.size > 0 else int(pct.size)

    # co2: last PC where drop between consecutive PCs exceeds 0.1%
    drops = pct[:-1] - pct[1:]
    co2_candidates = np.where(drops > 0.1)[0]
    co2 = int(co2_candidates[-1] + 2) if co2_candidates.size > 0 else 1

    return min(co1, co2)


def plot_hvg_panel(
    out_png: Path,
    stats: pd.DataFrame,
    n_top_genes: int,
    n_bins: int,
    dpi: int,
) -> None:
    hvg_mask = stats["highly_variable"].astype(bool).to_numpy()
    mean = stats["mean_log1p"].to_numpy(dtype=float)
    var = stats["var_log1p"].to_numpy(dtype=float)

    xb, yb = binned_trend(mean, var, n_bins=n_bins)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    # A: mean-variance trend
    axes[0].scatter(
        mean[~hvg_mask],
        var[~hvg_mask],
        s=6,
        alpha=0.18,
        c="#9CA3AF",
        linewidths=0,
        label="Non-HVG genes",
    )
    axes[0].scatter(
        mean[hvg_mask],
        var[hvg_mask],
        s=8,
        alpha=0.55,
        c="#E45756",
        linewidths=0,
        label="Selected HVGs",
    )
    if xb.size > 0:
        axes[0].plot(xb, yb, color="#1D4ED8", linewidth=2.0, label="Binned median trend")
    axes[0].set_xlabel("Mean log1p expression")
    axes[0].set_ylabel("Variance (log1p)")
    axes[0].set_title("Before HVG: mean-variance trend")
    axes[0].legend(frameon=False, loc="best")

    # B: mean vs normalized dispersion
    if "dispersions_norm" in stats.columns:
        y_disp = stats["dispersions_norm"].to_numpy(dtype=float)
        ylabel = "Normalized dispersion"
    elif "dispersions" in stats.columns:
        y_disp = stats["dispersions"].to_numpy(dtype=float)
        ylabel = "Dispersion"
    else:
        y_disp = np.full_like(mean, np.nan, dtype=float)
        ylabel = "Dispersion"

    # Some Scanpy runs can leave dispersion columns empty. Fall back to
    # variance/mean computed from the already log1p-normalized matrix.
    if not np.any(np.isfinite(y_disp)):
        eps = 1e-12
        y_disp = var / np.maximum(mean, eps)
        ylabel = "Dispersion proxy (var/mean)"

    axes[1].scatter(
        mean[~hvg_mask],
        y_disp[~hvg_mask],
        s=6,
        alpha=0.18,
        c="#9CA3AF",
        linewidths=0,
        label="Non-HVG genes",
    )
    axes[1].scatter(
        mean[hvg_mask],
        y_disp[hvg_mask],
        s=8,
        alpha=0.55,
        c="#E45756",
        linewidths=0,
        label="Selected HVGs",
    )
    axes[1].set_xlabel("Mean log1p expression")
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"Selected HVGs highlighted (target={n_top_genes})")
    axes[1].legend(frameon=False, loc="best")

    # C: histogram variance all vs hvg
    vmax = float(np.nanpercentile(var[np.isfinite(var)], 99.5)) if np.any(np.isfinite(var)) else 1.0
    bins = np.linspace(0, max(vmax, 1e-8), 80)
    axes[2].hist(var, bins=bins, alpha=0.55, color="#93C5FD", label="All genes")
    axes[2].hist(var[hvg_mask], bins=bins, alpha=0.65, color="#EF4444", label="Selected HVGs")
    axes[2].set_xlabel("Variance (log1p)")
    axes[2].set_ylabel("Gene count")
    axes[2].set_title("Variance distribution: all vs selected")
    axes[2].legend(frameon=False)

    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def extract_pca_variance_ratio(integration_dir: Path, condition: str) -> np.ndarray | None:
    cdir = integration_dir / condition
    preferred = cdir / f"{condition}_integrated.h5ad"
    h5ad_file = preferred if preferred.exists() else None
    if h5ad_file is None:
        matches = sorted(cdir.glob("*_integrated.h5ad"))
        if matches:
            h5ad_file = matches[0]
    if h5ad_file is None:
        return None

    adata = read_h5ad_compat(h5ad_file)

    ratio = None
    pca_uns = adata.uns.get("pca", {}) if isinstance(adata.uns.get("pca", {}), dict) else {}
    if "variance_ratio" in pca_uns:
        ratio = np.asarray(pca_uns["variance_ratio"], dtype=float)
    elif "X_pca" in adata.obsm:
        x = np.asarray(adata.obsm["X_pca"], dtype=float)
        var = x.var(axis=0, ddof=1)
        ratio = var / np.sum(var) if np.sum(var) > 0 else None

    if ratio is None or ratio.size == 0:
        return None
    return ratio


def plot_pca_scree(out_png: Path, out_csv: Path, var_ratio: np.ndarray, dpi: int) -> int:
    ratio = np.asarray(var_ratio, dtype=float)
    pcs = np.arange(1, ratio.size + 1)
    elbow = elbow_from_variance(ratio)

    df = pd.DataFrame(
        {
            "PC": pcs,
            "variance_ratio": ratio,
            "variance_percent": ratio * 100.0,
            "cumulative_variance_percent": np.cumsum(ratio) * 100.0,
        }
    )
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.plot(pcs, ratio * 100.0, marker="o", linewidth=1.6, markersize=3)
    ax.axvline(elbow, linestyle="--", linewidth=1.3, color="#DC2626")
    ax.text(
        elbow + 0.2,
        float(np.nanmax(ratio) * 100.0 * 0.92),
        f"Elbow PC={elbow}",
        color="#DC2626",
        fontsize=9,
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot")
    ax.grid(alpha=0.2)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return elbow


def run_condition(
    condition: str,
    pre_dir: Path,
    integration_dir: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    out_dir = out_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged_condition(
        pre_dir=pre_dir,
        condition=condition,
        target_sum=float(args.target_sum),
    )

    mean, var = compute_mean_var(merged.X)

    hvg_kwargs = {
        "n_top_genes": int(args.n_top_genes),
        "flavor": args.hvg_flavor,
        "subset": False,
        "inplace": True,
    }
    if args.batch_key in merged.obs.columns:
        hvg_kwargs["batch_key"] = args.batch_key

    sc.pp.highly_variable_genes(merged, **hvg_kwargs)

    if "highly_variable" in merged.var.columns:
        hv = merged.var["highly_variable"]
    else:
        hv = pd.Series(False, index=merged.var_names)
    stats = pd.DataFrame(
        {
            "gene": merged.var_names.astype(str),
            "mean_log1p": mean,
            "var_log1p": var,
            "highly_variable": hv.astype(bool).to_numpy(),
        }
    )

    for col in ["means", "dispersions", "dispersions_norm", "highly_variable_rank"]:
        if col in merged.var.columns:
            stats[col] = pd.to_numeric(merged.var[col], errors="coerce")

    stats_file = out_dir / f"{condition}_hvg_gene_stats.csv"
    stats.to_csv(stats_file, index=False)

    panel_file = out_dir / f"{condition}_hvg_diagnostic_panel.png"
    plot_hvg_panel(
        out_png=panel_file,
        stats=stats,
        n_top_genes=int(args.n_top_genes),
        n_bins=int(args.meanvar_bins),
        dpi=int(args.dpi),
    )

    hvg_mask = stats["highly_variable"].astype(bool)
    summary = {
        "condition": condition,
        "n_cells_merged": int(merged.n_obs),
        "n_genes_before_hvg": int(merged.n_vars),
        "n_hvg_selected": int(hvg_mask.sum()),
        "hvg_fraction": float(hvg_mask.mean()),
        "n_top_genes_target": int(args.n_top_genes),
        "batch_key_used": args.batch_key if args.batch_key in merged.obs.columns else "",
        "hvg_flavor": args.hvg_flavor,
        "mean_variance_all": float(np.nanmean(stats["var_log1p"].to_numpy(dtype=float))),
        "mean_variance_hvg": float(np.nanmean(stats.loc[hvg_mask, "var_log1p"].to_numpy(dtype=float))),
        "stats_file": str(stats_file),
        "panel_file": str(panel_file),
    }

    var_ratio = extract_pca_variance_ratio(integration_dir, condition)
    if var_ratio is not None:
        pca_csv = out_dir / f"{condition}_pca_variance_ratio.csv"
        pca_png = out_dir / f"{condition}_pca_scree_with_elbow.png"
        elbow = plot_pca_scree(pca_png, pca_csv, var_ratio, dpi=int(args.dpi))
        summary["pca_elbow_pc"] = int(elbow)
        summary["pca_variance_file"] = str(pca_csv)
        summary["pca_scree_file"] = str(pca_png)

    summary_file = out_dir / f"{condition}_hvg_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[{condition}] cells={summary['n_cells_merged']} genes_before={summary['n_genes_before_hvg']} "
        f"hvg={summary['n_hvg_selected']}"
    )
    return summary


def main() -> int:
    args = parse_args()
    pre_dir = resolve_base(args.preprocessed_dir)
    integration_dir = resolve_base(args.integration_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.list_conditions:
        names = list_conditions(pre_dir)
        if not names:
            print(f"No condition folders found in: {pre_dir}")
            return 1
        print("Available condition folders:")
        for name in names:
            print(f"- {name}")
        return 0

    try:
        targets = resolve_conditions(pre_dir, args.condition)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not targets:
        print(f"ERROR: No condition folders found in {pre_dir}")
        return 1

    summaries: list[dict[str, object]] = []
    for condition in targets:
        summaries.append(
            run_condition(
                condition=condition,
                pre_dir=pre_dir,
                integration_dir=integration_dir,
                out_root=out_root,
                args=args,
            )
        )

    pd.DataFrame(summaries).to_csv(out_root / "hvg_diagnostics_summary.csv", index=False)
    print(f"Done. Wrote diagnostics for {len(summaries)} condition(s) to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
