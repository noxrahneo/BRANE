#!/usr/bin/env python3
"""Generate plots from thesis-style DEG t-test outputs.

Expected inputs are per-contrast files from scripts/16_deg_ttest_logcpm.py:
- <contrast>/<contrast>_deg_stats.csv
- deg_contrast_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTRAST_LABELS: dict[str, str] = {
    "ER_tumor.vs.Normal": "ER+ tumour vs Normal",
    "HER2_tumor.vs.Normal": "HER2+ tumour vs Normal",
    "Triple_negative_BRCA1_tumor.vs.Normal": "TN (BRCA1) tumour vs Normal",
    "Triple_negative_tumor.vs.Normal": "Triple negative tumour vs Normal",
    "Normal_BRCA1_-_pre-neoplastic.vs.Normal": "N-BRCA1 pre-neoplastic vs Normal",
    "Triple_negative_BRCA1_tumor.vs.Normal_BRCA1_-_pre-neoplastic": "TN (BRCA1) tumour vs N-BRCA1",
}

PANEL_ORDER = [
    "ER_tumor.vs.Normal",
    "HER2_tumor.vs.Normal",
    "Triple_negative_BRCA1_tumor.vs.Normal",
    "Triple_negative_tumor.vs.Normal",
    "Normal_BRCA1_-_pre-neoplastic.vs.Normal",
    "Triple_negative_BRCA1_tumor.vs.Normal_BRCA1_-_pre-neoplastic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot volcano/MA/summary figures from DEG t-test outputs"
    )
    parser.add_argument(
        "--input-dir",
        default="results/10_deg_ttest",
        help="Root directory produced by 16_deg_ttest_logcpm.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/10_deg_ttest/figures",
        help="Output root for DEG plots",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold shown on plots",
    )
    parser.add_argument(
        "--min-abs-log2fc",
        type=float,
        default=0.2,
        help="Absolute log2FC threshold shown on plots",
    )
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=12,
        help="Number of top DEG genes to label on each volcano",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def list_contrast_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and ".vs." in p.name]
    )


def safe_neglog10(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.clip(arr, 1e-300, 1.0)
    return -np.log10(arr)


def load_stats(contrast_dir: Path) -> pd.DataFrame:
    matches = sorted(contrast_dir.glob("*_deg_stats.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_deg_stats.csv in {contrast_dir}")
    return pd.read_csv(matches[0])


def plot_volcano(
    df: pd.DataFrame,
    contrast: str,
    out_file: Path,
    fdr_threshold: float,
    min_abs_log2fc: float,
    label_top_n: int,
) -> None:
    fc = pd.to_numeric(df["lnFC"], errors="coerce").to_numpy()
    fdr = pd.to_numeric(df["fdr"], errors="coerce").to_numpy()
    y = safe_neglog10(fdr)

    sig = (
        np.isfinite(fdr)
        & (fdr <= float(fdr_threshold))
        & (np.abs(fc) >= float(min_abs_log2fc))
    )
    up = sig & (fc >= 0)
    down = sig & (fc < 0)
    ns = ~sig

    n_up = int(up.sum())
    n_down = int(down.sum())
    n_total = n_up + n_down
    label = CONTRAST_LABELS.get(contrast, contrast.replace(".", " ").replace("_", " "))

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(fc[ns], y[ns], s=6, alpha=0.25, color="#b0b8c4", rasterized=True, zorder=1)
    ax.scatter(fc[down], y[down], s=14, alpha=0.80, color="#2563eb", label=f"Down ({n_down:,})", rasterized=True, zorder=2)
    ax.scatter(fc[up], y[up], s=14, alpha=0.80, color="#dc2626", label=f"Up ({n_up:,})", rasterized=True, zorder=2)

    ax.axvline(float(min_abs_log2fc), color="#6b7280", linestyle="--", linewidth=0.8)
    ax.axvline(-float(min_abs_log2fc), color="#6b7280", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(float(fdr_threshold)), color="#6b7280", linestyle="--", linewidth=0.8)

    ax.set_xlabel(r"ln FC (tumour $-$ normal)", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}$(FDR)", fontsize=11)
    ax.set_title(
        f"{label}\n"
        f"{n_total:,} DEGs  ·  FDR $\\leq$ {fdr_threshold}  ·  |lnFC| $\\geq$ {min_abs_log2fc}",
        fontsize=11, fontweight="bold", linespacing=1.6,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.5, zorder=0)

    #label top genes: pick separately from up and down by fdr then |fc|
    if int(label_top_n) > 0:
        half = max(1, int(label_top_n) // 2)
        work = df.copy()
        work["lnFC"] = pd.to_numeric(work["lnFC"], errors="coerce")
        work["fdr"] = pd.to_numeric(work["fdr"], errors="coerce")
        degs = work[work["fdr"].le(float(fdr_threshold)) & work["lnFC"].abs().ge(float(min_abs_log2fc))].copy()
        if not degs.empty:
            up_genes = degs[degs["lnFC"] >= 0].nsmallest(half, "fdr")
            dn_genes = degs[degs["lnFC"] < 0].nsmallest(half, "fdr")
            for subset, ha in [(up_genes, "left"), (dn_genes, "right")]:
                for _, row in subset.iterrows():
                    x0 = float(row["lnFC"])
                    y0 = float(safe_neglog10(np.array([row["fdr"]]))[0])
                    xoff = 6 if ha == "left" else -6
                    ax.annotate(
                        str(row["gene"]),
                        (x0, y0),
                        xytext=(xoff, 3),
                        textcoords="offset points",
                        fontsize=7.5,
                        ha=ha,
                        va="bottom",
                        color="#111827",
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
                    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ma(
    df: pd.DataFrame,
    contrast: str,
    out_file: Path,
    fdr_threshold: float,
    min_abs_log2fc: float,
) -> None:
    mean_case = pd.to_numeric(df["mean_case_expr"], errors="coerce").to_numpy()
    mean_ctrl = pd.to_numeric(
        df["mean_control_expr"], errors="coerce"
    ).to_numpy()
    log2fc = pd.to_numeric(df["lnFC"], errors="coerce").to_numpy()
    fdr = pd.to_numeric(df["fdr"], errors="coerce").to_numpy()

    a = (mean_case + mean_ctrl) / 2.0
    sig = (
        np.isfinite(fdr)
        & (fdr <= float(fdr_threshold))
        & (np.abs(log2fc) >= float(min_abs_log2fc))
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(
        a[~sig],
        log2fc[~sig],
        s=8,
        alpha=0.30,
        color="#9aa4b2",
        label="not DEG",
    )
    ax.scatter(
        a[sig], log2fc[sig], s=11, alpha=0.75, color="#0f766e", label="DEG"
    )
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.axhline(
        float(min_abs_log2fc),
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
    )
    ax.axhline(
        -float(min_abs_log2fc),
        color="#6b7280",
        linestyle="--",
        linewidth=1.0,
    )
    ax.set_xlabel("A = (mean_case + mean_control) / 2")
    ax.set_ylabel("M = lnFC")
    ax.set_title(f"MA Plot | {contrast}")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_hist(df: pd.DataFrame, contrast: str, out_file: Path) -> None:
    pvals = pd.to_numeric(df["raw_pvalue"], errors="coerce").to_numpy()
    pvals = pvals[np.isfinite(pvals)]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.hist(pvals, bins=50, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Raw p-value")
    ax.set_ylabel("Gene count")
    ax.set_title(f"P-value Histogram | {contrast}")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_deg_count_summary(summary_df: pd.DataFrame, out_file: Path) -> None:
    plot_df = summary_df.copy()
    plot_df = plot_df.sort_values(
        "n_deg_total", ascending=False
    ).reset_index(drop=True)

    x = np.arange(plot_df.shape[0])
    up = pd.to_numeric(
        plot_df["n_up_deg"], errors="coerce"
    ).fillna(0).to_numpy()
    down = pd.to_numeric(
        plot_df["n_down_deg"], errors="coerce"
    ).fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.bar(x, up, label="up DEG", color="#d94841")
    ax.bar(x, down, bottom=up, label="down DEG", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["contrast"].astype(str), rotation=30, ha="right"
    )
    ax.set_ylabel("DEG count")
    ax.set_title("DEG Count per Contrast")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_volcano_panel(out_root: Path, panel_file: Path) -> None:
    import matplotlib.image as mpimg
    contrasts = [c for c in PANEL_ORDER if (out_root / c / f"{c}_volcano.png").exists()]
    if not contrasts:
        return
    n = len(contrasts)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 5.5))
    for i, (ax, contrast) in enumerate(zip(axes.flat, contrasts)):
        img = mpimg.imread(str(out_root / contrast / f"{contrast}_volcano.png"))
        ax.imshow(img, aspect="auto")
        ax.axis("off")
    for ax in axes.flat[len(contrasts):]:
        ax.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(panel_file), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    #resolve paths and create output root
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    #find all contrast directories
    cdirs = list_contrast_dirs(in_root)
    if not cdirs:
        raise FileNotFoundError(f"No contrast directories found in {in_root}")

    #generate per-contrast volcano, ma, and p-value histogram plots
    for cdir in cdirs:
        contrast = cdir.name
        df = load_stats(cdir)
        fig_dir = out_root / contrast
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_volcano(df=df, contrast=contrast, out_file=fig_dir / f"{contrast}_volcano.png",
                     fdr_threshold=float(args.fdr_threshold), min_abs_log2fc=float(args.min_abs_log2fc),
                     label_top_n=int(args.label_top_n))
        plot_ma(df=df, contrast=contrast, out_file=fig_dir / f"{contrast}_ma_plot.png",
                fdr_threshold=float(args.fdr_threshold), min_abs_log2fc=float(args.min_abs_log2fc))
        plot_pvalue_hist(df=df, contrast=contrast, out_file=fig_dir / f"{contrast}_pvalue_hist.png")
        print(f"[{contrast}] wrote DEG plots to {fig_dir}")

    #optional cross-contrast summary bar chart
    summary_file = in_root / "deg_contrast_summary.csv"
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        plot_deg_count_summary(summary_df=summary_df, out_file=out_root / "deg_count_summary.png")
        print(f"Wrote summary plot: {out_root / 'deg_count_summary.png'}")
    else:
        print(f"[warn] missing summary file: {summary_file}")

    #assemble multi-contrast volcano panel
    panel_file = out_root / "deg_volcano_panel.png"
    plot_volcano_panel(out_root, panel_file)
    print(f"Wrote volcano panel: {panel_file}")
    print(f"Done. DEG figures: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
