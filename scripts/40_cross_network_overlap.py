#!/usr/bin/env python3
"""
Script 54: Cross-network hub gene overlap analysis.

Compares hub gene lists across the 4 main cancer-vs-Normal pairs
(ER, HER2, TNBC, TNBC_BRCA1) within each tier (D, S_case, S_ctrl).

Produces:
  - Jaccard similarity heatmap (one panel per tier)
  - Presence/absence heatmap for recurring genes (2+ conditions)
  - Bar chart: genes shared in N conditions per tier
  - jaccard_matrix.csv, recurring_genes.csv per tier
  - universal_sctrl_genes.csv: the 15 pan-subtype normal-lost hubs

INPUTS:
  results/20_node_annotation/{pair}/{prefix}_hubs_{tier}.csv

OUTPUTS:
  results/27_cross_network/
"""
from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_ANNOT_DIR = REPO_ROOT / "results/20_node_annotation"
OUTPUT_ROOT = REPO_ROOT / "results/27_cross_network"

# 4 main cancer-vs-Normal pairs only (not BRCA1 progression)
PAIRS = {
    "ER_tumor__vs__Normal": "ER",
    "HER2_tumor__vs__Normal": "HER2",
    "Triple_negative_tumor__vs__Normal": "TNBC",
    "Triple_negative_BRCA1_tumor__vs__Normal": "TNBC_BRCA1",
}

TIERS = ["D", "S_case", "S_ctrl"]

TIER_LABELS = {
    "D": "Rewired (D)",
    "S_case": "Tumour-gained (S+)",
    "S_ctrl": "Normal-lost (S−)",
}

TIER_COLORS = {
    "D": "#e63946",
    "S_case": "#2a9d8f",
    "S_ctrl": "#457b9d",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _safe_genes(df: pd.DataFrame) -> set[str]:
    col = "approved_symbol" if "approved_symbol" in df.columns else "gene"
    return set(df[col].dropna().astype(str).str.strip().tolist())


def load_hub_sets() -> dict[tuple[str, str], set[str]]:
    """Return {(short_name, tier): set_of_genes}."""
    hubs: dict[tuple[str, str], set[str]] = {}
    for pair, short in PAIRS.items():
        for tier in TIERS:
            f = NODE_ANNOT_DIR / pair / f"{short}_hubs_{tier}.csv"
            if f.exists():
                hubs[(short, tier)] = _safe_genes(pd.read_csv(f))
            else:
                log.warning("Missing: %s", f)
    return hubs


# ---------------------------------------------------------------------------
# Jaccard heatmap
# ---------------------------------------------------------------------------

def plot_jaccard_heatmap(hubs: dict[tuple[str, str], set[str]], out_path: Path) -> None:
    labels = list(PAIRS.values())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, tier in zip(axes, TIERS):
        n = len(labels)
        mat = np.zeros((n, n))
        for i, l1 in enumerate(labels):
            for j, l2 in enumerate(labels):
                s1 = hubs.get((l1, tier), set())
                s2 = hubs.get((l2, tier), set())
                union = len(s1 | s2)
                mat[i, j] = len(s1 & s2) / union if union else 0.0

        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(TIER_LABELS[tier], fontsize=10, fontweight="bold",
                     color=TIER_COLORS[tier], pad=6)

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.5 else "#333333",
                        fontweight="bold" if val > 0.5 else "normal")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Jaccard similarity")

    fig.suptitle("Hub gene overlap across cancer subtypes (top-50 hubs per tier)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Bar chart: genes shared in N conditions
# ---------------------------------------------------------------------------

def plot_sharing_bars(hubs: dict[tuple[str, str], set[str]], out_path: Path) -> None:
    labels = list(PAIRS.values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, tier in zip(axes, TIERS):
        all_genes: Counter = Counter()
        for l in labels:
            for g in hubs.get((l, tier), set()):
                all_genes[g] += 1

        counts = Counter(all_genes.values())
        ns = [1, 2, 3, 4]
        heights = [counts.get(n, 0) for n in ns]
        bar_cols = ["#cccccc", "#a8c5e2", "#457b9d", "#1d3557"]
        bars = ax.bar([str(n) for n in ns], heights, color=bar_cols,
                      edgecolor="#333333", linewidth=0.6)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, str(h),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("Number of subtypes sharing the hub gene", fontsize=9)
        ax.set_ylabel("Number of genes", fontsize=9)
        ax.set_title(TIER_LABELS[tier], fontsize=10, fontweight="bold",
                     color=TIER_COLORS[tier])
        ax.set_ylim(0, max(heights) * 1.2 + 2 if heights else 5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Hub gene sharing across subtypes (ER, HER2, TNBC, TNBC_BRCA1)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Presence/absence heatmap for recurring genes
# ---------------------------------------------------------------------------

def plot_presence_absence(
    hubs: dict[tuple[str, str], set[str]],
    tier: str,
    min_conditions: int,
    out_path: Path,
    max_genes: int = 60,
) -> pd.DataFrame:
    labels = list(PAIRS.values())
    all_genes: Counter = Counter()
    for l in labels:
        for g in hubs.get((l, tier), set()):
            all_genes[g] += 1

    recurring = [g for g, n in all_genes.items() if n >= min_conditions]
    if not recurring:
        log.info("  %s: no genes in %d+ conditions — skipping", tier, min_conditions)
        return pd.DataFrame()

    # Sort by count desc, then alphabetically
    recurring = sorted(recurring, key=lambda g: (-all_genes[g], g))[:max_genes]

    mat = np.array([[1 if g in hubs.get((l, tier), set()) else 0
                     for l in labels]
                    for g in recurring])

    fig_h = max(4, len(recurring) * 0.28 + 1.5)
    fig, ax = plt.subplots(figsize=(5.5, fig_h))

    cmap = matplotlib.colors.ListedColormap(["#f0f4f8", TIER_COLORS[tier]])
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(recurring)))
    ax.set_yticklabels(recurring, fontsize=7.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Grid
    for x in np.arange(-0.5, len(labels), 1):
        ax.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, len(recurring), 1):
        ax.axhline(y, color="white", lw=0.8)

    # Count annotation on right
    for i, g in enumerate(recurring):
        n = all_genes[g]
        ax.text(len(labels) - 0.5 + 0.6, i, f"n={n}", va="center",
                fontsize=7, color="#555555")

    ax.set_title(
        f"{TIER_LABELS[tier]} — genes in ≥{min_conditions} subtypes  ({len(recurring)} shown)",
        fontsize=10, fontweight="bold", color=TIER_COLORS[tier], pad=12,
    )

    present_patch = mpatches.Patch(color=TIER_COLORS[tier], label="Hub in subtype")
    absent_patch = mpatches.Patch(color="#f0f4f8", label="Not a hub",
                                  edgecolor="#cccccc", linewidth=0.5)
    ax.legend(handles=[present_patch, absent_patch], loc="lower right",
              bbox_to_anchor=(1.35, 0), fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)

    # Return CSV-ready table
    rows = []
    for g in recurring:
        row = {"gene": g, "n_conditions": all_genes[g]}
        for l in labels:
            row[l] = int(g in hubs.get((l, tier), set()))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Jaccard CSV export
# ---------------------------------------------------------------------------

def export_jaccard_csv(hubs: dict[tuple[str, str], set[str]], tier: str, out_path: Path) -> None:
    labels = list(PAIRS.values())
    rows = []
    for l1, l2 in combinations(labels, 2):
        s1 = hubs.get((l1, tier), set())
        s2 = hubs.get((l2, tier), set())
        inter = sorted(s1 & s2)
        union = len(s1 | s2)
        jac = len(inter) / union if union else 0.0
        rows.append({
            "pair_A": l1, "pair_B": l2,
            "n_intersection": len(inter),
            "n_union": union,
            "jaccard": round(jac, 4),
            "shared_genes": "|".join(inter),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    hubs = load_hub_sets()

    # 1. Jaccard heatmap (all 3 tiers in one figure)
    plot_jaccard_heatmap(hubs, OUTPUT_ROOT / "jaccard_heatmap_all_tiers.png")

    # 2. Sharing bar chart
    plot_sharing_bars(hubs, OUTPUT_ROOT / "sharing_bar_chart.png")

    # 3. Per-tier presence/absence heatmaps + CSVs
    all_recurring: list[pd.DataFrame] = []
    for tier in TIERS:
        min_cond = 2
        df = plot_presence_absence(
            hubs, tier, min_conditions=min_cond,
            out_path=OUTPUT_ROOT / f"presence_absence_{tier}.png",
        )
        if not df.empty:
            df["tier"] = tier
            all_recurring.append(df)
            df.to_csv(OUTPUT_ROOT / f"recurring_genes_{tier}.csv", index=False)

        export_jaccard_csv(hubs, tier, OUTPUT_ROOT / f"jaccard_{tier}.csv")

    # 4. Combined recurring genes table
    if all_recurring:
        combined = pd.concat(all_recurring, ignore_index=True)
        combined.to_csv(OUTPUT_ROOT / "recurring_genes_all_tiers.csv", index=False)
        log.info("Combined recurring genes: %d rows", len(combined))

    # 5. Universal S_ctrl genes (all 4 subtypes)
    labels = list(PAIRS.values())
    universal = sorted(
        g for g in set().union(*[hubs.get((l, "S_ctrl"), set()) for l in labels])
        if all(g in hubs.get((l, "S_ctrl"), set()) for l in labels)
    )
    pd.DataFrame({"gene": universal}).to_csv(
        OUTPUT_ROOT / "universal_sctrl_genes.csv", index=False
    )
    log.info("Universal S_ctrl hub genes (all 4 subtypes): %d  %s", len(universal), universal)

    log.info("Done. Results in %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
