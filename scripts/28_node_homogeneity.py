#!/usr/bin/env python3
# flake8: noqa: E501
"""Node homogeneity analysis on CSD networks.

For each gene in the union network, computes homogeneity score H:

    H_i = sum_{j in {C,S,D}} (k_{j,i} / k_i)^2

H = 1  → node connected exclusively by one edge type (maximally homogeneous)
H = 1/3 → node has equal C, S, D connections (maximally mixed)

Outputs per pair (results/15_node_homogeneity/{pair}/):
  {pair}_node_homogeneity.csv   — per-gene H, degree, k_C, k_S, k_D
  {pair}_homogeneity_boxplot.png — H vs degree boxplot  (Fig A)
  {pair}_ternary_heatmap.png    — ternary heatmap of edge type fractions (Fig B)
  {pair}_venn_diagram.png        — Venn of genes per edge type (Fig C)

Input: results/14_csd_networks/{pair}/{pair}_differential_edges_permutation.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpltern
import numpy as np
import pandas as pd
from matplotlib_venn import venn3

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.network_utils import resolve_base

#constants
_BG   = "#fafafa"
_EDGE_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}
_DPI  = 150


def compute_homogeneity(edges_df: pd.DataFrame) -> pd.DataFrame:
    #compute per-gene H score and edge type degree breakdown
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"C": 0, "S": 0, "D": 0})
    for _, row in edges_df.iterrows():
        lt = row["link_type"]
        if lt not in ("C", "S", "D"):
            continue
        counts[row["gene_a"]][lt] += 1
        counts[row["gene_b"]][lt] += 1

    rows = []
    for gene, c in counts.items():
        kc, ks, kd = c["C"], c["S"], c["D"]
        k = kc + ks + kd
        if k == 0:
            continue
        H = (kc / k) ** 2 + (ks / k) ** 2 + (kd / k) ** 2
        rows.append({"gene": gene, "k": k, "k_C": kc, "k_S": ks, "k_D": kd,
                     "fC": kc / k, "fS": ks / k, "fD": kd / k, "H": H})

    return pd.DataFrame(rows).sort_values("k", ascending=False).reset_index(drop=True)


def plot_boxplot(df: pd.DataFrame, pair_name: str, out_path: Path) -> None:
    #h vs degree boxplot; bin individual degrees up to 10 then grouped
    #bin degrees: individual up to 10, then groups
    def bin_degree(k):
        if k <= 10:
            return str(k)
        elif k <= 20:
            return "11-20"
        elif k <= 50:
            return "21-50"
        elif k <= 100:
            return "51-100"
        else:
            return ">100"

    df = df[df["k"] >= 2].copy()
    df["degree_bin"] = df["k"].apply(bin_degree)

    bin_order = [str(i) for i in range(2, 11)] + ["11-20", "21-50", "51-100", ">100"]
    bin_order = [b for b in bin_order if b in df["degree_bin"].values]

    grouped = [df[df["degree_bin"] == b]["H"].values for b in bin_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    bp = ax.boxplot(grouped, patch_artist=True, medianprops=dict(color="#c0392b", linewidth=2),
                    boxprops=dict(facecolor="#d6eaf8", color="#555"),
                    whiskerprops=dict(color="#555"), capprops=dict(color="#555"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#888"))

    #overlay means as red squares
    for i, vals in enumerate(grouped):
        if len(vals):
            ax.plot(i + 1, np.mean(vals), "rs", markersize=5, zorder=5)

    ax.set_xticks(range(1, len(bin_order) + 1))
    ax.set_xticklabels(bin_order, fontsize=8)
    ax.set_xlabel("Node degree (k)", fontsize=10)
    ax.set_ylabel("Homogeneity score H", fontsize=10)
    ax.set_ylim(0.25, 1.05)
    ax.axhline(1 / 3, color="#aaa", linestyle="--", linewidth=0.8, label="H = 1/3 (fully mixed)")
    ax.axhline(1.0,   color="#aaa", linestyle=":",  linewidth=0.8, label="H = 1 (fully homogeneous)")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_title(f"Node homogeneity by degree\n{pair_name.replace('__vs__', ' vs ')}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


def plot_ternary(df: pd.DataFrame, pair_name: str, out_path: Path) -> None:
    #ternary scatter of (fC, fS, fD) fractions per gene
    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor(_BG)
    ax = fig.add_subplot(projection="ternary")

    #scatter each gene as a point on the ternary axes
    t = df["fC"].values   # top    = C
    l = df["fS"].values   # left   = S
    r = df["fD"].values   # right  = D

    ax.scatter(t, l, r, s=8, alpha=0.45, color="#2b6fb0", linewidths=0)

    ax.set_tlabel("C — Conserved", fontsize=10)
    ax.set_llabel("S — Specific", fontsize=10)
    ax.set_rlabel("D — Differentiated", fontsize=10)
    ax.taxis.set_label_position("tick1")
    ax.laxis.set_label_position("tick1")
    ax.raxis.set_label_position("tick1")

    ax.set_title(f"Edge type fractions per gene\n{pair_name.replace('__vs__', ' vs ')}",
                 fontsize=11, pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


def plot_venn(df: pd.DataFrame, pair_name: str, out_path: Path) -> None:
    #venn diagram of genes involved in C, S, D interactions
    set_C = set(df[df["k_C"] > 0]["gene"])
    set_S = set(df[df["k_S"] > 0]["gene"])
    set_D = set(df[df["k_D"] > 0]["gene"])

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    v = venn3([set_C, set_S, set_D], set_labels=("C — Conserved", "S — Specific", "D — Differentiated"),
              set_colors=(_EDGE_COLORS["C"], _EDGE_COLORS["S"], _EDGE_COLORS["D"]),
              alpha=0.55, ax=ax)

    ax.set_title(f"Genes per interaction type\n{pair_name.replace('__vs__', ' vs ')}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, facecolor=_BG)
    plt.close(fig)


def run_pair(pair_name: str, network_root: Path, out_root: Path) -> None:
    edges_file = network_root / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    if not edges_file.exists():
        print(f"[warn] {pair_name}: missing edge file, skipping")
        return

    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    print(f"[{pair_name}]")
    edges_df = pd.read_csv(edges_file)
    if edges_df.empty:
        print("  [warn] no edges, skipping")
        return

    df = compute_homogeneity(edges_df)
    print(f"  genes: {len(df)}  |  median H: {df['H'].median():.3f}  |  mean H: {df['H'].mean():.3f}")

    #save homogeneity csv and three diagnostic figures
    csv_path = pair_out / f"{pair_name}_node_homogeneity.csv"
    df.to_csv(csv_path, index=False)
    print(f"  saved → {csv_path}")

    plot_boxplot(df, pair_name, pair_out / f"{pair_name}_homogeneity_boxplot.png")
    print(f"  saved → boxplot")

    plot_ternary(df, pair_name, pair_out / f"{pair_name}_ternary_heatmap.png")
    print(f"  saved → ternary heatmap")

    plot_venn(df, pair_name, pair_out / f"{pair_name}_venn_diagram.png")
    print(f"  saved → venn diagram")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None, help="Run one pair only")
    parser.add_argument("--network-dir", default="results/14_csd_networks")
    parser.add_argument("--output-dir",  default="results/15_node_homogeneity")
    args = parser.parse_args()

    #resolve paths and discover pairs
    network_root = resolve_base(args.network_dir)
    out_root     = resolve_base(args.output_dir)

    pairs = [args.pair] if args.pair else sorted(p.name for p in network_root.iterdir() if p.is_dir())

    #run homogeneity analysis for each pair
    for pair in pairs:
        run_pair(pair, network_root, out_root)

    print(f"\nDone → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
