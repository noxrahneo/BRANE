#!/usr/bin/env python3
# flake8: noqa: E501
"""Check whether chemotherapy targets and GM1 biosynthesis genes appear in persistent network hubs.

Context
-------
Prior work investigated GM1 ganglioside microencapsulation of doxorubicin and paclitaxel
for enhanced drug delivery. This script asks whether the direct molecular targets of those
agents appear as hub genes in any of the BRANE persistent overlap networks.

Targets checked
---------------
Doxorubicin  : TOP2A, TOP2B  (Topoisomerase II — intercalation + topo-II inhibition)
Paclitaxel   : TUBB1, TUBB2A, TUBB3, TUBB4A, TUBA1A, TUBA1B  (tubulin stabilisation)

GM1 biosynthesis pathway (optional)
------------------------------------
ST3GAL5  (GM3 synthase)
B4GALNT1 (GM2/GD2 synthase)
ST8SIA1  (GD3 synthase)
B3GNT5   (lactosylceramide 3-alpha-N-acetyl-D-galactosaminyltransferase)

Outputs
-------
results/27_drug_targeting/gm1_connection_genes.csv
  columns: gene, drug_target_of, pair_name, is_hub, weighted_degree, hub_rank, all_pairs_node

Usage
-----
  python scripts/50_gm1_connection.py
  python scripts/50_gm1_connection.py --networks-root <custom_root> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.network_utils import resolve_base


DEFAULT_NETWORKS_ROOT = (
    "results/23_node_annotation/05_final_networks_with_lfc_ready"
)
DEFAULT_OUTPUT_DIR = "results/27_drug_targeting"
TOP_HUBS_N = 50

DOXORUBICIN_TARGETS = {"TOP2A", "TOP2B"}
PACLITAXEL_TARGETS = {"TUBB1", "TUBB2A", "TUBB3", "TUBB4A", "TUBA1A", "TUBA1B"}
GM1_BIOSYNTHESIS_GENES = {"ST3GAL5", "B4GALNT1", "ST8SIA1", "B3GNT5"}

DRUG_TARGET_MAP: dict[str, str] = {}
for _g in DOXORUBICIN_TARGETS:
    DRUG_TARGET_MAP[_g] = "doxorubicin"
for _g in PACLITAXEL_TARGETS:
    DRUG_TARGET_MAP[_g] = "paclitaxel"
for _g in GM1_BIOSYNTHESIS_GENES:
    DRUG_TARGET_MAP[_g] = "GM1_biosynthesis"

ALL_QUERY_GENES = set(DRUG_TARGET_MAP.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GM1/chemo target hub-network intersection")
    parser.add_argument("--networks-root", default=DEFAULT_NETWORKS_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-hubs", type=int, default=TOP_HUBS_N)
    return parser.parse_args()


def _gene_col(df: pd.DataFrame) -> str:
    """Return the gene column name (handles 'approved_symbol' or 'gene')."""
    for col in ("approved_symbol", "gene"):
        if col in df.columns:
            return col
    raise KeyError(f"No gene column found; available: {df.columns.tolist()}")


def discover_pairs(networks_root: Path) -> list[tuple[str, Path, Path]]:
    """Return (pair_name, edges_csv, tagged_csv) for each pair subfolder."""
    pairs = []
    for pair_dir in sorted(networks_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        viz_dir = pair_dir / "viz_inputs"
        if not viz_dir.is_dir():
            continue
        edges_files = list(viz_dir.glob("*_persistent_edges_edges_viz.csv"))
        tagged_files = list(viz_dir.glob("*_tagged_with_lfc.csv"))
        if edges_files and tagged_files:
            pairs.append((pair_dir.name, edges_files[0], tagged_files[0]))
    return pairs


def compute_weighted_degree(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns gene, weighted_degree sorted descending."""
    wa = edges_df.groupby("gene_a")["weight"].sum().rename("weighted_degree")
    wb = edges_df.groupby("gene_b")["weight"].sum().rename("weighted_degree")
    deg = wa.add(wb, fill_value=0.0).reset_index()
    deg.columns = ["gene", "weighted_degree"]
    return deg.sort_values("weighted_degree", ascending=False).reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    networks_root = Path(resolve_base(args.networks_root))
    output_dir = Path(resolve_base(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(networks_root)
    if not pairs:
        logging.warning("No pair folders found under %s", networks_root)
        return

    rows: list[dict] = []

    for pair_name, edges_path, tagged_path in pairs:
        edges_df = pd.read_csv(edges_path)
        tagged_df = pd.read_csv(tagged_path)
        gene_col = _gene_col(tagged_df)
        all_genes_in_pair = set(tagged_df[gene_col].dropna().str.upper().tolist())

        deg_df = compute_weighted_degree(edges_df)
        deg_df["gene_upper"] = deg_df["gene"].str.upper()
        hubs = set(deg_df.head(args.top_hubs)["gene_upper"].tolist())
        hub_rank = {row["gene_upper"]: i + 1 for i, row in deg_df.head(args.top_hubs).iterrows()}
        hub_degree = {row["gene_upper"]: row["weighted_degree"] for _, row in deg_df.iterrows()}

        for gene in ALL_QUERY_GENES:
            gene_upper = gene.upper()
            in_pair_nodes = gene_upper in all_genes_in_pair
            is_hub = gene_upper in hubs
            rows.append({
                "gene": gene,
                "drug_target_of": DRUG_TARGET_MAP[gene],
                "pair_name": pair_name,
                "is_hub": is_hub,
                "weighted_degree": hub_degree.get(gene_upper, 0.0),
                "hub_rank": hub_rank.get(gene_upper, None),
                "in_pair_nodes": in_pair_nodes,
            })

    result_df = pd.DataFrame(rows)

    output_path = output_dir / "gm1_connection_genes.csv"
    result_df.to_csv(output_path, index=False)
    logging.info("Wrote %s", output_path)

    found_hubs = result_df[result_df["is_hub"]]
    found_nodes = result_df[result_df["in_pair_nodes"] & ~result_df["is_hub"]]

    print("\n=== GM1 / Chemotherapy Target Network Intersection ===\n")

    if found_hubs.empty:
        print("No query genes appear as top-50 hubs in any pair.\n")
    else:
        print("Hub genes (top-50 by weighted degree):")
        for _, row in found_hubs.iterrows():
            print(
                f"  {row['gene']} ({row['drug_target_of']}) "
                f"| pair: {row['pair_name']} "
                f"| rank: {row['hub_rank']} "
                f"| weighted_degree: {row['weighted_degree']:.3f}"
            )
        print()

    if found_nodes.empty:
        print("No query genes appear as non-hub network nodes in any pair.\n")
    else:
        print("Network nodes (present but not top-50 hub):")
        for _, row in found_nodes.iterrows():
            print(
                f"  {row['gene']} ({row['drug_target_of']}) "
                f"| pair: {row['pair_name']} "
                f"| weighted_degree: {row['weighted_degree']:.3f}"
            )
        print()

    # A gene is truly absent only if it does not appear in ANY pair network
    present_any = set(result_df[result_df["in_pair_nodes"]]["gene"].str.upper().tolist())
    absent = [g for g in ALL_QUERY_GENES if g.upper() not in present_any]
    if absent:
        print(f"Not present in any pair network: {', '.join(sorted(absent))}\n")

    print(f"Full results written to: {output_path}")

    _plot_gm1_connection(result_df, output_dir)


def _plot_gm1_connection(result_df: pd.DataFrame, output_dir: Path) -> None:
    """Save a figure showing paclitaxel target and GM1 pathway gene network presence."""
    present = result_df[result_df["in_pair_nodes"]].copy()
    if present.empty:
        logging.info("No query genes present in any network — skipping plot")
        return

    # Short pair labels for x-axis
    pair_labels = {p: p.replace("_tumor__vs__Normal", "").replace("__vs__Normal", "").replace("_", " ").strip()
                   for p in result_df["pair_name"].unique()}

    # Colour by drug group
    colour_map = {
        "doxorubicin": "#e74c3c",
        "paclitaxel": "#2980b9",
        "GM1_biosynthesis": "#27ae60",
    }

    genes_present = sorted(present["gene"].unique())
    pairs_all = sorted(result_df["pair_name"].unique())
    n_genes = len(genes_present)
    n_pairs = len(pairs_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_genes * 0.7 + 2)))

    # LEFT PANEL: dot plot — gene × pair, dot size = weighted degree, colour = drug group
    ax = axes[0]
    for i, gene in enumerate(genes_present):
        drug_group = DRUG_TARGET_MAP.get(gene, "unknown")
        colour = colour_map.get(drug_group, "#888")
        for j, pair in enumerate(pairs_all):
            row = result_df[(result_df["gene"] == gene) & (result_df["pair_name"] == pair)]
            if row.empty or not row.iloc[0]["in_pair_nodes"]:
                continue
            wd = float(row.iloc[0]["weighted_degree"])
            is_hub = bool(row.iloc[0]["is_hub"])
            size = max(60, min(600, wd * 8))
            marker = "*" if is_hub else "o"
            ax.scatter(j, i, s=size, c=colour, marker=marker, edgecolors="black",
                       linewidths=1.2, alpha=0.85, zorder=3)

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(genes_present, fontsize=10)
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels([pair_labels[p] for p in pairs_all], rotation=35, ha="right", fontsize=8)
    ax.set_xlim(-0.5, n_pairs - 0.5)
    ax.set_ylim(-0.5, n_genes - 0.5)
    ax.set_title("Gene Presence Across Pairs\n(dot size = weighted degree, ★ = top-50 hub)", fontsize=11, fontweight="bold")
    ax.grid(axis="both", alpha=0.25)
    ax.set_axisbelow(True)

    # Add legend for drug group colours
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map["doxorubicin"],
               markersize=10, markeredgecolor="black", label="Doxorubicin target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map["paclitaxel"],
               markersize=10, markeredgecolor="black", label="Paclitaxel target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map["GM1_biosynthesis"],
               markersize=10, markeredgecolor="black", label="GM1 biosynthesis"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="grey",
               markersize=13, markeredgecolor="black", label="Top-50 hub"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8, framealpha=0.9)

    # RIGHT PANEL: bar chart — weighted degree per (gene, pair) for genes that are present
    ax2 = axes[1]
    bar_data = present.copy()
    bar_data["pair_short"] = bar_data["pair_name"].map(pair_labels)
    bar_data["label"] = bar_data["gene"] + "\n(" + bar_data["pair_short"] + ")"
    bar_data = bar_data.sort_values(["drug_target_of", "weighted_degree"], ascending=[True, False])

    colours = [colour_map.get(row["drug_target_of"], "#888") for _, row in bar_data.iterrows()]
    bars = ax2.barh(range(len(bar_data)), bar_data["weighted_degree"].values,
                    color=colours, edgecolor="black", linewidth=0.7, alpha=0.85)
    ax2.set_yticks(range(len(bar_data)))
    ax2.set_yticklabels(bar_data["label"].tolist(), fontsize=8)
    ax2.set_xlabel("Weighted Degree in Persistent Network", fontsize=10)
    ax2.set_title("Weighted Degree of Query Genes\nFound in Persistent Overlap Networks", fontsize=11, fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.25)

    # Annotate hub status
    for i, (_, row) in enumerate(bar_data.iterrows()):
        if row["is_hub"]:
            ax2.text(row["weighted_degree"] + 0.3, i, "★ hub", va="center", fontsize=7, color="#c0392b")

    plt.suptitle("Paclitaxel / Doxorubicin Targets and GM1 Biosynthesis Genes\nin BRANE Persistent Overlap Networks",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path = output_dir / "gm1_connection_plot.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Wrote %s", fig_path)
    print(f"Plot written to: {fig_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
