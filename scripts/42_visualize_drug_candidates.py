#!/usr/bin/env python3
"""
Visualize drug candidate results with multiple plot types.

Usage:
  python 51_visualize_drug_candidates.py \
    --input-dir results/24_drug_targets \
    --output-dir results/24_drug_targets/visualizations \
    --top-n 15
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import networkx as nx

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["font.size"] = 10


def load_data(input_dir):
    """Load drug candidates and summary data."""
    input_path = Path(input_dir)
    
    # Load ranked candidates
    candidates_file = input_path / "05_drug_candidates_ranked.csv"
    if not candidates_file.exists():
        raise FileNotFoundError(f"Missing: {candidates_file}")
    
    candidates_df = pd.read_csv(candidates_file)
    
    # Load summary
    summary_file = input_path / "06_summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    
    # Load interactions
    interactions_file = input_path / "03_hub_gene_drug_interactions_raw.csv"
    interactions_df = None
    if interactions_file.exists():
        interactions_df = pd.read_csv(interactions_file)
    
    return candidates_df, summary, interactions_df


def plot_top_candidates_bar(candidates_df, top_n=15, ax=None):
    """Bar chart of top N candidates by score."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    top_df = candidates_df.head(top_n).copy()
    top_df = top_df.sort_values("candidate_score", ascending=True)  # For horizontal bar
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_df)))
    bars = ax.barh(range(len(top_df)), top_df["candidate_score"], color=colors)
    
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["canonical_drug_name"], fontsize=9)
    ax.set_xlabel("Candidate Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Drug Candidates", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    # Add score labels
    for i, (idx, row) in enumerate(top_df.iterrows()):
        ax.text(row["candidate_score"] + 1, i, f"{row['candidate_score']:.1f}", va="center", fontsize=9)
    
    return ax.get_figure() if ax is None else None


def plot_approval_status(candidates_df, ax=None):
    """Pie chart of drug approval status."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Count approval statuses - check if we have true/false values
    approval_counts = candidates_df["approved_any"].value_counts()
    
    colors = {True: "#2ecc71", False: "#95a5a6"}
    pie_colors = [colors.get(status, "#95a5a6") for status in approval_counts.index]
    labels = ["Approved" if status else "Not Approved" for status in approval_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        approval_counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=90,
        textprops={"fontsize": 10},
    )
    
    ax.set_title("Drug Approval Status Distribution", fontsize=12, fontweight="bold")
    
    return ax.get_figure() if ax is None else None


def plot_score_composition(candidates_df, top_n=10, ax=None):
    """Stacked bar showing score component breakdown."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    top_df = candidates_df.head(top_n).copy()
    
    # Estimate component contributions
    components = {
        "hub_targets": top_df["hub_genes_targeted"] * 2.2,
        "pair_coverage": top_df["pair_coverage"] * 1.5,
        "interaction_score": top_df["mean_interaction_score"] * 4.0,
        "evidence": top_df["mean_evidence_score"] * 0.15,
    }
    
    x = np.arange(len(top_df))
    bottom = np.zeros(len(top_df))
    
    colors_stack = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    
    for (component, values), color in zip(components.items(), colors_stack):
        ax.bar(x, values, bottom=bottom, label=component.replace("_", " ").title(), color=color)
        bottom += values
    
    ax.set_xticks(x)
    ax.set_xticklabels(top_df["canonical_drug_name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score Contribution", fontsize=11, fontweight="bold")
    ax.set_title("Score Component Breakdown (Top 10)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    return ax.get_figure() if ax is None else None


def plot_hub_count_vs_score(candidates_df, ax=None):
    """Scatter plot: hub count vs rank score."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by approval status
    colors = ["#2ecc71" if approved else "#95a5a6" for approved in candidates_df["approved_any"]]
    
    scatter = ax.scatter(
        candidates_df["hub_genes_targeted"].astype(int),
        candidates_df["candidate_score"],
        s=candidates_df["interactions_count"] * 3,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    
    ax.set_xlabel("Number of Hub Targets", fontsize=11, fontweight="bold")
    ax.set_ylabel("Candidate Score", fontsize=11, fontweight="bold")
    ax.set_title("Hub Count vs Score (size = interaction count)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Add legend for approval status
    ax.scatter([], [], s=100, c="#2ecc71", label="Approved", edgecolors="black", linewidth=0.5)
    ax.scatter([], [], s=100, c="#95a5a6", label="Not Approved", edgecolors="black", linewidth=0.5)
    ax.legend(title="Status", fontsize=9)
    
    return ax.get_figure() if ax is None else None


def plot_interaction_source_distribution(candidates_df, ax=None):
    """Bar chart showing DGIdb interaction sources."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Parse interaction_sources (comma-separated)
    source_counts = {}
    for sources_str in candidates_df["source_dbs"].fillna(""):
        if isinstance(sources_str, str):
            for source in sources_str.split("|"):
                source = source.strip()
                if source:
                    source_counts[source] = source_counts.get(source, 0) + 1
    
    if source_counts:
        sources = sorted(source_counts.keys(), key=lambda x: source_counts[x], reverse=True)
        counts = [source_counts[s] for s in sources]
        
        bars = ax.barh(sources, counts, color=plt.cm.Set3(np.linspace(0, 1, len(sources))))
        ax.set_xlabel("Number of Drugs from Source", fontsize=11, fontweight="bold")
        ax.set_title("DGIdb Interaction Sources", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        for i, (source, count) in enumerate(zip(sources, counts)):
            ax.text(count + 0.5, i, str(count), va="center", fontsize=10)
    
    return ax.get_figure() if ax is None else None


def _bipartite_positions(G, left_nodes, right_nodes, left_x=-1.4, right_x=1.4, y_span=None):
    """Create deterministic two-column positions for bipartite graphs.

    y_span is computed dynamically so that nodes always have at least 0.28 units
    of vertical separation regardless of how many nodes are on the taller side.
    """
    left_sorted = sorted(left_nodes, key=lambda n: (-G.degree(n), str(n)))
    right_sorted = sorted(right_nodes, key=lambda n: (-G.degree(n), str(n)))

    if y_span is None:
        n_max = max(len(left_sorted), len(right_sorted), 1)
        y_span = max(1.8, n_max * 0.28)

    pos = {}

    if left_sorted:
        left_y = np.linspace(y_span, -y_span, len(left_sorted))
        for node, y_val in zip(left_sorted, left_y):
            pos[node] = (left_x, y_val)

    if right_sorted:
        right_y = np.linspace(y_span, -y_span, len(right_sorted))
        for node, y_val in zip(right_sorted, right_y):
            pos[node] = (right_x, y_val)

    return pos


def plot_drug_gene_network(candidates_df, top_n=15, ax=None):
    """Network diagram showing drug-gene connections in a readable bipartite layout."""
    if ax is None:
        # Scale figure height so nodes have room: at least 0.5 inches per drug node
        n_drugs = min(top_n, len(candidates_df))
        fig_height = max(14, n_drugs * 0.55)
        fig, ax = plt.subplots(figsize=(22, fig_height))
    
    # Create network graph
    G = nx.Graph()
    
    # Use top candidates
    top_drugs = candidates_df.head(top_n)
    
    # Parse gene targets from hub_genes column
    for _, drug_row in top_drugs.iterrows():
        drug_name = drug_row["canonical_drug_name"]
        # targeted_hub_genes is comma-separated or single gene
        hub_genes_str = drug_row.get("targeted_hub_genes", "")
        if isinstance(hub_genes_str, str) and hub_genes_str:
            genes = [g.strip() for g in str(hub_genes_str).split("|")]
            for gene in genes:
                if gene:
                    G.add_edge(drug_name, gene, weight=drug_row["candidate_score"])
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "No network data available", ha="center", va="center", transform=ax.transAxes)
        return ax.get_figure() if ax is None else None
    
    # Separate drugs and genes
    drug_nodes = [n for n in G.nodes() if n in top_drugs["canonical_drug_name"].values]
    gene_nodes = [n for n in G.nodes() if n not in drug_nodes]
    
    # Deterministic bipartite layout: drugs on left, genes on right.
    pos = _bipartite_positions(G, drug_nodes, gene_nodes, left_x=-1.4, right_x=1.4)
    
    # Draw edges with varying thickness based on score
    edge_weights = [min(3.5, max(0.6, G[u][v]["weight"] / 20.0)) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.22,
        width=edge_weights,
        edge_color="#6b7280",
        connectionstyle="arc3,rad=0.06",
    )
    
    # Get colors for drugs based on approval status
    drug_colors = {}
    for _, drug_row in top_drugs.iterrows():
        if drug_row["approved_any"]:
            drug_colors[drug_row["canonical_drug_name"]] = "#2ecc71"
        else:
            drug_colors[drug_row["canonical_drug_name"]] = "#95a5a6"
    
    drug_node_colors = [drug_colors.get(n, "#95a5a6") for n in drug_nodes]
    gene_node_colors = ["#3498db"] * len(gene_nodes)
    
    # Draw drugs - larger nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=drug_nodes,
        node_color=drug_node_colors,
        node_size=1150,
        ax=ax,
        edgecolors="black",
        linewidths=1.6,
        alpha=0.95,
    )
    
    # Draw genes - smaller square nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=gene_nodes,
        node_color=gene_node_colors,
        node_size=760,
        ax=ax,
        node_shape="s",
        edgecolors="black",
        linewidths=1.2,
        alpha=0.95,
    )
    
    # Put labels outside nodes to avoid stacking text over marker shapes.
    for node in drug_nodes:
        x_val, y_val = pos[node]
        label = node[:28] + "..." if len(node) > 28 else node
        ax.text(
            x_val - 0.11,
            y_val,
            label,
            ha="right",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.8},
        )

    for node in gene_nodes:
        x_val, y_val = pos[node]
        label = node[:20] + "..." if len(node) > 20 else node
        ax.text(
            x_val + 0.11,
            y_val,
            label,
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.8},
        )

    ax.set_xlim(-2.55, 2.55)
    ax.margins(y=0.06)

    ax.set_title(f"Drug-Gene Network (Top {top_n} Candidates, Bipartite Layout)", fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=12, label="Approved Drug", markeredgecolor="black", markeredgewidth=1.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=12, label="Other Drug", markeredgecolor="black", markeredgewidth=1.5),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db", markersize=10, label="Hub Gene", markeredgecolor="black", markeredgewidth=1.5),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.95)
    
    return ax.get_figure() if ax is None else None


def plot_drug_gene_heatmap(candidates_df, top_n=20, ax=None):
    """Heatmap of drugs × genes with interaction counts."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get top candidates
    top_drugs = candidates_df.head(top_n).copy()
    
    # Build matrix: drugs × genes
    drugs = []
    genes_all = set()
    drug_gene_counts = {}
    
    for _, drug_row in top_drugs.iterrows():
        drug_name = drug_row["canonical_drug_name"]
        drugs.append(drug_name)
        hub_genes_str = drug_row.get("targeted_hub_genes", "")
        
        if isinstance(hub_genes_str, str) and hub_genes_str:
            genes = [g.strip() for g in str(hub_genes_str).split("|")]
            genes_all.update(genes)
            
            for gene in genes:
                if gene:
                    key = (drug_name, gene)
                    drug_gene_counts[key] = drug_row.get("mean_interaction_score", 1.0)
    
    genes = sorted(list(genes_all))
    
    if len(genes) == 0 or len(drugs) == 0:
        ax.text(0.5, 0.5, "No heatmap data available", ha="center", va="center", transform=ax.transAxes)
        return ax.get_figure() if ax is None else None
    
    # Create matrix
    matrix = np.zeros((len(drugs), len(genes)))
    for i, drug in enumerate(drugs):
        for j, gene in enumerate(genes):
            matrix[i, j] = drug_gene_counts.get((drug, gene), 0)
    
    # Plot heatmap
    sns.heatmap(matrix, xticklabels=genes, yticklabels=drugs, cmap="YlOrRd", 
               ax=ax, cbar_kws={"label": "Interaction Score"}, annot=True, fmt=".1f", linewidths=0.5)
    
    ax.set_xlabel("Hub Genes", fontsize=11, fontweight="bold")
    ax.set_ylabel("Drug Candidates", fontsize=11, fontweight="bold")
    ax.set_title(f"Drug-Gene Interaction Heatmap (Top {top_n})", fontsize=12, fontweight="bold")
    
    return ax.get_figure() if ax is None else None


def plot_summary_stats(summary, ax=None):
    """Text summary of key statistics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axis("off")
    
    # Format text
    stats_text = f"""
DRUG CANDIDATE ANALYSIS SUMMARY

Pairs Processed:           {summary.get('pairs_processed', 'N/A')}
Combined Hub Genes:        {summary.get('combined_hubs', 'N/A')}
Raw DGIdb Interactions:    {summary.get('raw_interactions', 'N/A')}
Synonym Dictionary Rows:   {summary.get('synonym_rows', 'N/A')}
Final Drug Candidates:     {summary.get('drug_candidates', 'N/A')}

Pipeline Status:           {summary.get('status', 'N/A').upper()}
"""
    
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    return ax.get_figure() if ax is None else None


def create_comprehensive_figure(candidates_df, summary, output_path):
    """Create a comprehensive multi-panel figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel 1: Top candidates
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    top_df = candidates_df.head(15).copy()
    top_df = top_df.sort_values("candidate_score", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_df)))
    ax1.barh(range(len(top_df)), top_df["candidate_score"], color=colors)
    ax1.set_yticks(range(len(top_df)))
    ax1.set_yticklabels(top_df["canonical_drug_name"], fontsize=9)
    ax1.set_xlabel("Score", fontweight="bold")
    ax1.set_title("Top 15 Drug Candidates", fontweight="bold", fontsize=12)
    ax1.grid(axis="x", alpha=0.3)
    
    # Panel 2: Approval status pie
    ax2 = fig.add_subplot(gs[0, 2])
    approval_counts = candidates_df["approved_any"].value_counts()
    colors_pie = ["#2ecc71", "#95a5a6"]
    labels_pie = ["Approved" if idx else "Not Approved" for idx in approval_counts.index]
    ax2.pie(approval_counts.values, labels=labels_pie, autopct="%1.0f%%", colors=colors_pie[:len(approval_counts)])
    ax2.set_title("Approval Status", fontweight="bold", fontsize=11)
    
    # Panel 3: Hub count vs score
    ax3 = fig.add_subplot(gs[1, 2])
    colors_scatter = ["#2ecc71" if approved else "#95a5a6" for approved in candidates_df["approved_any"]]
    ax3.scatter(candidates_df["hub_genes_targeted"].astype(int), candidates_df["candidate_score"], 
               s=candidates_df["interactions_count"] * 2, c=colors_scatter, alpha=0.6, edgecolors="black", linewidth=0.5)
    ax3.set_xlabel("Hub Count", fontsize=9, fontweight="bold")
    ax3.set_ylabel("Score", fontsize=9, fontweight="bold")
    ax3.set_title("Hubs vs Score", fontweight="bold", fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    stats_text = f"""Pairs: {summary.get('pairs_processed', 'N/A')} | Hubs: {summary.get('combined_hubs', 'N/A')} | Raw Interactions: {summary.get('raw_interactions', 'N/A')} | Candidates: {summary.get('drug_candidates', 'N/A')} | Status: {summary.get('status', 'N/A')}"""
    ax4.text(0.5, 0.5, stats_text, ha="center", va="center", fontsize=11, fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    
    plt.suptitle("Drug Candidate Analysis Summary", fontsize=14, fontweight="bold", y=0.995)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize drug candidate results")
    parser.add_argument("--input-dir", required=True, help="Input directory with drug results")
    parser.add_argument("--output-dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--top-n", type=int, default=15, help="Top N candidates to show in details")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_dir}...", file=sys.stderr)
    candidates_df, summary, interactions_df = load_data(args.input_dir)
    print(f"  Loaded {len(candidates_df)} candidates", file=sys.stderr)
    
    # Create comprehensive figure
    print("Creating comprehensive summary figure...", file=sys.stderr)
    fig = create_comprehensive_figure(candidates_df, summary, output_path)
    comprehensive_path = output_path / "01_comprehensive_summary.png"
    fig.savefig(comprehensive_path, dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: {comprehensive_path}", file=sys.stderr)
    plt.close(fig)
    
    # Create individual plots
    print("Creating individual detail plots...", file=sys.stderr)
    
    # Plot 1: Top candidates
    fig = plt.figure(figsize=(12, 7))
    plot_top_candidates_bar(candidates_df, top_n=args.top_n, ax=plt.gca())
    fig.savefig(output_path / "02_top_candidates_bar.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 02_top_candidates_bar.png", file=sys.stderr)
    plt.close(fig)
    
    # Plot 2: Approval status
    fig = plt.figure(figsize=(8, 6))
    plot_approval_status(candidates_df, ax=plt.gca())
    fig.savefig(output_path / "03_approval_status_pie.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 03_approval_status_pie.png", file=sys.stderr)
    plt.close(fig)
    
    # Plot 3: Hub vs score
    fig = plt.figure(figsize=(10, 7))
    plot_hub_count_vs_score(candidates_df, ax=plt.gca())
    fig.savefig(output_path / "04_hub_count_vs_score.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 04_hub_count_vs_score.png", file=sys.stderr)
    plt.close(fig)
    
    # Plot 4: Score composition
    fig = plt.figure(figsize=(12, 7))
    plot_score_composition(candidates_df, top_n=args.top_n, ax=plt.gca())
    fig.savefig(output_path / "05_score_composition.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 05_score_composition.png", file=sys.stderr)
    plt.close(fig)
    
    # Plot 5: Interaction sources
    if not candidates_df["source_dbs"].isna().all():
        fig = plt.figure(figsize=(12, 7))
        plot_interaction_source_distribution(candidates_df, ax=plt.gca())
        fig.savefig(output_path / "06_interaction_sources.png", dpi=args.dpi, bbox_inches="tight")
        print(f"  Saved: 06_interaction_sources.png", file=sys.stderr)
        plt.close(fig)
    
    # Plot 6: Summary stats
    fig = plt.figure(figsize=(10, 6))
    plot_summary_stats(summary, ax=plt.gca())
    fig.savefig(output_path / "07_summary_stats.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 07_summary_stats.png", file=sys.stderr)
    plt.close(fig)
    
    # Plot 7: Drug-gene network
    print("Creating drug-gene network diagram...", file=sys.stderr)
    n_net = min(15, len(candidates_df))
    n_drug_nodes = n_net
    fig_h = max(14, n_drug_nodes * 0.55)
    fig = plt.figure(figsize=(22, fig_h))
    plot_drug_gene_network(candidates_df, top_n=n_net, ax=plt.gca())
    fig.savefig(output_path / "08_drug_gene_network.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 08_drug_gene_network.png", file=sys.stderr)
    plt.close(fig)
    
    # Plot 8: Drug-gene heatmap
    print("Creating drug-gene heatmap...", file=sys.stderr)
    fig = plt.figure(figsize=(14, 8))
    plot_drug_gene_heatmap(candidates_df, top_n=min(20, len(candidates_df)), ax=plt.gca())
    fig.savefig(output_path / "09_drug_gene_heatmap.png", dpi=args.dpi, bbox_inches="tight")
    print(f"  Saved: 09_drug_gene_heatmap.png", file=sys.stderr)
    plt.close(fig)
    
    print(f"\nAll visualizations saved to: {output_path}", file=sys.stderr)
    print("Files created:", file=sys.stderr)
    print("  01_comprehensive_summary.png - Main summary figure", file=sys.stderr)
    print("  02_top_candidates_bar.png - Ranked bar chart", file=sys.stderr)
    print("  03_approval_status_pie.png - Drug approval distribution", file=sys.stderr)
    print("  04_hub_count_vs_score.png - Scatter plot analysis", file=sys.stderr)
    print("  05_score_composition.png - Score breakdown", file=sys.stderr)
    print("  06_interaction_sources.png - DGIdb sources", file=sys.stderr)
    print("  07_summary_stats.png - Pipeline statistics", file=sys.stderr)
    print("  08_drug_gene_network.png - Drug-gene network diagram", file=sys.stderr)
    print("  09_drug_gene_heatmap.png - Drug-gene interaction heatmap", file=sys.stderr)


if __name__ == "__main__":
    main()
