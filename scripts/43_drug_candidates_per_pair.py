"""Per-network drug candidate rankings and summary tables."""


import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import networkx as nx

#set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 12)
plt.rcParams["font.size"] = 10


def get_pair_names(pair_dirs):
    #extract pair names from directory structure
    pair_path = Path(pair_dirs)
    pair_folders = sorted([d.name for d in pair_path.iterdir() if d.is_dir() and d.name.startswith("pair_")])
    return pair_folders


def load_interactions_file(input_dir):
    #load interactions file to get drug-gene-pair mappings
    interactions_file = Path(input_dir) / "03_hub_gene_drug_interactions_raw.csv"
    if not interactions_file.exists():
        return None
    return pd.read_csv(interactions_file)


def get_drugs_per_pair(candidates_df, interactions_df, pair_name):
    #get drugs and their scores for a specific pair
    if interactions_df is None:
        return None
    
    #filter interactions for this pair - check if pair_name is in the hub_pairs string
    pair_interactions = interactions_df[interactions_df["hub_pairs"].str.contains(pair_name, na=False)]
    
    if len(pair_interactions) == 0:
        return None
    
    #get unique drugs in this pair
    pair_drugs = pair_interactions["drug_name"].unique()
    
    #filter candidates to only those in this pair
    pair_candidates = candidates_df[candidates_df["canonical_drug_name"].isin(pair_drugs)].copy()
    
    if len(pair_candidates) == 0:
        return None
    
    return pair_candidates.sort_values("candidate_score", ascending=False)


def plot_per_pair_top_drugs(candidates_df, interactions_df, pair_names, output_path, top_n=10):
    #create figure showing top drugs for each pair
    n_pairs = len(pair_names)
    
    fig = plt.figure(figsize=(18, 3 * n_pairs))
    gs = GridSpec(n_pairs, 1, figure=fig, hspace=0.4)
    
    pair_drug_data = {}
    
    for idx, pair_name in enumerate(pair_names):
        ax = fig.add_subplot(gs[idx])
        
        pair_candidates = get_drugs_per_pair(candidates_df, interactions_df, pair_name)
        
        if pair_candidates is None or len(pair_candidates) == 0:
            ax.text(0.5, 0.5, f"No drugs for {pair_name}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        
        pair_drug_data[pair_name] = pair_candidates
        
        #top N drugs for this pair
        top_pair = pair_candidates.head(top_n).copy()
        top_pair = top_pair.sort_values("candidate_score", ascending=True)
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_pair)))
        ax.barh(range(len(top_pair)), top_pair["candidate_score"], color=colors)
        
        ax.set_yticks(range(len(top_pair)))
        ax.set_yticklabels(top_pair["canonical_drug_name"], fontsize=9)
        ax.set_xlabel("Candidate Score", fontweight="bold")
        ax.set_title(f"{pair_name} — Top {min(top_n, len(pair_candidates))} Drug Candidates", fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        
        #add score labels
        for i, (idx_row, row) in enumerate(top_pair.iterrows()):
            ax.text(row["candidate_score"] + 0.5, i, f"{row['candidate_score']:.1f}", va="center", fontsize=8)
    
    plt.suptitle("Drug Candidates per Tumor Condition/Pair", fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(output_path / "10_per_pair_top_drugs.png", dpi=300, bbox_inches="tight")
    print(f"Saved: 10_per_pair_top_drugs.png", file=sys.stderr)
    plt.close(fig)
    
    return pair_drug_data


def _bipartite_positions(G, left_nodes, right_nodes, left_x=-1.3, right_x=1.3, y_span=None):
    """Create deterministic two-column positions for bipartite graphs.

    y_span scales dynamically so that nodes always have at least 0.28 units of
    vertical separation regardless of how many nodes are on the taller side.
    """
    left_sorted = sorted(left_nodes, key=lambda n: (-G.degree(n), str(n)))
    right_sorted = sorted(right_nodes, key=lambda n: (-G.degree(n), str(n)))

    if y_span is None:
        n_max = max(len(left_sorted), len(right_sorted), 1)
        y_span = max(1.7, n_max * 0.28)

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


def _draw_pair_network(ax, G, pair_candidates, pair_name, label_size=6):
    #render one pair network using a readable bipartite layout
    drug_nodes = [n for n in G.nodes() if n in pair_candidates["canonical_drug_name"].values]
    gene_nodes = [n for n in G.nodes() if n not in drug_nodes]

    pos = _bipartite_positions(G, drug_nodes, gene_nodes, left_x=-1.35, right_x=1.35)

    edge_weights = [min(3.2, max(0.5, G[u][v]["weight"] / 22.0)) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.2,
        width=edge_weights,
        edge_color="#6b7280",
        connectionstyle="arc3,rad=0.05",
    )

    drug_color_map = {}
    for _, drug_row in pair_candidates.iterrows():
        if drug_row["canonical_drug_name"] in drug_nodes:
            drug_color_map[drug_row["canonical_drug_name"]] = "#2ecc71" if drug_row["approved_any"] else "#95a5a6"

    drug_colors = [drug_color_map.get(n, "#95a5a6") for n in drug_nodes]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=drug_nodes,
        node_color=drug_colors,
        node_size=920,
        ax=ax,
        edgecolors="black",
        linewidths=1.3,
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=gene_nodes,
        node_color="#3498db",
        node_size=620,
        ax=ax,
        node_shape="s",
        edgecolors="black",
        linewidths=1.0,
        alpha=0.95,
    )

    for node in drug_nodes:
        x_val, y_val = pos[node]
        label = node[:24] + "..." if len(node) > 24 else node
        ax.text(
            x_val - 0.10,
            y_val,
            label,
            ha="right",
            va="center",
            fontsize=label_size,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.6},
        )

    for node in gene_nodes:
        x_val, y_val = pos[node]
        label = node[:16] + "..." if len(node) > 16 else node
        ax.text(
            x_val + 0.10,
            y_val,
            label,
            ha="left",
            va="center",
            fontsize=label_size,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.6},
        )

    ax.set_xlim(-2.45, 2.45)
    title = f"{pair_name} ({len(drug_nodes)} drugs, {len(gene_nodes)} genes)"
    ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
    ax.axis("off")


def plot_per_pair_networks(candidates_df, interactions_df, pair_names, output_path):
    #create network diagrams for each pair with readable bipartite layout
    n_pairs = len(pair_names)

    #keep each panel large enough for readable labels.
    #height per row scales with the number of drugs in the largest pair (cap at 15).
    MAX_DRUGS_PER_PAIR = 15
    def _n_pair_drugs(p):
        df = get_drugs_per_pair(candidates_df, interactions_df, p)
        return 0 if df is None else min(MAX_DRUGS_PER_PAIR, len(df))

    max_drugs = max((_n_pair_drugs(p) for p in pair_names), default=10)
    row_height = max(8.0, max_drugs * 0.55)
    n_cols = 1
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(25, row_height * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.32, wspace=0.12)

    for idx, pair_name in enumerate(pair_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        pair_candidates = get_drugs_per_pair(candidates_df, interactions_df, pair_name)
        if pair_candidates is not None:
            pair_candidates = pair_candidates.head(MAX_DRUGS_PER_PAIR)
        
        if pair_candidates is None or len(pair_candidates) == 0:
            ax.text(0.5, 0.5, f"No drugs for {pair_name}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        
        #build network for this pair
        G = nx.Graph()
        
        #add nodes and edges
        for _, drug_row in pair_candidates.iterrows():
            drug_name = drug_row["canonical_drug_name"]
            hub_genes_str = drug_row.get("targeted_hub_genes", "")
            
            if isinstance(hub_genes_str, str) and hub_genes_str:
                genes = [g.strip() for g in str(hub_genes_str).split("|")]
                for gene in genes:
                    if gene:
                        G.add_edge(drug_name, gene, weight=drug_row["candidate_score"])
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, f"No network for {pair_name}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        
        _draw_pair_network(ax, G, pair_candidates, pair_name, label_size=6)

        #save full-size per-pair image as well.
        n_drug_nodes = len([n for n in G.nodes() if n in pair_candidates["canonical_drug_name"].values])
        single_height = max(12, n_drug_nodes * 0.6)
        single_fig, single_ax = plt.subplots(figsize=(24, single_height))
        _draw_pair_network(single_ax, G, pair_candidates, pair_name, label_size=8)
        safe_pair = pair_name.replace(" ", "_").replace("|", "_").replace("/", "_")
        single_file = output_path / f"11_network_{safe_pair}.png"
        single_fig.savefig(single_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {single_file.name}", file=sys.stderr)
        plt.close(single_fig)
    
    plt.suptitle("Per-Pair Drug-Gene Networks (Bipartite Layout)", fontsize=16, fontweight="bold", y=0.997)
    fig.savefig(output_path / "11_per_pair_networks.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: 11_per_pair_networks.png", file=sys.stderr)
    plt.close(fig)


def plot_drug_scores_across_pairs(candidates_df, interactions_df, pair_names, output_path, top_n=15):
    #heatmap showing drug scores across all pairs
    #get top global drugs
    top_global = candidates_df.head(top_n).copy()
    
    #build matrix: drugs × pairs
    matrix_data = []
    drugs = []
    
    for _, drug_row in top_global.iterrows():
        drug_name = drug_row["canonical_drug_name"]
        drugs.append(drug_name)
        pair_scores = []
        
        for pair_name in pair_names:
            #find interactions for this drug in this pair
            pair_interactions = interactions_df[interactions_df["hub_pairs"].str.contains(pair_name, na=False)]
            pair_drug_interactions = pair_interactions[pair_interactions["drug_name"] == drug_name]
            
            if len(pair_drug_interactions) > 0:
                #use mean interaction score for this pair
                score = pair_drug_interactions["interaction_score"].mean()
            else:
                score = 0
            
            pair_scores.append(score)
        
        matrix_data.append(pair_scores)
    
    matrix = np.array(matrix_data)
    
    fig, ax = plt.subplots(figsize=(12, len(drugs) * 0.5 + 2))
    
    sns.heatmap(matrix, xticklabels=[p[:25] for p in pair_names], yticklabels=drugs, cmap="YlOrRd", 
               ax=ax, cbar_kws={"label": "Mean Interaction Score"}, annot=True, fmt=".2f", linewidths=0.5)
    
    ax.set_xlabel("Tumor Condition/Pair", fontsize=11, fontweight="bold")
    ax.set_ylabel("Drug Candidates", fontsize=11, fontweight="bold")
    ax.set_title(f"Drug Scores Across Conditions (Top {top_n} Global Drugs)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    
    fig.savefig(output_path / "12_drug_scores_across_pairs.png", dpi=300, bbox_inches="tight")
    print(f"Saved: 12_drug_scores_across_pairs.png", file=sys.stderr)
    plt.close(fig)


def export_per_pair_tables(candidates_df, interactions_df, pair_names, output_path, top_n=10):
    #export per-pair drug rankings to CSV files
    summary_rows = []
    
    for pair_name in pair_names:
        pair_candidates = get_drugs_per_pair(candidates_df, interactions_df, pair_name)
        
        if pair_candidates is None or len(pair_candidates) == 0:
            continue
        
        #save per-pair CSV
        output_file = output_path / f"pair_{pair_name.replace(' ', '_').replace('|', '_')}_top_drugs.csv"
        pair_candidates.head(top_n).to_csv(output_file, index=False)
        print(f"Saved: {output_file.name}", file=sys.stderr)
        
        #collect summary (top 5)
        for rank, (_, drug_row) in enumerate(pair_candidates.head(5).iterrows(), 1):
            summary_rows.append({
                "pair": pair_name,
                "rank": rank,
                "drug_name": drug_row["canonical_drug_name"],
                "score": drug_row["candidate_score"],
                "approved": drug_row["approved_any"],
                "hub_targets": drug_row["targeted_hub_genes"],
                "interactions": drug_row["interactions_count"],
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / "00_per_pair_summary.csv", index=False)
    print(f"Saved: 00_per_pair_summary.csv", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-pair/condition drug analysis")
    parser.add_argument("--input-dir", required=True, help="Input directory with drug results")
    parser.add_argument("--pair-dirs", required=True, help="Path to pair directories")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis")
    parser.add_argument("--top-n", type=int, default=10, help="Top N drugs per pair to show")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    #load data
    print(f"Loading drug candidates...", file=sys.stderr)
    candidates_file = Path(args.input_dir) / "05_drug_candidates_ranked.csv"
    candidates_df = pd.read_csv(candidates_file)
    
    print(f"Loading interactions...", file=sys.stderr)
    interactions_df = load_interactions_file(args.input_dir)
    
    if interactions_df is None:
        print("ERROR: Could not find interactions file", file=sys.stderr)
        sys.exit(1)
    
    print(f"Identifying pairs...", file=sys.stderr)
    #extract unique pairs from the hub_pairs column (pipe-separated)
    all_pairs = set()
    for pairs_str in interactions_df["hub_pairs"].fillna(""):
        if isinstance(pairs_str, str):
            for pair in pairs_str.split("|"):
                pair = pair.strip()
                if pair:
                    all_pairs.add(pair)
    
    pair_names = sorted(list(all_pairs))
    print(f"  Found {len(pair_names)} pairs: {', '.join(pair_names)}", file=sys.stderr)
    
    #generate visualizations
    print(f"Generating per-pair drug rankings...", file=sys.stderr)
    plot_per_pair_top_drugs(candidates_df, interactions_df, pair_names, output_path, top_n=args.top_n)
    
    print(f"Generating per-pair networks...", file=sys.stderr)
    plot_per_pair_networks(candidates_df, interactions_df, pair_names, output_path)
    
    print(f"Generating drug scores heatmap...", file=sys.stderr)
    plot_drug_scores_across_pairs(candidates_df, interactions_df, pair_names, output_path, top_n=15)
    
    print(f"Exporting per-pair tables...", file=sys.stderr)
    export_per_pair_tables(candidates_df, interactions_df, pair_names, output_path, top_n=args.top_n)
    
    print(f"\nPer-pair analysis complete!", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)
    print(f"Files created:", file=sys.stderr)
    print(f"  00_per_pair_summary.csv - Summary table of top 5 drugs per pair", file=sys.stderr)
    print(f"  10_per_pair_top_drugs.png - Top {args.top_n} drugs per pair", file=sys.stderr)
    print(f"  11_per_pair_networks.png - Drug-gene networks per pair", file=sys.stderr)
    print(f"  12_drug_scores_across_pairs.png - Heat map of drug scores across conditions", file=sys.stderr)
    print(f"  pair_*_top_drugs.csv - Individual CSV for each pair", file=sys.stderr)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
