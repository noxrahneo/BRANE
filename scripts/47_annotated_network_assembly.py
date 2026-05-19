"""Earlier annotated network assembly; superseded by script 36."""


from __future__ import annotations

import argparse
import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]

NET_DIR   = REPO_ROOT / "results" / "14_csd_networks"
LFC_DIR   = REPO_ROOT / "results" / "20_node_annotation" / "zzz_03_output_with_lfc"
OUT_ROOT  = REPO_ROOT / "results" / "20_node_annotation"
VIZ_DIR   = REPO_ROOT / "results" / "16_network_viz"

TOP_N_HUBS = 50
DPI        = 200
SEED       = 7

PAIRS = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

# ── colour palettes ────────────────────────────────────────────────────────────

CELL_TYPE_COLOURS: dict[str, str] = {
    "Epithelial":     "#E63946",
    "Basal_Epi":      "#FF6B6B",
    "Luminal_Epi":    "#FF9F9F",
    "Fibroblast":     "#2A9D8F",
    "Stroma":         "#52B788",
    "Endothelial":    "#457B9D",
    "BCell":          "#A8DADC",
    "TCell":          "#6A4C93",
    "NK":             "#9B72CF",
    "Myeloid":        "#F4A261",
    "Plasma":         "#E9C46A",
    "Cycling":        "#264653",
    "Lymphoid_mixed": "#8E9AAF",
    "Unassigned":     "#CCCCCC",
}

MODULE_PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#D4E09B","#F2D7EE","#CBE896","#A8DFF0","#FFB347",
    "#C0C0C0","#B5EAD7","#FF6961","#AEC6CF","#FDFD96",
]

#edge colours matching 16_network_viz style
EDGE_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}

def module_colour(mod_id: int | float | str) -> str:
    try:
        return MODULE_PALETTE[int(mod_id) % len(MODULE_PALETTE)]
    except Exception:
        return "#CCCCCC"

def cell_type_colour(ct: str | float) -> str:
    if pd.isna(ct) or ct == "Unassigned":
        return CELL_TYPE_COLOURS["Unassigned"]
    return CELL_TYPE_COLOURS.get(str(ct), "#AAAAAA")

def cancer_label(row: pd.Series, gene_name: str | None = None) -> str:
    #build label with cancer marker. If row doesn't have 'gene' column, use gene_name param
    if gene_name is None:
        gene_name = str(row.get("approved_symbol", row.get("gene", "GENE")))
    else:
        gene_name = str(row.get("approved_symbol", gene_name))
    
    if row.get("oncokb_is_oncogene") == 1 or row.get("uniprot_oncogene") == 1:
        return "★ " + gene_name
    elif row.get("oncokb_is_tsg") == 1 or row.get("uniprot_tsg") == 1:
        return "▼ " + gene_name
    return gene_name

def lfc_suffix(direction: str | float) -> str:
    if pd.isna(direction):
        return ""
    d = str(direction).lower()
    if d == "up":   return " ↑"
    if d == "down": return " ↓"
    return ""

# ── data loading ───────────────────────────────────────────────────────────────

def load_pair_data(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    #return (nodes_df, edges_df) with all layers merged
    net = NET_DIR / pair

    #network structure
    hom  = pd.read_csv(net / f"{pair}_node_homogeneity_permutation.csv")
    edges = pd.read_csv(net / f"{pair}_differential_edges_permutation.csv")

    #annotation (tagged_with_lfc)
    lfc_path = LFC_DIR / f"{pair}_tagged_with_lfc.csv"
    if lfc_path.exists():
        ann = pd.read_csv(lfc_path)
    else:
        ann = pd.DataFrame({"gene": list(set(edges["gene_a"]) | set(edges["gene_b"]))})

    #merge: homogeneity has degree/module/etc; annotation has HGNC/cancer/cell-type/LFC
    nodes = hom.merge(ann, on="gene", how="left")
    return nodes, edges

def compute_tier_hubs(edges: pd.DataFrame, nodes: pd.DataFrame, top_n: int = TOP_N_HUBS) -> dict[str, pd.DataFrame]:
    #compute per-tier hub rankings from edge file
    def degree_rank(sub_edges: pd.DataFrame) -> pd.DataFrame:
        from collections import Counter
        counts = Counter()
        for _, row in sub_edges.iterrows():
            counts[row["gene_a"]] += 1
            counts[row["gene_b"]] += 1
        df = pd.DataFrame(counts.most_common(top_n), columns=["gene", "tier_degree"])
        df = df.merge(
            nodes[["gene","approved_symbol","known_cancer_gene","cancer_role",
                   "oncokb_is_oncogene","oncokb_is_tsg","cell_type","major_compartment",
                   "direction","module"]].drop_duplicates("gene"),
            on="gene", how="left"
        )
        df.insert(0, "rank", range(1, len(df)+1))
        return df

    d_edges  = edges[edges["link_type"] == "D"]
    s_edges  = edges[edges["link_type"] == "S"]

    #split S into case-gained (rho_case > rho_control) and ctrl-gained (rho_control > rho_case)
    s_case = s_edges[s_edges["rho_case"] >= s_edges["rho_control"]]
    s_ctrl = s_edges[s_edges["rho_control"] > s_edges["rho_case"]]

    return {
        "D":      degree_rank(d_edges),
        "S_case": degree_rank(s_case),
        "S_ctrl": degree_rank(s_ctrl),
    }

def compute_module_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    #module-level summary: size, top genes, cancer gene count, dominant cell type
    rows = []
    for mod_id, grp in nodes.groupby("module"):
        known_cancer = int(grp["known_cancer_gene"].sum()) if "known_cancer_gene" in grp.columns else 0
        top_by_degree = grp.nlargest(5, "degree")["gene"].tolist() if "degree" in grp.columns else []
        dominant_ct = (
            grp["cell_type"].value_counts().index[0]
            if "cell_type" in grp.columns and grp["cell_type"].notna().any()
            else "unknown"
        )
        rows.append({
            "module":           mod_id,
            "n_genes":          len(grp),
            "known_cancer_genes": known_cancer,
            "dominant_cell_type": dominant_ct,
            "top_genes_by_degree": "|".join(str(g) for g in top_by_degree),
        })
    return pd.DataFrame(rows).sort_values("module")

# ── community layout (matching 32_csd_visualization.py) ───────────────────────

def _community_layout(g: nx.Graph, node_to_module: dict, seed: int = SEED) -> dict:
    #communities arranged via spring layout on contracted meta-graph; spring within each
    buckets: dict[int, list] = {}
    for node, mod in node_to_module.items():
        if node in g:
            buckets.setdefault(mod, []).append(node)
    mods = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_mods = len(mods)
    if n_mods == 0:
        return nx.spring_layout(g, seed=seed)
    if n_mods == 1:
        _, nodes = mods[0]
        return dict(nx.spring_layout(g.subgraph(nodes), seed=seed, weight="weight", iterations=80))

    node_to_mod = {n: m for m, nodes in mods for n in nodes}
    meta = nx.Graph()
    for m, _ in mods:
        meta.add_node(m)
    for u, v, d in g.edges(data=True):
        mu, mv = node_to_mod.get(u), node_to_mod.get(v)
        if mu is not None and mv is not None and mu != mv:
            w = d.get("weight", 1.0)
            if meta.has_edge(mu, mv):
                meta[mu][mv]["weight"] += w
            else:
                meta.add_edge(mu, mv, weight=w)

    spread = max(100.0, n_mods * 18.0)
    meta_pos = nx.spring_layout(meta, seed=seed, weight="weight",
                                k=spread / math.sqrt(n_mods), iterations=150)
    xs = [x for x, y in meta_pos.values()]; ys = [y for x, y in meta_pos.values()]
    scale = spread / max(max(xs) - min(xs) or 1.0, max(ys) - min(ys) or 1.0)
    centres = {m: (x * scale, y * scale) for m, (x, y) in meta_pos.items()}
    max_mod_scale = spread / (1.5 * math.sqrt(n_mods))

    pos: dict = {}
    for local_idx, (mod_id, nodes) in enumerate(mods):
        cx, cy = centres[mod_id]
        n = len(nodes)
        subg = g.subgraph(nodes)
        if n == 1:
            pos[nodes[0]] = (cx, cy)
        elif n == 2:
            pos[nodes[0]] = (cx - 0.5, cy); pos[nodes[1]] = (cx + 0.5, cy)
        else:
            module_scale = min(max_mod_scale, max(4.0, math.sqrt(n) * 2.0))
            sub_pos = nx.spring_layout(subg, seed=seed + local_idx, weight="weight",
                                       k=18.0 / math.sqrt(n), iterations=80)
            for node, (x, y) in sub_pos.items():
                pos[node] = (cx + x * module_scale, cy + y * module_scale)
    return pos


def _degree_scaled_sizes(g: nx.Graph, lo: float = 20.0, hi: float = 180.0) -> dict:
    dw = dict(g.degree(weight="weight"))
    vals = np.array(list(dw.values()), dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        return {n: lo + (hi - lo) * (v - vmin) / (vmax - vmin) for n, v in dw.items()}
    return {n: (lo + hi) / 2 for n in dw}


# ── interactive HTML ───────────────────────────────────────────────────────────

def build_interactive_html(pair: str, nodes: pd.DataFrame, edges: pd.DataFrame, out_path: Path) -> None:
    #generate pyvis html with module fill, cell-type border, cancer markers, and LFC arrow
    try:
        from pyvis.network import Network
    except ImportError:
        logging.warning("pyvis not available — skipping HTML for %s", pair)
        return

    net = Network(height="900px", width="100%", bgcolor="#fafafa", font_color="#1d3557",
                  notebook=False, directed=False)
    net.show_buttons(filter_=["physics"])

    node_map = nodes.set_index("gene")

    for _, row in nodes.iterrows():
        gene = row["gene"]
        fill   = module_colour(row.get("module", 0))
        border = cell_type_colour(row.get("cell_type"))
        label  = cancer_label(row, gene_name=gene) + lfc_suffix(row.get("direction"))
        deg    = int(row.get("degree", 1))
        size   = max(8, min(45, 5 + deg * 0.18))

        tooltip = (
            f"<b>{gene}</b><br>"
            f"Module: {row.get('module', '?')}<br>"
            f"Degree: {deg}<br>"
            f"Cell type: {row.get('cell_type', 'unassigned')}<br>"
            f"Compartment: {row.get('major_compartment', '?')}<br>"
            f"Cancer role: {row.get('cancer_role', 'none')}<br>"
            f"LFC direction: {row.get('direction', '?')}<br>"
            f"Homogeneity H: {row.get('homogeneity', '?')}"
        )

        net.add_node(
            gene, label=label, title=tooltip,
            color={"background": fill, "border": border,
                   "highlight": {"background": fill, "border": "#FFD700"}},
            size=size, borderWidth=3,
        )

    for _, row in edges.iterrows():
        ga, gb = row["gene_a"], row["gene_b"]
        if ga not in node_map.index or gb not in node_map.index:
            continue
        lt = row.get("link_type", "C")
        ec = EDGE_COLORS.get(lt, "#AAAAAA")
        wt = float(row.get("weight", 1.0))
        net.add_edge(ga, gb,
                     color={"color": ec, "opacity": 0.45},
                     width=max(0.5, min(4.0, wt * 1.5)))

    net.set_options("""{
      "nodes": {"font": {"size": 11, "color": "#1d3557"}},
      "edges": {"smooth": {"type": "continuous"}},
      "physics": {
        "barnesHut": {"springLength": 100, "springConstant": 0.02, "damping": 0.85},
        "stabilization": {"iterations": 300, "fit": true}
      }
    }""")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_path))
    logging.info("  HTML → %s", out_path.name)

# ── stable PNGs ───────────────────────────────────────────────────────────────

MIN_MODULE_DISPLAY = 15   # hide modules smaller than this in full network
TOP_LABEL_N = 20          # hub genes labelled in full network

def build_stable_pngs(pair: str, nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: Path) -> None:
    #generate full network and per-tier PNG figures
    out_dir.mkdir(parents=True, exist_ok=True)
    node_map = nodes.set_index("gene")

    def build_graph(sub_edges: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, r in sub_edges.iterrows():
            G.add_edge(str(r["gene_a"]), str(r["gene_b"]),
                       weight=float(r.get("weight", 1.0)),
                       link_type=str(r.get("link_type", "C")))
        return G

    def draw_network(sub_edges: pd.DataFrame, title: str, fname: str,
                     n_top_label: int = TOP_LABEL_N,
                     figsize: tuple[float, float] = (28, 22)) -> None:
        G = build_graph(sub_edges)
        if G.number_of_nodes() == 0:
            return

        #module membership for layout
        mod_map = {n: int(node_map.loc[n, "module"]) if n in node_map.index else 0
                   for n in G.nodes()}

        #for full network, hide tiny modules (matching 16_network_viz)
        if fname.endswith("full_annotated.png"):
            mod_sizes: dict[int, int] = {}
            for m in mod_map.values():
                mod_sizes[m] = mod_sizes.get(m, 0) + 1
            large_mods = {m for m, sz in mod_sizes.items() if sz >= MIN_MODULE_DISPLAY}
            display_nodes = {n for n, m in mod_map.items() if m in large_mods}
            small_comp = [n for comp in nx.connected_components(G.subgraph(display_nodes))
                          if len(comp) <= 2 for n in comp]
            display_nodes -= set(small_comp)
            if display_nodes:
                G = G.subgraph(display_nodes).copy()
                mod_map = {n: m for n, m in mod_map.items() if n in G}

        if G.number_of_nodes() == 0:
            return

        pos = _community_layout(G, mod_map, seed=SEED)
        sizes = _degree_scaled_sizes(G, lo=20.0, hi=200.0)
        node_list = list(G.nodes())

        fig, ax = plt.subplots(figsize=figsize, facecolor="#fafafa")
        ax.set_facecolor("#fafafa")

        #edges coloured by CSD type
        for lt, ec in EDGE_COLORS.items():
            elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("link_type") == lt]
            if elist:
                nx.draw_networkx_edges(G, pos, edgelist=elist, ax=ax,
                                       edge_color=ec, alpha=0.22, width=0.5)

        #nodes: module fill, cell-type border
        fills   = [module_colour(mod_map.get(n, 0)) for n in node_list]
        borders = [cell_type_colour(node_map.loc[n, "cell_type"] if n in node_map.index else None)
                   for n in node_list]
        sz      = [sizes.get(n, 30.0) for n in node_list]

        nx.draw_networkx_nodes(G, pos, nodelist=node_list, ax=ax,
                               node_color=fills, edgecolors=borders,
                               node_size=sz, linewidths=2.5, alpha=0.90)

        #hub gene labels with cancer/LFC annotations — dark text matching 16_network_viz
        dw = dict(G.degree(weight="weight"))
        hubs = sorted(dw, key=lambda x: dw[x], reverse=True)[:n_top_label]
        
        hub_labels = {}
        for hub in hubs:
            if hub in pos and hub in node_map.index:
                row = node_map.loc[hub]
                label = cancer_label(row, gene_name=hub) + lfc_suffix(row.get("direction"))
                hub_labels[hub] = label
            elif hub in pos:
                hub_labels[hub] = hub
        
        nx.draw_networkx_labels(G, pos, labels=hub_labels,
                                ax=ax, font_size=5.5, font_color="#1d3557",
                                font_weight="bold")

        #legends
        edge_patches = [mpatches.Patch(color=c, label=f"{t} — {'Conserved' if t=='C' else 'Specific' if t=='S' else 'Differentiated'}")
                        for t, c in EDGE_COLORS.items()]
        present_cts = set(node_map["cell_type"].dropna().unique()) if "cell_type" in node_map.columns else set()
        ct_patches = [mpatches.Patch(color=c, label=ct)
                      for ct, c in CELL_TYPE_COLOURS.items() if ct in present_cts]

        ax.legend(handles=edge_patches, loc="lower left", fontsize=8, ncol=1,
                  bbox_to_anchor=(0.00, 0.00), frameon=True, framealpha=0.85,
                  title="Edge type", title_fontsize=8)
        if ct_patches:
            ax.add_artist(ax.get_legend())
            fig.legend(handles=ct_patches, loc="lower right", fontsize=7, ncol=2,
                       bbox_to_anchor=(1.0, 0.00), frameon=True, framealpha=0.85,
                       title="Cell type (node border)", title_fontsize=7)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        ax.set_title(
            f"{pair.replace('__vs__', ' vs ')}\n{title}\n"
            f"{n_nodes:,} nodes · {n_edges:,} edges  |  top {n_top_label} hubs labelled (★ oncogene, ▼ TSG, ↑↓ LFC)",
            fontsize=10, pad=8,
        )
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=DPI, bbox_inches="tight", facecolor="#fafafa")
        plt.close(fig)
        logging.info("  PNG → %s", fname)

    #full network (all edges, filtered to large modules) — larger figure size, top 20 labels
    draw_network(edges, "Full annotated network (all tiers)", f"{pair}_full_annotated.png",
                 n_top_label=20, figsize=(28, 26))

    #per-tier subnetworks — smaller figure size, top 8 labels
    for lt, label, fname_suffix in [
        ("D", "Tier I — Rewired co-expression (D edges)",       "tier_D"),
        ("S", "Tier II/III — Condition-specific (S edges)",     "tier_S"),
        ("C", "Conserved co-expression (C edges)",              "tier_C"),
    ]:
        sub = edges[edges["link_type"] == lt]
        if len(sub) > 0:
            draw_network(sub, label, f"{pair}_{fname_suffix}.png",
                        n_top_label=8, figsize=(22, 20))

# ── main ──────────────────────────────────────────────────────────────────────

def process_pair(pair: str) -> None:
    logging.info("=" * 60)
    logging.info("Pair: %s", pair)

    out_dir = OUT_ROOT / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    #load and merge
    nodes, edges = load_pair_data(pair)
    logging.info("  Nodes: %d   Edges: %d", len(nodes), len(edges))

    # 1. Full node table
    nodes_out = out_dir / f"{pair}_nodes.csv"
    nodes.to_csv(nodes_out, index=False)
    logging.info("  nodes.csv → %d rows", len(nodes))

    # 2. Per-tier hub tables
    tier_hubs = compute_tier_hubs(edges, nodes)
    for tier, df in tier_hubs.items():
        p = out_dir / f"{pair}_hubs_{tier}.csv"
        df.to_csv(p, index=False)
        logging.info("  hubs_%s.csv → %d genes", tier, len(df))

    #overall hubs (from pre-computed file)
    overall_hubs_src = NET_DIR / pair / f"{pair}_top_hubs_permutation.csv"
    if overall_hubs_src.exists():
        import shutil
        shutil.copy(overall_hubs_src, out_dir / f"{pair}_hubs_overall.csv")
        logging.info("  hubs_overall.csv → copied from 14_csd_networks")

    # 3. Module summary
    mod_summary = compute_module_summary(nodes)
    mod_summary.to_csv(out_dir / f"{pair}_modules.csv", index=False)
    logging.info("  modules.csv → %d modules", len(mod_summary))

    # 4. Interactive HTML
    build_interactive_html(pair, nodes, edges, out_dir / f"{pair}_network_annotated.html")

    # 5. Stable PNGs
    build_stable_pngs(pair, nodes, edges, out_dir / "stable_pngs")

    logging.info("  Done: %s", pair)


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotated network assembly for 20_node_annotation")
    parser.add_argument("--pairs", nargs="*", default=PAIRS, help="Pairs to process (default: all)")
    parser.add_argument("--skip-html", action="store_true", help="Skip interactive HTML generation")
    parser.add_argument("--skip-pngs", action="store_true", help="Skip stable PNG generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    if args.skip_html:
        global build_interactive_html
        build_interactive_html = lambda *a, **kw: None  # noqa
    if args.skip_pngs:
        global build_stable_pngs
        build_stable_pngs = lambda *a, **kw: None  # noqa

    t0 = __import__("time").time()
    for pair in args.pairs:
        process_pair(pair)

    elapsed = __import__("time").time() - t0
    logging.info("=" * 60)
    logging.info("All pairs complete in %.1fs", elapsed)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
