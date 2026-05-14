#!/usr/bin/env python3
"""Regenerate stable PNGs using interactive HTML node styles.

Uses node/edge payloads embedded in each interactive HTML to preserve:
- community/module fill colors
- cell-type border colors
- oncogene/TSG + direction badges (added to labels)

Outputs in each pair folder:
- stable_pngs/full_network_stable.png
- stable_pngs/module_01..module_04_largest_cc.png
- stable_pngs/ego_hub_01..ego_hub_05.png
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[1]
READY = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "12_tagging"
    / "05_final_networks_with_lfc_ready"
)

MAX_MODULES = 4
MAX_EGOS = 5

COMMUNITY_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
    "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AB", "#1F77B4", "#2CA02C",
    "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22",
    "#17BECF", "#6A3D9A", "#33A02C", "#FB9A99", "#FDBF6F", "#A6CEE3",
    "#B2DF8A", "#CAB2D6", "#FF7F00", "#B15928",
]

CELL_TYPE_COLOR_MAP = {
    "bcell": "#0072B2",
    "cycling": "#E74C3C",
    "epithelial": "#2ECC71",
    "endo": "#F39C12",
    "endothelial": "#F39C12",
    "fibroblast": "#6D4C41",
    "fibro": "#00ACC1",
    "fibro2": "#00897B",
    "myeloid": "#9B59B6",
    "macro": "#E67E22",
    "tcell": "#3498DB",
    "tcell2": "#3498DB",
    "t_cell": "#3498DB",
    "luminal_epi": "#27AE60",
    "luminal": "#27AE60",
    "basal_epi": "#16A085",
    "basal": "#16A085",
    "plasma": "#E74C3C",
    "nk": "#F1C40F",
    "unknown": "#9CA3AF",
}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _extract_json_array(text: str, key: str) -> list[dict]:
    # key is either 'nodes' or 'edges'
    pat = re.compile(rf"\b{key}\s*=\s*new\s+vis\.DataSet\((\[.*?\])\);", re.DOTALL)
    m = pat.search(text)
    if not m:
        raise ValueError(f"Could not find DataSet array for {key}")
    payload = m.group(1)
    return json.loads(payload)


def _extract_metadata_map(text: str) -> dict[str, dict]:
    pat = re.compile(
        r'<script type="application/json" id="cell-type-metadata">\s*(\{.*?\})\s*</script>',
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}


def _group_sort_key(group_name: str) -> tuple[int, str]:
    m = re.match(r"^C(\d+)$", str(group_name or ""))
    if m:
        return (int(m.group(1)), str(group_name))
    return (10_000, str(group_name))


def _build_group_color_map(nodes: list[dict]) -> dict[str, str]:
    groups = sorted({str(n.get("group") or "Ungrouped") for n in nodes}, key=_group_sort_key)
    out: dict[str, str] = {}
    for i, g in enumerate(groups):
        out[g] = COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)]
    return out


def _is_truthy_flag(value) -> bool:
    if value is True or value == 1:
        return True
    s = str(value or "").strip().lower()
    return s in {"1", "true", "yes"}


def _badge_for_meta(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ""
    role = str(meta.get("cancer_role") or "").strip().lower()
    is_onc = _is_truthy_flag(meta.get("oncokb_is_oncogene")) or role == "oncogene"
    is_tsg = _is_truthy_flag(meta.get("oncokb_is_tsg")) or role in {"tsg", "tumor_suppressor", "tumor suppressor"}
    direction = str(meta.get("direction") or "").strip().upper()

    marks = ""
    if direction == "UP":
        marks += "↑"
    if direction == "DOWN":
        marks += "↓"
    if is_onc:
        marks += "★"
    if is_tsg:
        marks += "▼"
    return marks


def _normalize_color_spec(node: dict) -> tuple[str, str]:
    # returns (fill, border)
    fill = "#9ca3af"
    border = "#9ca3af"

    color = node.get("color")
    if isinstance(color, dict):
        fill = str(color.get("background") or color.get("color") or fill)
        border = str(color.get("border") or border)
    elif isinstance(color, str) and color.strip():
        fill = color.strip()

    # if explicit cell_color is available, prioritize as border (cell type border)
    cell_color = node.get("cell_color")
    if isinstance(cell_color, str) and cell_color.strip():
        border = cell_color.strip()

    return fill, border


def _cell_type_color(cell_type: str) -> str:
    key = str(cell_type or "Unknown").strip().lower()
    return CELL_TYPE_COLOR_MAP.get(key, "#9CA3AF")


def _edge_color(edge: dict) -> str:
    c = edge.get("color")
    if isinstance(c, dict):
        return str(c.get("color") or "#ef8f79")
    if isinstance(c, str) and c.strip():
        return c.strip()
    return "#ef8f79"


def _build_graph(nodes: list[dict], edges: list[dict], meta_map: dict[str, dict]) -> nx.Graph:
    g = nx.Graph()
    group_color_map = _build_group_color_map(nodes)

    for n in nodes:
        node_id = str(n.get("id") if n.get("id") is not None else n.get("label") or "").strip()
        if not node_id:
            continue
        fill, border = _normalize_color_spec(n)
        group = str(n.get("group") or "Ungrouped")
        fill = group_color_map.get(group, fill)
        label = str(n.get("label") or node_id)

        meta = meta_map.get(node_id) or {}
        badge = _badge_for_meta(meta)
        if badge and not label.endswith(f" {badge}"):
            label = f"{label} {badge}".strip()

        cell_type = str(meta.get("cell_type") or n.get("cell_type") or "Unknown")
        border = _cell_type_color(cell_type)

        g.add_node(
            node_id,
            label=label,
            fill=fill,
            border=border,
            size=float(n.get("size", 12) or 12),
            group=group,
            cell_type=cell_type,
        )

    for e in edges:
        a = str(e.get("from") or "").strip()
        b = str(e.get("to") or "").strip()
        if not a or not b:
            continue
        if a not in g or b not in g:
            continue
        w = e.get("value", e.get("width", e.get("weight", 1.0)))
        try:
            weight = float(w)
        except Exception:
            weight = 1.0
        g.add_edge(a, b, weight=weight, color=_edge_color(e))

    return g


def _draw_graph(
    g: nx.Graph,
    out_png: Path,
    title: str,
    seed: int,
    label_mode: str,
) -> None:
    if g.number_of_nodes() == 0:
        return

    plt.figure(figsize=(16, 12), dpi=220)
    n_nodes = g.number_of_nodes()
    if n_nodes > 900:
        pos = nx.spring_layout(g, seed=seed, weight=None, iterations=28)
    elif n_nodes > 500:
        pos = nx.spring_layout(g, seed=seed, weight=None, iterations=40)
    elif n_nodes > 200:
        pos = nx.spring_layout(g, seed=seed, weight="weight", iterations=60)
    else:
        pos = nx.spring_layout(g, seed=seed, weight="weight", iterations=100)

    nodes = list(g.nodes())
    fills = [g.nodes[n].get("fill", "#9ca3af") for n in nodes]
    borders = [g.nodes[n].get("border", "#9ca3af") for n in nodes]
    sizes_raw = [float(g.nodes[n].get("size", 12)) for n in nodes]
    sizes = [max(12.0, min(220.0, s * 6.0)) for s in sizes_raw]

    edge_list = list(g.edges())
    edge_colors = [g[u][v].get("color", "#ef8f79") for u, v in edge_list]
    edge_weights = [float(g[u][v].get("weight", 1.0)) for u, v in edge_list]
    max_w = max(edge_weights) if edge_weights else 1.0
    widths = [0.10 + 1.0 * (w / max_w) for w in edge_weights] if edge_weights else 0.15

    nx.draw_networkx_edges(g, pos, edgelist=edge_list, edge_color=edge_colors, alpha=0.14, width=widths)
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        node_color=fills,
        edgecolors=borders,
        linewidths=1.2,
        node_size=sizes,
        alpha=0.98,
    )

    if label_mode == "none":
        pass
    else:
        labels = {n: g.nodes[n].get("label", n) for n in nodes}
        fs = 5 if label_mode == "small" else 7
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=fs, font_color="#111827")

    plt.title(title)
    plt.axis("off")

    # Legend 1: community fill colors (top by count)
    by_group: dict[str, int] = {}
    for n in nodes:
        grp = str(g.nodes[n].get("group") or "Ungrouped")
        by_group[grp] = by_group.get(grp, 0) + 1
    top_groups = sorted(by_group.items(), key=lambda kv: kv[1], reverse=True)[:8]
    group_handles = []
    for grp, cnt in top_groups:
        color = g.nodes[next(iter([n for n in nodes if g.nodes[n].get("group") == grp]))].get("fill", "#9ca3af")
        group_handles.append(mpatches.Patch(facecolor=color, edgecolor="#374151", label=f"{grp} ({cnt})"))

    # Legend 2: cell-type border colors (top by count)
    by_ct: dict[str, int] = {}
    ct_color: dict[str, str] = {}
    for n in nodes:
        ct = str(g.nodes[n].get("cell_type") or "Unknown")
        by_ct[ct] = by_ct.get(ct, 0) + 1
        ct_color[ct] = g.nodes[n].get("border", "#9CA3AF")
    top_ct = sorted(by_ct.items(), key=lambda kv: kv[1], reverse=True)[:8]
    ct_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffffff", markeredgecolor=ct_color.get(ct, "#9CA3AF"), markeredgewidth=2.0, markersize=8, label=f"{ct} ({cnt})")
        for ct, cnt in top_ct
    ]

    if group_handles:
        lg1 = plt.legend(handles=group_handles, title="Module Fill Colors", loc="upper left", frameon=True, fontsize=8, title_fontsize=9)
        plt.gca().add_artist(lg1)
    if ct_handles:
        lg2 = plt.legend(handles=ct_handles, title="Cell-Type Border Colors", loc="upper right", frameon=True, fontsize=8, title_fontsize=9)
        plt.gca().add_artist(lg2)

    # Badge legend
    badge_text = "Symbols: ↑ UP   ↓ DOWN   ★ Oncogene   ▼ TSG"
    plt.gcf().text(0.5, 0.02, badge_text, ha="center", va="bottom", fontsize=9, color="#111827")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def _top_hubs(g: nx.Graph, n: int) -> list[str]:
    strengths = {k: sum(d.get("weight", 1.0) for _, _, d in g.edges(k, data=True)) for k in g.nodes()}
    return [k for k, _ in sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:n]]


def process_pair(pair_dir: Path) -> None:
    pair = pair_dir.name
    htmls = sorted(pair_dir.glob("*_network_interactive.html"))
    if not htmls:
        logging.warning("No interactive HTML in %s", pair)
        return

    html_path = htmls[0]
    text = html_path.read_text(encoding="utf-8")

    nodes = _extract_json_array(text, "nodes")
    edges = _extract_json_array(text, "edges")
    meta_map = _extract_metadata_map(text)

    g = _build_graph(nodes, edges, meta_map)
    if g.number_of_nodes() == 0:
        logging.warning("Empty graph for %s", pair)
        return

    out_dir = pair_dir / "stable_pngs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full network (no labels to reduce clutter)
    _draw_graph(
        g,
        out_dir / "full_network_stable.png",
        f"{pair} | full network | nodes={g.number_of_nodes()} edges={g.number_of_edges()}",
        seed=42,
        label_mode="none",
    )

    # Modules: top connected components, labels only on smaller components
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    for idx, comp in enumerate(components[:MAX_MODULES], start=1):
        sg = g.subgraph(comp).copy()
        label_mode = "small" if sg.number_of_nodes() <= 140 else "none"
        _draw_graph(
            sg,
            out_dir / f"module_{idx:02d}_largest_cc.png",
            f"{pair} | module {idx} | nodes={sg.number_of_nodes()} edges={sg.number_of_edges()}",
            seed=100 + idx,
            label_mode=label_mode,
        )

    # Ego networks around weighted-degree hubs, with labels and badges
    for i, hub in enumerate(_top_hubs(g, MAX_EGOS), start=1):
        ego = nx.ego_graph(g, hub, radius=1)
        _draw_graph(
            ego,
            out_dir / f"ego_hub_{i:02d}_{hub}.png",
            f"{pair} | ego around {hub} | nodes={ego.number_of_nodes()} edges={ego.number_of_edges()}",
            seed=200 + i,
            label_mode="large",
        )

    logging.info("Regenerated styled PNGs for %s", pair)


def main() -> None:
    configure_logging()
    pairs = sorted([p for p in READY.iterdir() if p.is_dir()])
    if not pairs:
        logging.error("No pair folders found in %s", READY)
        return

    done = 0
    for pair_dir in pairs:
        try:
            process_pair(pair_dir)
            done += 1
        except Exception as exc:
            logging.exception("Failed %s: %s", pair_dir.name, exc)

    logging.info("Styled PNG regeneration complete: %s/%s", done, len(pairs))


if __name__ == "__main__":
    main()
