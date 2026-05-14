#!/usr/bin/env python3
"""
regenerate_interactive_slim.py

Re-generates the six interactive HTML co-expression network files using
template.html (vis.js CDN, not PyVis), with a stricter visual threshold
(mean + 1 SD of permutation maxima per CSD tier) applied as a
display-only filter.

This script does NOT modify any pipeline results or existing scripts.

Outputs: interactive_networks/{prefix}_network_annotated.html

Usage:
    python3 BRANE/scripts/regenerate_interactive_slim.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_HTML = Path(__file__).parent / "template.html"

NET_DIR   = REPO_ROOT / "results" / "14_csd_networks"
THR_DIR   = REPO_ROOT / "results" / "13_csd_thresholds"
LFC_DIR   = REPO_ROOT / "results" / "20_node_annotation" / "03_output_with_lfc"
HOM_DIR   = NET_DIR   # node_homogeneity files live alongside edges

PAIRS = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

PAIR_PREFIX: dict[str, str] = {
    "ER_tumor__vs__Normal":                                            "ER",
    "HER2_tumor__vs__Normal":                                          "HER2",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal":                       "NormalBRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal":                         "TNBC_BRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic":  "TNBC_BRCA1_vs_NormalBRCA1",
    "Triple_negative_tumor__vs__Normal":                               "TNBC",
}

EDGE_COLORS = {"C": "#2b6fb0", "D": "#e63946", "S": "#2a9d8f"}

# Identical to _MODULE_PALETTE in script 36 — keeps node fill colours consistent
_MODULE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#d3d3d3", "#888888",
]


def module_fill_color(mod_id: int) -> str:
    return _MODULE_PALETTE[(int(mod_id) - 1) % len(_MODULE_PALETTE)]

CELL_TYPE_BORDER_COLORS = {
    "bcell": "#0072B2", "cycling": "#E74C3C", "epithelial": "#2ECC71",
    "endo": "#F39C12", "endothelial": "#F39C12", "fibroblast": "#6D4C41",
    "fibro": "#00ACC1", "myeloid": "#9B59B6", "tcell": "#3498DB",
    "t_cell": "#3498DB", "luminal_epi": "#27AE60", "luminal": "#27AE60",
    "basal_epi": "#16A085", "basal": "#16A085", "plasma": "#E91E63",
    "nk": "#F1C40F", "unknown": "#9CA3AF",
}


def cell_type_border_color(ct: str | None) -> str:
    if not ct or str(ct).lower() in {"nan", "none", ""}:
        return "#9CA3AF"
    return CELL_TYPE_BORDER_COLORS.get(str(ct).lower().replace(" ", "_"), "#9CA3AF")


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def compute_slim_thresholds(pair: str) -> dict[str, float]:
    """Return {C, S, D}: mean + 1 SD of the 500 per-permutation maxima."""
    thr_file = THR_DIR / pair / f"{pair}_permutation_thresholds.json"
    with open(thr_file) as f:
        d = json.load(f)
    thresholds: dict[str, float] = {}
    for key in ("C", "S", "D"):
        rm = np.array(d[f"running_mean_{key}"])
        n = np.arange(1, len(rm) + 1)
        maxima = np.diff(rm * n, prepend=0)
        thresholds[key] = float(maxima.mean() + maxima.std())
    return thresholds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_edges(pair: str) -> pd.DataFrame:
    path = NET_DIR / pair / f"{pair}_differential_edges_permutation.csv"
    df = pd.read_csv(path)
    def _detail(row: pd.Series) -> str:
        if row["link_type"] != "S":
            return str(row["link_type"])
        return "S_case" if float(row.get("rho_case", 0)) > float(row.get("rho_control", 0)) else "S_ctrl"
    df["link_type_detail"] = df.apply(_detail, axis=1)
    return df


EDGE_CAP = 5_000


def filter_edges(edges: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    mask = (
        ((edges["link_type"] == "C") & (edges["C"] > thresholds["C"])) |
        ((edges["link_type"] == "S") & (edges["S"] > thresholds["S"])) |
        ((edges["link_type"] == "D") & (edges["D"] > thresholds["D"]))
    )
    filtered = edges[mask].copy()
    if len(filtered) <= EDGE_CAP:
        return filtered
    # Equal budget per tier so all three colours are visually prominent
    per_tier = EDGE_CAP // 3
    kept = []
    for lt, col in (("C", "C"), ("S", "S"), ("D", "D")):
        tier = filtered[filtered["link_type"] == lt]
        if tier.empty or col not in tier.columns:
            continue
        kept.append(tier.nlargest(min(per_tier, len(tier)), col))
    return pd.concat(kept).copy()


def load_modules(pair: str) -> dict[str, int]:
    path = NET_DIR / pair / f"{pair}_leiden_modules.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if "gene" not in df.columns or "module" not in df.columns:
        return {}
    return {str(r["gene"]): int(r["module"]) for _, r in df.iterrows()}


def load_nodes(pair: str) -> pd.DataFrame:
    """Merge homogeneity + LFC annotation into a single node table."""
    hom_path = NET_DIR / pair / f"{pair}_node_homogeneity_permutation.csv"
    hom = pd.read_csv(hom_path)

    lfc_path = LFC_DIR / f"{pair}_tagged_with_lfc.csv"
    if lfc_path.exists():
        keep = [
            "gene", "approved_symbol", "hgnc_id", "entrez_id", "full_name",
            "known_cancer_gene", "cancer_role", "evidence_tier",
            "oncokb_is_oncogene", "oncokb_is_tsg",
            "cell_type", "major_compartment",
            "lfc", "direction",
        ]
        ann = pd.read_csv(lfc_path)
        present = [c for c in keep if c in ann.columns]
        ann = ann[present].copy()
        ann_key = "approved_symbol" if "approved_symbol" in ann.columns else "gene"
        df = hom.merge(ann, left_on="gene", right_on=ann_key, how="left", suffixes=("", "_ann"))
        if "gene_ann" in df.columns:
            df = df.drop(columns=["gene_ann"])
    else:
        df = hom.copy()
    return df


# ---------------------------------------------------------------------------
# HTML object builders (matching script 36 style)
# ---------------------------------------------------------------------------

def _html_node_obj(row: pd.Series, modules_map: dict[str, int]) -> dict:
    gene = str(row.get("gene", ""))
    wd = float(row["weighted_degree"]) if "weighted_degree" in row and pd.notna(row.get("weighted_degree")) else 0.0
    mod = int(modules_map.get(gene, row.get("module", 1) if pd.notna(row.get("module", None)) else 1))
    ct = str(row.get("cell_type", "")) if pd.notna(row.get("cell_type", "")) else ""
    cell_color = cell_type_border_color(ct if ct not in {"nan", "None", ""} else None)

    lnfc_val = row.get("lnFC", float("nan"))
    lnfc_str = f"{float(lnfc_val):+.3f}" if pd.notna(lnfc_val) else "n/a"
    cancer_role = str(row.get("cancer_role", "not_classified"))
    is_deg = bool(row.get("is_deg", False))
    deg_dir = str(row.get("deg_direction", ""))
    known_cancer = bool(row.get("known_cancer_gene", False))

    cancer_sym = {"oncogene": "★", "tsg": "▲"}.get(cancer_role.lower(), "")
    dir_sym = {"up": "↑", "down": "↓", "unchanged": "−"}.get(deg_dir.lower(), "")
    label = f"{gene}{cancer_sym}{dir_sym}" if (known_cancer or is_deg) else gene

    # Use literal \n separators (not <br>) — vis-network 9.x renders title as
    # plain text; the template CSS white-space:pre-line then handles line breaks.
    # The template's _cleanBaseTitle also expects this separator format.
    deg_str = f"Yes ({deg_dir})" if is_deg else "No"
    tooltip = (
        f"{gene}"
        f"\\nModule: {mod}"
        f"\\nCell type: {ct or 'Unknown'}"
        f"\\nWeighted degree: {wd:.4f}"
        f"\\nlnFC (DEG): {lnfc_str}"
        f"\\nDEG: {deg_str}"
        f"\\nCancer: {cancer_role}"
    )
    fill = module_fill_color(mod)
    return {
        "id": gene,
        "label": label,
        "original_label": gene,
        "group": f"C{mod}",
        "color": {
            "background": fill,
            "border": cell_color,
            "highlight": {"background": fill, "border": cell_color},
            "hover": {"background": fill, "border": cell_color},
        },
        "size": 10.0 + 30.0 * min(wd / 200.0, 1.0),
        "wd": wd,
        "cell_color": cell_color,
        "cell_type": ct or "Unknown",
        "borderWidth": 4,
        "borderWidthSelected": 6,
        "shape": "dot",
        "font": {"color": "#222222", "bold": {"color": "#1d3557"}},
        "title": tooltip,
    }


def _html_edge_obj(row: pd.Series) -> dict:
    lt = str(row.get("link_type_detail", "C"))
    w = float(row.get("weight", 1.0)) if pd.notna(row.get("weight")) else 1.0
    lt_visual = "S" if lt.startswith("S") else lt
    edge_label = {"C": "Conserved", "D": "Differential", "S": "Condition-specific"}.get(lt_visual, lt)
    return {
        "from": str(row["gene_a"]),
        "to": str(row["gene_b"]),
        "color": EDGE_COLORS.get(lt_visual, "#8d99ae"),
        "edge_type": lt_visual,
        "value": float(row.get("wTO", w)),
        "ew": w,
        "width": min(0.5 + 6.0 * w / 2.0, 8.0),
        "title": f"Type: {edge_label}\\nWeight: {w:.4f}",
    }


# ---------------------------------------------------------------------------
# HTML injection
# ---------------------------------------------------------------------------

def build_html(pair: str, edges_df: pd.DataFrame, nodes_df: pd.DataFrame,
               modules_map: dict[str, int]) -> str:
    if not TEMPLATE_HTML.exists():
        raise FileNotFoundError(f"template.html not found at {TEMPLATE_HTML}")

    template = TEMPLATE_HTML.read_text(encoding="utf-8")

    nodes_list = [_html_node_obj(row, modules_map) for _, row in nodes_df.iterrows()]
    edges_list = [_html_edge_obj(row) for _, row in edges_df.iterrows()]

    nodes_json = json.dumps(nodes_list, separators=(",", ":"))
    edges_json = json.dumps(edges_list, separators=(",", ":"))

    html = re.sub(
        r"nodes\s*=\s*new vis\.DataSet\(\[.*?\]\);",
        lambda _: f"nodes = new vis.DataSet({nodes_json});",
        template, count=1, flags=re.DOTALL,
    )
    html = re.sub(
        r"edges\s*=\s*new vis\.DataSet\(\[.*?\]\);",
        lambda _: f"edges = new vis.DataSet({edges_json});",
        html, count=1, flags=re.DOTALL,
    )
    display_name = pair.replace("__vs__", " vs ")

    # Hide the two <h1> header bars (dark background, visible behind search bar)
    html = re.sub(r"<h1>.*?</h1>", '<h1 style="display:none"></h1>', html, flags=re.DOTALL)

    # Inject correct network name into the graph-title badge (top-left corner)
    html = re.sub(
        r'<h2 class="graph-title">[^<]*</h2>',
        f'<h2 class="graph-title">{display_name}</h2>',
        html, count=1,
    )

    # Set initial gene/edge sliders to 50% so the network loads faster
    html = html.replace(
        'id="geneTopPct" type="range" min="5" max="100" step="1" value="100"',
        'id="geneTopPct" type="range" min="5" max="100" step="1" value="50"',
    )
    html = html.replace(
        'id="edgeKeepPct" type="range" min="5" max="100" step="1" value="100"',
        'id="edgeKeepPct" type="range" min="5" max="100" step="1" value="50"',
    )
    html = html.replace('<span id="genePctLabel">100%</span>', '<span id="genePctLabel">50%</span>')
    html = html.replace('<span id="edgePctLabel">100%</span>', '<span id="edgePctLabel">50%</span>')

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_root = REPO_ROOT.parent / "interactive_networks"
    out_root.mkdir(parents=True, exist_ok=True)

    for pair in PAIRS:
        prefix = PAIR_PREFIX[pair]
        logging.info("Processing %s …", pair)

        thresholds = compute_slim_thresholds(pair)
        logging.info(
            "  Slim thresholds  C=%.4f  S=%.4f  D=%.4f",
            thresholds["C"], thresholds["S"], thresholds["D"],
        )

        edges = load_edges(pair)
        n_before = len(edges)
        edges = filter_edges(edges, thresholds)
        n_after = len(edges)
        logging.info("  Edges: %d → %d (%.1f%%)", n_before, n_after, 100 * n_after / n_before)

        nodes = load_nodes(pair)
        modules_map = load_modules(pair)

        html = build_html(pair, edges, nodes, modules_map)

        out_path = out_root / f"{prefix}_network_annotated.html"
        out_path.write_text(html, encoding="utf-8")
        logging.info("  → %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)

    logging.info("Done. Output in %s", out_root)


if __name__ == "__main__":
    main()
