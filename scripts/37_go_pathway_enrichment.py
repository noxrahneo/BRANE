#!/usr/bin/env python3
"""Clean Stage-13 GO/pathway enrichment for network and hub-level analyses."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

NODE_ANNOT_DIR = REPO_ROOT / "results/20_node_annotation"
CSD_NETWORKS_DIR = REPO_ROOT / "results/14_csd_networks"
OUTPUT_ROOT = REPO_ROOT / "results/21_enrichment"

PAIR_NAMES = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

PAIR_SHORT = {
    "ER_tumor__vs__Normal": "ER",
    "HER2_tumor__vs__Normal": "HER2",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": "NormalBRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal": "TNBC_BRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": "TNBC_BRCA1_vs_NormalBRCA1",
    "Triple_negative_tumor__vs__Normal": "TNBC",
}

TIERS = ["D", "S_case", "S_ctrl"]
ANALYSIS_TYPES = ["hub_genes", "all_genes"]

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
GPROFILER_ORGANISM = "hsapiens"
GPROFILER_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]
FDR_THRESHOLD = 0.05
TOP_HUB_COUNT = 50
MIN_QUERY_GENES_HUB = 3
DOTPLOT_TOP_N = 15
TOP_MODULES_N = 6
MIN_MODULE_GENES = 10


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_symbol_list(series: pd.Series) -> list[str]:
    return (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def _link_type_detail(edges_df: pd.DataFrame) -> pd.Series:
    """Derive S_case/S_ctrl split from rho columns; return link_type_detail series."""
    def _detail(row: pd.Series) -> str:
        if row["link_type"] != "S":
            return str(row["link_type"])
        return "S_case" if float(row.get("rho_case", 0)) > float(row.get("rho_control", 0)) else "S_ctrl"
    return edges_df.apply(_detail, axis=1)


def load_tier_genes(pair_name: str, tier: str) -> tuple[list[str], list[str]]:
    """Return (hub_genes, all_tier_genes) for the given pair and tier.

    hub_genes   — pre-computed top-50 hubs from 20_node_annotation
    all_tier_genes — all genes that appear in edges of this tier
    """
    prefix = PAIR_SHORT.get(pair_name, pair_name)
    hub_file = NODE_ANNOT_DIR / pair_name / f"{prefix}_hubs_{tier}.csv"
    if not hub_file.exists():
        raise FileNotFoundError(f"Hub file not found: {hub_file}")

    hubs_df = pd.read_csv(hub_file, low_memory=False)
    sym_col = "approved_symbol" if "approved_symbol" in hubs_df.columns else "gene"
    hub_genes = _safe_symbol_list(hubs_df[sym_col])

    # All genes in this tier's edges
    edge_file = CSD_NETWORKS_DIR / pair_name / f"{pair_name}_differential_edges_permutation.csv"
    edges_df = pd.read_csv(edge_file, low_memory=False)
    edges_df["link_type_detail"] = _link_type_detail(edges_df)
    tier_edges = edges_df[edges_df["link_type_detail"] == tier]
    all_genes = list(
        set(tier_edges["gene_a"].dropna().tolist()) |
        set(tier_edges["gene_b"].dropna().tolist())
    )
    return hub_genes, all_genes


def load_module_genes(
    pair_name: str,
    top_n: int = TOP_MODULES_N,
) -> list[tuple[int, int, str, list[str]]]:
    """Return top-N modules by size as list of (module_id, n_genes, cell_type, gene_list)."""
    prefix = PAIR_SHORT.get(pair_name, pair_name)
    nodes_file = NODE_ANNOT_DIR / pair_name / f"{prefix}_nodes.csv"
    modules_file = NODE_ANNOT_DIR / pair_name / f"{prefix}_modules.csv"

    if not nodes_file.exists() or not modules_file.exists():
        raise FileNotFoundError(f"Node/module files not found for {pair_name}")

    nodes_df = pd.read_csv(nodes_file, low_memory=False)
    modules_df = pd.read_csv(modules_file, low_memory=False)

    sym_col = "approved_symbol" if "approved_symbol" in nodes_df.columns else "gene"
    top_module_ids = (
        modules_df.sort_values("n_genes", ascending=False)
        .head(top_n)["module"]
        .tolist()
    )

    result = []
    for mod_id in top_module_ids:
        genes = _safe_symbol_list(
            nodes_df.loc[nodes_df["module"] == mod_id, sym_col]
        )
        if len(genes) < MIN_MODULE_GENES:
            continue
        row = modules_df[modules_df["module"] == mod_id].iloc[0]
        cell_type = str(row.get("dominant_cell_type", "")) if "dominant_cell_type" in row else ""
        result.append((int(mod_id), len(genes), cell_type, genes))

    return result


def module_output_dir(pair_name: str, module_id: int) -> Path:
    return OUTPUT_ROOT / pair_name / "modules" / f"module_{module_id}"


def run_module_analysis(
    pair_name: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Run enrichment on each of the top-N Leiden modules for a pair."""
    summary_rows: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []

    try:
        modules = load_module_genes(pair_name)
    except FileNotFoundError as exc:
        logging.warning("%s: module enrichment skipped — %s", pair_name, exc)
        return summary_rows, combined_frames

    for mod_id, n_genes, cell_type, genes in modules:
        label = f"module_{mod_id}" + (f" ({cell_type})" if cell_type else "")
        analysis_type = f"modules/module_{mod_id}"

        if len(genes) < MIN_QUERY_GENES_HUB:
            logging.info("%s | %s: skipped (too few genes)", pair_name, label)
            continue

        raw_df = call_gprofiler(genes, None, FDR_THRESHOLD)
        if raw_df.attrs.get("api_error"):
            logging.error("%s | %s: API error", pair_name, label)
            continue

        result_df = postprocess(raw_df, pair_name, analysis_type, "genome", len(genes))
        logging.info("%s | %s | n_genes=%d | n_significant=%d",
                     pair_name, label, len(genes), len(result_df))

        out_dir = module_output_dir(pair_name, mod_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_dir / "enrichment_full.csv", index=False)
        result_df.head(20).to_csv(out_dir / "enrichment_top20.csv", index=False)

        title = f"{pair_name.replace('__vs__', ' vs ')} | Module {mod_id}"
        if cell_type:
            title += f" ({cell_type})"
        plot_dotplot(result_df, pair_name, title, out_dir / "dotplot.png")

        row = _summary_row_from_df(
            pair_name=pair_name,
            analysis_type=analysis_type,
            n_query_genes=len(genes),
            n_background_genes=0,
            background_source="genome",
            df=result_df,
        )
        summary_rows.append(row)
        if not result_df.empty:
            combined_frames.append(result_df)

    return summary_rows, combined_frames


def _build_gprofiler_payload(
    query_genes: list[str],
    background_genes: list[str] | None,
    threshold: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "organism": GPROFILER_ORGANISM,
        "query": query_genes,
        "sources": GPROFILER_SOURCES,
        "user_threshold": threshold,
        "significance_threshold_method": "fdr",
        "no_iea": True,
        "all_results": True,
        "domain_scope": "custom" if background_genes else "annotated",
    }
    if background_genes:
        payload["background"] = background_genes
    return payload


def _request_gprofiler(
    query_genes: list[str],
    background_genes: list[str] | None,
    threshold: float,
) -> dict[str, Any] | None:
    payload = _build_gprofiler_payload(
        query_genes,
        background_genes,
        threshold,
    )
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "breast_cancer_network_enrichment_v2",
    }
    try:
        response = requests.post(
            GPROFILER_URL,
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logging.error("g:Profiler request failed: %s", exc)
        return None
    return response.json()


def call_gprofiler(
    query_genes: list[str],
    background_genes: list[str] | None,
    threshold: float = FDR_THRESHOLD,
) -> pd.DataFrame:
    response_json = _request_gprofiler(
        query_genes,
        background_genes,
        threshold,
    )
    if response_json is None:
        df = pd.DataFrame()
        df.attrs["api_error"] = True
        return df

    if "result" not in response_json:
        logging.error("g:Profiler response missing 'result' key")
        df = pd.DataFrame()
        df.attrs["api_error"] = True
        return df

    df = pd.DataFrame(response_json["result"])
    if df.empty:
        return df

    if "significant" not in df.columns:
        logging.error("g:Profiler response missing 'significant' column")
        err_df = pd.DataFrame()
        err_df.attrs["api_error"] = True
        return err_df

    # Critical fix: all_results=True, then significant filter on client side.
    df = df[df["significant"].eq(True)].copy()
    return df


def postprocess(
    df: pd.DataFrame,
    pair_name: str,
    analysis_type: str,
    background_source: str,
    n_query: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.rename(
        columns={"native": "term_id", "name": "term_name"},
        inplace=True,
    )

    out["pair_name"] = pair_name
    out["analysis_type"] = analysis_type
    out["background_source"] = background_source
    out["gene_ratio"] = out["intersection_size"] / max(n_query, 1)

    if "adjusted_p" not in out.columns and "p_value" in out.columns:
        out["adjusted_p"] = out["p_value"]

    def _genes_string(value: Any) -> str:
        if isinstance(value, list):
            return "|".join(str(x) for x in value)
        return ""

    out["genes_in_term"] = out["intersections"].apply(_genes_string)
    out.sort_values(
        ["adjusted_p", "gene_ratio"],
        ascending=[True, False],
        inplace=True,
    )

    if "significant" in out.columns:
        out.drop(columns=["significant"], inplace=True)

    return out


def analysis_output_dir(pair_name: str, tier: str, analysis_type: str) -> Path:
    return OUTPUT_ROOT / pair_name / tier / analysis_type


def save_analysis_outputs(
    result_df: pd.DataFrame,
    pair_name: str,
    tier: str,
    analysis_type: str,
) -> None:
    out_dir = analysis_output_dir(pair_name, tier, analysis_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(out_dir / "enrichment_full.csv", index=False)
    top20 = result_df.head(20) if not result_df.empty else pd.DataFrame()
    top20.to_csv(out_dir / "enrichment_top20.csv", index=False)


def _filter_for_dotplot(df: pd.DataFrame) -> pd.DataFrame:
    """Remove overly generic terms, deduplicate synonyms, limit per source."""
    out = df.copy()
    # Drop terms where >40% of query genes hit the term — too broad to interpret
    if "gene_ratio" in out.columns:
        out = out[out["gene_ratio"] <= 0.4]
    # Within each source, drop terms that share the same adjusted_p as a
    # higher-ranked term — these are typically GO synonyms / parent-child pairs
    if "source" in out.columns and "adjusted_p" in out.columns:
        out = out.sort_values("adjusted_p")
        out = out.drop_duplicates(subset=["source", "adjusted_p"], keep="first")
    # Keep at most 3 terms per source so no single database dominates
    if "source" in out.columns:
        out = (
            out.sort_values("adjusted_p")
            .groupby("source", sort=False)
            .head(3)
            .sort_values("adjusted_p")
        )
    return out.head(DOTPLOT_TOP_N)


def plot_dotplot(
    df: pd.DataFrame,
    pair_name: str,
    analysis_type: str,
    out_path: Path,
) -> None:
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    if df.empty or len(df) < 1:
        logging.info("%s | %s: dotplot skipped (no rows)", pair_name, analysis_type)
        return

    plot_df = _filter_for_dotplot(df)
    if plot_df.empty:
        logging.info(
            "%s | %s: dotplot skipped (all terms filtered as too generic)",
            pair_name, analysis_type,
        )
        return

    plot_df = plot_df.copy()
    # Sort: most significant at the top
    plot_df = plot_df.sort_values("adjusted_p", ascending=False).reset_index(drop=True)

    n_terms = len(plot_df)
    fig_height = max(4, 0.45 * n_terms + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Colour scale: p.adjust, red = most significant, blue = least
    p_vals = plot_df["adjusted_p"].clip(lower=1e-300)
    norm = mcolors.Normalize(vmin=p_vals.min(), vmax=p_vals.max())
    cmap = cm.get_cmap("RdYlBu")  # red (low p) \u2192 yellow \u2192 blue (high p)
    colors = [cmap(norm(p)) for p in p_vals]

    # Dot size: gene_ratio (percentage of query genes in term); scale to visible pts
    gene_ratios = plot_df["gene_ratio"].fillna(0)
    max_ratio = gene_ratios.max() if gene_ratios.max() > 0 else 1.0
    dot_sizes = (gene_ratios / max_ratio) * 250 + 30

    sc = ax.scatter(
        x=plot_df["intersection_size"],
        y=range(n_terms),
        c=p_vals,
        cmap="RdYlBu",
        norm=norm,
        s=dot_sizes,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )

    ax.set_yticks(range(n_terms))
    ax.set_yticklabels([t[:65] for t in plot_df["term_name"]], fontsize=8.5)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title(
        f"{pair_name.replace('__vs__', ' vs ')} | {analysis_type}",
        fontsize=10, pad=8,
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

    # Colour bar for p.adjust
    cbar = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("p.adjust", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Size legend for Percentage
    ratio_ticks = [0.2, 0.4, 0.6]
    size_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#888888",
               markersize=np.sqrt((r / max_ratio) * 250 + 30) * 0.6,
               label=f"{int(r * 100)}%")
        for r in ratio_ticks
        if r <= max_ratio + 0.05
    ]
    if size_handles:
        ax.legend(
            handles=size_handles,
            title="Percentage",
            fontsize=7.5,
            title_fontsize=8,
            loc="lower right",
            framealpha=0.85,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _empty_summary_row(
    pair_name: str,
    analysis_type: str,
    status: str,
    skip_reason: str,
    n_query_genes: int,
    n_background_genes: int,
    background_source: str,
) -> dict[str, Any]:
    return {
        "pair_name": pair_name,
        "analysis_type": analysis_type,
        "status": status,
        "skip_reason": skip_reason,
        "n_query_genes": n_query_genes,
        "n_background_genes": n_background_genes,
        "background_source": background_source,
        "fdr_threshold": FDR_THRESHOLD,
        "n_significant_terms": 0,
        "n_GO_BP": 0,
        "n_GO_MF": 0,
        "n_GO_CC": 0,
        "n_KEGG": 0,
        "n_REAC": 0,
        "top_term_name": "",
        "top_term_source": "",
        "top_term_adjusted_p": None,
    }


def _summary_row_from_df(
    pair_name: str,
    analysis_type: str,
    n_query_genes: int,
    n_background_genes: int,
    background_source: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    if df.empty:
        return _empty_summary_row(
            pair_name,
            analysis_type,
            "completed",
            "",
            n_query_genes,
            n_background_genes,
            background_source,
        )

    top = df.iloc[0]
    return {
        "pair_name": pair_name,
        "analysis_type": analysis_type,
        "status": "completed",
        "skip_reason": "",
        "n_query_genes": n_query_genes,
        "n_background_genes": n_background_genes,
        "background_source": background_source,
        "fdr_threshold": FDR_THRESHOLD,
        "n_significant_terms": int(len(df)),
        "n_GO_BP": int((df["source"] == "GO:BP").sum()),
        "n_GO_MF": int((df["source"] == "GO:MF").sum()),
        "n_GO_CC": int((df["source"] == "GO:CC").sum()),
        "n_KEGG": int((df["source"] == "KEGG").sum()),
        "n_REAC": int((df["source"] == "REAC").sum()),
        "top_term_name": str(top["term_name"]),
        "top_term_source": str(top["source"]),
        "top_term_adjusted_p": float(top["adjusted_p"]),
    }


def _run_single_analysis(
    pair_name: str,
    tier: str,
    analysis_type: str,
    query: list[str],
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Run one g:Profiler call and return (summary_row, result_df)."""
    if len(query) < MIN_QUERY_GENES_HUB:
        reason = f"fewer than {MIN_QUERY_GENES_HUB} query genes ({len(query)})"
        logging.info("%s | %s | %s skipped: %s", pair_name, tier, analysis_type, reason)
        save_analysis_outputs(pd.DataFrame(), pair_name, tier, analysis_type)
        return _empty_summary_row(pair_name, f"{tier}/{analysis_type}", "skipped", reason,
                                  len(query), 0, "genome"), None

    raw_df = call_gprofiler(query, None, FDR_THRESHOLD)
    if raw_df.attrs.get("api_error"):
        logging.error("%s | %s | %s API error", pair_name, tier, analysis_type)
        save_analysis_outputs(pd.DataFrame(), pair_name, tier, analysis_type)
        return _empty_summary_row(pair_name, f"{tier}/{analysis_type}", "api_error",
                                  "gprofiler_request_failed", len(query), 0, "genome"), None

    result_df = postprocess(raw_df, pair_name, f"{tier}/{analysis_type}", "genome", len(query))
    logging.info("%s | %s | %s | n_query=%d | n_significant=%d",
                 pair_name, tier, analysis_type, len(query), len(result_df))
    save_analysis_outputs(result_df, pair_name, tier, analysis_type)
    plot_dotplot(result_df, pair_name, f"{tier} / {analysis_type}",
                 analysis_output_dir(pair_name, tier, analysis_type) / "dotplot.png")
    row = _summary_row_from_df(pair_name=pair_name, analysis_type=f"{tier}/{analysis_type}",
                               n_query_genes=len(query), n_background_genes=0,
                               background_source="genome", df=result_df)
    return row, result_df if not result_df.empty else None


def run_pair(
    pair_name: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    summary_rows: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []

    for tier in TIERS:
        try:
            hub_genes, all_genes = load_tier_genes(pair_name, tier)
        except FileNotFoundError as exc:
            logging.warning("%s | %s: skipping — %s", pair_name, tier, exc)
            continue

        for analysis_type, query in [("hub_genes", hub_genes), ("all_genes", all_genes)]:
            row, df = _run_single_analysis(pair_name, tier, analysis_type, query)
            summary_rows.append(row)
            if df is not None:
                combined_frames.append(df)

    return summary_rows, combined_frames


def save_summary(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "enrichment_summary.csv", index=False)


def save_combined(frames: list[pd.DataFrame]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(OUTPUT_ROOT / "enrichment_combined.csv", index=False)


def print_test_gprofiler_structure() -> int:
    """Single live API test: print raw response structure and required keys."""
    pair_name = PAIR_NAMES[0]
    try:
        hub_genes, _ = load_tier_genes(pair_name, "D")
    except Exception as exc:
        logging.error("Test setup failed for %s: %s", pair_name, exc)
        return 1

    test_genes = hub_genes[:8]
    if len(test_genes) < 3:
        logging.error("Test gene list too small from %s", pair_name)
        return 1

    response_json = _request_gprofiler(
        test_genes,
        background_genes=None,
        threshold=FDR_THRESHOLD,
    )
    if response_json is None:
        logging.error("Test call failed: no response")
        return 1

    print("TEST_GPROFILER_RESPONSE_KEYS", sorted(response_json.keys()))
    has_result = "result" in response_json
    print("TEST_HAS_RESULT_KEY", has_result)

    if not has_result:
        return 1

    raw_df = pd.DataFrame(response_json["result"])
    print("TEST_RESULT_ROWS", len(raw_df))
    print("TEST_RESULT_COLUMNS", list(raw_df.columns))
    print("TEST_HAS_SIGNIFICANT_COLUMN", "significant" in raw_df.columns)

    return 0


def main(modules_only: bool = False) -> int:
    setup_logging()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []

    for pair_name in tqdm(PAIR_NAMES, desc="Pairs", unit="pair"):
        try:
            if not modules_only:
                summary_rows, frames = run_pair(pair_name)
                all_summary_rows.extend(summary_rows)
                all_frames.extend(frames)

            mod_rows, mod_frames = run_module_analysis(pair_name)
            all_summary_rows.extend(mod_rows)
            all_frames.extend(mod_frames)
        except Exception as exc:
            logging.error("%s: unexpected failure (%s)", pair_name, exc)
            continue

    if modules_only:
        # Append to existing summary rather than overwrite tier results
        existing_summary = OUTPUT_ROOT / "enrichment_summary.csv"
        existing_combined = OUTPUT_ROOT / "enrichment_combined.csv"
        if existing_summary.exists():
            existing_rows = pd.read_csv(existing_summary).to_dict("records")
            all_summary_rows = existing_rows + all_summary_rows
        if existing_combined.exists() and all_frames:
            existing_df = pd.read_csv(existing_combined, low_memory=False)
            all_frames = [existing_df] + all_frames

    save_combined(all_frames)
    save_summary(all_summary_rows)

    logging.info("Done. Outputs written to %s", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-gprofiler-structure",
        action="store_true",
        help="Run one single API call and print raw response structure.",
    )
    parser.add_argument(
        "--modules-only",
        action="store_true",
        help="Run only module-level enrichment (top-6 Leiden modules per pair). "
             "Appends results to existing tier summary/combined CSVs.",
    )
    args = parser.parse_args()

    setup_logging()
    if args.test_gprofiler_structure:
        raise SystemExit(print_test_gprofiler_structure())

    raise SystemExit(main(modules_only=args.modules_only))
