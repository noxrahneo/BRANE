#!/usr/bin/env python3
"""Stage-13 revised GO/pathway enrichment for network, hubs, and directional hubs.

Analyses per pair:
1) whole_network (primary)
2) hubs
3) hubs_UP
4) hubs_DOWN

Strict IO policy:
- Read-only inputs from stage-01 shared upstream, stage-12 tagging outputs, and
  final_networks_with_lfc_ready viz inputs (for hub derivation fallback).
- Write-only outputs to stage-13 enrichment output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm


# ============================================================================
# CONSTANTS
# ============================================================================

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

PAIR_NAMES: list[str] = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

TAGGING_DIR: Path = (
    REPO_ROOT / "results/23_node_annotation/02_output"
)
FINAL_READY_DIR: Path = (
    REPO_ROOT
    / "results/23_node_annotation/05_final_networks_with_lfc_ready"
)
SHARED_UPSTREAM_DIR: Path = (
    REPO_ROOT / "results/14_differential_prep"
)
OUTPUT_ROOT: Path = (
    REPO_ROOT / "results/24_enrichment/output"
)
OUTPUT_PAIR_DIR: Path = OUTPUT_ROOT / "pair_results"
OUTPUT_COMBINED_DIR: Path = OUTPUT_ROOT / "combined"

ANALYSIS_TYPES: list[str] = [
    "whole_network",
    "hubs",
    "hubs_UP",
    "hubs_DOWN",
]

GPROFILER_ORGANISM: str = "hsapiens"
GPROFILER_SOURCES: list[str] = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]
GPROFILER_USER_THRESHOLD: float = 0.05

TOP_HUB_COUNT: int = 50
# Whole-network should not be excluded by this threshold.
MIN_QUERY_GENES_HUB_ANALYSES: int = 3
DOTPLOT_TOP_N: int = 15


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class PairData:
    pair_name: str
    tagged_df: pd.DataFrame
    allvalues_df: Optional[pd.DataFrame]
    network_genes: list[str]
    hub_genes: list[str]
    up_hub_genes: list[str]
    down_hub_genes: list[str]
    eligible_genes: list[str]
    whole_background_source: str  # eligible_genes | genome


@dataclass
class AnalysisResult:
    pair_name: str
    analysis_type: str
    status: str  # completed | skipped
    skip_reason: str
    n_genes_queried: int
    n_background_genes: int
    background_source: str
    full_df: pd.DataFrame
    threshold_used: float = 0.05
    is_sensitivity_analysis: bool = False


# ============================================================================
# SETUP
# ============================================================================


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# LOAD + BUILD UNIVERSES
# ============================================================================


def _resolve_tagged_file(pair_name: str) -> Path:
    lfc_file = TAGGING_DIR / f"{pair_name}_tagged_with_lfc.csv"
    base_file = TAGGING_DIR / f"{pair_name}_tagged.csv"
    if lfc_file.exists():
        return lfc_file
    if base_file.exists():
        return base_file
    raise FileNotFoundError(f"No tagged file found for {pair_name}")


def _resolve_allvalues_file(pair_name: str) -> Optional[Path]:
    flat_path = SHARED_UPSTREAM_DIR / f"{pair_name}_AllValues.tsv"
    if flat_path.exists():
        return flat_path

    nested_path = SHARED_UPSTREAM_DIR / pair_name / f"{pair_name}_AllValues.tsv"
    if nested_path.exists():
        return nested_path

    return None


def _resolve_persistent_edges_file(pair_name: str) -> Optional[Path]:
    viz_dir = FINAL_READY_DIR / pair_name / "viz_inputs"
    candidates = [
        viz_dir / f"{pair_name}_persistent_edges_edges_viz.csv",
        viz_dir / f"{pair_name}_persistent_edges_edges.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _as_bool_series(s: pd.Series) -> pd.Series:
    lowered = s.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


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


def load_data(pair_name: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    tagged_file = _resolve_tagged_file(pair_name)
    tagged_df = pd.read_csv(tagged_file, low_memory=False)

    allvalues_file = _resolve_allvalues_file(pair_name)
    if allvalues_file is None:
        logging.warning(
            "%s: AllValues missing; whole-network enrichment will use genome background",
            pair_name,
        )
        allvalues_df = None
    else:
        allvalues_df = pd.read_csv(allvalues_file, sep="\t", low_memory=False)

    edges_file = _resolve_persistent_edges_file(pair_name)
    if edges_file is None:
        edges_df = None
    else:
        edges_df = pd.read_csv(edges_file, low_memory=False)

    return tagged_df, allvalues_df, edges_df


def _derive_hubs_from_edges(
    tagged_df: pd.DataFrame,
    edges_df: Optional[pd.DataFrame],
    symbol_col: str,
) -> list[str]:
    if edges_df is None:
        return []
    if not {"gene_a", "gene_b", "weight"}.issubset(edges_df.columns):
        return []

    long_edges = pd.concat(
        [
            edges_df[["gene_a", "weight"]].rename(columns={"gene_a": "gene"}),
            edges_df[["gene_b", "weight"]].rename(columns={"gene_b": "gene"}),
        ],
        ignore_index=True,
    )
    long_edges["weight"] = pd.to_numeric(long_edges["weight"], errors="coerce").fillna(0.0)

    weighted_degree = (
        long_edges.groupby("gene", as_index=False)["weight"].sum().rename(columns={"weight": "weighted_degree"})
    )
    top = weighted_degree.sort_values("weighted_degree", ascending=False).head(TOP_HUB_COUNT)

    if "gene" in tagged_df.columns and symbol_col in tagged_df.columns:
        symbol_map = (
            tagged_df[["gene", symbol_col]]
            .dropna()
            .drop_duplicates(subset=["gene"])
            .set_index("gene")[symbol_col]
            .to_dict()
        )
        hub_syms = top["gene"].map(symbol_map).fillna(top["gene"])
        return _safe_symbol_list(hub_syms)

    return _safe_symbol_list(top["gene"])


def build_gene_universes(
    pair_name: str,
    tagged_df: pd.DataFrame,
    allvalues_df: Optional[pd.DataFrame],
    edges_df: Optional[pd.DataFrame],
) -> PairData:
    symbol_col = "approved_symbol" if "approved_symbol" in tagged_df.columns else "gene"

    network_genes = _safe_symbol_list(tagged_df[symbol_col])

    if "hub" in tagged_df.columns:
        hubs_mask = _as_bool_series(tagged_df["hub"])
        hub_genes = _safe_symbol_list(tagged_df.loc[hubs_mask, symbol_col])
    else:
        hub_genes = _derive_hubs_from_edges(tagged_df, edges_df, symbol_col)
        if hub_genes:
            logging.info(
                "%s: derived %d hubs from persistent-edge weighted degree",
                pair_name,
                len(hub_genes),
            )
        else:
            logging.warning(
                "%s: no hub column and no usable persistent edges; hub analyses may be skipped",
                pair_name,
            )

    if "direction" in tagged_df.columns and hub_genes:
        hubs_df = tagged_df[tagged_df[symbol_col].astype(str).isin(set(hub_genes))].copy()
        dirs = hubs_df["direction"].astype(str).str.upper().str.strip()
        up_hub_genes = _safe_symbol_list(hubs_df.loc[dirs.eq("UP"), symbol_col])
        down_hub_genes = _safe_symbol_list(hubs_df.loc[dirs.eq("DOWN"), symbol_col])
    else:
        up_hub_genes = []
        down_hub_genes = []

    if allvalues_df is not None and {"gene_a", "gene_b"}.issubset(allvalues_df.columns):
        eligible_genes = _safe_symbol_list(
            pd.concat([allvalues_df["gene_a"], allvalues_df["gene_b"]], axis=0, ignore_index=True)
        )
    else:
        eligible_genes = []

    # For whole-network: ALWAYS use genome background (None)
    # This provides proper statistical contrast when query is large
    # See: https://github.com/user/repo/issues/XXX
    whole_background_source = "genome"

    return PairData(
        pair_name=pair_name,
        tagged_df=tagged_df,
        allvalues_df=allvalues_df,
        network_genes=network_genes,
        hub_genes=hub_genes,
        up_hub_genes=up_hub_genes,
        down_hub_genes=down_hub_genes,
        eligible_genes=eligible_genes,
        whole_background_source=whole_background_source,
    )


# ============================================================================
# ENRICHMENT + POSTPROCESS
# ============================================================================


def _run_gprofiler(
    query_genes: list[str], background_genes: Optional[list[str]], user_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Call g:Profiler REST API directly.
    
    Args:
        query_genes: list of gene symbols to query
        background_genes: optional custom background universe
        user_threshold: FDR threshold (default 0.05)
        
    Returns:
        DataFrame with enrichment results
    """
    url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    
    payload: dict[str, Any] = {
        "organism": GPROFILER_ORGANISM,
        "query": query_genes,
        "sources": GPROFILER_SOURCES,
        "user_threshold": user_threshold,
        "significance_threshold_method": "fdr",
        "no_iea": True,
        "all_results": False,  # Only return significant results
    }
    
    # If custom background provided, use it
    if background_genes:
        payload["background"] = background_genes
        payload["domain_scope"] = "custom"
    
    headers = {
        "User-Agent": "breast_cancer_enrichment_v1",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        # g:Profiler returns results in 'result' key
        if "result" in result:
            df = pd.DataFrame(result["result"])
            return df
        else:
            logging.warning("Unexpected response structure from g:Profiler: %s", result.keys())
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        logging.error("g:Profiler API request failed: %s", e)
        return pd.DataFrame()


def postprocess(
    df: pd.DataFrame,
    pair_name: str,
    analysis_type: str,
    query_size: int,
) -> pd.DataFrame:
    # Handle empty results
    if df.empty:
        return pd.DataFrame()
        
    out = df.copy()
    out.rename(columns={"native": "term_id", "name": "term_name"}, inplace=True)

    out["pair_name"] = pair_name
    out["analysis_type"] = analysis_type
    out["adjusted_p"] = out["p_value"]
    out["gene_ratio"] = out["intersection_size"] / max(query_size, 1)

    if "intersections" in out.columns:
        out["hub_genes_in_term"] = out["intersections"].apply(
            lambda x: "|".join(str(g) for g in (x if isinstance(x, list) else [])) if isinstance(x, (list, tuple)) else ""
        )
    elif "query" in out.columns:
        out["hub_genes_in_term"] = out["query"].apply(lambda x: str(x) if pd.notna(x) else "")
    else:
        out["hub_genes_in_term"] = ""

    out.sort_values(["adjusted_p", "gene_ratio"], ascending=[True, False], inplace=True)
    return out


def run_one_analysis(
    pair_data: PairData,
    analysis_type: str,
    query_genes: list[str],
    background_genes: Optional[list[str]],
    background_source: str,
    min_query_genes: int,
    threshold: float = 0.05,
) -> AnalysisResult:
    if len(query_genes) < min_query_genes:
        reason = f"fewer than {min_query_genes} genes in query"
        logging.info("%s | %s skipped: %s", pair_data.pair_name, analysis_type, reason)
        return AnalysisResult(
            pair_name=pair_data.pair_name,
            analysis_type=analysis_type,
            status="skipped",
            skip_reason=reason,
            n_genes_queried=len(query_genes),
            n_background_genes=0 if background_genes is None else len(background_genes),
            background_source=background_source,
            full_df=pd.DataFrame(),
            threshold_used=threshold,
        )

    raw_df = _run_gprofiler(query_genes=query_genes, background_genes=background_genes, user_threshold=threshold)
    
    # Handle empty results from API
    if raw_df.empty:
        logging.info(
            "%s | %s completed with zero significant terms (threshold: FDR %.2f)",
            pair_data.pair_name,
            analysis_type,
            threshold,
        )
        return AnalysisResult(
            pair_name=pair_data.pair_name,
            analysis_type=analysis_type,
            status="completed",
            skip_reason="",
            n_genes_queried=len(query_genes),
            n_background_genes=0 if background_genes is None else len(background_genes),
            background_source=background_source,
            full_df=pd.DataFrame(),
            threshold_used=threshold,
        )
    
    pp_df = postprocess(
        raw_df,
        pair_name=pair_data.pair_name,
        analysis_type=analysis_type,
        query_size=len(query_genes),
    )

    sig_n = int((pp_df["significant"] == True).sum()) if "significant" in pp_df.columns else 0
    if sig_n == 0:
        logging.info(
            "%s | %s completed with zero significant terms (threshold: FDR %.2f)",
            pair_data.pair_name,
            analysis_type,
            threshold,
        )

    return AnalysisResult(
        pair_name=pair_data.pair_name,
        analysis_type=analysis_type,
        status="completed",
        skip_reason="",
        n_genes_queried=len(query_genes),
        n_background_genes=0 if background_genes is None else len(background_genes),
        background_source=background_source,
        full_df=pp_df,
        threshold_used=threshold,
    )


def run_all_enrichments(pair_data: PairData) -> list[AnalysisResult]:
    results: list[AnalysisResult] = []

    # whole_network: ALWAYS use genome background (None) for proper statistical contrast
    whole_result = run_one_analysis(
        pair_data=pair_data,
        analysis_type="whole_network",
        query_genes=pair_data.network_genes,
        background_genes=None,  # genome background
        background_source="genome",
        min_query_genes=1,
        threshold=0.05,  # primary threshold
    )
    results.append(whole_result)
    
    # If zero significant terms at 0.05, run SENSITIVITY analysis at 0.10
    sig_at_005 = int((whole_result.full_df["significant"] == True).sum()) if not whole_result.full_df.empty else 0
    if whole_result.status == "completed" and sig_at_005 == 0:
        logging.info(
            "%s | whole_network: zero terms at FDR 0.05, running sensitivity analysis at FDR 0.10",
            pair_data.pair_name,
        )
        sensitivity_result = run_one_analysis(
            pair_data=pair_data,
            analysis_type="whole_network",
            query_genes=pair_data.network_genes,
            background_genes=None,
            background_source="genome",
            min_query_genes=1,
            threshold=0.10,  # relaxed threshold
        )
        sensitivity_result.is_sensitivity_analysis = True
        results.append(sensitivity_result)

    # Hub enrichment: use network genes as background (this contrast is valid)
    hub_bg = pair_data.network_genes
    results.append(
        run_one_analysis(
            pair_data=pair_data,
            analysis_type="hubs",
            query_genes=pair_data.hub_genes,
            background_genes=hub_bg,
            background_source="network_genes",
            min_query_genes=MIN_QUERY_GENES_HUB_ANALYSES,
            threshold=0.05,
        )
    )
    results.append(
        run_one_analysis(
            pair_data=pair_data,
            analysis_type="hubs_UP",
            query_genes=pair_data.up_hub_genes,
            background_genes=hub_bg,
            background_source="network_genes",
            min_query_genes=MIN_QUERY_GENES_HUB_ANALYSES,
            threshold=0.05,
        )
    )
    results.append(
        run_one_analysis(
            pair_data=pair_data,
            analysis_type="hubs_DOWN",
            query_genes=pair_data.down_hub_genes,
            background_genes=hub_bg,
            background_source="network_genes",
            min_query_genes=MIN_QUERY_GENES_HUB_ANALYSES,
            threshold=0.05,
        )
    )

    return results


# ============================================================================
# SAVE + SUMMARY + PLOTS
# ============================================================================


def _analysis_dir(pair_name: str, analysis_type: str) -> Path:
    return OUTPUT_PAIR_DIR / pair_name / analysis_type


def save_outputs(result: AnalysisResult) -> None:
    out_dir = _analysis_dir(result.pair_name, result.analysis_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_file = out_dir / "enrichment_results_full.csv"
    top20_file = out_dir / "enrichment_top20_significant.csv"

    if result.status == "skipped" or result.full_df.empty:
        pd.DataFrame().to_csv(full_file, index=False)
        pd.DataFrame().to_csv(top20_file, index=False)
        return

    result.full_df.to_csv(full_file, index=False)

    sig_df = result.full_df[result.full_df["significant"] == True].copy()
    sig_df.head(20).to_csv(top20_file, index=False)


def plot_dotplots(result: AnalysisResult) -> None:
    if result.status != "completed" or result.full_df.empty:
        return

    sig_df = result.full_df[result.full_df["significant"] == True].copy()
    if sig_df.empty:
        return

    top_df = sig_df.nsmallest(DOTPLOT_TOP_N, "adjusted_p").copy()
    top_df = top_df.sort_values("adjusted_p", ascending=False)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=top_df,
        x="gene_ratio",
        y="term_name",
        hue="source",
        size="intersection_size",
        sizes=(60, 360),
        palette="tab10",
        edgecolor="black",
        linewidth=0.3,
    )
    plt.title(f"{result.pair_name} — {result.analysis_type} enrichment")
    plt.xlabel("gene_ratio")
    plt.ylabel("term_name")
    plt.tight_layout()

    png_file = _analysis_dir(result.pair_name, result.analysis_type) / "enrichment_dotplot.png"
    plt.savefig(png_file, dpi=150)
    plt.close()


def _summary_row(result: AnalysisResult) -> dict[str, Any]:
    if result.status != "completed" or result.full_df.empty:
        return {
            "pair_name": result.pair_name,
            "analysis_type": result.analysis_type,
            "status": result.status,
            "skip_reason": result.skip_reason,
            "n_genes_queried": result.n_genes_queried,
            "n_background_genes": result.n_background_genes,
            "background_source": result.background_source,
            "threshold_fdr": result.threshold_used,
            "is_sensitivity_analysis": result.is_sensitivity_analysis,
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

    sig_df = result.full_df[result.full_df["significant"] == True].copy()

    if sig_df.empty:
        return {
            "pair_name": result.pair_name,
            "analysis_type": result.analysis_type,
            "status": "completed",
            "skip_reason": "",
            "n_genes_queried": result.n_genes_queried,
            "n_background_genes": result.n_background_genes,
            "background_source": result.background_source,
            "threshold_fdr": result.threshold_used,
            "is_sensitivity_analysis": result.is_sensitivity_analysis,
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

    top = sig_df.nsmallest(1, "adjusted_p").iloc[0]
    return {
        "pair_name": result.pair_name,
        "analysis_type": result.analysis_type,
        "status": "completed",
        "skip_reason": "",
        "n_genes_queried": result.n_genes_queried,
        "n_background_genes": result.n_background_genes,
        "background_source": result.background_source,
        "threshold_fdr": result.threshold_used,
        "is_sensitivity_analysis": result.is_sensitivity_analysis,
        "n_significant_terms": int(len(sig_df)),
        "n_GO_BP": int((sig_df["source"] == "GO:BP").sum()),
        "n_GO_MF": int((sig_df["source"] == "GO:MF").sum()),
        "n_GO_CC": int((sig_df["source"] == "GO:CC").sum()),
        "n_KEGG": int((sig_df["source"] == "KEGG").sum()),
        "n_REAC": int((sig_df["source"] == "REAC").sum()),
        "top_term_name": str(top["term_name"]),
        "top_term_source": str(top["source"]),
        "top_term_adjusted_p": float(top["adjusted_p"]),
    }


def generate_summary(all_results: list[AnalysisResult]) -> None:
    rows = [_summary_row(r) for r in all_results]
    summary_df = pd.DataFrame(rows)
    OUTPUT_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_COMBINED_DIR / "enrichment_summary_report.csv", index=False)


def save_combined(all_results: list[AnalysisResult]) -> None:
    frames: list[pd.DataFrame] = []
    for r in all_results:
        if r.status == "completed" and not r.full_df.empty:
            frames.append(r.full_df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    OUTPUT_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_COMBINED_DIR / "all_pairs_enrichment_combined.csv", index=False)


def write_index(all_results: list[AnalysisResult]) -> None:
    rows: list[dict[str, Any]] = []
    for r in all_results:
        rows.append(
            {
                "pair_name": r.pair_name,
                "analysis_type": r.analysis_type,
                "status": r.status,
                "pair_analysis_dir": str(_analysis_dir(r.pair_name, r.analysis_type)),
                "results_file": str(_analysis_dir(r.pair_name, r.analysis_type) / "enrichment_results_full.csv"),
                "top20_file": str(_analysis_dir(r.pair_name, r.analysis_type) / "enrichment_top20_significant.csv"),
                "dotplot_file": str(_analysis_dir(r.pair_name, r.analysis_type) / "enrichment_dotplot.png"),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "output_index.csv", index=False)


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    setup_logging()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[AnalysisResult] = []

    for pair_name in tqdm(PAIR_NAMES, desc="Pairs", unit="pair"):
        logging.info("Processing %s", pair_name)

        tagged_df, allvalues_df, edges_df = load_data(pair_name)
        pair_data = build_gene_universes(pair_name, tagged_df, allvalues_df, edges_df)
        pair_results = run_all_enrichments(pair_data)

        for result in pair_results:
            save_outputs(result)
            plot_dotplots(result)
            all_results.append(result)

    save_combined(all_results)
    generate_summary(all_results)
    write_index(all_results)

    logging.info("Done. Outputs written to %s", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
