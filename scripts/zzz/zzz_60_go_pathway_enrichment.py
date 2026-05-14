#!/usr/bin/env python3
"""Stage-13 GO and pathway enrichment from final viz_inputs assets.

Strict IO policy:
- Read-only inputs from final_networks_with_lfc_ready/viz_inputs and tagging output fallback.
- Write-only outputs to 24_enrichment/output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

try:
    from gprofiler import GProfiler
except ImportError:
    print("ERROR: gprofiler-official not installed. Install with:")
    print("  pip install gprofiler-official")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]

PAIR_NAMES = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

FINAL_READY_DIR = (
    REPO_ROOT
    / "results/23_node_annotation/05_final_networks_with_lfc_ready"
)
TAGGING_FALLBACK_DIR = (
    REPO_ROOT / "results/23_node_annotation/output"
)
OUTPUT_DIR = (
    REPO_ROOT / "results/24_enrichment/output"
)

MIN_HUB_GENES = 10
TOP_HUB_COUNT = 50
GP_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_tagged_file(pair_name: str) -> Path:
    viz_dir = FINAL_READY_DIR / pair_name / "viz_inputs"
    lfc = viz_dir / f"{pair_name}_tagged_with_lfc.csv"
    if lfc.exists():
        return lfc
    plain = viz_dir / f"{pair_name}_tagged.csv"
    if plain.exists():
        return plain

    lfc_fb = TAGGING_FALLBACK_DIR / f"{pair_name}_tagged_with_lfc.csv"
    if lfc_fb.exists():
        return lfc_fb
    plain_fb = TAGGING_FALLBACK_DIR / f"{pair_name}_tagged.csv"
    if plain_fb.exists():
        return plain_fb
    raise FileNotFoundError(f"No tagged file found for {pair_name}")


def find_edges_file(pair_name: str) -> Path:
    viz_dir = FINAL_READY_DIR / pair_name / "viz_inputs"
    viz_edges = viz_dir / f"{pair_name}_persistent_edges_edges_viz.csv"
    if viz_edges.exists():
        return viz_edges
    edges = viz_dir / f"{pair_name}_persistent_edges_edges.csv"
    if edges.exists():
        return edges
    raise FileNotFoundError(f"No persistent edges file found in viz_inputs for {pair_name}")


def load_hub_and_background(pair_name: str) -> tuple[list[str], list[str], pd.DataFrame]:
    tagged_file = find_tagged_file(pair_name)
    edges_file = find_edges_file(pair_name)

    tagged_df = pd.read_csv(tagged_file, low_memory=False)
    edges_df = pd.read_csv(edges_file, low_memory=False)

    symbol_col = "approved_symbol" if "approved_symbol" in tagged_df.columns else "gene"
    symbol_map = (
        tagged_df[["gene", symbol_col]]
        .dropna()
        .drop_duplicates(subset=["gene"])
        .set_index("gene")[symbol_col]
        .to_dict()
    )

    all_genes = tagged_df[symbol_col].dropna().astype(str).drop_duplicates().tolist()

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

    top_hubs = weighted_degree.sort_values("weighted_degree", ascending=False).head(TOP_HUB_COUNT).copy()
    top_hubs["approved_symbol"] = top_hubs["gene"].map(symbol_map).fillna(top_hubs["gene"])

    meta_cols = [c for c in ["gene", "direction", "cell_type", "lfc"] if c in tagged_df.columns]
    hub_meta = top_hubs.merge(
        tagged_df[meta_cols].drop_duplicates(subset=["gene"]),
        on="gene",
        how="left",
    )
    if symbol_col != "approved_symbol":
        hub_meta.rename(columns={symbol_col: "approved_symbol"}, inplace=True)
    hub_meta["approved_symbol"] = hub_meta["approved_symbol"].fillna(hub_meta["gene"]).astype(str)
    if "direction" in hub_meta.columns:
        hub_meta["direction"] = hub_meta["direction"].astype(str).str.upper().replace({"UNCHANGED": "STABLE"})

    hub_genes = hub_meta["approved_symbol"].dropna().drop_duplicates().tolist()
    return hub_genes, all_genes, hub_meta


def run_gp(query_genes: list[str], background_genes: list[str]) -> Optional[pd.DataFrame]:
    if len(query_genes) < MIN_HUB_GENES:
        return None
    gp = GProfiler(return_dataframe=True)
    return gp.profile(
        organism="hsapiens",
        query=query_genes,
        background=background_genes,
        sources=GP_SOURCES,
        significance_threshold_method="fdr",
        user_threshold=0.05,
        no_iea=True,
        ordered=False,
    )


def postprocess(df: pd.DataFrame, pair_name: str, query_size: int) -> pd.DataFrame:
    out = df.copy()
    out.rename(columns={"native": "term_id", "name": "term_name"}, inplace=True)
    out["pair_name"] = pair_name
    out["adjusted_p"] = out["p_value"]
    out["gene_ratio"] = out["intersection_size"] / max(query_size, 1)
    if "intersections" in out.columns:
        out["hub_genes_in_term"] = out["intersections"].apply(
            lambda x: "|".join(sorted(x)) if isinstance(x, list) else ""
        )
    elif "query" in out.columns:
        out["hub_genes_in_term"] = out["query"].apply(
            lambda x: "|".join(sorted(str(x).split(","))) if pd.notna(x) else ""
        )
    else:
        out["hub_genes_in_term"] = ""
    out.sort_values(["adjusted_p", "gene_ratio"], ascending=[True, False], inplace=True)
    return out


def save_pair_outputs(pair_name: str, full_df: pd.DataFrame) -> None:
    full_file = OUTPUT_DIR / f"{pair_name}_enrichment_results.csv"
    full_df.to_csv(full_file, index=False)

    sig_df = full_df[full_df["significant"] == True].copy()
    top20_by_source = (
        sig_df.sort_values(["source", "adjusted_p", "gene_ratio"], ascending=[True, True, False])
        .groupby("source", as_index=False, group_keys=False)
        .head(20)
    )
    top20_file = OUTPUT_DIR / f"{pair_name}_enrichment_top20.csv"
    top20_by_source.to_csv(top20_file, index=False)


def save_direction_outputs(pair_name: str, direction: str, df: pd.DataFrame) -> None:
    out_file = OUTPUT_DIR / f"{pair_name}_enrichment_{direction}_hubs.csv"
    df.to_csv(out_file, index=False)


def make_dotplot(pair_name: str, full_df: pd.DataFrame) -> None:
    sig_df = full_df[full_df["significant"] == True].copy()
    if sig_df.empty:
        return
    plot_df = sig_df.nsmallest(15, "adjusted_p").copy()
    plot_df.sort_values("adjusted_p", ascending=False, inplace=True)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x="gene_ratio",
        y="term_name",
        hue="source",
        size="intersection_size",
        sizes=(60, 360),
        palette="tab10",
        legend="brief",
    )
    plt.title(f"Hub gene enrichment — {pair_name}")
    plt.xlabel("gene_ratio")
    plt.ylabel("term_name")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{pair_name}_enrichment_dotplot.png", dpi=150)
    plt.close()


def write_summary(summary_rows: list[dict[str, object]]) -> None:
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "enrichment_summary_report.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-13 GO/pathway enrichment from viz_inputs")
    parser.add_argument("--pair", action="append", default=[], help="Pair name to process")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    pairs = args.pair if args.pair else PAIR_NAMES

    combined: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for pair_name in tqdm(pairs, desc="Pairs", unit="pair"):
        logging.info("Processing %s", pair_name)
        try:
            hub_genes, background_genes, hub_meta = load_hub_and_background(pair_name)
        except Exception as exc:
            logging.error("Failed to load inputs for %s: %s", pair_name, exc)
            continue

        if len(hub_genes) < MIN_HUB_GENES:
            logging.warning("%s has only %d hubs (<%d), skipping", pair_name, len(hub_genes), MIN_HUB_GENES)
            continue

        gp_df = run_gp(hub_genes, background_genes)
        if gp_df is None:
            logging.warning("No enrichment results for %s", pair_name)
            continue

        pp = postprocess(gp_df, pair_name, len(hub_genes))
        save_pair_outputs(pair_name, pp)
        make_dotplot(pair_name, pp)
        combined.append(pp)

        for direction in ["UP", "DOWN"]:
            if "direction" not in hub_meta.columns:
                continue
            d_hubs = hub_meta[hub_meta["direction"] == direction]["approved_symbol"].dropna().drop_duplicates().tolist()
            if len(d_hubs) < MIN_HUB_GENES:
                continue
            d_gp = run_gp(d_hubs, background_genes)
            if d_gp is None:
                continue
            d_pp = postprocess(d_gp, pair_name, len(d_hubs))
            save_direction_outputs(pair_name, direction, d_pp)

        sig = pp[pp["significant"] == True]
        top_row = sig.nsmallest(1, "adjusted_p") if not sig.empty else pd.DataFrame()
        summary_rows.append(
            {
                "pair_name": pair_name,
                "n_hub_genes_queried": len(hub_genes),
                "n_significant_terms_total": int(len(sig)),
                "n_GO_BP_terms": int((sig["source"] == "GO:BP").sum()),
                "n_GO_MF_terms": int((sig["source"] == "GO:MF").sum()),
                "n_GO_CC_terms": int((sig["source"] == "GO:CC").sum()),
                "n_KEGG_terms": int((sig["source"] == "KEGG").sum()),
                "n_REAC_terms": int((sig["source"] == "REAC").sum()),
                "top_term_name": "" if top_row.empty else str(top_row.iloc[0]["term_name"]),
                "top_term_source": "" if top_row.empty else str(top_row.iloc[0]["source"]),
                "top_term_adjusted_p": None if top_row.empty else float(top_row.iloc[0]["adjusted_p"]),
            }
        )

    if combined:
        pd.concat(combined, ignore_index=True).to_csv(OUTPUT_DIR / "all_pairs_enrichment_combined.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "all_pairs_enrichment_combined.csv", index=False)
    write_summary(summary_rows)
    logging.info("Done. Outputs written to %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
