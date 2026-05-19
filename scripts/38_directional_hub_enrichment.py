#!/usr/bin/env python3
"""g:Profiler enrichment on UP/DOWN hub subsets per contrast across D, S_case, and S_ctrl tiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

PAIRS: list[tuple[str, str]] = [
    ("ER_tumor__vs__Normal",                                          "ER"),
    ("HER2_tumor__vs__Normal",                                        "HER2"),
    ("Normal_BRCA1_-_pre-neoplastic__vs__Normal",                     "NormalBRCA1"),
    ("Triple_negative_BRCA1_tumor__vs__Normal",                       "TNBC_BRCA1"),
    ("Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic","TNBC_BRCA1_vs_NormalBRCA1"),
    ("Triple_negative_tumor__vs__Normal",                             "TNBC"),
]

TIERS = ["D", "S_case", "S_ctrl"]
HUB_DIR  = REPO_ROOT / "results/20_node_annotation"
OUT_DIR  = REPO_ROOT / "results/21_enrichment"
MIN_GENES = 3

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
GPROFILER_SOURCES = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def clean_genes(series: pd.Series) -> list[str]:
    return (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def load_hubs(pair: str, short: str) -> pd.DataFrame:
    frames = []
    for tier in TIERS:
        f = HUB_DIR / pair / f"{short}_hubs_{tier}.csv"
        if f.exists():
            df = pd.read_csv(f)[["approved_symbol", "deg_direction", "lnFC"]]
            df["tier"] = tier
            frames.append(df)
        else:
            logging.warning("Missing hub file: %s", f)
    if not frames:
        return pd.DataFrame(columns=["approved_symbol", "deg_direction", "lnFC", "tier"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("approved_symbol")
    combined["approved_symbol"] = combined["approved_symbol"].astype(str).str.strip()
    combined = combined[combined["approved_symbol"].notna() & (combined["approved_symbol"] != "nan")]
    return combined


def run_gprofiler(genes: list[str], background: list[str]) -> pd.DataFrame:
    payload = {
        "organism": "hsapiens",
        "query": genes,
        "sources": GPROFILER_SOURCES,
        "user_threshold": 0.05,
        "significance_threshold_method": "fdr",
        "no_iea": True,
        "all_results": False,
        "background": background,
        "domain_scope": "custom",
    }
    headers = {"Content-Type": "application/json", "User-Agent": "directional_hub_enrichment_v1"}
    try:
        resp = requests.post(GPROFILER_URL, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if "result" in data:
            return pd.DataFrame(data["result"])
        return pd.DataFrame()
    except Exception as e:
        logging.error("g:Profiler request failed: %s", e)
        return pd.DataFrame()


def postprocess(df: pd.DataFrame, pair: str, direction: str, n_query: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.rename(columns={"native": "term_id", "name": "term_name"}, inplace=True)
    out["pair_name"] = pair
    out["direction"] = direction
    out["gene_ratio"] = out["intersection_size"] / max(n_query, 1)
    out.sort_values(["p_value", "gene_ratio"], ascending=[True, False], inplace=True)
    return out


def run_pair(pair: str, short: str) -> list[dict]:
    hubs = load_hubs(pair, short)
    if hubs.empty:
        logging.warning("%s: no hubs found, skipping", pair)
        return []

    all_hub_genes = clean_genes(hubs["approved_symbol"])
    summary_rows = []

    for direction in ("up", "down"):
        subset = clean_genes(hubs.loc[hubs["deg_direction"] == direction, "approved_symbol"])
        label = direction.upper()
        out_dir = OUT_DIR / pair / f"hubs_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(subset) < MIN_GENES:
            logging.info("%s | hubs_%s: only %d genes, skipping", pair, label, len(subset))
            pd.DataFrame().to_csv(out_dir / "enrichment_full.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "enrichment_top20.csv", index=False)
            summary_rows.append({
                "pair_name": pair, "direction": label, "n_genes": len(subset),
                "status": "skipped", "n_significant": 0,
                "top_term": "", "top_source": "", "top_q": None,
            })
            continue

        logging.info("%s | hubs_%s: %d genes", pair, label, len(subset))
        raw = run_gprofiler(subset, all_hub_genes)
        result = postprocess(raw, pair, label, len(subset))

        result.to_csv(out_dir / "enrichment_full.csv", index=False)
        top20 = result.head(20) if not result.empty else result
        top20.to_csv(out_dir / "enrichment_top20.csv", index=False)

        if result.empty:
            summary_rows.append({
                "pair_name": pair, "direction": label, "n_genes": len(subset),
                "status": "completed", "n_significant": 0,
                "top_term": "", "top_source": "", "top_q": None,
            })
        else:
            top = result.iloc[0]
            summary_rows.append({
                "pair_name": pair, "direction": label, "n_genes": len(subset),
                "status": "completed", "n_significant": len(result),
                "top_term": top.get("term_name", ""),
                "top_source": top.get("source", ""),
                "top_q": float(top.get("p_value", float("nan"))),  # g:Profiler returns adjusted p as p_value when fdr method used
            })

    return summary_rows


def main() -> int:
    setup_logging()
    all_summary = []
    for pair, short in PAIRS:
        logging.info("=== %s ===", pair)
        rows = run_pair(pair, short)
        all_summary.extend(rows)

    summary_df = pd.DataFrame(all_summary)
    summary_path = OUT_DIR / "directional_hub_enrichment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Summary written to %s", summary_path)
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
