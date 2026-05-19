#!/usr/bin/env python3
"""Normalise network gene symbols against HGNC via REST API with caching."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

HGNC_API_URL = "https://rest.genenames.org/fetch/status/Approved"
HGNC_HEADERS = {"Accept": "application/json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalise gene symbols against HGNC"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with a 'gene' column",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output normalised CSV",
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path for output normalisation log",
    )
    parser.add_argument(
        "--cache",
        default="hgnc_cache.json",
        help="Path to HGNC cache JSON",
    )
    parser.add_argument(
        "--cache-max-age",
        type=int,
        default=30,
        help="Maximum cache age in days before re-fetching HGNC data",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    #resolve relative/absolute path
    return Path(path_text).expanduser().resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def cache_is_fresh(cache_path: Path, max_age_days: int) -> bool:
    #return True if cache exists and is newer than max-age threshold
    if not cache_path.exists():
        return False
    max_age = timedelta(days=int(max_age_days))
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) <= max_age


def format_cache_date(cache_path: Path) -> str:
    #format cache file mtime as iso string
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    return mtime.isoformat()


def fetch_hgnc_docs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    #fetch all approved HGNC entries from REST API with pagination
    print("[HGNC] Fetching approved HGNC entries from REST API...")

    docs: list[dict[str, Any]] = []
    start = 0
    rows = 5000
    total_found: int | None = None

    while True:
        params = {"start": start, "rows": rows}
        try:
            response = requests.get(
                HGNC_API_URL,
                headers=HGNC_HEADERS,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "HGNC API request failed. Check internet connection "
                "or try again later."
            ) from exc

        try:
            body = payload["response"]
            page_docs = body.get("docs", [])
            if total_found is None:
                total_found = int(body.get("numFound", len(page_docs)))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Unexpected HGNC API response structure."
            ) from exc

        docs.extend(page_docs)
        print(f"[HGNC] Retrieved {len(docs)} / {total_found} entries...")

        if not page_docs:
            break
        if total_found is not None and len(docs) >= total_found:
            break

        start += rows

    raw_payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": HGNC_API_URL,
        "response": {
            "numFound": len(docs),
            "docs": docs,
        },
    }
    return docs, raw_payload


def load_hgnc_data(
    cache_path: Path,
    cache_max_age: int,
    log_lines: list[str],
) -> list[dict[str, Any]]:
    #load hgnc docs from cache if fresh, otherwise fetch from api and refresh cache
    if cache_is_fresh(cache_path, cache_max_age):
        print(f"[HGNC] Using fresh cache: {cache_path}")
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            docs = raw.get("response", {}).get("docs", [])
            if not isinstance(docs, list):
                raise ValueError("Invalid cache docs payload")
            msg = (
                "Using cached HGNC data from "
                f"{format_cache_date(cache_path)}"
            )
            log_lines.append(msg)
            return docs
        except Exception:  # noqa: BLE001
            print(
                "[HGNC] Cache appears corrupted. Deleting and "
                f"re-fetching: {cache_path}"
            )
            try:
                cache_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to remove corrupted cache file: "
                    f"{cache_path}"
                ) from exc

    docs, raw_payload = fetch_hgnc_docs()
    ensure_parent_dir(cache_path)
    cache_path.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    msg = f"Fetched {len(docs)} entries from HGNC API, saved to cache"
    log_lines.append(msg)
    print(f"[HGNC] {msg}")
    return docs


def safe_symbol_list(entry: dict[str, Any], key: str) -> list[str]:
    #return cleaned list of symbols from a hgnc entry field that may be str or list
    value = entry.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def build_hgnc_dictionaries(
    docs: list[dict[str, Any]],
    log_lines: list[str],
) -> tuple[set[str], dict[str, str], dict[str, dict[str, Any]]]:
    #build approved symbol set, alias->approved mapping, and per-symbol metadata dict
    print("[HGNC] Building symbol dictionaries...")

    approved_set: set[str] = set()
    alias_dict: dict[str, str] = {}
    hgnc_dict: dict[str, dict[str, Any]] = {}
    conflict_warnings: list[str] = []

    for entry in docs:
        symbol = str(entry.get("symbol", "")).strip()
        if not symbol:
            continue

        approved_set.add(symbol)

        aliases = safe_symbol_list(entry, "alias_symbol")
        prevs = safe_symbol_list(entry, "prev_symbol")

        hgnc_dict[symbol] = {
            "hgnc_id": str(entry.get("hgnc_id", "")).strip() or None,
            "entrez_id": (str(entry.get("entrez_id", "")).strip() or None),
            "ensembl_gene_id": (
                str(entry.get("ensembl_gene_id", "")).strip() or None
            ),
            "alias_symbols": "|".join(aliases),
            "prev_symbols": "|".join(prevs),
        }

        for alias in aliases + prevs:
            if not alias:
                continue
            existing = alias_dict.get(alias)
            if existing is None:
                alias_dict[alias] = symbol
            elif existing != symbol:
                warning = (
                    f"[warn] alias conflict: {alias} -> {existing} "
                    f"(ignored new mapping to {symbol})"
                )
                conflict_warnings.append(warning)

    if conflict_warnings:
        print(
            "[HGNC] Alias conflicts: "
            f"{len(conflict_warnings)} (kept first mapping)"
        )
        log_lines.extend(conflict_warnings)

    return approved_set, alias_dict, hgnc_dict


def normalise_gene_symbol(
    gene: str,
    approved_set: set[str],
    alias_dict: dict[str, str],
) -> tuple[str, str]:
    #return (approved_symbol, status) for one gene: unchanged, normalised, or unresolved
    if gene in approved_set:
        return gene, "unchanged"
    if gene in alias_dict:
        return alias_dict[gene], "normalised"
    return gene, "unresolved"


def apply_normalisation(
    df_in: pd.DataFrame,
    approved_set: set[str],
    alias_dict: dict[str, str],
    hgnc_dict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    #normalise each gene symbol row-wise and append hgnc metadata columns
    print("[HGNC] Normalising symbols...")

    rows: list[dict[str, Any]] = []
    for gene_raw in df_in["gene"].astype(str).tolist():
        gene = gene_raw.strip()
        approved_symbol, status = normalise_gene_symbol(
            gene,
            approved_set,
            alias_dict,
        )

        meta = hgnc_dict.get(approved_symbol, None)
        if meta is None:
            row = {
                "gene": gene,
                "approved_symbol": approved_symbol,
                "hgnc_id": None,
                "entrez_id": None,
                "ensembl_gene_id": None,
                "alias_symbols": "",
                "prev_symbols": "",
                "normalisation_status": status,
            }
        else:
            row = {
                "gene": gene,
                "approved_symbol": approved_symbol,
                "hgnc_id": meta.get("hgnc_id"),
                "entrez_id": meta.get("entrez_id"),
                "ensembl_gene_id": meta.get("ensembl_gene_id"),
                "alias_symbols": meta.get("alias_symbols", ""),
                "prev_symbols": meta.get("prev_symbols", ""),
                "normalisation_status": status,
            }
        rows.append(row)

    return pd.DataFrame(rows)


def pct(count: int, total: int) -> float:
    #safe percentage: 0.0 when total is zero
    if total <= 0:
        return 0.0
    return 100.0 * float(count) / float(total)


def write_log(
    log_path: Path,
    log_lines: list[str],
    df_out: pd.DataFrame,
) -> None:
    #write run log with status counts, normalised mappings, and unresolved symbols
    print(f"[HGNC] Writing log: {log_path}")

    ensure_parent_dir(log_path)

    total = int(df_out.shape[0])
    unchanged_df = df_out[df_out["normalisation_status"] == "unchanged"]
    normalised_df = df_out[df_out["normalisation_status"] == "normalised"]
    unresolved_df = df_out[df_out["normalisation_status"] == "unresolved"]

    unchanged_n = int(unchanged_df.shape[0])
    normalised_n = int(normalised_df.shape[0])
    unresolved_n = int(unresolved_df.shape[0])

    normalised_pairs = sorted(
        {
            (r["gene"], r["approved_symbol"])
            for _, r in normalised_df.iterrows()
        }
    )
    unresolved_genes = sorted(set(unresolved_df["gene"].astype(str).tolist()))

    entrez_present_df = df_out[
        df_out["entrez_id"].notna()
        & (df_out["entrez_id"].astype(str).str.strip() != "")
    ]
    entrez_missing_df = df_out[~df_out.index.isin(entrez_present_df.index)]

    entrez_present_n = int(entrez_present_df.shape[0])
    entrez_missing_n = int(entrez_missing_df.shape[0])
    missing_entrez_genes = sorted(
        set(entrez_missing_df["approved_symbol"].astype(str).tolist())
    )

    content: list[str] = []
    content.append(f"Run timestamp: {datetime.now(timezone.utc).isoformat()}")
    content.extend(log_lines)
    content.append("")
    content.append(f"Total genes in network: {total}")
    content.append(
        f"Unchanged: {unchanged_n} ({pct(unchanged_n, total):.2f}%)"
    )
    content.append(
        f"Normalised: {normalised_n} ({pct(normalised_n, total):.2f}%)"
    )
    content.append(
        f"Unresolved: {unresolved_n} ({pct(unresolved_n, total):.2f}%)"
    )
    content.append("")

    content.append("Normalised mappings (old_symbol -> new_symbol):")
    if normalised_pairs:
        for old_symbol, new_symbol in normalised_pairs:
            content.append(f"  {old_symbol} -> {new_symbol}")
    else:
        content.append("  (none)")
    content.append("")

    content.append("Unresolved symbols:")
    if unresolved_genes:
        for g in unresolved_genes:
            content.append(f"  {g}")
    else:
        content.append("  (none)")
    content.append("")

    content.append(f"Genes with entrez_id present: {entrez_present_n}")
    content.append(f"Genes missing entrez_id: {entrez_missing_n}")
    content.append(
        "entrez_id coverage: "
        f"{entrez_present_n}/{total} "
        f"({pct(entrez_present_n, total):.2f}%)"
    )
    content.append("")

    content.append("Genes missing entrez_id:")
    if missing_entrez_genes:
        for g in missing_entrez_genes:
            content.append(f"  {g}")
    else:
        content.append("  (none)")

    log_path.write_text("\n".join(content) + "\n", encoding="utf-8")


def run_validation(df_in: pd.DataFrame, df_out: pd.DataFrame) -> None:
    #validate row count match, no null approved_symbols, and warn on high unresolved rate
    print("[HGNC] Running validation checks...")

    if int(df_out.shape[0]) != int(df_in.shape[0]):
        raise AssertionError(
            "Validation failed: output row count "
            f"({df_out.shape[0]}) != input row count ({df_in.shape[0]})."
        )

    if df_out["approved_symbol"].isna().any():
        raise AssertionError(
            "Validation failed: approved_symbol contains null values."
        )

    unresolved_n = int((df_out["normalisation_status"] == "unresolved").sum())
    total = int(df_out.shape[0])
    unresolved_rate = (
        (float(unresolved_n) / float(total)) if total > 0 else 0.0
    )
    if unresolved_rate > 0.20:
        print(
            "[warn] Unresolved rate exceeds 20%. Please check whether "
            "input symbols are HGNC-style gene symbols "
            "(and not Ensembl IDs or Entrez integer IDs)."
        )


def main() -> int:
    args = parse_args()

    #resolve all paths
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    log_path = resolve_path(args.log)
    cache_path = resolve_path(args.cache)

    print(f"[HGNC] Input: {input_path}")
    print(f"[HGNC] Output: {output_path}")
    print(f"[HGNC] Log: {log_path}")
    print(f"[HGNC] Cache: {cache_path}")

    #load and validate input csv
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    print("[HGNC] Loading input CSV...")
    df_in = pd.read_csv(input_path)
    if "gene" not in df_in.columns:
        raise ValueError("Input CSV must contain a 'gene' column.")

    log_lines: list[str] = []

    #load hgnc data and build lookup dictionaries
    docs = load_hgnc_data(cache_path=cache_path, cache_max_age=int(args.cache_max_age), log_lines=log_lines)
    approved_set, alias_dict, hgnc_dict = build_hgnc_dictionaries(docs=docs, log_lines=log_lines)

    #normalise symbols, validate, and write outputs
    df_out = apply_normalisation(df_in=df_in, approved_set=approved_set, alias_dict=alias_dict, hgnc_dict=hgnc_dict)
    run_validation(df_in=df_in, df_out=df_out)

    print(f"[HGNC] Writing output CSV: {output_path}")
    ensure_parent_dir(output_path)
    ordered_cols = ["gene", "approved_symbol", "hgnc_id", "entrez_id", "ensembl_gene_id", "alias_symbols", "prev_symbols", "normalisation_status"]
    df_out = df_out[ordered_cols]
    df_out.to_csv(output_path, index=False)

    write_log(log_path=log_path, log_lines=log_lines, df_out=df_out)
    print("[HGNC] Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[HGNC][error] {exc}", file=sys.stderr)
        raise
