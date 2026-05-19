#!/usr/bin/env python3
"""Fetch NCBI Gene metadata for HGNC-normalised node tables."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd
import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

GENE_TYPE_MAP = {
    "1": "unknown",
    "2": "tRNA",
    "3": "rRNA",
    "4": "snRNA",
    "5": "scRNA",
    "6": "snoRNA",
    "7": "protein-coding",
    "8": "pseudo",
    "9": "transposon",
    "10": "miscRNA",
    "11": "ncRNA",
    "255": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NCBI Gene annotations")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV from 29_hgnc_normalization.py",
    )
    parser.add_argument("--output", required=True, help="Output annotated CSV")
    parser.add_argument("--log", required=True, help="Output log file path")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional NCBI API key",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Entrez IDs per efetch request",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    #resolve relative/absolute path
    return Path(path_text).expanduser().resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_api_key(arg_api_key: str | None) -> tuple[str | None, bool]:
    #return api key from arg or NCBI_API_KEY env var, and whether one was found
    key = arg_api_key or os.getenv("NCBI_API_KEY")
    if key is not None:
        key = key.strip()
    if not key:
        return None, False
    return key, True


def chunked(items: list[int], size: int) -> list[list[int]]:
    #split list into fixed-size chunks
    if size <= 0:
        raise ValueError("batch-size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]


def safe_text(elem: ET.Element | None, default: str = "") -> str:
    #return stripped text from xml element or default
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def first_text(root: ET.Element, paths: list[str]) -> str:
    #try multiple xpath-like paths and return first non-empty text
    for path in paths:
        node = root.find(path)
        text = safe_text(node)
        if text:
            return text
    return ""


def parse_gene_type(gene_elem: ET.Element) -> str:
    #parse ncbi gene type from Entrezgene_type/@value
    node = gene_elem.find(".//Entrezgene_type")
    if node is None:
        return ""
    val = node.attrib.get("value", "").strip()
    if not val:
        return ""
    return GENE_TYPE_MAP.get(val, f"other({val})")


def parse_go_terms(gene_elem: ET.Element) -> tuple[str, str, str]:
    #parse go ids, term names, and categories from ncbi xml gene-commentary sections
    tuples: set[tuple[str, str, str]] = set()

    for go_root in gene_elem.findall(".//Gene-commentary"):
        heading = safe_text(go_root.find("Gene-commentary_heading"))
        if heading.lower() != "geneontology":
            continue

        for category_node in go_root.findall(".//Gene-commentary"):
            category_heading = safe_text(
                category_node.find("Gene-commentary_heading")
            )
            category = category_heading.lower().replace(" ", "_")

            for term_node in category_node.findall(".//Gene-commentary"):
                term_name = safe_text(term_node.find("Gene-commentary_text"))

                go_id = ""
                for dbtag in term_node.findall(".//Dbtag"):
                    db_name = safe_text(dbtag.find("Dbtag_db"))
                    if db_name != "GO":
                        continue
                    go_str = safe_text(dbtag.find(".//Object-id_str"))
                    go_num = safe_text(dbtag.find(".//Object-id_id"))
                    if go_str:
                        go_id = go_str
                    elif go_num:
                        go_id = f"GO:{go_num}"
                    if go_id:
                        break

                if go_id or term_name:
                    tuples.add((go_id, term_name, category))

    if not tuples:
        return "", "", ""

    sorted_items = sorted(tuples)
    go_ids = "|".join([x[0] for x in sorted_items if x[0]])
    go_names = "|".join([x[1] for x in sorted_items if x[1]])
    go_cats = "|".join([x[2] for x in sorted_items if x[2]])
    return go_ids, go_names, go_cats


def parse_omim_ids(gene_elem: ET.Element) -> str:
    #parse omim/mim ids from dbtag sections
    omim: set[str] = set()
    for dbtag in gene_elem.findall(".//Dbtag"):
        db_name = safe_text(dbtag.find("Dbtag_db"))
        if db_name != "MIM":
            continue
        val = safe_text(dbtag.find(".//Object-id_id"))
        if not val:
            val = safe_text(dbtag.find(".//Object-id_str"))
        if val:
            omim.add(val)
    return "|".join(sorted(omim))


def parse_gene_record(
    gene_elem: ET.Element,
) -> tuple[str, dict[str, str]] | None:
    #parse one Entrezgene xml record into annotation dict keyed by entrez_id
    entrez_id = first_text(
        gene_elem,
        [
            ".//Entrezgene_track-info/Gene-track/Gene-track_geneid",
            ".//Gene-track_geneid",
        ],
    )
    if not entrez_id:
        return None

    ncbi_symbol = first_text(
        gene_elem,
        [
            ".//Entrezgene_gene/Gene-ref/Gene-ref_locus",
            ".//Gene-ref_locus",
        ],
    )
    full_name = first_text(
        gene_elem,
        [
            ".//Entrezgene_gene/Gene-ref/Gene-ref_desc",
            ".//Gene-ref_desc",
        ],
    )
    summary = first_text(gene_elem, [".//Entrezgene_summary"])
    chromosome = first_text(
        gene_elem,
        [
            ".//Entrezgene_location/Maps/Maps_display-str",
            ".//Maps_display-str",
        ],
    )
    map_location = first_text(
        gene_elem,
        [
            ".//Entrezgene_gene/Gene-ref/Gene-ref_maploc",
            ".//Gene-ref_maploc",
        ],
    )
    gene_type = parse_gene_type(gene_elem)
    go_ids, go_terms, go_categories = parse_go_terms(gene_elem)
    omim_ids = parse_omim_ids(gene_elem)

    annotation = {
        "ncbi_symbol": ncbi_symbol,
        "full_name": full_name,
        "gene_type": gene_type,
        "summary": summary,
        "chromosome": chromosome,
        "map_location": map_location,
        "go_ids": go_ids,
        "go_term_names": go_terms,
        "go_categories": go_categories,
        "omim_ids": omim_ids,
    }
    return entrez_id, annotation


def fetch_batch_xml(
    batch_ids: list[int],
    api_key: str | None,
    batch_index: int,
    total_batches: int,
) -> str | None:
    #fetch one ncbi efetch xml batch with one retry on failure
    id_text = ",".join([str(x) for x in batch_ids])
    params = {
        "db": "gene",
        "id": id_text,
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = EUTILS_BASE
    for attempt in [1, 2]:
        try:
            response = requests.get(url, params=params, timeout=90)
            if response.status_code == 200:
                return response.text
            if attempt == 1:
                print(
                    f"Batch {batch_index}/{total_batches}: HTTP "
                    f"{response.status_code}, retrying in 5s..."
                )
                time.sleep(5)
        except Exception as exc:  # noqa: BLE001
            if attempt == 1:
                print(
                    f"Batch {batch_index}/{total_batches}: request error "
                    f"({exc}), retrying in 5s..."
                )
                time.sleep(5)
            else:
                return None
    return None


def parse_batch_xml(
    xml_text: str,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    #parse efetch xml payload into per-entrez annotations and list of malformed ids
    result: dict[str, dict[str, str]] = {}
    malformed_ids: list[str] = []

    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to parse NCBI efetch XML response."
        ) from exc

    for gene_elem in root.findall(".//Entrezgene"):
        try:
            parsed = parse_gene_record(gene_elem)
            if parsed is None:
                malformed_ids.append("<missing_geneid>")
                continue
            entrez_id, annotation = parsed
            result[str(entrez_id)] = annotation
        except Exception:  # noqa: BLE001
            bad_id = first_text(gene_elem, [".//Gene-track_geneid"])
            malformed_ids.append(bad_id or "<unknown>")

    return result, malformed_ids


def build_empty_annotation() -> dict[str, str]:
    #return empty annotation dict for genes with no ncbi record
    return {
        "ncbi_symbol": "",
        "full_name": "",
        "gene_type": "",
        "summary": "",
        "chromosome": "",
        "map_location": "",
        "go_ids": "",
        "go_term_names": "",
        "go_categories": "",
        "omim_ids": "",
    }


def main() -> int:
    args = parse_args()

    #resolve paths
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    log_path = resolve_path(args.log)

    print(f"[NCBI] Input: {input_path}")
    print(f"[NCBI] Output: {output_path}")
    print(f"[NCBI] Log: {log_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    df_in = pd.read_csv(input_path)
    required = ["gene", "approved_symbol", "entrez_id"]
    for col in required:
        if col not in df_in.columns:
            raise ValueError(f"Input CSV is missing required column: {col}")

    api_key, has_key = get_api_key(args.api_key)
    sleep_seconds = 0.1 if has_key else 0.34
    print(f"[NCBI] API key: {'found' if has_key else 'not found'}")
    print(f"[NCBI] Rate limit sleep: {sleep_seconds}s between batches")

    #separate genes with and without entrez ids
    has_entrez_mask = df_in["entrez_id"].notna() & (
        df_in["entrez_id"].astype(str).str.strip() != ""
    )
    no_entrez_mask = ~has_entrez_mask

    has_entrez_df = df_in[has_entrez_mask].copy()
    no_entrez_df = df_in[no_entrez_mask].copy()

    print(f"[NCBI] Total genes: {len(df_in)}")
    print(f"[NCBI] With entrez_id: {len(has_entrez_df)}")
    print(f"[NCBI] Without entrez_id: {len(no_entrez_df)}")

    #prepare unique entrez ids and batch them
    query_ids: list[int] = []
    gene_by_entrez: dict[str, list[str]] = {}
    for _, row in has_entrez_df.iterrows():
        raw = str(row["entrez_id"]).strip()
        try:
            eid = int(float(raw))
        except Exception:
            continue
        query_ids.append(eid)
        gene_by_entrez.setdefault(str(eid), []).append(str(row["gene"]))

    unique_query_ids = sorted(set(query_ids))
    batches = chunked(unique_query_ids, int(args.batch_size))

    fetched_annotations: dict[str, dict[str, str]] = {}
    malformed_geneids: list[str] = []
    failed_ids: set[str] = set()
    successful_batch_count = 0

    #batch fetch loop
    total_batches = len(batches)
    print(f"[NCBI] Total batches: {total_batches}")
    for i, batch in enumerate(batches, start=1):
        xml_text = fetch_batch_xml(
            batch_ids=batch,
            api_key=api_key,
            batch_index=i,
            total_batches=total_batches,
        )
        if xml_text is None:
            for eid in batch:
                failed_ids.add(str(eid))
            print(
                f"Batch {i}/{total_batches}: failed after retry "
                f"({len(batch)} IDs)"
            )
            time.sleep(sleep_seconds)
            continue

        parsed, malformed = parse_batch_xml(xml_text)
        fetched_annotations.update(parsed)
        malformed_geneids.extend(malformed)
        successful_batch_count += 1
        print(
            f"Batch {i}/{total_batches}: fetched {len(parsed)} records"
        )
        time.sleep(sleep_seconds)

    #merge annotations back to all input rows
    fetched_ids = set(fetched_annotations.keys())
    rows_out: list[dict[str, Any]] = []
    no_record_genes: list[str] = []
    fetch_failed_genes: list[str] = []
    symbol_mismatch_rows: list[tuple[str, str, str]] = []

    empty_anno = build_empty_annotation()

    for _, row in df_in.iterrows():
        out = {str(k): v for k, v in row.to_dict().items()}
        gene = str(out.get("gene", ""))
        approved_symbol = str(out.get("approved_symbol", ""))
        entrez_raw = out.get("entrez_id", None)

        status = ""
        anno = empty_anno.copy()

        if pd.isna(entrez_raw) or str(entrez_raw).strip() == "":
            status = "no_entrez_id"
        else:
            try:
                eid = str(int(float(str(entrez_raw).strip())))
            except Exception:
                eid = ""

            if not eid:
                status = "no_entrez_id"
            elif eid in failed_ids:
                status = "fetch_failed"
                fetch_failed_genes.append(f"{gene} ({eid})")
            elif eid in fetched_annotations:
                status = "fetched"
                anno = fetched_annotations[eid]
                ncbi_symbol = anno.get("ncbi_symbol", "")
                if (
                    ncbi_symbol
                    and approved_symbol
                    and ncbi_symbol != approved_symbol
                ):
                    symbol_mismatch_rows.append(
                        (gene, approved_symbol, ncbi_symbol)
                    )
            else:
                status = "no_record"
                no_record_genes.append(f"{gene} ({eid})")

        out.update(anno)
        out["ncbi_fetch_status"] = status
        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    #validate row count and required columns
    if len(df_out) != len(df_in):
        raise AssertionError(
            "Output row count differs from input row count."
        )
    if df_out["gene"].isna().any() or df_out["approved_symbol"].isna().any():
        raise AssertionError(
            "Null values found in required columns (gene/approved_symbol)."
        )

    fetched_n = int((df_out["ncbi_fetch_status"] == "fetched").sum())
    total_n = int(len(df_out))
    success_rate = (100.0 * fetched_n / total_n) if total_n else 0.0
    print(f"[NCBI] Fetch success rate: {fetched_n} of {total_n} genes fetched")
    if success_rate < 90.0:
        print("[warn] Fetch success rate below 90%.")

    #coverage stats
    empty_summary_n = int((df_out["summary"].fillna("") == "").sum())
    no_go_n = int((df_out["go_ids"].fillna("") == "").sum())
    no_omim_n = int((df_out["omim_ids"].fillna("") == "").sum())

    #write output csv
    ensure_parent_dir(output_path)
    output_cols_input = [
        "gene",
        "approved_symbol",
        "hgnc_id",
        "entrez_id",
        "ensembl_gene_id",
        "alias_symbols",
        "prev_symbols",
        "normalisation_status",
    ]
    output_cols_new = [
        "ncbi_symbol",
        "full_name",
        "gene_type",
        "summary",
        "chromosome",
        "map_location",
        "go_ids",
        "go_term_names",
        "go_categories",
        "omim_ids",
        "ncbi_fetch_status",
    ]
    for col in output_cols_input + output_cols_new:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out = df_out[output_cols_input + output_cols_new]
    df_out.to_csv(output_path, index=False)
    print(f"[NCBI] Wrote output CSV: {output_path}")

    #write log
    ensure_parent_dir(log_path)
    log_lines: list[str] = []
    log_lines.append(
        f"Run timestamp: {datetime.now(timezone.utc).isoformat()}"
    )
    log_lines.append(f"api_key status: {'found' if has_key else 'not found'}")
    log_lines.append(f"total genes in input: {total_n}")
    log_lines.append(f"genes with entrez_id (queried): {len(has_entrez_df)}")
    log_lines.append(f"genes without entrez_id (kept): {len(no_entrez_df)}")
    log_lines.append(f"total batches sent: {total_batches}")
    log_lines.append(f"successful batches: {successful_batch_count}")
    log_lines.append(f"total records fetched: {len(fetched_ids)}")
    log_lines.append(
        f"fetch success rate: {fetched_n}/{total_n} ({success_rate:.2f}%)"
    )
    log_lines.append("")

    log_lines.append("genes with no NCBI record found:")
    if no_record_genes:
        for item in sorted(set(no_record_genes)):
            log_lines.append(f"  {item}")
    else:
        log_lines.append("  (none)")

    log_lines.append("")
    log_lines.append("genes where fetch failed:")
    if fetch_failed_genes:
        for item in sorted(set(fetch_failed_genes)):
            log_lines.append(f"  {item}")
    else:
        log_lines.append("  (none)")

    log_lines.append("")
    log_lines.append(f"genes with empty summary: {empty_summary_n}")
    log_lines.append(f"genes with no GO terms: {no_go_n}")
    log_lines.append(f"genes with no OMIM IDs: {no_omim_n}")

    log_lines.append("")
    log_lines.append(
        "entrez_id symbol mismatches "
        "(approved_symbol vs ncbi_symbol):"
    )
    if symbol_mismatch_rows:
        for gene, approved, ncbi in sorted(set(symbol_mismatch_rows)):
            log_lines.append(
                f"  {gene}: approved_symbol={approved}, ncbi_symbol={ncbi}"
            )
    else:
        log_lines.append("  (none)")

    if malformed_geneids:
        log_lines.append("")
        log_lines.append("malformed gene records skipped:")
        for gid in sorted(set(malformed_geneids)):
            log_lines.append(f"  {gid}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[NCBI] Wrote log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[NCBI][error] {exc}", file=sys.stderr)
        raise
