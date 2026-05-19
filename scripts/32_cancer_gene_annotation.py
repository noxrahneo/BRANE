#!/usr/bin/env python3
"""Annotate persistent network genes with cancer evidence tiers."""

from __future__ import annotations

import argparse
import base64
import getpass
import gzip
import hashlib
import io
import json
import logging
import os
import re
import sys
import tarfile
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

COSMIC_DOWNLOAD_API = (
    "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted"
    "?path=grch38/cosmic/v103/Cosmic_CancerGeneCensus_Tsv_v103_GRCh38.tar"
    "&bucket=downloads"
)
COSMIC_EXPECTED_MD5 = "BA1B771995FE10DFD7E69EEFB9CBDDD5"
ONCOKB_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList.txt"
NCG_URL = "http://ncg.kcl.ac.uk/download/NCG_cancergenes.tsv"
NCG_DOWNLOAD_PAGE_URL = "http://network-cancer-genes.org/download.php"
OPEN_TARGETS_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
OPEN_TARGETS_EFO = "EFO_0000305"
OPEN_TARGETS_PAGE_SIZE = 500
OPEN_TARGETS_THRESHOLD = 0.3
INTOGEN_URLS_TO_TRY = [
    "https://intogen.org/api/drivers/download?released=2024-09-20",
    "https://www.intogen.org/download/drivers.tsv",
    "https://bbglab.irbbarcelona.org/intogen/release/2024/drivers.tsv",
]
ONCOVAR_URLS = {
    "TCGA": "https://oncovar.org/Data/Download/Onco_genes_OncoVar_TCGA/TCGA.BRCA.onco.genes.OncoVar.tsv.gz",
    "ICGC": "https://oncovar.org/Data/Download/Onco_genes_OncoVar_ICGC/ICGC.BRCA.onco.genes.OncoVar.tsv.gz",
}
CANCERMINE_URL = (
    "https://zenodo.org/record/7689627/files/cancermine_collated.tsv"
)
OMIM_GENEMAP2_URL_TEMPLATE = (
    "https://data.omim.org/downloads/{api_key}/genemap2.txt"
)
OMIM_CANCER_KEYWORDS = [
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "leukemia",
    "lymphoma",
    "melanoma",
    "sarcoma",
    "glioma",
    "blastoma",
    "neoplasm",
    "malignant",
    "oncogenesis",
]
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ONCOGENE_KW = "KW-0656"
UNIPROT_TSG_KW = "KW-0043"
COSMIC_TABLE_SUFFIXES = (
    ".tsv",
    ".txt",
    ".csv",
    ".tsv.gz",
    ".txt.gz",
    ".csv.gz",
)


def score_cosmic_member_name(member_name: str) -> int:
    #heuristic score for selecting the real COSMIC census table file
    name = member_name.lower()
    score = 0
    if "cancergenecensus" in name or "cancer_gene_census" in name:
        score += 50
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        score += 20
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        score += 10
    if "readme" in name or "description" in name or "schema" in name:
        score -= 30
    return score


def looks_like_cosmic_table(content: bytes) -> bool:
    #light validation to distinguish census table from prose docs
    text = content.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    low = first_line.lower()
    return (
        "gene symbol" in low
        or "gene_symbol" in low
        or (
            "tier" in low
            and ("somatic" in low or "germline" in low)
            and ("\t" in first_line or "," in first_line)
        )
    )


def has_required_cosmic_columns(columns: list[str]) -> bool:
    #return True if columns match known COSMIC CGC header variants
    cols = set(columns)
    human = {
        "Gene Symbol",
        "Role in Cancer",
        "Tier",
        "Somatic",
        "Germline",
        "Tumour Types(Somatic)",
        "Tumour Types(Germline)",
    }
    machine = {
        "GENE_SYMBOL",
        "ROLE_IN_CANCER",
        "TIER",
        "SOMATIC",
        "GERMLINE",
        "TUMOUR_TYPES_SOMATIC",
        "TUMOUR_TYPES_GERMLINE",
    }
    return human.issubset(cols) or machine.issubset(cols)


def is_parseable_cosmic_table(content: bytes) -> bool:
    #check if bytes decode into a COSMIC CGC table with known headers
    for sep in ("\t", ","):
        try:
            df_hdr = pd.read_csv(io.BytesIO(content), sep=sep, nrows=0)
            if has_required_cosmic_columns(list(df_hdr.columns)):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


class ScriptError(RuntimeError):
    pass  #user-facing script failure with clean stderr message


def parse_args() -> argparse.Namespace:
    #parse command-line arguments
    parser = argparse.ArgumentParser(description="Cancer gene annotation")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV from script 48",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output master annotation CSV",
    )
    parser.add_argument("--log", required=True, help="Output log file path")
    parser.add_argument(
        "--cosmic-cache",
        default="data/reference/cosmic_cgc_v103_grch38.tsv",
        help="Cached COSMIC CGC TSV path",
    )
    parser.add_argument(
        "--oncokb-cache",
        default="data/reference/oncokb_cancer_genes.tsv",
        help="Cached OncoKB file path",
    )
    parser.add_argument(
        "--ot-cache",
        default="data/reference/open_targets_breast_cancer.json",
        help="Cached Open Targets JSON path",
    )
    parser.add_argument(
        "--intogen-cache",
        default="data/2024-06-18_IntOGen-Drivers/Compendium_Cancer_Genes.tsv",
        help="Cached IntOGen TSV path",
    )
    parser.add_argument(
        "--oncovar-cache",
        default=None,
        help="Cached OncoVar BRCA TSV path (optional)",
    )
    parser.add_argument(
        "--cancermine-cache",
        default="data/reference/cancermine_collated.tsv",
        help="Cached CancerMine TSV path",
    )
    parser.add_argument(
        "--uniprot-cache",
        default="data/reference/uniprot_cancer_keywords.json",
        help="Cached UniProt keyword JSON path",
    )
    parser.add_argument(
        "--cache-max-age",
        type=int,
        default=30,
        help="Max cache age (days) for OncoKB and Open Targets",
    )
    parser.add_argument(
        "--cancermine-min-citations",
        type=int,
        default=3,
        help="Minimum citation count for CancerMine entries",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    #resolve relative and user paths into absolute paths
    return Path(path_text).expanduser().resolve()


def ensure_parent_dir(path: Path) -> None:
    #create parent directories for a target path
    path.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    #current UTC timestamp string
    return datetime.now(timezone.utc).isoformat()


def cache_is_fresh(path: Path, max_age_days: int) -> bool:
    #return True when cache exists and is not older than threshold
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) <= timedelta(days=max_age_days)


def normalize_symbol(symbol: Any) -> str:
    #normalize a gene symbol for stable matching
    if symbol is None:
        return ""
    return str(symbol).strip().upper()


def must_get_cosmic_credentials() -> tuple[str, str]:
    #read COSMIC credentials from env, or prompt interactively
    email = os.getenv("COSMIC_EMAIL", "").strip()
    password = os.getenv("COSMIC_PASSWORD", "").strip()

    if email and password:
        return email, password

    if not sys.stdin.isatty():
        raise ScriptError(
            "Missing COSMIC credentials. Set COSMIC_EMAIL/COSMIC_PASSWORD "
            "or run interactively to enter them when prompted."
        )

    print("COSMIC_EMAIL not set. Enter COSMIC email:")
    email = input().strip()
    if not email:
        raise ScriptError("COSMIC email is required.")

    password = getpass.getpass("Enter COSMIC password: ").strip()
    if not password:
        raise ScriptError("COSMIC password is required.")

    return email, password


def role_normalize_cosmic(role_raw: str) -> str:
    #normalize COSMIC role labels to controlled values
    role = role_raw.strip().lower()
    if role == "oncogene":
        return "oncogene"
    if role == "tsg":
        return "TSG"
    if role == "oncogene, tsg":
        return "oncogene, TSG"
    if role == "fusion":
        return "fusion"
    return "other"


def role_from_oncokb(is_oncogene: bool, is_tsg: bool) -> str:
    #derive role from OncoKB binary flags
    if is_oncogene and is_tsg:
        return "oncogene, TSG"
    if is_oncogene:
        return "oncogene"
    if is_tsg:
        return "TSG"
    return "other"


def cancer_role_from_source(source_role: str) -> str:
    #map source role into final cancer_role taxonomy
    role = source_role.strip().lower()
    if role == "oncogene":
        return "oncogene"
    if role == "tsg":
        return "TSG"
    if role == "fusion":
        return "fusion"
    if role == "both":
        return "both"
    if role == "oncogene, tsg":
        return "both"
    return "not_classified"


def parse_bool_like(value: Any) -> bool:
    #parse bool-like values used in TSV files
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def download_cosmic_tsv(cosmic_cache: Path, log_lines: list[str]) -> Path:
    #authenticate with COSMIC, download tar, extract and cache TSV
    email, password = must_get_cosmic_credentials()

    credentials = f"{email}:{password}".encode("utf-8")
    auth_string = base64.b64encode(credentials).decode("utf-8")

    headers = {"Authorization": f"Basic {auth_string}"}
    try:
        response = requests.get(
            COSMIC_DOWNLOAD_API,
            headers=headers,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(
            "Failed to contact COSMIC scripted-download API."
        ) from exc

    if response.status_code != 200:
        raise ScriptError(
            "COSMIC authentication/download URL request failed with status "
            f"{response.status_code}. Check credentials and account access."
        )

    try:
        download_url = response.json()["url"]
    except Exception as exc:  # noqa: BLE001
        raise ScriptError("COSMIC response missing download URL.") from exc

    log_lines.append("Obtained COSMIC download URL (valid for 1 hour)")

    try:
        tar_response = requests.get(download_url, timeout=180)
    except Exception as exc:  # noqa: BLE001
        raise ScriptError("Failed to download COSMIC archive.") from exc

    if tar_response.status_code != 200:
        raise ScriptError(
            "COSMIC archive download failed with status "
            f"{tar_response.status_code}."
        )

    tar_bytes = tar_response.content
    md5 = hashlib.md5(tar_bytes).hexdigest().upper()  # noqa: S324
    if md5 != COSMIC_EXPECTED_MD5:
        warning = (
            "WARNING: COSMIC MD5 mismatch. expected="
            f"{COSMIC_EXPECTED_MD5}, got={md5}"
        )
        print(warning)
        log_lines.append(warning)
    else:
        log_lines.append(f"COSMIC MD5 checksum verified: {md5}")

    try:
        tar_obj = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*")
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(
            "Downloaded COSMIC archive is not a valid tar file."
        ) from exc

    extracted_tsv: bytes | None = None
    scanned_members: list[str] = []
    best_candidates: list[tuple[int, str, bytes]] = []
    with tar_obj:
        for member in tar_obj.getmembers():
            if not member.isfile():
                continue

            member_name = member.name
            scanned_members.append(member_name)
            lowered = member_name.lower()
            if not lowered.endswith(COSMIC_TABLE_SUFFIXES):
                continue

            extracted = tar_obj.extractfile(member)
            if extracted is None:
                continue
            raw_data = extracted.read()
            if lowered.endswith(".gz"):
                try:
                    candidate_bytes = gzip.decompress(raw_data)
                except Exception as exc:  # noqa: BLE001
                    raise ScriptError(
                        "Found compressed table in COSMIC archive but "
                        "failed to "
                        f"decompress: {member_name}"
                    ) from exc
            else:
                candidate_bytes = raw_data

            if looks_like_cosmic_table(candidate_bytes):
                if is_parseable_cosmic_table(candidate_bytes):
                    extracted_tsv = candidate_bytes
                    log_lines.append(
                        f"COSMIC archive member used: {member_name}"
                    )
                    break

            score = score_cosmic_member_name(member_name)
            best_candidates.append((score, member_name, candidate_bytes))

        if extracted_tsv is None and best_candidates:
            best_candidates.sort(key=lambda x: x[0], reverse=True)
            for _, member_name, candidate_bytes in best_candidates:
                if is_parseable_cosmic_table(candidate_bytes):
                    extracted_tsv = candidate_bytes
                    log_lines.append(
                        "COSMIC archive member fallback used: "
                        f"{member_name}"
                    )
                    break

    if extracted_tsv is None:
        preview = ", ".join(scanned_members[:10])
        raise ScriptError(
            "No table file (.tsv/.txt/.csv, optional .gz) found inside "
            "COSMIC archive. "
            f"Scanned members (first 10): {preview}"
        )

    ensure_parent_dir(cosmic_cache)
    cosmic_cache.write_bytes(extracted_tsv)
    log_lines.append(
        "COSMIC Cancer Gene Census downloaded and cached: "
        f"{cosmic_cache}"
    )
    return cosmic_cache


def load_cosmic(
    cosmic_cache: Path,
    log_lines: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    #load COSMIC cache (or download first), then parse lookup structures
    if cosmic_cache.exists():
        cache_bytes = cosmic_cache.read_bytes()
        if is_parseable_cosmic_table(cache_bytes):
            log_lines.append(f"Using cached COSMIC file: {cosmic_cache}")
        else:
            log_lines.append(
                "Cached COSMIC file is not a parseable CGC table; "
                "re-downloading"
            )
            cosmic_cache.unlink(missing_ok=True)
            download_cosmic_tsv(cosmic_cache, log_lines)
    else:
        log_lines.append("COSMIC cache not found, downloading")
        download_cosmic_tsv(cosmic_cache, log_lines)

    try:
        df = pd.read_csv(cosmic_cache, sep="\t", dtype=str)
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(
            f"Failed to read COSMIC TSV: {cosmic_cache}"
        ) from exc

    if not has_required_cosmic_columns(list(df.columns)):
        try:
            df = pd.read_csv(
                cosmic_cache,
                sep=",",
                dtype=str,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:  # noqa: BLE001
            raise ScriptError(
                "Failed to parse COSMIC cache as comma-delimited table: "
                f"{cosmic_cache}"
            ) from exc

    def pick_col(*names: str) -> str:
        for col in names:
            if col in df.columns:
                return col
        raise ScriptError(
            "COSMIC table missing required columns; looked for any of: "
            f"{names}. Found columns: {list(df.columns)[:12]}"
        )

    col_gene = pick_col("Gene Symbol", "GENE_SYMBOL")
    col_role = pick_col("Role in Cancer", "ROLE_IN_CANCER")
    col_tier = pick_col("Tier", "TIER")
    col_somatic = pick_col("Somatic", "SOMATIC")
    col_germline = pick_col("Germline", "GERMLINE")
    col_ttum_s = pick_col("Tumour Types(Somatic)", "TUMOUR_TYPES_SOMATIC")
    col_ttum_g = pick_col("Tumour Types(Germline)", "TUMOUR_TYPES_GERMLINE")

    cosmic_genes: set[str] = set()
    cosmic_data: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        gene = normalize_symbol(row.get(col_gene, ""))
        if not gene:
            continue

        role_raw = str(row.get(col_role, "") or "")
        tier_raw = str(row.get(col_tier, "") or "").strip()
        somatic = str(row.get(col_somatic, "") or "").strip()
        germline = str(row.get(col_germline, "") or "").strip()
        tumour_s = str(row.get(col_ttum_s, "") or "").strip()
        tumour_g = str(row.get(col_ttum_g, "") or "").strip()

        cosmic_genes.add(gene)
        cosmic_data[gene] = {
            "cosmic_role": role_normalize_cosmic(role_raw),
            "cosmic_tier": tier_raw,
            "somatic": somatic,
            "germline": germline,
            "tumour_types_somatic": tumour_s,
            "tumour_types_germline": tumour_g,
        }

    log_lines.append(
        f"COSMIC Cancer Gene Census loaded: {len(cosmic_genes)} genes"
    )
    return cosmic_genes, cosmic_data


def fetch_oncokb(oncokb_cache: Path) -> None:
    #fetch OncoKB cancerGeneList and save to cache file
    try:
        response = requests.get(ONCOKB_URL, timeout=60)
    except Exception as exc:  # noqa: BLE001
        raise ScriptError("Failed to connect to OncoKB API.") from exc

    if response.status_code != 200:
        raise ScriptError(
            f"OncoKB API request failed with status {response.status_code}."
        )

    ensure_parent_dir(oncokb_cache)
    oncokb_cache.write_text(response.text, encoding="utf-8")


def load_oncokb(
    oncokb_cache: Path,
    cache_max_age: int,
    log_lines: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    #load or fetch OncoKB and parse lookup structures
    if cache_is_fresh(oncokb_cache, cache_max_age):
        log_lines.append("Using cached OncoKB file")
    else:
        fetch_oncokb(oncokb_cache)
        log_lines.append("Fetched OncoKB cancer gene list from API")

    try:
        df = pd.read_csv(oncokb_cache, sep="\t", dtype=str)
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(
            f"Failed to read OncoKB TSV: {oncokb_cache}"
        ) from exc

    required = ["hugoSymbol", "isOncogene", "isTSG"]
    missing = [c for c in required if c not in df.columns]

    oncokb_genes: set[str] = set()
    oncokb_data: dict[str, dict[str, Any]] = {}

    if not missing:
        for _, row in df.iterrows():
            gene = normalize_symbol(row.get("hugoSymbol", ""))
            if not gene:
                continue

            is_oncogene = parse_bool_like(row.get("isOncogene", "False"))
            is_tsg = parse_bool_like(row.get("isTSG", "False"))

            oncokb_genes.add(gene)
            oncokb_data[gene] = {
                "oncokb_is_oncogene": is_oncogene,
                "oncokb_is_tsg": is_tsg,
                "oncokb_role": role_from_oncokb(is_oncogene, is_tsg),
            }
    else:
        #newer endpoint format: uses "Hugo Symbol" + "Gene Type".
        if "Hugo Symbol" in df.columns and "Gene Type" in df.columns:
            df2 = df
        else:
            try:
                df2 = pd.read_csv(
                    oncokb_cache,
                    sep=r"\s{2,}",
                    engine="python",
                    dtype=str,
                )
            except Exception as exc:  # noqa: BLE001
                raise ScriptError(
                    f"OncoKB TSV missing required columns: {missing}"
                ) from exc

            if (
                "Hugo Symbol" not in df2.columns
                or "Gene Type" not in df2.columns
            ):
                raise ScriptError(
                    f"OncoKB TSV missing required columns: {missing}"
                )

        for _, row in df2.iterrows():
            gene = normalize_symbol(row.get("Hugo Symbol", ""))
            if not gene:
                continue

            gene_type = str(row.get("Gene Type", "") or "").upper()
            is_oncogene = "ONCOGENE" in gene_type
            is_tsg = "TSG" in gene_type

            oncokb_genes.add(gene)
            oncokb_data[gene] = {
                "oncokb_is_oncogene": is_oncogene,
                "oncokb_is_tsg": is_tsg,
                "oncokb_role": role_from_oncokb(is_oncogene, is_tsg),
            }

    log_lines.append(
        f"OncoKB cancer gene list loaded: {len(oncokb_genes)} genes"
    )
    return oncokb_genes, oncokb_data


def fetch_open_targets_all() -> list[dict[str, Any]]:
    #fetch all breast cancer associated targets via paginated GraphQL
    query = """
    query AssociatedTargets($efoId: String!, $index: Int!, $size: Int!) {
      disease(efoId: $efoId) {
        associatedTargets(page: {index: $index, size: $size}) {
          count
          rows {
            target {
              approvedSymbol
              approvedName
            }
            score
          }
        }
      }
    }
    """

    out: list[dict[str, Any]] = []
    page_index = 0

    while True:
        variables = {
            "efoId": OPEN_TARGETS_EFO,
            "index": page_index,
            "size": OPEN_TARGETS_PAGE_SIZE,
        }
        payload = {"query": query, "variables": variables}

        try:
            response = requests.post(
                OPEN_TARGETS_ENDPOINT,
                json=payload,
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001
            raise ScriptError(
                "Failed to connect to Open Targets API."
            ) from exc

        if response.status_code != 200:
            raise ScriptError(
                "Open Targets API request failed with status "
                f"{response.status_code}."
            )

        try:
            content = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ScriptError(
                "Open Targets API returned non-JSON payload."
            ) from exc

        if "errors" in content:
            raise ScriptError(
                f"Open Targets GraphQL error: {content['errors']}"
            )

        disease = content.get("data", {}).get("disease", {})
        assoc = disease.get("associatedTargets", {}) if disease else {}
        rows = assoc.get("rows", []) if assoc else []

        if not rows:
            break

        for row in rows:
            target = row.get("target", {}) or {}
            symbol = str(target.get("approvedSymbol", "") or "").strip()
            if not symbol:
                continue
            score_val = row.get("score", 0.0)
            try:
                score = float(score_val)
            except (TypeError, ValueError):
                score = 0.0
            out.append({
                "approvedSymbol": symbol,
                "score": score,
            })

        page_index += 1

    return out


def load_open_targets(
    ot_cache: Path,
    cache_max_age: int,
    log_lines: list[str],
) -> dict[str, float]:
    #load or fetch Open Targets and create symbol->score map
    if cache_is_fresh(ot_cache, cache_max_age):
        log_lines.append("Using cached Open Targets data")
        try:
            records = json.loads(ot_cache.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ScriptError(
                f"Failed to parse Open Targets cache: {ot_cache}"
            ) from exc
    else:
        records = fetch_open_targets_all()
        ensure_parent_dir(ot_cache)
        ot_cache.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_lines.append(
            "Fetched Open Targets breast cancer associations from API"
        )

    if not isinstance(records, list):
        raise ScriptError("Open Targets cached content is not a list.")

    ot_data: dict[str, float] = {}
    for rec in records:
        symbol = normalize_symbol(rec.get("approvedSymbol", ""))
        if not symbol:
            continue
        score_val = rec.get("score", 0.0)
        try:
            score = float(score_val)
        except (TypeError, ValueError):
            score = 0.0
        existing = ot_data.get(symbol, 0.0)
        if score > existing:
            ot_data[symbol] = score

    n_above = sum(
        1 for score in ot_data.values() if score >= OPEN_TARGETS_THRESHOLD
    )
    log_lines.append(
        "Open Targets: fetched "
        f"{len(ot_data)} breast cancer gene associations"
    )
    log_lines.append(
        f"Open Targets threshold: {OPEN_TARGETS_THRESHOLD}"
    )
    log_lines.append(
        f"Open Targets genes with score >= {OPEN_TARGETS_THRESHOLD}: {n_above}"
    )

    return ot_data


def safe_float(value: Any, default: float = 0.0) -> float:
    #convert value to float safely, returning default on failure
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def map_intogen_role(role_raw: str) -> str:
    #map IntOGen role values to script role labels
    role = str(role_raw).strip()
    if role == "Act":
        return "oncogene"
    if role == "LoF":
        return "TSG"
    if role == "ambiguous":
        return "both"
    return "other"


def load_intogen(
    intogen_cache: Path,
    log_lines: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, int]]:
    #load IntOGen drivers from cache or API and build lookup maps
    print("[IntOGen] Loading...")
    if intogen_cache.exists():
        log_lines.append("Using cached IntOGen file")
    else:
        print("[IntOGen] Downloading...")
        last_error = ""
        downloaded = False
        for url in INTOGEN_URLS_TO_TRY:
            try:
                response = requests.get(url, timeout=90)
            except Exception as exc:  # noqa: BLE001
                last_error = f"request error for {url}: {exc}"
                log_lines.append(
                    f"IntOGen download attempt failed: {last_error}"
                )
                continue

            if response.status_code == 200 and response.text.strip():
                ensure_parent_dir(intogen_cache)
                intogen_cache.write_text(response.text, encoding="utf-8")
                log_lines.append(
                    f"Downloaded IntOGen cancer drivers from: {url}"
                )
                downloaded = True
                break

            last_error = f"status={response.status_code} for {url}"
            log_lines.append(f"IntOGen download attempt failed: {last_error}")

        if not downloaded:
            raise ScriptError(
                "IntOGen download failed for all known URLs. "
                "Please manually download from "
                "https://www.intogen.org/download and save to "
                f"{intogen_cache}. Last error: {last_error}"
            )

    df = pd.read_csv(intogen_cache, sep="\t", dtype=str)
    log_lines.append(f"IntOGen columns: {list(df.columns)}")

    required = [
        "SYMBOL",
        "CANCER_TYPE",
        "ROLE",
        "METHODS",
        "QVALUE_COMBINATION",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ScriptError(f"IntOGen TSV missing required columns: {missing}")

    best: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        symbol = normalize_symbol(row.get("SYMBOL", ""))
        if not symbol:
            continue

        cancer_type = str(row.get("CANCER_TYPE", "") or "").strip()
        is_brca = 1 if cancer_type == "BRCA" else 0
        qvalue = safe_float(row.get("QVALUE_COMBINATION", None), default=1.0)
        role = map_intogen_role(str(row.get("ROLE", "") or ""))

        cand = {
            "intogen_role": role,
            "intogen_breast_specific": is_brca,
            "intogen_qvalue": qvalue,
        }
        cur = best.get(symbol)
        if cur is None:
            best[symbol] = cand
            continue

        if cand["intogen_breast_specific"] > cur["intogen_breast_specific"]:
            best[symbol] = cand
            continue
        if (
            cand["intogen_breast_specific"]
            == cur["intogen_breast_specific"]
            and cand["intogen_qvalue"] < cur["intogen_qvalue"]
        ):
            best[symbol] = cand

    intogen_genes = set(best.keys())
    intogen_brca = sum(
        1 for v in best.values() if v.get("intogen_breast_specific", 0) == 1
    )
    log_lines.append(
        f"IntOGen: {len(intogen_genes)} total driver genes loaded"
    )
    log_lines.append(f"IntOGen: {intogen_brca} breast-specific (BRCA) genes")
    return intogen_genes, best, {
        "total": len(intogen_genes),
        "breast": intogen_brca,
    }


def load_oncovar_brca_genes(
    oncovar_cache: Path | None,
    log_lines: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    #load oncovar brca oncogenic driver gene files; returns empty data if cache missing
    if oncovar_cache is None or not oncovar_cache.exists():
        return set(), {}

    print("[OncoVar] Loading...")
    log_lines.append("Using cached OncoVar BRCA file")

    try:
        df = pd.read_csv(oncovar_cache, sep="\t", dtype=str)
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"[OncoVar] Could not read cache file: {exc}")
        return set(), {}

    #normalize column names — OncoVar TSV columns vary
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "_")
        if cl in ("gene", "gene_symbol", "symbol", "hugo_symbol", "gene_name"):
            col_map[col] = "gene_symbol"
        elif "oncovar_score" in cl or ("score" in cl and "level" not in cl):
            col_map[col] = "oncovar_score"
        elif "level" in cl or "driver_level" in cl or "confidence_level" in cl:
            col_map[col] = "driver_level"
        elif "role" in cl or "cancer_role" in cl or ("type" in cl and "source" not in cl):
            col_map[col] = "cancer_role"
        elif "source" in cl and "oncovar_source" not in df.columns:
            col_map[col] = "oncovar_source"
    df = df.rename(columns=col_map)

    if "gene_symbol" not in df.columns:
        log_lines.append(f"[OncoVar] WARNING: Could not identify gene symbol column. Available: {list(df.columns)}")
        return set(), {}

    result = {}
    for _, row in df.iterrows():
        raw_sym = str(row.get("gene_symbol", "")).strip()
        if not raw_sym or raw_sym == "nan":
            continue

        #use raw symbol; will be normalized during per-gene matching
        approved = raw_sym

        score = row.get("oncovar_score", None)
        try:
            score = float(score) if score and str(score) not in ("nan", "") else None
        except (ValueError, TypeError):
            score = None

        level = str(row.get("driver_level", "Level4")).strip()
        role = str(row.get("cancer_role", "unknown")).strip().lower()
        src = str(row.get("oncovar_source", "Local")).strip()

        #only keep Level 4 genes (high-confidence) if filtering requested
        if "level4" not in level.lower() and "4" not in level and level not in ("nan", ""):
            continue

        #only keep if not already present with higher-confidence source, or merge
        if approved not in result:
            result[approved] = {
                "oncovar_score": score,
                "oncovar_level": level,
                "oncovar_role": role if role not in ("nan", "", "none") else "driver",
                "oncovar_sources": {src},
            }
        else:
            #merge sources; keep max score
            result[approved]["oncovar_sources"].add(src)
            if score is not None and (result[approved]["oncovar_score"] is None
                                       or float(score) > float(result[approved]["oncovar_score"] or 0)):
                result[approved]["oncovar_score"] = score

    #convert sets to sorted strings for serialization
    for sym in result:
        result[sym]["oncovar_sources"] = ";".join(sorted(result[sym]["oncovar_sources"]))

    log_lines.append(f"[OncoVar] {len(result)} unique BRCA driver genes loaded.")
    print(f"[OncoVar] {len(result)} unique BRCA driver genes loaded.")
    return set(result.keys()), result


def resolve_hgnc_symbol(
    symbol: Any,
    alias_dict: dict[str, str],
    approved_set: set[str],
) -> str:
    #resolve a symbol to approved HGNC, keeping unresolved symbols as-is
    sym = normalize_symbol(symbol)
    if not sym:
        return ""
    if sym in approved_set:
        return sym
    mapped = normalize_symbol(alias_dict.get(sym, ""))
    if mapped:
        return mapped
    return sym


def build_alias_and_approved_from_df(
    df: pd.DataFrame,
) -> tuple[dict[str, str], set[str]]:
    #build alias->approved map and approved HGNC symbol set from input table
    alias_dict: dict[str, str] = {}
    approved_set: set[str] = set()

    if "approved_symbol" not in df.columns:
        return alias_dict, approved_set

    alias_cols = [c for c in ("alias_symbols", "prev_symbols") if c in df.columns]

    for _, row in df.iterrows():
        approved = normalize_symbol(row.get("approved_symbol", ""))
        if not approved:
            continue

        approved_set.add(approved)
        alias_dict[approved] = approved

        for col in alias_cols:
            raw = str(row.get(col, "") or "")
            if not raw or raw.lower() == "nan":
                continue
            for token in re.split(r"[|,;]", raw):
                alias = normalize_symbol(token)
                if alias:
                    alias_dict[alias] = approved

    return alias_dict, approved_set


def load_oncovar_local_gene_sets(
    brca_path: Path,
    pancancer_path: Path,
    alias_dict: dict[str, str],
    approved_set: set[str],
    log_lines: list[str],
) -> tuple[set[str], set[str]]:
    #load local OncoVar BRCA/PanCancer files and return approved symbol sets
    #oncovar
    try:
        df_brca = pd.read_csv(brca_path, sep="\t", compression="gzip", dtype=str)
    except FileNotFoundError as exc:
        raise ScriptError(
            f"OncoVar BRCA file not found: {brca_path}. "
            "Expected local file at data/references/TCGA.BRCA.onco.genes.OncoVar.tsv.gz"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(f"Failed to read OncoVar BRCA file: {brca_path}. Reason: {exc}") from exc

    try:
        df_pan = pd.read_csv(
            pancancer_path,
            sep="\t",
            compression="gzip",
            dtype=str,
        )
    except FileNotFoundError as exc:
        raise ScriptError(
            f"OncoVar PanCancer file not found: {pancancer_path}. "
            "Expected local file at data/references/TCGA.PanCancer.onco.genes.OncoVar.tsv.gz"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ScriptError(
            f"Failed to read OncoVar PanCancer file: {pancancer_path}. Reason: {exc}"
        ) from exc

    print(f"[OncoVar BRCA] columns: {list(df_brca.columns)}")
    print(f"[OncoVar PanCancer] columns: {list(df_pan.columns)}")
    log_lines.append(f"OncoVar BRCA columns: {list(df_brca.columns)}")
    log_lines.append(f"OncoVar PanCancer columns: {list(df_pan.columns)}")

    def extract_symbol_set(df_src: pd.DataFrame, dataset_name: str) -> set[str]:
        symbol_col = None
        normalized_cols = {
            str(c).strip().lower().replace(" ", "_"): c
            for c in df_src.columns
        }
        for candidate in (
            "gene",
            "gene_symbol",
            "symbol",
            "hugo_symbol",
            "gene_name",
        ):
            if candidate in normalized_cols:
                symbol_col = normalized_cols[candidate]
                break

        if symbol_col is None:
            raise ScriptError(
                f"OncoVar {dataset_name} is missing a recognizable gene symbol column. "
                f"Found columns: {list(df_src.columns)}"
            )

        resolved: set[str] = set()
        for raw in df_src[symbol_col].tolist():
            resolved_symbol = resolve_hgnc_symbol(raw, alias_dict, approved_set)
            if resolved_symbol:
                resolved.add(resolved_symbol)
        return resolved

    oncovar_brca_set = extract_symbol_set(df_brca, "BRCA")
    oncovar_pan_set = extract_symbol_set(df_pan, "PanCancer")

    logging.info("OncoVar BRCA genes loaded: %d", len(oncovar_brca_set))
    logging.info("OncoVar PanCancer genes loaded: %d", len(oncovar_pan_set))
    log_lines.append(f"OncoVar BRCA genes loaded: {len(oncovar_brca_set)}")
    log_lines.append(f"OncoVar PanCancer genes loaded: {len(oncovar_pan_set)}")

    return oncovar_brca_set, oncovar_pan_set


def download_ncg_reference(ncg_path: Path) -> None:
    #download NCG gene table and cache locally
    #the legacy direct URL now redirects to HTML; use form download endpoint.
    response = requests.post(
        NCG_DOWNLOAD_PAGE_URL,
        data={"downloadcancergenes": "Download"},
        timeout=120,
    )
    if response.status_code != 200 or "entrez" not in response.text.lower():
        fallback = requests.get(NCG_URL, timeout=120)
        if fallback.status_code != 200:
            raise ScriptError(
                "Failed to download NCG reference from known endpoints. "
                f"POST {NCG_DOWNLOAD_PAGE_URL} status={response.status_code}; "
                f"GET {NCG_URL} status={fallback.status_code}."
            )
        response = fallback

    ensure_parent_dir(ncg_path)
    ncg_path.write_text(response.text, encoding="utf-8")


def load_ncg_gene_set(
    ncg_path: Path,
    alias_dict: dict[str, str],
    approved_set: set[str],
    log_lines: list[str],
) -> tuple[set[str], dict[str, str]]:
    #load NCG symbols and optional role labels
    #ncg
    if not ncg_path.exists():
        try:
            download_ncg_reference(ncg_path)
            log_lines.append(f"Downloaded NCG reference to: {ncg_path}")
        except Exception as exc:  # noqa: BLE001
            raise ScriptError(
                "NCG file not found and download failed. "
                f"Expected local file: {ncg_path}. Reason: {exc}"
            ) from exc

    def _read_ncg_table(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t", dtype=str)

    try:
        df_ncg = _read_ncg_table(ncg_path)
    except FileNotFoundError as exc:
        raise ScriptError(f"NCG file not found: {ncg_path}") from exc
    except Exception as exc:  # noqa: BLE001
        raw_preview = ncg_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[:1024].lower()
        if "<html" in raw_preview or "<!doctype html" in raw_preview:
            download_ncg_reference(ncg_path)
            log_lines.append(
                "Refreshed NCG cache after detecting cached HTML page"
            )
            try:
                df_ncg = _read_ncg_table(ncg_path)
            except Exception as retry_exc:  # noqa: BLE001
                raise ScriptError(
                    "NCG cache was HTML and refresh failed to produce a valid "
                    f"TSV. File: {ncg_path}. Reason: {retry_exc}"
                ) from retry_exc
        else:
            raise ScriptError(
                f"Failed to read NCG reference file: {ncg_path}. Reason: {exc}"
            ) from exc

    print(f"[NCG] columns: {list(df_ncg.columns)}")
    log_lines.append(f"NCG columns: {list(df_ncg.columns)}")

    normalized_cols = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df_ncg.columns
    }

    symbol_col = None
    for candidate in ("symbol", "gene_symbol", "hugo_symbol", "gene"):
        if candidate in normalized_cols:
            symbol_col = normalized_cols[candidate]
            break
    if symbol_col is None:
        raise ScriptError(
            "NCG file is missing a recognizable gene symbol column. "
            f"Found columns: {list(df_ncg.columns)}"
        )

    role_col = None
    for candidate in ("type", "role", "cancer_type"):
        if candidate in normalized_cols:
            role_col = normalized_cols[candidate]
            break

    ncg_gene_set: set[str] = set()
    ncg_role_data: dict[str, str] = {}
    for _, row in df_ncg.iterrows():
        resolved_symbol = resolve_hgnc_symbol(
            row.get(symbol_col, ""),
            alias_dict,
            approved_set,
        )
        if not resolved_symbol:
            continue

        ncg_gene_set.add(resolved_symbol)
        raw_role = ""
        if role_col is not None:
            raw_role = str(row.get(role_col, "") or "").strip()
        role_final = raw_role if raw_role else "cancer_gene"
        if (
            resolved_symbol not in ncg_role_data
            or ncg_role_data[resolved_symbol] == "cancer_gene"
        ):
            ncg_role_data[resolved_symbol] = role_final

    logging.info("NCG genes loaded: %d", len(ncg_gene_set))
    log_lines.append(f"NCG genes loaded: {len(ncg_gene_set)}")
    return ncg_gene_set, ncg_role_data


def download_omim_genemap2(omim_path: Path, api_key: str) -> None:
    #download OMIM genemap2 using OMIM API key
    url = OMIM_GENEMAP2_URL_TEMPLATE.format(api_key=api_key)
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise ScriptError(
            "Failed to download OMIM genemap2.txt "
            f"(status={response.status_code})."
        )
    ensure_parent_dir(omim_path)
    omim_path.write_text(response.text, encoding="utf-8")


def load_omim_cancer_mim_set(
    omim_path: Path,
    log_lines: list[str],
) -> set[str]:
    #parse OMIM genemap2 and collect cancer phenotype MIM IDs
    #omim
    lines = omim_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("#") and "Phenotypes" in line and "Mim Number" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ScriptError(
            "OMIM genemap2 header not found (expected commented header "
            "with Mim Number and Phenotypes columns)."
        )

    header_cols = [
        col.strip()
        for col in lines[header_idx].lstrip("#").strip().split("\t")
    ]
    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < len(header_cols):
            parts.extend([""] * (len(header_cols) - len(parts)))
        rows.append(parts[: len(header_cols)])

    df_omim = pd.DataFrame(rows, columns=header_cols)
    print(f"[OMIM] columns: {list(df_omim.columns)}")
    log_lines.append(f"OMIM genemap2 columns: {list(df_omim.columns)}")

    normalized_cols = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df_omim.columns
    }
    mim_col = normalized_cols.get("mim_number")
    phen_col = normalized_cols.get("phenotypes")
    if mim_col is None or phen_col is None:
        raise ScriptError(
            "OMIM genemap2 is missing required columns Mim Number and/or "
            f"Phenotypes. Found columns: {list(df_omim.columns)}"
        )

    pattern = re.compile(
        "|".join(OMIM_CANCER_KEYWORDS),
        flags=re.IGNORECASE,
    )
    cancer_df = df_omim[
        df_omim[phen_col].fillna("").astype(str).str.contains(pattern, na=False)
    ]

    omim_cancer_mim_set: set[str] = set()
    for raw in cancer_df[mim_col].fillna("").astype(str).tolist():
        mim = re.sub(r"\D", "", raw)
        if mim:
            omim_cancer_mim_set.add(mim)

    logging.info(
        "OMIM cancer phenotype MIM IDs loaded: %d",
        len(omim_cancer_mim_set),
    )
    log_lines.append(
        "OMIM cancer phenotype MIM IDs loaded: "
        f"{len(omim_cancer_mim_set)}"
    )
    return omim_cancer_mim_set


def map_cancermine_role(role_raw: str) -> str:
    #map CancerMine role labels to script role labels
    role = str(role_raw).strip().lower()
    if role == "oncogene":
        return "oncogene"
    if role == "tumor_suppressor":
        return "TSG"
    if role == "driver":
        return "driver"
    return "other"


def load_cancermine(
    cancermine_cache: Path,
    cache_max_age: int,
    min_citations: int,
    log_lines: list[str],
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, int]]:
    #load CancerMine collated table with citation and role aggregation
    print("[CancerMine] Loading...")
    if cache_is_fresh(cancermine_cache, cache_max_age):
        log_lines.append("Using cached CancerMine file")
    else:
        print("[CancerMine] Downloading...")
        try:
            response = requests.get(CANCERMINE_URL, timeout=120)
            if response.status_code != 200:
                raise RuntimeError(f"status={response.status_code}")
            ensure_parent_dir(cancermine_cache)
            cancermine_cache.write_text(response.text, encoding="utf-8")
            log_lines.append("Downloaded CancerMine collated file")
        except Exception as exc:  # noqa: BLE001
            log_lines.append(
                "WARNING: CancerMine download failed. Please manually "
                "download from http://bionlp.bcgsc.ca/cancermine/ and save "
                f"to {cancermine_cache}. Reason: {exc}"
            )
            return set(), {}, {"total": 0, "breast": 0}

    df = pd.read_csv(cancermine_cache, sep="\t", dtype=str)
    log_lines.append(f"CancerMine columns: {list(df.columns)}")

    required = [
        "gene_normalized",
        "cancer_normalized",
        "role",
        "citation_count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ScriptError(
            f"CancerMine TSV missing required columns: {missing}"
        )

    work = df.copy()
    work["citation_count"] = pd.to_numeric(
        work["citation_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    work = work[work["citation_count"] >= int(min_citations)].copy()
    if work.empty:
        log_lines.append(
            "CancerMine: 0 genes after citation filter "
            f"(>={min_citations})"
        )
        log_lines.append("CancerMine: 0 breast-specific genes")
        return set(), {}, {"total": 0, "breast": 0}

    work["symbol_norm"] = work["gene_normalized"].map(normalize_symbol)
    work = work[work["symbol_norm"] != ""].copy()
    work["role_mapped"] = work["role"].map(map_cancermine_role)
    work["is_breast"] = (
        work["cancer_normalized"].fillna("").str.contains(
            "breast",
            case=False,
            regex=True,
        )
    )

    cancermine_data: dict[str, dict[str, Any]] = {}
    for symbol, sub in work.groupby("symbol_norm"):
        symbol_key = normalize_symbol(symbol)
        if not symbol_key:
            continue
        sub = sub.copy()
        breast_sub = sub[sub["is_breast"]]
        use = breast_sub if not breast_sub.empty else sub
        roles = set(use["role_mapped"].dropna().tolist())

        if "oncogene" in roles and "TSG" in roles:
            role_final = "both"
        elif roles == {"driver"}:
            role_final = "driver"
        else:
            top = use.sort_values(
                ["citation_count", "role_mapped"],
                ascending=[False, True],
            ).iloc[0]
            role_final = str(top["role_mapped"])

        max_citations = (
            int(use["citation_count"].max()) if not use.empty else 0
        )
        cancermine_data[symbol_key] = {
            "cancermine_role": role_final,
            "cancermine_breast_specific": 1 if not breast_sub.empty else 0,
            "cancermine_citation_count": max_citations,
        }

    cancermine_genes = set(cancermine_data.keys())
    cancermine_breast = sum(
        1
        for v in cancermine_data.values()
        if v.get("cancermine_breast_specific", 0) == 1
    )
    log_lines.append(
        "CancerMine: "
        f"{len(cancermine_genes)} genes after citation filter "
        f"(>={min_citations})"
    )
    log_lines.append(
        f"CancerMine: {cancermine_breast} breast-specific genes"
    )
    return cancermine_genes, cancermine_data, {
        "total": len(cancermine_genes),
        "breast": cancermine_breast,
    }


def extract_uniprot_next_link(link_header: str) -> str | None:
    #extract next-page URL from UniProt Link header
    if not link_header:
        return None
    match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
    if match:
        return match.group(1)
    return None


def fetch_uniprot_keyword_symbols(keyword_id: str, label: str) -> set[str]:
    #fetch UniProt reviewed human symbols for one cancer keyword
    print(f"[UniProt] Fetching {label}...")
    symbols: set[str] = set()
    params = {
        "query": (
            f"keyword:{keyword_id} AND organism_id:9606 AND reviewed:true"
        ),
        "fields": "gene_names,keyword",
        "format": "tsv",
        "size": 500,
    }

    next_url: str | None = UNIPROT_SEARCH_URL
    first = True
    while next_url is not None:
        response = requests.get(
            next_url,
            params=params if first else None,
            timeout=90,
        )
        first = False
        if response.status_code != 200:
            raise ScriptError(
                f"UniProt request failed for {label}: {response.status_code}"
            )

        lines = response.text.splitlines()
        if lines:
            for line in lines[1:]:
                parts = line.split("\t")
                if not parts:
                    continue
                gene_names = parts[0].strip() if len(parts) >= 1 else ""
                if not gene_names:
                    continue
                primary = gene_names.split()[0].strip()
                symbol = normalize_symbol(primary)
                if symbol:
                    symbols.add(symbol)

        next_url = extract_uniprot_next_link(response.headers.get("Link", ""))

    return symbols


def load_uniprot(
    uniprot_cache: Path,
    cache_max_age: int,
    log_lines: list[str],
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    #load UniProt oncogene/TSG keyword sets from cache or API
    print("[UniProt] Loading...")
    if cache_is_fresh(uniprot_cache, cache_max_age):
        log_lines.append("Using cached UniProt data")
        payload = json.loads(uniprot_cache.read_text(encoding="utf-8"))
        oncogenes = {
            normalize_symbol(x)
            for x in payload.get("oncogenes", [])
            if normalize_symbol(x)
        }
        tsgs = {
            normalize_symbol(x)
            for x in payload.get("tsgs", [])
            if normalize_symbol(x)
        }
    else:
        oncogenes = fetch_uniprot_keyword_symbols(
            UNIPROT_ONCOGENE_KW,
            "proto-oncogenes",
        )
        tsgs = fetch_uniprot_keyword_symbols(
            UNIPROT_TSG_KW,
            "tumour suppressors",
        )
        payload = {
            "fetched_at": now_iso(),
            "oncogenes": sorted(oncogenes),
            "tsgs": sorted(tsgs),
        }
        ensure_parent_dir(uniprot_cache)
        uniprot_cache.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    log_lines.append(
        "UniProt proto-oncogenes (KW-0656): "
        f"{len(oncogenes)} human reviewed"
    )
    log_lines.append(
        "UniProt tumour suppressors (KW-0043): "
        f"{len(tsgs)} human reviewed"
    )

    combined = sorted(oncogenes | tsgs)
    data: dict[str, dict[str, Any]] = {}
    for symbol in combined:
        is_onco = 1 if symbol in oncogenes else 0
        is_tsg = 1 if symbol in tsgs else 0
        role = ""
        if is_onco and is_tsg:
            role = "both"
        elif is_onco:
            role = "oncogene"
        elif is_tsg:
            role = "TSG"
        data[symbol] = {
            "uniprot_oncogene": is_onco,
            "uniprot_tsg": is_tsg,
            "uniprot_role": role,
        }

    return oncogenes, tsgs, data


def pct(count: int, total: int) -> str:
    #format percent with one decimal place
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * count / total):.1f}%"


def build_summary_table(rows: list[tuple[str, int, str]]) -> str:
    #build all-sources boxed summary table with count and percentage
    col1 = "Source"
    col2 = "Genes"
    col3 = "% of network"

    w1 = max([len(col1)] + [len(r[0]) for r in rows])
    w2 = max([len(col2)] + [len(str(r[1])) for r in rows])
    w3 = max([len(col3)] + [len(r[2]) for r in rows])

    top = "┌" + "─" * (w1 + w2 + w3 + 10) + "┐"
    title_text = " Cancer Gene Annotation Summary -- All Sources "
    title = "│" + title_text.ljust(w1 + w2 + w3 + 10) + "│"
    sep_header = (
        "├" + "─" * (w1 + 2) + "┬" + "─" * (w2 + 2) + "┬"
        + "─" * (w3 + 2) + "┤"
    )
    sep_rows = (
        "├" + "─" * (w1 + 2) + "┼" + "─" * (w2 + 2) + "┼"
        + "─" * (w3 + 2) + "┤"
    )
    header = (
        f"│ {col1.ljust(w1)} │ {col2.ljust(w2)} │ {col3.ljust(w3)} │"
    )

    out = [top, title, sep_header, header, sep_rows]
    for src, n, p in rows:
        out.append(f"│ {src.ljust(w1)} │ {str(n).ljust(w2)} │ {p.ljust(w3)} │")

    out.append(sep_rows)
    bot = (
        "└" + "─" * (w1 + 2) + "┴" + "─" * (w2 + 2)
        + "┴" + "─" * (w3 + 2) + "┘"
    )
    out.append(bot)
    return "\n".join(out)


def main() -> int:
    #run cancer evidence annotation workflow
    args = parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    log_path = resolve_path(args.log)
    cosmic_cache = resolve_path(args.cosmic_cache)
    oncokb_cache = resolve_path(args.oncokb_cache)
    ot_cache = resolve_path(args.ot_cache)
    intogen_cache = resolve_path(args.intogen_cache)
    cancermine_cache = resolve_path(args.cancermine_cache)
    uniprot_cache = resolve_path(args.uniprot_cache)

    if not input_path.exists():
        raise ScriptError(f"Input CSV does not exist: {input_path}")

    log_lines: list[str] = [
        f"Run timestamp: {now_iso()}",
        f"Input: {input_path}",
        f"Output: {output_path}",
        f"Log: {log_path}",
    ]

    df = pd.read_csv(input_path)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    required_cols = ["gene", "approved_symbol"]
    missing_input = [c for c in required_cols if c not in df.columns]
    if missing_input:
        raise ScriptError(
            f"Input CSV missing required columns: {missing_input}"
        )

    if df["approved_symbol"].isna().any() or df["gene"].isna().any():
        raise ScriptError("Input CSV has null gene/approved_symbol values.")

    alias_dict, approved_set = build_alias_and_approved_from_df(df)

    cosmic_genes: set[str] = set()
    cosmic_data: dict[str, dict[str, Any]] = {}
    oncokb_genes: set[str] = set()
    oncokb_data: dict[str, dict[str, Any]] = {}
    ot_data: dict[str, float] = {}
    intogen_genes: set[str] = set()
    intogen_data: dict[str, dict[str, Any]] = {}
    oncovar_brca_genes: set[str] = set()
    oncovar_pancancer_genes: set[str] = set()
    ncg_gene_set: set[str] = set()
    ncg_role_data: dict[str, str] = {}
    omim_cancer_mim_set: set[str] = set()
    cancermine_genes: set[str] = set()
    cancermine_data: dict[str, dict[str, Any]] = {}
    uniprot_oncogenes: set[str] = set()
    uniprot_tsgs: set[str] = set()
    uniprot_data: dict[str, dict[str, Any]] = {}

    intogen_stats = {"total": 0, "breast": 0}
    cancermine_stats = {"total": 0, "breast": 0}

    print("[COSMIC] Loading...")
    try:
        cosmic_genes, cosmic_data = load_cosmic(cosmic_cache, log_lines)
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: COSMIC unavailable: {exc}")
        print(f"[COSMIC] WARNING: {exc}")

    print("[OncoKB] Loading...")
    try:
        oncokb_genes, oncokb_data = load_oncokb(
            oncokb_cache,
            args.cache_max_age,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: OncoKB unavailable: {exc}")
        print(f"[OncoKB] WARNING: {exc}")

    print("[OpenTargets] Loading...")
    try:
        ot_data = load_open_targets(ot_cache, args.cache_max_age, log_lines)
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: Open Targets unavailable: {exc}")
        print(f"[OpenTargets] WARNING: {exc}")

    try:
        intogen_genes, intogen_data, intogen_stats = load_intogen(
            intogen_cache,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: IntOGen unavailable: {exc}")
        print(f"[IntOGen] WARNING: {exc}")

    #oncovar
    oncovar_brca_path = resolve_path(
        "data/references/TCGA.BRCA.onco.genes.OncoVar.tsv.gz"
    )
    oncovar_pancancer_path = resolve_path(
        "data/references/TCGA.PanCancer.onco.genes.OncoVar.tsv.gz"
    )
    try:
        (
            oncovar_brca_genes,
            oncovar_pancancer_genes,
        ) = load_oncovar_local_gene_sets(
            oncovar_brca_path,
            oncovar_pancancer_path,
            alias_dict,
            approved_set,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: OncoVar unavailable: {exc}")
        print(f"[OncoVar] WARNING: {exc}")

    #ncg
    ncg_path = resolve_path("data/references/NCG_cancergenes.tsv")
    try:
        ncg_gene_set, ncg_role_data = load_ncg_gene_set(
            ncg_path,
            alias_dict,
            approved_set,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: NCG unavailable: {exc}")
        print(f"[NCG] WARNING: {exc}")

    #omim
    omim_genemap_path = resolve_path("data/references/omim_genemap2.txt")
    try:
        omim_api_key = os.getenv("OMIM_API_KEY", "").strip()
        if not omim_genemap_path.exists():
            if not omim_api_key:
                raise ScriptError(
                    "OMIM_API_KEY is not set and OMIM cache file is missing. "
                    f"Expected: {omim_genemap_path}"
                )
            download_omim_genemap2(omim_genemap_path, omim_api_key)
            log_lines.append(
                "Downloaded OMIM genemap2 to: "
                f"{omim_genemap_path}"
            )
        omim_cancer_mim_set = load_omim_cancer_mim_set(
            omim_genemap_path,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: OMIM unavailable: {exc}")
        print(f"[OMIM] WARNING: {exc}")
        omim_cancer_mim_set = set()

    try:
        (
            cancermine_genes,
            cancermine_data,
            cancermine_stats,
        ) = load_cancermine(
            cancermine_cache,
            args.cache_max_age,
            args.cancermine_min_citations,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: CancerMine unavailable: {exc}")
        print(f"[CancerMine] WARNING: {exc}")

    try:
        (
            uniprot_oncogenes,
            uniprot_tsgs,
            uniprot_data,
        ) = load_uniprot(
            uniprot_cache,
            args.cache_max_age,
            log_lines,
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"WARNING: UniProt unavailable: {exc}")
        print(f"[UniProt] WARNING: {exc}")

    symbols_norm = df["approved_symbol"].map(normalize_symbol)

    annotation_rows: list[dict[str, Any]] = []
    tier1a_matches: set[str] = set()
    tier1b_matches: set[str] = set()
    tier1c_matches: set[str] = set()
    tier1d_matches: set[str] = set()
    tier2_matches: set[str] = set()
    tier3_matches: set[str] = set()
    tier3b_matches: set[str] = set()
    tier4_matches: set[str] = set()
    tier4b_matches: set[str] = set()
    tier5_matches: set[str] = set()
    breast_cosmic_matches: set[str] = set()
    intogen_network_matches: set[str] = set()
    cancermine_network_matches: set[str] = set()
    uniprot_network_matches: set[str] = set()

    known_before_count = 0
    omim_annotated_count = 0

    if "omim_ids" not in df.columns:
        df["omim_ids"] = ""

    for idx, symbol in enumerate(symbols_norm.tolist()):
        gene = normalize_symbol(df.iloc[idx].get("gene", ""))
        approved_sym = symbol if symbol else gene

        cosmic_match = approved_sym in cosmic_genes
        intogen_match = approved_sym in intogen_genes
        oncovar_brca_match = approved_sym in oncovar_brca_genes
        oncovar_pancancer_match = approved_sym in oncovar_pancancer_genes
        oncokb_match = approved_sym in oncokb_genes
        cancermine_match = approved_sym in cancermine_genes
        ncg_match = approved_sym in ncg_gene_set
        uniprot_onco = 1 if approved_sym in uniprot_oncogenes else 0
        uniprot_tsg = 1 if approved_sym in uniprot_tsgs else 0
        ot_score = float(ot_data.get(approved_sym, 0.0))
        ot_match = ot_score >= OPEN_TARGETS_THRESHOLD
        omim_text = str(df.iloc[idx].get("omim_ids", "") or "").strip()
        omim_is_nan = str(omim_text).lower() == "nan"
        omim_annotated = 1 if (omim_text and not omim_is_nan) else 0
        gene_omim_ids = set()
        if omim_annotated == 1:
            for token in re.split(r"[;|,\s]+", omim_text):
                tok = token.strip()
                if not tok or tok.lower() == "nan":
                    continue
                mim = re.sub(r"\D", "", tok)
                if mim:
                    gene_omim_ids.add(mim)
        omim_cancer_phenotype = (
            1 if any(mim in omim_cancer_mim_set for mim in gene_omim_ids)
            else 0
        )
        if omim_annotated == 1:
            omim_annotated_count += 1

        tier1 = 1 if cosmic_match else 0
        intogen_driver = 1 if intogen_match else 0
        #oncovar
        oncovar_brca_gene = 1 if oncovar_brca_match else 0
        oncovar_pancancer_gene = 1 if oncovar_pancancer_match else 0
        tier2 = 1 if oncokb_match else 0
        cancermine_flag = 1 if cancermine_match else 0
        ncg_gene = 1 if ncg_match else 0
        tier4_flag = 1 if (uniprot_onco == 1 or uniprot_tsg == 1) else 0
        tier3 = 1 if ot_match else 0
        known_before = 1 if (cosmic_match or oncokb_match) else 0
        known = 1 if (
            cosmic_match
            or intogen_match
            or oncovar_brca_match
            or oncovar_pancancer_match
            or oncokb_match
            or cancermine_match
            or ncg_match
        ) else 0
        if known_before == 1:
            known_before_count += 1

        evidence_tier = "Unannotated"
        if cosmic_match:
            evidence_tier = "Tier1a_COSMIC"
            tier1a_matches.add(approved_sym)
        elif intogen_match:
            evidence_tier = "Tier1b_IntOGen"
            tier1b_matches.add(approved_sym)
        elif oncovar_brca_match:
            evidence_tier = "Tier1c_OncoVar_BRCA"
            tier1c_matches.add(approved_sym)
        elif oncovar_pancancer_match and not oncovar_brca_match:
            evidence_tier = "Tier1d_OncoVar_PanCancer"
            tier1d_matches.add(approved_sym)
        elif oncokb_match:
            evidence_tier = "Tier2_OncoKB"
            tier2_matches.add(approved_sym)
        elif cancermine_match:
            evidence_tier = "Tier3_CancerMine"
            tier3_matches.add(approved_sym)
        elif ncg_match:
            evidence_tier = "Tier3b_NCG"
            tier3b_matches.add(approved_sym)
        elif tier4_flag == 1:
            evidence_tier = "Tier4_UniProt"
            tier4_matches.add(approved_sym)
        elif omim_cancer_phenotype == 1:
            evidence_tier = "Tier4b_OMIM"
            tier4b_matches.add(approved_sym)
        elif ot_match:
            evidence_tier = "Tier5_OpenTargets"
            tier5_matches.add(approved_sym)

        cosmic_role = ""
        cosmic_tier = None
        cosmic_somatic = ""
        cosmic_germline = ""
        cosmic_tumour_types = ""

        oncokb_is_oncogene: bool | None = None
        oncokb_is_tsg: bool | None = None
        intogen_role = ""
        intogen_breast_specific = 0
        intogen_qvalue: float | None = None
        oncovar_score: float | None = None
        oncovar_level = ""
        oncovar_role = ""
        oncovar_sources = ""
        ncg_role = ""
        cancermine_role = ""
        cancermine_breast_specific = 0
        cancermine_citation_count = 0
        uniprot_role = ""
        source_role_for_final = ""

        if cosmic_match:
            cdat = cosmic_data.get(approved_sym, {})
            cosmic_role = str(cdat.get("cosmic_role", ""))
            cosmic_tier = cdat.get("cosmic_tier")
            cosmic_somatic = str(cdat.get("somatic", ""))
            cosmic_germline = str(cdat.get("germline", ""))
            cosmic_tumour_types = str(cdat.get("tumour_types_somatic", ""))
            source_role_for_final = cosmic_role

            tt_s = str(cdat.get("tumour_types_somatic", "") or "")
            tt_g = str(cdat.get("tumour_types_germline", "") or "")
            if "breast" in tt_s.lower() or "breast" in tt_g.lower():
                breast_cosmic_matches.add(approved_sym)

        if intogen_match:
            idat = intogen_data.get(approved_sym, {})
            intogen_role = str(idat.get("intogen_role", ""))
            intogen_breast_specific = int(
                idat.get("intogen_breast_specific", 0)
            )
            qv = idat.get("intogen_qvalue", None)
            intogen_qvalue = (
                None if qv is None else safe_float(qv, default=0.0)
            )
            intogen_network_matches.add(approved_sym)
            if not source_role_for_final:
                source_role_for_final = intogen_role

        if oncovar_brca_match or oncovar_pancancer_match:
            #oncovar
            oncovar_level = (
                "BRCA"
                if oncovar_brca_match
                else "PanCancer"
            )
            oncovar_role = "driver"
            oncovar_sources = "OncoVar_local"
            if not source_role_for_final:
                source_role_for_final = oncovar_role

        if oncokb_match:
            odat = oncokb_data.get(approved_sym, {})
            oncokb_is_oncogene = bool(odat.get("oncokb_is_oncogene", False))
            oncokb_is_tsg = bool(odat.get("oncokb_is_tsg", False))
            if not source_role_for_final:
                source_role_for_final = str(odat.get("oncokb_role", ""))
        if cancermine_match:
            cmdat = cancermine_data.get(approved_sym, {})
            cancermine_role = str(cmdat.get("cancermine_role", ""))
            cancermine_breast_specific = int(
                cmdat.get("cancermine_breast_specific", 0)
            )
            cancermine_citation_count = int(
                cmdat.get("cancermine_citation_count", 0)
            )
            cancermine_network_matches.add(approved_sym)
            if not source_role_for_final:
                source_role_for_final = cancermine_role

        if ncg_match:
            ncg_role = str(ncg_role_data.get(approved_sym, "cancer_gene"))
            if not source_role_for_final:
                source_role_for_final = ncg_role

        if tier4_flag == 1:
            udat = uniprot_data.get(approved_sym, {})
            uniprot_role = str(udat.get("uniprot_role", ""))
            uniprot_network_matches.add(approved_sym)
            if not source_role_for_final:
                source_role_for_final = uniprot_role

        #keep hierarchy output stable.
        if evidence_tier == "Tier2_OncoKB":
            tier2_matches.add(approved_sym)
        if evidence_tier == "Tier3_CancerMine":
            tier3_matches.add(approved_sym)
        if evidence_tier == "Tier3b_NCG":
            tier3b_matches.add(approved_sym)
        if evidence_tier == "Tier4_UniProt":
            tier4_matches.add(approved_sym)
        if evidence_tier == "Tier4b_OMIM":
            tier4b_matches.add(approved_sym)
        if evidence_tier == "Tier5_OpenTargets":
            tier5_matches.add(approved_sym)

        cancer_role = cancer_role_from_source(source_role_for_final)

        annotation_rows.append(
            {
                "tier1_cosmic": tier1,
                "cosmic_tier": cosmic_tier,
                "cosmic_role": cosmic_role,
                "cosmic_somatic": cosmic_somatic,
                "cosmic_germline": cosmic_germline,
                "cosmic_tumour_types": cosmic_tumour_types,
                "intogen_driver": intogen_driver,
                "intogen_role": intogen_role,
                "intogen_breast_specific": intogen_breast_specific,
                "intogen_qvalue": intogen_qvalue,
                "oncovar_brca_gene": oncovar_brca_gene,
                "oncovar_pancancer_gene": oncovar_pancancer_gene,
                "oncovar_score": oncovar_score,
                "oncovar_level": oncovar_level,
                "oncovar_role": oncovar_role,
                "oncovar_sources": oncovar_sources,
                "tier2_oncokb": tier2,
                "oncokb_is_oncogene": oncokb_is_oncogene,
                "oncokb_is_tsg": oncokb_is_tsg,
                "cancermine_cancer_gene": cancermine_flag,
                "ncg_gene": ncg_gene,
                "ncg_role": ncg_role,
                "cancermine_role": cancermine_role,
                "cancermine_breast_specific": cancermine_breast_specific,
                "cancermine_citation_count": cancermine_citation_count,
                "omim_annotated": omim_annotated,
                "omim_cancer_phenotype": omim_cancer_phenotype,
                "uniprot_oncogene": uniprot_onco,
                "uniprot_tsg": uniprot_tsg,
                "uniprot_role": uniprot_role,
                "open_targets_score": ot_score,
                "tier3_open_targets": tier3,
                "known_cancer_gene": known,
                "cancer_role": cancer_role,
                "evidence_tier": evidence_tier,
            }
        )

    ann = pd.DataFrame(annotation_rows)
    out_df = pd.concat([df.reset_index(drop=True), ann], axis=1)

    identity_cols = [
        "gene",
        "approved_symbol",
        "hgnc_id",
        "entrez_id",
        "ensembl_gene_id",
        "alias_symbols",
        "prev_symbols",
        "normalisation_status",
    ]
    bio_cols = [
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
    ann_cols = [
        "tier1_cosmic",
        "cosmic_tier",
        "cosmic_role",
        "cosmic_somatic",
        "cosmic_germline",
        "cosmic_tumour_types",
        "intogen_driver",
        "intogen_role",
        "intogen_breast_specific",
        "intogen_qvalue",
        "oncovar_brca_gene",
        "oncovar_pancancer_gene",
        "tier2_oncokb",
        "oncokb_is_oncogene",
        "oncokb_is_tsg",
        "cancermine_cancer_gene",
        "ncg_gene",
        "ncg_role",
        "cancermine_role",
        "cancermine_breast_specific",
        "cancermine_citation_count",
        "omim_annotated",
        "omim_cancer_phenotype",
        "uniprot_oncogene",
        "uniprot_tsg",
        "uniprot_role",
        "open_targets_score",
        "tier3_open_targets",
        "known_cancer_gene",
        "cancer_role",
        "evidence_tier",
    ]

    expected_cols = identity_cols + bio_cols + ann_cols
    for col in expected_cols:
        if col not in out_df.columns:
            out_df[col] = ""

    out_df = out_df[expected_cols]

    #validation checks.
    if len(out_df) != len(df):
        raise ScriptError(
            "Validation failed: output row count differs from input."
        )

    if out_df["gene"].isna().any() or out_df["approved_symbol"].isna().any():
        raise ScriptError(
            "Validation failed: nulls present in gene/approved_symbol."
        )

    allowed_known = {0, 1}
    known_vals = set(
        out_df["known_cancer_gene"].dropna().astype(int).unique()
    )
    if not known_vals.issubset(allowed_known):
        raise ScriptError(
            "Validation failed: known_cancer_gene contains invalid values."
        )

    allowed_tiers = {
        "Tier1a_COSMIC",
        "Tier1b_IntOGen",
        "Tier1c_OncoVar_BRCA",
        "Tier1d_OncoVar_PanCancer",
        "Tier2_OncoKB",
        "Tier3_CancerMine",
        "Tier3b_NCG",
        "Tier4_UniProt",
        "Tier4b_OMIM",
        "Tier5_OpenTargets",
        "Unannotated",
    }
    if not set(out_df["evidence_tier"].dropna().astype(str).unique()).issubset(
        allowed_tiers
    ):
        raise ScriptError(
            "Validation failed: evidence_tier contains invalid values."
        )

    ensure_parent_dir(output_path)
    out_df.to_csv(output_path, index=False)

    total = len(out_df)
    n_tier1a = int((out_df["evidence_tier"] == "Tier1a_COSMIC").sum())
    n_tier1b = int((out_df["evidence_tier"] == "Tier1b_IntOGen").sum())
    n_tier1c = int((out_df["evidence_tier"] == "Tier1c_OncoVar_BRCA").sum())
    n_tier1d = int((out_df["evidence_tier"] == "Tier1d_OncoVar_PanCancer").sum())
    n_tier2 = int((out_df["evidence_tier"] == "Tier2_OncoKB").sum())
    n_tier3 = int((out_df["evidence_tier"] == "Tier3_CancerMine").sum())
    n_tier3b = int((out_df["evidence_tier"] == "Tier3b_NCG").sum())
    n_tier4 = int((out_df["evidence_tier"] == "Tier4_UniProt").sum())
    n_tier4b = int((out_df["evidence_tier"] == "Tier4b_OMIM").sum())
    n_tier5 = int((out_df["evidence_tier"] == "Tier5_OpenTargets").sum())
    n_un = int((out_df["evidence_tier"] == "Unannotated").sum())
    n_known = int((out_df["known_cancer_gene"] == 1).sum())
    n_known_before = int(known_before_count)
    n_increase = n_known - n_known_before
    p_increase = (
        (100.0 * n_increase / n_known_before) if n_known_before > 0 else 0.0
    )

    n_intogen_network = len(intogen_network_matches)
    n_oncovar_brca_flag = int((out_df["oncovar_brca_gene"] == 1).sum())
    n_oncovar_pancancer_flag = int((out_df["oncovar_pancancer_gene"] == 1).sum())
    n_ncg_flag = int((out_df["ncg_gene"] == 1).sum())
    n_omim_cancer_flag = int((out_df["omim_cancer_phenotype"] == 1).sum())
    n_cancermine_network = len(cancermine_network_matches)
    n_uniprot_network = len(uniprot_network_matches)

    n_onco = int((out_df["cancer_role"] == "oncogene").sum())
    n_tsg = int((out_df["cancer_role"] == "TSG").sum())
    n_fusion = int((out_df["cancer_role"] == "fusion").sum())
    n_both = int((out_df["cancer_role"] == "both").sum())

    log_lines.append(f"Network genes total: {total}")
    log_lines.append(
        f"Tier 1a COSMIC matches: {n_tier1a} ({pct(n_tier1a, total)})"
    )
    log_lines.append(
        "Tier 1a matched symbols: " + ", ".join(sorted(tier1a_matches))
    )
    log_lines.append(
        f"Tier 1b IntOGen matches: {n_tier1b} ({pct(n_tier1b, total)})"
    )
    log_lines.append(
        "Tier 1b matched symbols: " + ", ".join(sorted(tier1b_matches))
    )
    log_lines.append(
        f"Tier 1c OncoVar BRCA matches: {n_tier1c} ({pct(n_tier1c, total)})"
    )
    log_lines.append(
        f"Tier 1c matched symbols: {', '.join(sorted(tier1c_matches))}"
    )
    log_lines.append(
        "Tier 1d OncoVar PanCancer matches: "
        f"{n_tier1d} ({pct(n_tier1d, total)})"
    )
    log_lines.append(
        f"Tier 1d matched symbols: {', '.join(sorted(tier1d_matches))}"
    )
    log_lines.append(
        f"Tier 2 OncoKB matches: {n_tier2} ({pct(n_tier2, total)})"
    )
    log_lines.append(
        "Tier 2 matched symbols: " + ", ".join(sorted(tier2_matches))
    )
    log_lines.append(
        f"Tier 3 CancerMine matches: {n_tier3} ({pct(n_tier3, total)})"
    )
    log_lines.append(
        f"Tier 3b NCG matches: {n_tier3b} ({pct(n_tier3b, total)})"
    )
    log_lines.append(
        f"Tier 4 UniProt matches: {n_tier4} ({pct(n_tier4, total)})"
    )
    log_lines.append(
        f"Tier 4b OMIM matches: {n_tier4b} ({pct(n_tier4b, total)})"
    )
    log_lines.append(
        f"Tier 5 Open Targets matches: {n_tier5} ({pct(n_tier5, total)})"
    )
    log_lines.append(f"Unannotated: {n_un} ({pct(n_un, total)})")
    log_lines.append(
        "IntOGen: loaded total="
        f"{intogen_stats['total']}, breast-specific={intogen_stats['breast']}"
    )
    log_lines.append(
        "IntOGen network matches: "
        f"{n_intogen_network} ({pct(n_intogen_network, total)})"
    )
    log_lines.append(
        "OncoVar BRCA flag count: "
        f"{n_oncovar_brca_flag} ({pct(n_oncovar_brca_flag, total)})"
    )
    log_lines.append(
        "OncoVar PanCancer flag count: "
        f"{n_oncovar_pancancer_flag} ({pct(n_oncovar_pancancer_flag, total)})"
    )
    log_lines.append(
        f"NCG flag count: {n_ncg_flag} ({pct(n_ncg_flag, total)})"
    )
    log_lines.append(
        "OMIM cancer phenotype flag count: "
        f"{n_omim_cancer_flag} ({pct(n_omim_cancer_flag, total)})"
    )
    log_lines.append(
        "CancerMine: loaded total="
        f"{cancermine_stats['total']}, "
        f"breast-specific={cancermine_stats['breast']}"
    )
    log_lines.append(
        "CancerMine network matches: "
        f"{n_cancermine_network} ({pct(n_cancermine_network, total)})"
    )
    log_lines.append(
        f"Genes with any OMIM annotation: {omim_annotated_count}"
    )
    log_lines.append(
        "UniProt loaded oncogenes="
        f"{len(uniprot_oncogenes)}, TSGs={len(uniprot_tsgs)}"
    )
    log_lines.append(
        "UniProt network matches: "
        f"{n_uniprot_network} ({pct(n_uniprot_network, total)})"
    )
    log_lines.append(
        f"known_cancer_gene = 1 total: {n_known} ({pct(n_known, total)})"
    )
    log_lines.append(
        "known_cancer_gene before extension (COSMIC+OncoKB only): "
        f"{n_known_before} ({pct(n_known_before, total)})"
    )
    log_lines.append(
        "known_cancer_gene after extension (all sources): "
        f"{n_known} ({pct(n_known, total)})"
    )
    log_lines.append(
        f"increase: {n_increase} ({p_increase:.1f}%)"
    )
    log_lines.append(f"Oncogenes: {n_onco}")
    log_lines.append(f"TSGs: {n_tsg}")
    log_lines.append(f"Fusion genes: {n_fusion}")
    log_lines.append(f"Both oncogene and TSG: {n_both}")
    log_lines.append(
        "Genes in network AND in COSMIC tumour types containing "
        f"'breast': {len(breast_cosmic_matches)}"
    )

    ensure_parent_dir(log_path)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    summary_rows = [
        ("Tier 1a COSMIC", n_tier1a, pct(n_tier1a, total)),
        ("Tier 1b IntOGen only", n_tier1b, pct(n_tier1b, total)),
        ("Tier 1c OncoVar BRCA only", n_tier1c, pct(n_tier1c, total)),
        (
            "Tier 1d OncoVar PanCancer only",
            n_tier1d,
            pct(n_tier1d, total),
        ),
        ("Tier 2  OncoKB only", n_tier2, pct(n_tier2, total)),
        ("Tier 3  CancerMine only", n_tier3, pct(n_tier3, total)),
        ("Tier 3b NCG only", n_tier3b, pct(n_tier3b, total)),
        ("Tier 4  UniProt only", n_tier4, pct(n_tier4, total)),
        ("Tier 4b OMIM only", n_tier4b, pct(n_tier4b, total)),
        ("Tier 5  Open Targets only", n_tier5, pct(n_tier5, total)),
        ("Unannotated", n_un, pct(n_un, total)),
        (
            "known_cancer_gene (all tiers)",
            n_known,
            pct(n_known, total),
        ),
    ]
    print(build_summary_table(summary_rows))
    print(
        "known_cancer_gene before extension: "
        f"{n_known_before} ({pct(n_known_before, total)})"
    )
    print(
        "known_cancer_gene after extension: "
        f"{n_known} ({pct(n_known, total)})"
    )
    print(f"increase: {n_increase} ({p_increase:.1f}%)")
    print(f"Wrote output CSV: {output_path}")
    print(f"Wrote log file: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScriptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)
