"""Build drug-target candidate tables from DGIdb, ChEMBL/RxNorm synonym expansion, and composite scoring."""


from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils.network_utils import resolve_base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NETWORKS_ROOT = str(REPO_ROOT / "results/20_node_annotation")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "results/24_drug_targets")
DEFAULT_SURVIVAL_PATH = str(REPO_ROOT / "results/23_survival/top_prognostic_genes.csv")
DEFAULT_CROSS_NETWORK_PATH = str(REPO_ROOT / "results/27_cross_network/recurring_genes_all_tiers.csv")

#dGIdb concept IDs with these prefixes are pharmacological compounds
CHEMBL_CONCEPT_PREFIX = "chembl:"

DGIDB_GRAPHQL_URL = "https://dgidb.org/api/graphql"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
RXNORM_DRUGS_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"

DGIDB_INTERACTIONS_QUERY = """
query($geneNames:[String!], $first:Int, $after:String) {
  interactions(geneNames:$geneNames, first:$first, after:$after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      interactionScore
      evidenceScore
      interactionTypes { type }
      sources { sourceDbName }
      gene { name }
      drug {
        name
        approved
        antiNeoplastic
        immunotherapy
        conceptId
        drugAliases { alias }
      }
    }
  }
}
""".strip()

CHEMBL_ID_RE = re.compile(r"\b(CHEMBL\d+)\b", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


@dataclass
class PairInputFiles:
    #resolved input files for one case-vs-control pair

    pair_name: str
    edges_csv: Path
    tagged_csv: Path


def parse_args() -> argparse.Namespace:
    #parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Prioritize drug candidates targeting persistent-network hubs"
    )
    parser.add_argument(
        "--networks-root",
        default=DEFAULT_NETWORKS_ROOT,
        help="Root folder containing pair subfolders with viz_inputs/*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for candidate tables",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=50,
        help="Top hubs per persistent network and in combined ranking",
    )
    parser.add_argument(
        "--dgidb-page-size",
        type=int,
        default=250,
        help="DGIdb interactions page size for each hub gene",
    )
    parser.add_argument(
        "--dgidb-max-pages",
        type=int,
        default=30,
        help="Safety cap for DGIdb pagination per gene",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=45,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.08,
        help="Sleep between API calls to stay polite with external services",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/reference/drug_synonym_cache",
        help="Directory for ChEMBL/RxNorm cache files",
    )
    parser.add_argument(
        "--approved-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only DGIdb interactions with approved drugs",
    )
    parser.add_argument(
        "--anti-neoplastic-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only drugs marked anti-neoplastic in DGIdb",
    )
    parser.add_argument(
        "--require-chembl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only drugs whose DGIdb conceptId starts with 'chembl:'. "
             "NOTE: DGIdb stores most approved drugs under NCIT concept IDs, so enabling this "
             "will discard the majority of clinical candidates. Use only for strict small-molecule filtering.",
    )
    parser.add_argument(
        "--survival-path",
        default=DEFAULT_SURVIVAL_PATH,
        help="Path to top_prognostic_genes.csv from Script 62 (FDR<0.1 hub genes); used to add survival signal to ranking",
    )
    parser.add_argument(
        "--cross-network-path",
        default=DEFAULT_CROSS_NETWORK_PATH,
        help="Path to recurring_genes_all_tiers.csv from Script 40; restricts drug query to cross-network recurring genes (D+S_case, 2+ subtypes)",
    )
    return parser.parse_args()


def normalise_gene_key(text: Any) -> str:
    #upper-case stable key for gene joins
    if text is None:
        return ""
    return str(text).strip().upper()


def normalise_drug_alias(text: Any) -> str:
    #normalize drug synonym strings for cross-source matching
    if text is None:
        return ""
    s = str(text).strip().upper()
    if not s:
        return ""
    s = NON_ALNUM_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def ensure_dir(path: Path) -> None:
    #create directory tree if missing
    path.mkdir(parents=True, exist_ok=True)


PAIR_SHORT = {
    "ER_tumor__vs__Normal": "ER",
    "HER2_tumor__vs__Normal": "HER2",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": "NormalBRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal": "TNBC_BRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": "TNBC_BRCA1_vs_NormalBRCA1",
    "Triple_negative_tumor__vs__Normal": "TNBC",
}
DRUG_TIERS = ["D", "S_case"]  # tiers eligible for drug-target integration
CSD_NETWORKS_DIR = REPO_ROOT / "results/14_csd_networks"
LFC_DIR = REPO_ROOT / "results/20_node_annotation/03_output_with_lfc"


def discover_pair_inputs(networks_root: Path) -> list[PairInputFiles]:
    #find pair directories with hub CSVs for D and S_case tiers
    node_annot_root = REPO_ROOT / "results/20_node_annotation"
    out: list[PairInputFiles] = []
    for pair_name, short in PAIR_SHORT.items():
        pair_dir = node_annot_root / pair_name
        if not pair_dir.exists():
            logging.warning("Skipping %s (pair dir missing)", pair_name)
            continue
        tagged_csv = LFC_DIR / f"{pair_name}_tagged_with_lfc.csv"
        if not tagged_csv.exists():
            logging.warning("Skipping %s (tagged CSV missing)", pair_name)
            continue
        #use differential edges from CSD network for metadata (edges not used for hub ranking)
        edge_csv = CSD_NETWORKS_DIR / pair_name / f"{pair_name}_differential_edges_permutation.csv"
        if not edge_csv.exists():
            logging.warning("Skipping %s (edge CSV missing)", pair_name)
            continue
        out.append(PairInputFiles(pair_name=pair_name, edges_csv=edge_csv, tagged_csv=tagged_csv))
    if not out:
        raise RuntimeError("No valid pair folders found")
    return out


def compute_weighted_degrees(edges_df: pd.DataFrame) -> pd.DataFrame:
    #compute weighted degree per gene from persistent edge table
    required = {"gene_a", "gene_b", "weight"}
    if not required.issubset(edges_df.columns):
        raise ValueError(f"Edge CSV missing required columns: {required}")

    deg: dict[str, float] = defaultdict(float)
    work = edges_df.copy()
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)
    work = work[(work["gene_a"].notna()) & (work["gene_b"].notna())].copy()

    for row in work[["gene_a", "gene_b", "weight"]].itertuples(index=False):
        a = str(row.gene_a).strip()
        b = str(row.gene_b).strip()
        w = float(row.weight)
        if not a or not b:
            continue
        deg[a] += w
        deg[b] += w

    if not deg:
        return pd.DataFrame(columns=["gene", "weighted_degree", "degree_rank"])

    out = pd.DataFrame(
        {
            "gene": list(deg.keys()),
            "weighted_degree": list(deg.values()),
        }
    )
    out = out.sort_values("weighted_degree", ascending=False).reset_index(drop=True)
    out["degree_rank"] = out.index + 1
    return out


def build_gene_metadata_index(tagged_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    #index tagged metadata by both original and approved symbol
    index: dict[str, dict[str, Any]] = {}
    for row in tagged_df.to_dict(orient="records"):
        row_dict = {str(k): v for k, v in row.items()}
        key_gene = normalise_gene_key(row_dict.get("gene"))
        key_appr = normalise_gene_key(row_dict.get("approved_symbol"))
        if key_gene and key_gene not in index:
            index[key_gene] = row_dict
        if key_appr and key_appr not in index:
            index[key_appr] = row_dict
    return index


def attach_metadata(
    pair_name: str,
    top_hubs_df: pd.DataFrame,
    tagged_df: pd.DataFrame,
) -> pd.DataFrame:
    #attach cancer and annotation metadata to pair hub table
    if top_hubs_df.empty:
        return pd.DataFrame()

    meta_idx = build_gene_metadata_index(tagged_df)
    rows: list[dict[str, Any]] = []
    for row in top_hubs_df.to_dict(orient="records"):
        gene = str(row.get("gene", "")).strip()
        meta = meta_idx.get(normalise_gene_key(gene), {})
        out = {
            "pair_name": pair_name,
            "gene": gene,
            "weighted_degree": float(row.get("weighted_degree", 0.0)),
            "degree_rank": int(row.get("degree_rank", 0)),
            "approved_symbol": meta.get("approved_symbol", ""),
            "ensembl_gene_id": meta.get("ensembl_gene_id", ""),
            "entrez_id": meta.get("entrez_id", ""),
            "known_cancer_gene": meta.get("known_cancer_gene", ""),
            "cancer_role": meta.get("cancer_role", ""),
            "evidence_tier": meta.get("evidence_tier", ""),
            "direction": meta.get("direction", ""),
            "lfc": meta.get("lfc", ""),
            "cell_type": meta.get("cell_type", ""),
            "cell_type_ref": meta.get("cell_type_ref", ""),
        }
        rows.append(out)
    return pd.DataFrame(rows)


def aggregate_combined_hubs(pair_hubs_df: pd.DataFrame, top_hubs: int) -> pd.DataFrame:
    #aggregate per-pair hub ranks into one combined top-hub table
    if pair_hubs_df.empty:
        return pd.DataFrame()

    work = pair_hubs_df.copy()
    work["canonical_gene"] = work["approved_symbol"].astype(str).str.strip()
    work.loc[work["canonical_gene"] == "", "canonical_gene"] = work["gene"]
    work["canonical_gene"] = work["canonical_gene"].astype(str)

    grouped = work.groupby("canonical_gene", dropna=False)
    rows: list[dict[str, Any]] = []
    for canonical_gene, gdf in grouped:
        pairs = sorted(set(gdf["pair_name"].astype(str).tolist()))
        rows.append(
            {
                "canonical_gene": canonical_gene,
                "n_pairs_present": int(len(pairs)),
                "pair_list": "|".join(pairs),
                "mean_weighted_degree": float(pd.to_numeric(gdf["weighted_degree"], errors="coerce").mean()),
                "sum_weighted_degree": float(pd.to_numeric(gdf["weighted_degree"], errors="coerce").sum()),
                "best_rank": int(pd.to_numeric(gdf["degree_rank"], errors="coerce").min()),
                "ensembl_gene_id": first_nonempty(gdf["ensembl_gene_id"].tolist()),
                "entrez_id": first_nonempty(gdf["entrez_id"].tolist()),
                "known_cancer_gene": first_nonempty(gdf["known_cancer_gene"].tolist()),
                "cancer_role": first_nonempty(gdf["cancer_role"].tolist()),
                "evidence_tier": first_nonempty(gdf["evidence_tier"].tolist()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(
        ["n_pairs_present", "sum_weighted_degree", "mean_weighted_degree", "best_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["combined_rank"] = out.index + 1
    #return the full unique set — the top_hubs cap applies only per-pair, not to the
    # combined aggregation. Capping here would silently discard hub genes from pairs
    # with low overlap and skew the drug query toward a subset of network conditions.
    return out


def first_nonempty(values: list[Any]) -> str:
    #return first non-empty string-like value
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def load_cross_network_genes(cross_network_path: Path) -> set[str]:
    #return uppercase gene names recurring in 2+ subtypes in D or S_case tiers
    if not cross_network_path.exists():
        raise FileNotFoundError(
            f"Cross-network overlap output not found: {cross_network_path}\n"
            "Run Script 40 (cross_network_overlap) before this script."
        )
    df = pd.read_csv(cross_network_path)
    mask = df["tier"].isin(DRUG_TIERS) & (df["n_conditions"] >= 2)
    genes = set(df.loc[mask, "gene"].dropna().astype(str).str.strip().str.upper())
    logging.info(
        "Cross-network recurring genes (D+S_case, 2+ subtypes): %d", len(genes)
    )
    return genes


def safe_request_json(
    session: requests.Session,
    method: str,
    url: str,
    timeout: int,
    **kwargs: Any,
) -> dict[str, Any] | None:
    #hTTP helper with logging and graceful failure
    try:
        response = session.request(method=method, url=url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Request failed (%s): %s", url, exc)
        return None


def fetch_dgidb_interactions_for_gene(
    session: requests.Session,
    gene_name: str,
    page_size: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    #fetch DGIdb interactions for one gene with cursor pagination
    all_rows: list[dict[str, Any]] = []
    after: str | None = None
    pages = 0
    gene_key = normalise_gene_key(gene_name)

    while pages < int(max_pages):
        variables: dict[str, Any] = {
            "geneNames": [gene_name],
            "first": int(page_size),
            "after": after,
        }
        payload = safe_request_json(
            session=session,
            method="POST",
            url=DGIDB_GRAPHQL_URL,
            timeout=timeout,
            json={"query": DGIDB_INTERACTIONS_QUERY, "variables": variables},
        )
        if payload is None:
            break
        if payload.get("errors"):
            logging.warning("DGIdb GraphQL errors for %s: %s", gene_name, payload["errors"])
            break

        interactions = payload.get("data", {}).get("interactions", {})
        nodes = interactions.get("nodes", []) or []
        for node in nodes:
            node_gene = normalise_gene_key(node.get("gene", {}).get("name", ""))
            if node_gene != gene_key:
                continue
            all_rows.append(node)

        page_info = interactions.get("pageInfo", {}) or {}
        has_next = bool(page_info.get("hasNextPage", False))
        after = page_info.get("endCursor", None)
        pages += 1
        if not has_next or not after:
            break
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    return all_rows


def extract_chembl_ids(texts: list[str]) -> set[str]:
    #extract CHEMBL IDs from alias strings
    out: set[str] = set()
    for text in texts:
        for match in CHEMBL_ID_RE.findall(str(text)):
            out.add(match.upper())
    return out


def fetch_chembl_synonyms(
    session: requests.Session,
    chembl_id: str,
    cache_dir: Path,
    timeout: int,
) -> set[str]:
    #get synonym set from one ChEMBL molecule, with local JSON cache
    ensure_dir(cache_dir)
    cache_file = cache_dir / f"chembl_{chembl_id.upper()}.json"

    payload: dict[str, Any] | None = None
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = None

    if payload is None:
        payload = safe_request_json(
            session=session,
            method="GET",
            url=CHEMBL_MOLECULE_URL.format(chembl_id=chembl_id.upper()),
            timeout=timeout,
        )
        if payload is not None:
            cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    syns: set[str] = set()
    if payload is None:
        return syns

    pref_name = payload.get("pref_name")
    if pref_name:
        syns.add(str(pref_name))

    for row in payload.get("molecule_synonyms", []) or []:
        raw = row.get("molecule_synonym")
        if raw:
            syns.add(str(raw))

    return {x for x in syns if str(x).strip()}


def fetch_rxnorm_synonyms(
    session: requests.Session,
    drug_name: str,
    cache_dir: Path,
    timeout: int,
) -> set[str]:
    #fetch RxNorm lexical variants for a drug name, with local cache
    ensure_dir(cache_dir)
    cache_key = normalise_drug_alias(drug_name).replace(" ", "_")
    cache_file = cache_dir / f"rxnorm_{cache_key}.json"

    payload: dict[str, Any] | None = None
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = None

    if payload is None:
        payload = safe_request_json(
            session=session,
            method="GET",
            url=RXNORM_DRUGS_URL,
            timeout=timeout,
            params={"name": drug_name},
        )
        if payload is not None:
            cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out: set[str] = set()
    if payload is None:
        return out

    groups = payload.get("drugGroup", {}).get("conceptGroup", []) or []
    for grp in groups:
        for item in grp.get("conceptProperties", []) or []:
            n1 = item.get("name")
            n2 = item.get("synonym")
            if n1:
                out.add(str(n1))
            if n2:
                out.add(str(n2))
    return {x for x in out if str(x).strip()}


class UnionFind:
    #union-find for synonym-overlap merging of drug identities

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def load_survival_genes(survival_path: str) -> set[str]:
    #load fdr-significant prognostic hub genes from 39_survival_analysis.py output

    Returns an uppercase set of gene symbols. Returns empty set if file is missing.
    #p = Path(survival_path)Compute final ranking score for merged drug candidates.

    All count-based terms are normalised to [0, 1] before weighting so that
    no single dimension dominates due to its natural scale:

      hub_norm    = hub_genes_targeted / max(hub_genes_targeted)
      pair_norm   = pair_coverage / 6          (6 = total number of pairs)
      count_norm  = log1p(count) / log1p(max_count)
      norm_inter  = mean_interaction_score / (mean_interaction_score + 1)  [Michaelis form]

    score = 2.2 * hub_norm
          + 1.5 * pair_norm
          + 2.0 * norm_inter
          + 1.0 * count_norm
          + 1.8 * approved_any
          + 1.5 * survival_fraction

    anti_neoplastic_any is applied as a pre-filter (all candidates retained here
    pass it), not as a scoring term — adding a constant offset to every row has
    no effect on ranking.
    """
    if group_df.empty:
        return group_df
    if survival_genes is None:
        survival_genes = set()

    N_PAIRS = 6  # total comparison pairs in this study

    work = group_df.copy()
    work["mean_interaction_score"] = pd.to_numeric(work["mean_interaction_score"], errors="coerce").fillna(0.0)
    work["hub_genes_targeted"] = pd.to_numeric(work["hub_genes_targeted"], errors="coerce").fillna(0)
    work["interactions_count"] = pd.to_numeric(work["interactions_count"], errors="coerce").fillna(0)
    work["pair_coverage"] = pd.to_numeric(work["pair_coverage"], errors="coerce").fillna(0)

    #normalise all scoring terms to [0, 1] before weighting
    hub_max = work["hub_genes_targeted"].max()
    hub_norm = work["hub_genes_targeted"] / hub_max if hub_max > 0 else work["hub_genes_targeted"] * 0.0

    pair_norm = work["pair_coverage"] / N_PAIRS

    count_log = work["interactions_count"].apply(lambda x: math.log1p(float(x)))
    count_max = count_log.max()
    count_norm = count_log / count_max if count_max > 0 else count_log * 0.0

    norm_interaction = work["mean_interaction_score"] / (work["mean_interaction_score"] + 1.0)

    #survival bonus: proportion of targeted hub genes that are FDR-significant prognostic
    def _survival_fraction(targeted_str: str) -> float:
        if not isinstance(targeted_str, str) or not targeted_str.strip():
            return 0.0
        genes = [g.strip().upper() for g in targeted_str.split("|") if g.strip()]
        if not genes:
            return 0.0
        return sum(1 for g in genes if g in survival_genes) / len(genes)

    work["survival_fraction"] = work["targeted_hub_genes"].apply(_survival_fraction)

    #annotation columns (no scoring weight — for biological interpretation)
    def _prognostic_genes(targeted_str: str) -> str:
        if not isinstance(targeted_str, str) or not targeted_str.strip():
            return ""
        return "|".join(
            g.strip() for g in targeted_str.split("|")
            if g.strip().upper() in survival_genes
        )

    work["targeted_genes_prognostic"] = work["targeted_hub_genes"].apply(_prognostic_genes)
    work["targeted_genes_discordant_prognostic"] = work["targeted_genes_prognostic"]

    score = (
        2.2 * hub_norm
        + 1.5 * pair_norm
        + 2.0 * norm_interaction
        + 1.0 * count_norm
        + 1.8 * work["approved_any"].astype(int)
        + 1.5 * work["survival_fraction"]
    )
    work["candidate_score"] = score
    work = work.sort_values(
        ["candidate_score", "hub_genes_targeted", "pair_coverage"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    work["candidate_rank"] = work.index + 1
    return work


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    #run full hub->drug candidate pipeline and write outputs
    networks_root = resolve_base(str(args.networks_root))
    output_dir = resolve_base(str(args.output_dir))
    cache_dir = resolve_base(str(args.cache_dir))
    ensure_dir(output_dir)
    ensure_dir(cache_dir)

    cross_network_path = Path(args.cross_network_path)
    if not cross_network_path.is_absolute():
        cross_network_path = REPO_ROOT / cross_network_path
    cross_network_genes = load_cross_network_genes(cross_network_path)

    pair_inputs = discover_pair_inputs(networks_root)
    logging.info("Found %d pair folders", len(pair_inputs))

    pair_hub_tables: list[pd.DataFrame] = []
    for pair in pair_inputs:
        tagged_df = pd.read_csv(pair.tagged_csv)
        short = PAIR_SHORT.get(pair.pair_name, pair.pair_name)

        for tier in DRUG_TIERS:
            hub_csv = REPO_ROOT / "results/20_node_annotation" / pair.pair_name / f"{short}_hubs_{tier}.csv"
            if not hub_csv.exists():
                logging.warning("Skipping %s / %s (hub CSV missing)", pair.pair_name, tier)
                continue
            hubs_df = pd.read_csv(hub_csv)
            #normalise column names to what attach_metadata expects
            if "tier_degree" in hubs_df.columns and "weighted_degree" not in hubs_df.columns:
                hubs_df = hubs_df.rename(columns={"tier_degree": "weighted_degree"})
            hubs_df["tier"] = tier
            hubs_with_meta = attach_metadata(pair.pair_name, hubs_df, tagged_df)
            hubs_with_meta["tier"] = tier
            pair_hub_tables.append(hubs_with_meta)

    pair_hubs_df = pd.concat(pair_hub_tables, ignore_index=True) if pair_hub_tables else pd.DataFrame()

    #restrict to cross-network recurring genes (D+S_case, 2+ subtypes)
    if not pair_hubs_df.empty and cross_network_genes:
        gene_upper = pair_hubs_df["gene"].astype(str).str.upper()
        appr_upper = pair_hubs_df.get("approved_symbol", pd.Series(dtype=str)).astype(str).str.upper()
        mask = gene_upper.isin(cross_network_genes) | appr_upper.isin(cross_network_genes)
        before = len(pair_hubs_df)
        pair_hubs_df = pair_hubs_df[mask].reset_index(drop=True)
        logging.info(
            "Cross-network filter: %d → %d hub-gene rows (%d genes kept)",
            before, len(pair_hubs_df), pair_hubs_df["gene"].nunique(),
        )

    pair_hubs_path = output_dir / "01_pair_hubs_topN.csv"
    pair_hubs_df.to_csv(pair_hubs_path, index=False)

    combined_hubs_df = aggregate_combined_hubs(pair_hubs_df, int(args.top_hubs))
    combined_hubs_path = output_dir / "02_combined_hubs_topN.csv"
    combined_hubs_df.to_csv(combined_hubs_path, index=False)

    if combined_hubs_df.empty:
        raise RuntimeError("Combined hub table is empty; cannot continue to drug targeting")

    session = requests.Session()
    timeout = int(args.request_timeout)
    sleep_seconds = float(args.sleep_seconds)

    interaction_rows: list[dict[str, Any]] = []
    synonym_rows: list[dict[str, Any]] = []
    drug_synonyms_by_key: dict[str, set[str]] = defaultdict(set)
    drug_display_by_key: dict[str, str] = {}
    drug_flags_by_key: dict[str, dict[str, bool]] = defaultdict(lambda: {
        "approved": False,
        "anti_neoplastic": False,
        "immunotherapy": False,
    })

    for hub in combined_hubs_df.to_dict(orient="records"):
        gene = str(hub.get("canonical_gene", "")).strip()
        if not gene:
            continue

        nodes = fetch_dgidb_interactions_for_gene(
            session=session,
            gene_name=gene,
            page_size=int(args.dgidb_page_size),
            max_pages=int(args.dgidb_max_pages),
            timeout=timeout,
            sleep_seconds=sleep_seconds,
        )

        logging.info("%s: %d DGIdb interactions", gene, len(nodes))
        for node in nodes:
            drug = node.get("drug", {}) or {}
            drug_name = str(drug.get("name", "")).strip()
            if not drug_name:
                continue

            approved = bool(drug.get("approved", False))
            anti_neoplastic = bool(drug.get("antiNeoplastic", False))
            immunotherapy = bool(drug.get("immunotherapy", False))

            if bool(args.approved_only) and not approved:
                continue
            if bool(args.anti_neoplastic_only) and not anti_neoplastic:
                continue

            concept_id = str(drug.get("conceptId", "")).strip()

            if bool(args.require_chembl) and not concept_id.lower().startswith(CHEMBL_CONCEPT_PREFIX):
                continue
            drug_key = concept_id if concept_id else f"NAME::{normalise_drug_alias(drug_name)}"

            drug_display_by_key.setdefault(drug_key, drug_name)
            flags = drug_flags_by_key[drug_key]
            flags["approved"] = flags["approved"] or approved
            flags["anti_neoplastic"] = flags["anti_neoplastic"] or anti_neoplastic
            flags["immunotherapy"] = flags["immunotherapy"] or immunotherapy

            aliases_dgidb: set[str] = {drug_name}
            for alias_row in drug.get("drugAliases", []) or []:
                alias = alias_row.get("alias")
                if alias:
                    aliases_dgidb.add(str(alias))

            chembl_ids = extract_chembl_ids(list(aliases_dgidb))
            aliases_chembl: set[str] = set()
            for chembl_id in chembl_ids:
                aliases_chembl |= fetch_chembl_synonyms(
                    session=session,
                    chembl_id=chembl_id,
                    cache_dir=cache_dir,
                    timeout=timeout,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            aliases_rxnorm = fetch_rxnorm_synonyms(
                session=session,
                drug_name=drug_name,
                cache_dir=cache_dir,
                timeout=timeout,
            )

            all_aliases = set(aliases_dgidb) | aliases_chembl | aliases_rxnorm
            for alias in all_aliases:
                alias_norm = normalise_drug_alias(alias)
                if not alias_norm:
                    continue
                drug_synonyms_by_key[drug_key].add(alias_norm)

            for alias in aliases_dgidb:
                alias_norm = normalise_drug_alias(alias)
                if alias_norm:
                    synonym_rows.append(
                        {
                            "drug_key": drug_key,
                            "canonical_drug_name": drug_name,
                            "alias": alias,
                            "alias_norm": alias_norm,
                            "source": "DGIdb",
                        }
                    )
            for alias in aliases_chembl:
                alias_norm = normalise_drug_alias(alias)
                if alias_norm:
                    synonym_rows.append(
                        {
                            "drug_key": drug_key,
                            "canonical_drug_name": drug_name,
                            "alias": alias,
                            "alias_norm": alias_norm,
                            "source": "ChEMBL",
                        }
                    )
            for alias in aliases_rxnorm:
                alias_norm = normalise_drug_alias(alias)
                if alias_norm:
                    synonym_rows.append(
                        {
                            "drug_key": drug_key,
                            "canonical_drug_name": drug_name,
                            "alias": alias,
                            "alias_norm": alias_norm,
                            "source": "RxNorm",
                        }
                    )

            interaction_types = sorted(
                {
                    str(x.get("type", "")).strip().lower()
                    for x in (node.get("interactionTypes", []) or [])
                    if str(x.get("type", "")).strip()
                }
            )
            sources = sorted(
                {
                    str(x.get("sourceDbName", "")).strip()
                    for x in (node.get("sources", []) or [])
                    if str(x.get("sourceDbName", "")).strip()
                }
            )

            interaction_rows.append(
                {
                    "hub_gene": gene,
                    "hub_combined_rank": int(hub.get("combined_rank", 0)),
                    "hub_pairs": str(hub.get("pair_list", "")),
                    "hub_n_pairs_present": int(hub.get("n_pairs_present", 0)),
                    "hub_sum_weighted_degree": float(hub.get("sum_weighted_degree", 0.0)),
                    "drug_key": drug_key,
                    "drug_name": drug_name,
                    "approved": approved,
                    "anti_neoplastic": anti_neoplastic,
                    "immunotherapy": immunotherapy,
                    "interaction_score": float(node.get("interactionScore", 0.0) or 0.0),
                    "evidence_score": float(node.get("evidenceScore", 0.0) or 0.0),
                    "interaction_types": "|".join(interaction_types),
                    "source_dbs": "|".join(sources),
                }
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    interactions_df = pd.DataFrame(interaction_rows)
    interactions_path = output_dir / "03_hub_gene_drug_interactions_raw.csv"
    interactions_df.to_csv(interactions_path, index=False)

    synonyms_df = pd.DataFrame(synonym_rows)
    if not synonyms_df.empty:
        synonyms_df = synonyms_df.drop_duplicates().reset_index(drop=True)
    synonyms_path = output_dir / "04_drug_synonym_dictionary.csv"
    synonyms_df.to_csv(synonyms_path, index=False)

    if interactions_df.empty:
        summary = {
            "status": "no_interactions",
            "message": "No DGIdb interactions passed filters",
            "pairs_processed": len(pair_inputs),
            "combined_hubs": int(combined_hubs_df.shape[0]),
        }
        (output_dir / "06_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    uf = UnionFind()
    for key in interactions_df["drug_key"].astype(str).tolist():
        uf.add(key)

    alias_owner: dict[str, str] = {}
    for key, synset in drug_synonyms_by_key.items():
        uf.add(key)
        for alias_norm in synset:
            owner = alias_owner.get(alias_norm)
            if owner is None:
                alias_owner[alias_norm] = key
            else:
                uf.union(owner, key)

    interactions_df["drug_group_id"] = interactions_df["drug_key"].astype(str).apply(uf.find)

    group_rows: list[dict[str, Any]] = []
    for group_id, gdf in interactions_df.groupby("drug_group_id", dropna=False):
        group_keys = sorted(set(gdf["drug_key"].astype(str).tolist()))
        hub_genes = sorted(set(gdf["hub_gene"].astype(str).tolist()))
        pair_set: set[str] = set()
        for pairs_text in gdf["hub_pairs"].astype(str).tolist():
            for p in str(pairs_text).split("|"):
                p = p.strip()
                if p:
                    pair_set.add(p)

        canonical_name = choose_group_name(group_keys, drug_display_by_key)
        synset_merged: set[str] = set()
        approved_any = False
        anti_any = False
        immuno_any = False
        for k in group_keys:
            synset_merged |= set(drug_synonyms_by_key.get(k, set()))
            flags = drug_flags_by_key.get(k, {})
            approved_any = approved_any or bool(flags.get("approved", False))
            anti_any = anti_any or bool(flags.get("anti_neoplastic", False))
            immuno_any = immuno_any or bool(flags.get("immunotherapy", False))

        interaction_types = sorted(
            {
                t.strip().lower()
                for text in gdf["interaction_types"].astype(str).tolist()
                for t in text.split("|")
                if t.strip()
            }
        )
        source_dbs = sorted(
            {
                s.strip()
                for text in gdf["source_dbs"].astype(str).tolist()
                for s in text.split("|")
                if s.strip()
            }
        )

        group_rows.append(
            {
                "drug_group_id": group_id,
                "canonical_drug_name": canonical_name,
                "hub_genes_targeted": len(hub_genes),
                "targeted_hub_genes": "|".join(hub_genes),
                "pair_coverage": len(pair_set),
                "pairs": "|".join(sorted(pair_set)),
                "interactions_count": int(gdf.shape[0]),
                "mean_interaction_score": float(pd.to_numeric(gdf["interaction_score"], errors="coerce").mean()),
                "max_interaction_score": float(pd.to_numeric(gdf["interaction_score"], errors="coerce").max()),
                "mean_evidence_score": float(pd.to_numeric(gdf["evidence_score"], errors="coerce").mean()),
                "max_evidence_score": float(pd.to_numeric(gdf["evidence_score"], errors="coerce").max()),
                "approved_any": bool(approved_any),
                "anti_neoplastic_any": bool(anti_any),
                "immunotherapy_any": bool(immuno_any),
                "interaction_types": "|".join(interaction_types),
                "source_dbs": "|".join(source_dbs),
                "synonym_count": len(synset_merged),
                "synonyms_norm": "|".join(sorted(synset_merged)),
                "merged_drug_keys": "|".join(group_keys),
            }
        )

    survival_genes = load_survival_genes(resolve_base(str(args.survival_path)))

    candidates_df = pd.DataFrame(group_rows)
    candidates_df = rank_drug_groups(candidates_df, survival_genes=survival_genes)
    candidates_df = filter_non_drug_candidates(candidates_df)
    candidates_path = output_dir / "05_drug_candidates_ranked.csv"
    candidates_df.to_csv(candidates_path, index=False)

    summary = {
        "status": "ok",
        "pairs_processed": len(pair_inputs),
        "pair_hub_rows": int(pair_hubs_df.shape[0]),
        "combined_hubs": int(combined_hubs_df.shape[0]),
        "raw_interactions": int(interactions_df.shape[0]),
        "synonym_rows": int(synonyms_df.shape[0]),
        "drug_candidates": int(candidates_df.shape[0]),
        "top_candidate_preview": candidates_df.head(10).to_dict(orient="records"),
        "outputs": {
            "pair_hubs": str(pair_hubs_path),
            "combined_hubs": str(combined_hubs_path),
            "raw_interactions": str(interactions_path),
            "synonym_dictionary": str(synonyms_path),
            "candidates": str(candidates_path),
        },
    }
    (output_dir / "06_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


_NON_DRUG_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in [
    #non-pharmaceutical entities
    r"\(recombinant",          # recombinant protein replacement therapies
    r"^protein s$",            # endogenous anticoagulant, not a therapeutic
    r"^h2o2$",                 # hydrogen peroxide
    r"^glutamine$",            # amino acid supplement
    r"^glucarpidase$",         # enzyme rescue agent (methotrexate toxicity)
    #cardiovascular / non-oncology drug classes
    r"statin",                 # statins (atorvastatin, simvastatin, …)
    r"pril\b",                 #ace inhibitors
    r"^warfarin",              # anticoagulant
    #cNS / endocrine drugs
    r"^haloperidol",           # antipsychotic
    r"^levodopa$",             # Parkinson's dopamine precursor
    r"^thyrotropin$",          # thyroid-stimulating hormone
    #topical-only corticosteroids (interact with ANXA1 but have no systemic oncology use)
    r"^(?:desonide|halobetasol|alclometasone|loteprednol|prednicarbate|clocortolone"
    r"|rimexolone|desoximetasone|amcinonide|diflorasone|hydrocortamate|clobetasol"
    r"|flumethasone|fluocinolone|flurandrenolide|halcinonide)$",
    r"(?:pivalate|acetonide|dipropionate|diacetate)$",  # esterified topical steroid forms
    r"^phorbol",               # phorbol esters are lab tumor promoters, not therapeutics
]]


def is_pharmacological_drug(name: str) -> bool:
    #return False for known non-drug or clearly off-topic entities
    n = str(name).strip()
    return not any(p.search(n) for p in _NON_DRUG_PATTERNS)


def filter_non_drug_candidates(df: pd.DataFrame) -> pd.DataFrame:
    #remove non-pharmaceutical entities from the ranked candidate table
    if df.empty:
        return df
    mask = df["canonical_drug_name"].apply(is_pharmacological_drug)
    removed = (~mask).sum()
    if removed:
        logging.info("Post-filter: removed %d non-drug entries", removed)
    return df[mask].reset_index(drop=True)


def choose_group_name(group_keys: list[str], name_map: dict[str, str]) -> str:
    #pick stable display name for a merged drug group
    names = [str(name_map.get(k, "")).strip() for k in group_keys]
    names = [n for n in names if n]
    if not names:
        return group_keys[0] if group_keys else ""
    counts = pd.Series(names).value_counts()
    return str(counts.index[0])


def configure_logging() -> None:
    #set script logger format
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    #cLI entrypoint
    configure_logging()
    args = parse_args()
    summary = run_pipeline(args)
    logging.info("Finished. Status=%s", summary.get("status"))
    if summary.get("status") == "ok":
        logging.info(
            "Pairs=%s | Hubs=%s | Interactions=%s | Candidates=%s",
            summary.get("pairs_processed"),
            summary.get("combined_hubs"),
            summary.get("raw_interactions"),
            summary.get("drug_candidates"),
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
