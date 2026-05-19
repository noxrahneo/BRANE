#!/usr/bin/env python3
"""Extended z-score cell-type tagging for persistent network genes.

Pipeline order: preflight → z-score matrices (fine + coarse) → label assignment
→ per-pair tagged tables → coverage + hub diagnostics.
Reads results/04_annotation (read-only); writes to results/20_node_annotation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE04_ANNOTATION_DIR = REPO_ROOT / "results" / "04_annotation"
CANCER_ANNOTATION_DIR = REPO_ROOT / "results" / "19_cancer_gene_annotation"
PERMUTATION_NETWORK_DIR = REPO_ROOT / "results" / "14_csd_networks"

TAGGING_DIR = REPO_ROOT / "results" / "20_node_annotation"
EXTENDED_ZSCORE_DIR = TAGGING_DIR / "01_input_data" / "extended_zscore"
OUTPUT_DIR = TAGGING_DIR / "02_output"

CONDITIONS: list[str] = [
    "ER_tumor",
    "HER2_tumor",
    "Normal",
    "Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_BRCA1_tumor",
    "Triple_negative_tumor",
]


@dataclass
class PreflightConditionResult:
    condition: str
    network_gene_count: int
    h5ad_gene_count: int
    overlap_count: int
    overlap_ratio: float
    sample_max: float
    normalization_path: str
    low_label_warning: bool


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extended z-score cell-type tagging for persistent network genes"
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight diagnostics only, then exit",
    )
    parser.add_argument(
        "--min-cells-per-type",
        type=int,
        default=50,
        help="Minimum cells required to keep a label when building z-score matrices",
    )
    parser.add_argument(
        "--min-top-zscore",
        type=float,
        default=0.5,
        help="Minimum top z-score to assign gene label",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.3,
        help="Minimum top1-top2 z-score margin to assign gene label",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    EXTENDED_ZSCORE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pairs() -> list[str]:
    if not CANCER_ANNOTATION_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {CANCER_ANNOTATION_DIR}")
    pairs = sorted(
        d.name for d in CANCER_ANNOTATION_DIR.iterdir() if d.is_dir() and "__vs__" in d.name
    )
    if not pairs:
        raise RuntimeError(f"No pair folders found in {CANCER_ANNOTATION_DIR}")
    return pairs


def parse_pair_name(pair_name: str) -> tuple[str, str]:
    parts = pair_name.split("__vs__")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair name format: {pair_name}")
    return parts[0], parts[1]


def get_master_annotation_path(pair_name: str) -> Path:
    pair_dir = CANCER_ANNOTATION_DIR / pair_name
    preferred = sorted(pair_dir.glob("*_master_annotation.csv"))
    if preferred:
        return preferred[0]
    fallback = sorted(pair_dir.glob("*_cancer_gene_master_annotation.csv"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"No master annotation CSV found for pair: {pair_name}")


def read_master_annotation(pair_name: str) -> pd.DataFrame:
    path = get_master_annotation_path(pair_name)
    df = pd.read_csv(path)
    if "approved_symbol" not in df.columns:
        raise ValueError(f"approved_symbol column missing in {path}")
    return df


def collect_network_genes_by_condition(pairs: list[str]) -> dict[str, set[str]]:
    by_condition: dict[str, set[str]] = {c: set() for c in CONDITIONS}

    for pair in pairs:
        test_cond, ref_cond = parse_pair_name(pair)
        df = read_master_annotation(pair)
        genes = set(df["approved_symbol"].dropna().astype(str).str.strip())
        genes.discard("")

        if test_cond in by_condition:
            by_condition[test_cond].update(genes)
        if ref_cond in by_condition:
            by_condition[ref_cond].update(genes)

    return by_condition


def _sample_max_from_matrix(x: Any, n_rows: int = 100, n_cols: int = 100) -> float:
    rows = min(n_rows, x.shape[0])
    cols = min(n_cols, x.shape[1])
    block = x[:rows, :cols]

    if sparse.issparse(block):
        return float(block.max()) if block.nnz > 0 else 0.0

    arr = np.asarray(block)
    if arr.size == 0:
        return 0.0
    return float(np.nanmax(arr))


def _normalization_path_from_sample_max(sample_max: float) -> str:
    if sample_max > 100:
        return "raw_counts_detected_normalize_total_log1p"
    if sample_max <= 20:
        return "already_log_normalized_use_directly"
    return "ambiguous_scale_assume_log_like_use_directly"


def _warn_small_labels(label_counts: pd.Series, condition: str, label_name: str) -> bool:
    low = label_counts[label_counts < 50]
    if low.empty:
        return False

    logging.warning(
        "[%s] %s labels with fewer than 50 cells:\n%s",
        condition,
        label_name,
        low.to_string(),
    )
    return True


def run_preflight(
    pairs: list[str],
    network_genes_by_condition: dict[str, set[str]],
) -> tuple[list[PreflightConditionResult], dict[str, str]]:
    logging.info("=== Step 0: Preflight diagnostics ===")

    results: list[PreflightConditionResult] = []
    normalization_paths: dict[str, str] = {}

    for cond in tqdm(CONDITIONS, desc="Preflight conditions"):
        h5ad_path = STAGE04_ANNOTATION_DIR / cond / f"{cond}_annotated.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(f"Missing h5ad for condition {cond}: {h5ad_path}")

        adata = sc.read_h5ad(h5ad_path, backed="r")
        try:
            if "cell_type_annot" not in adata.obs.columns:
                raise ValueError(f"[{cond}] Missing obs column: cell_type_annot")
            if "major_compartment" not in adata.obs.columns:
                raise ValueError(f"[{cond}] Missing obs column: major_compartment")

            network_genes = network_genes_by_condition.get(cond, set())
            h5ad_genes = set(pd.Index(adata.var_names).astype(str))
            overlap = network_genes & h5ad_genes
            overlap_ratio = (
                float(len(overlap) / len(network_genes)) if network_genes else 1.0
            )

            sample_max = _sample_max_from_matrix(adata.X, n_rows=100, n_cols=100)
            norm_path = _normalization_path_from_sample_max(sample_max)
            normalization_paths[cond] = norm_path

            cell_type_counts = adata.obs["cell_type_annot"].astype(str).value_counts()
            major_counts = adata.obs["major_compartment"].astype(str).value_counts()

            logging.info(
                "[%s] Symbol overlap: %d/%d = %.4f",
                cond,
                len(overlap),
                len(network_genes),
                overlap_ratio,
            )
            logging.info("[%s] Expression sample_max(100x100): %.4f", cond, sample_max)
            logging.info("[%s] Scale path: %s", cond, norm_path)
            logging.info("[%s] cell_type_annot counts:\n%s", cond, cell_type_counts.to_string())
            logging.info("[%s] major_compartment counts:\n%s", cond, major_counts.to_string())

            low_warn_cell = _warn_small_labels(cell_type_counts, cond, "cell_type_annot")
            low_warn_major = _warn_small_labels(major_counts, cond, "major_compartment")

            results.append(
                PreflightConditionResult(
                    condition=cond,
                    network_gene_count=len(network_genes),
                    h5ad_gene_count=len(h5ad_genes),
                    overlap_count=len(overlap),
                    overlap_ratio=overlap_ratio,
                    sample_max=sample_max,
                    normalization_path=norm_path,
                    low_label_warning=(low_warn_cell or low_warn_major),
                )
            )
        finally:
            if getattr(adata, "isbacked", False) and adata.file is not None:
                adata.file.close()

    #threshold lowered from 0.70 — network genes not in HVG set simply get no cell type label
    low_overlap = [r for r in results if r.overlap_ratio < 0.40]
    if low_overlap:
        msg = "\n".join(
            f"{r.condition}: overlap={r.overlap_ratio:.4f} ({r.overlap_count}/{r.network_gene_count})"
            for r in low_overlap
        )
        raise RuntimeError(
            "Preflight failed: overlap < 0.40 detected. Stop and fix symbol matching first.\n"
            + msg
        )

    summary_df = pd.DataFrame([r.__dict__ for r in results])
    summary_path = OUTPUT_DIR / "preflight_diagnostics.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info("Preflight summary saved: %s", summary_path)

    return results, normalization_paths


def build_extended_zscore(
    h5ad_path: Path,
    network_genes: list[str],
    cell_type_col: str,
    normalization_path: str,
    min_cells_per_type: int = 50,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    #build row-z-score matrix (genes x labels) and map labels to cell counts
    adata = sc.read_h5ad(h5ad_path)

    try:
        if cell_type_col not in adata.obs.columns:
            raise ValueError(f"Missing obs column '{cell_type_col}' in {h5ad_path}")

        genes_present = [g for g in network_genes if g in set(map(str, adata.var_names))]
        if not genes_present:
            out = pd.DataFrame()
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(output_path)
            return out, {}

        adata = adata[:, genes_present].copy()

        if normalization_path == "raw_counts_detected_normalize_total_log1p":
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        labels = adata.obs[cell_type_col].astype(str).fillna("Unknown")
        label_counts = labels.value_counts()
        keep_labels = label_counts[label_counts >= int(min_cells_per_type)].index.tolist()

        if not keep_labels:
            cols = ["__no_label_passed_min_cells__"]
            z = pd.DataFrame(0.0, index=genes_present, columns=cols)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                z.to_csv(output_path)
            return z, {}

        x = adata.X
        if sparse.issparse(x):
            x = x.tocsr()

        means_by_label: dict[str, np.ndarray] = {}
        for label in keep_labels:
            idx = np.where(labels.values == label)[0]
            if idx.size == 0:
                continue
            if sparse.issparse(x):
                mu = np.asarray(x[idx, :].mean(axis=0)).ravel()
            else:
                mu = np.asarray(x[idx, :], dtype=float).mean(axis=0)
            means_by_label[label] = mu

        if not means_by_label:
            z = pd.DataFrame(0.0, index=genes_present, columns=["__no_means__"])
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                z.to_csv(output_path)
            return z, {}

        mean_df = pd.DataFrame(means_by_label, index=genes_present)
        values = mean_df.to_numpy(dtype=float)
        row_mean = values.mean(axis=1, keepdims=True)
        row_std = values.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        z_values = (values - row_mean) / row_std

        z_df = pd.DataFrame(z_values, index=mean_df.index, columns=mean_df.columns)

        #build cell counts dict for passed labels
        cell_counts_dict = {str(label): int(label_counts[label]) for label in keep_labels}

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            z_df.to_csv(output_path)

        return z_df, cell_counts_dict
    finally:
        del adata


def assign_cell_type(
    zscore_matrix: pd.DataFrame,
    cell_counts: dict[str, int] | None = None,
    min_top_zscore: float = 0.5,
    min_margin: float = 0.3,
    apply_confidence_penalty: bool = True,
) -> pd.DataFrame:
    #assign cell types by top z-score + margin; rare types penalized by count/median confidence
    if zscore_matrix.empty:
        return pd.DataFrame(
            columns=["gene", "cell_type", "top_zscore", "margin", "ct_source", "confidence_factor"]
        )

    arr = zscore_matrix.to_numpy(dtype=float)
    col_names = np.array(zscore_matrix.columns, dtype=object)

    #apply confidence penalty if cell counts provided
    confidence_factors = np.ones(len(col_names), dtype=float)
    if apply_confidence_penalty and cell_counts is not None:
        counts_arr = np.array([cell_counts.get(str(ct), 0) for ct in col_names], dtype=float)
        valid_counts = counts_arr[counts_arr > 0]
        if len(valid_counts) > 0:
            median_count = float(np.median(valid_counts))
            confidence_factors = np.minimum(1.0, counts_arr / median_count)
            logging.debug(
                "Cell count confidence factors: median=%.1f, range=[%.3f, %.3f]",
                median_count, confidence_factors.min(), confidence_factors.max()
            )
    
    #apply confidence factor to each column
    weighted_arr = arr * confidence_factors[np.newaxis, :]

    # argmax/top score on weighted array
    top_idx = np.argmax(weighted_arr, axis=1)
    top_scores_weighted = weighted_arr[np.arange(weighted_arr.shape[0]), top_idx]
    top_scores_raw = arr[np.arange(arr.shape[0]), top_idx]  # Keep raw for output
    top_labels = col_names[top_idx]
    top_confidence = confidence_factors[top_idx]

    # second highest score via sort (on weighted array)
    if weighted_arr.shape[1] > 1:
        part = np.partition(weighted_arr, kth=weighted_arr.shape[1] - 2, axis=1)
        second_scores = part[:, -2]
    else:
        second_scores = np.zeros(weighted_arr.shape[0], dtype=float)

    margins = top_scores_weighted - second_scores
    assign_mask = (top_scores_weighted >= float(min_top_zscore)) & (margins >= float(min_margin))

    assigned = np.where(assign_mask, top_labels, None)
    source = np.where(assign_mask, "extended_zscore", None)

    out = pd.DataFrame(
        {
            "gene": zscore_matrix.index.astype(str),
            "cell_type": assigned,
            "top_zscore": top_scores_raw,
            "weighted_zscore": top_scores_weighted,
            "margin": margins,
            "confidence_factor": top_confidence,
            "ct_source": source,
        }
    )
    return out


def resolve_pair_annotation(
    pair_name: str,
    fine_assignments: dict[str, pd.DataFrame],
    coarse_assignments: dict[str, pd.DataFrame],
    master_annotation_path: Path,
) -> pd.DataFrame:
    test_cond, ref_cond = parse_pair_name(pair_name)

    master = pd.read_csv(master_annotation_path)
    if "approved_symbol" not in master.columns:
        raise ValueError(f"approved_symbol column missing in {master_annotation_path}")

    test_fine = fine_assignments[test_cond].rename(
        columns={
            "gene": "approved_symbol",
            "cell_type": "cell_type",
            "top_zscore": "top_zscore",
            "margin": "margin",
            "ct_source": "ct_source",
        }
    )[["approved_symbol", "cell_type", "top_zscore", "margin", "ct_source"]]

    ref_fine = fine_assignments[ref_cond].rename(
        columns={"gene": "approved_symbol", "cell_type": "cell_type_ref"}
    )[["approved_symbol", "cell_type_ref"]]

    test_coarse = coarse_assignments[test_cond].rename(
        columns={"gene": "approved_symbol", "cell_type": "major_compartment"}
    )[["approved_symbol", "major_compartment"]]

    ref_coarse = coarse_assignments[ref_cond].rename(
        columns={"gene": "approved_symbol", "cell_type": "major_compartment_ref"}
    )[["approved_symbol", "major_compartment_ref"]]

    out = master.merge(test_fine, on="approved_symbol", how="left")
    out = out.merge(ref_fine, on="approved_symbol", how="left")
    out = out.merge(test_coarse, on="approved_symbol", how="left")
    out = out.merge(ref_coarse, on="approved_symbol", how="left")

    out["ct_switched"] = (
        out["cell_type"].notna()
        & out["cell_type_ref"].notna()
        & (out["cell_type"].astype(str) != out["cell_type_ref"].astype(str))
    )

    return out


def _edge_file_for_pair(pair_name: str) -> Path | None:
    fname = f"{pair_name}_differential_edges_permutation.csv"
    candidate = PERMUTATION_NETWORK_DIR / pair_name / fname
    if candidate.exists():
        return candidate
    return None


def _compute_hubs(pair_name: str, top_n: int = 50) -> list[str]:
    edge_file = _edge_file_for_pair(pair_name)
    if edge_file is None:
        return []

    e = pd.read_csv(edge_file)
    if e.empty or "gene_a" not in e.columns or "gene_b" not in e.columns:
        return []

    use_weight = "weight" in e.columns and pd.to_numeric(e["weight"], errors="coerce").notna().any()

    if use_weight:
        weights = pd.to_numeric(e["weight"], errors="coerce").fillna(1.0)
        deg: dict[str, float] = {}
        for a, b, w in zip(e["gene_a"].astype(str), e["gene_b"].astype(str), weights):
            deg[a] = deg.get(a, 0.0) + float(w)
            deg[b] = deg.get(b, 0.0) + float(w)
        return [g for g, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]

    # fallback to unweighted degree
    deg2: dict[str, int] = {}
    for a, b in zip(e["gene_a"].astype(str), e["gene_b"].astype(str)):
        deg2[a] = deg2.get(a, 0) + 1
        deg2[b] = deg2.get(b, 0) + 1
    return [g for g, _ in sorted(deg2.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]


def _build_fine_to_coarse_map(conditions: Iterable[str]) -> dict[str, str]:
    mapping_counts: dict[str, dict[str, int]] = {}

    for cond in conditions:
        path = STAGE04_ANNOTATION_DIR / cond / f"{cond}_cluster_to_celltype_mapping.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "predicted_cell_type" not in df.columns or "major_compartment" not in df.columns:
            continue

        for _, row in df.iterrows():
            fine = str(row["predicted_cell_type"]).strip()
            coarse = str(row["major_compartment"]).strip()
            if not fine or fine.lower() == "nan":
                continue
            if not coarse or coarse.lower() == "nan":
                continue
            mapping_counts.setdefault(fine, {})
            mapping_counts[fine][coarse] = mapping_counts[fine].get(coarse, 0) + 1

    out: dict[str, str] = {}
    for fine, counts in mapping_counts.items():
        out[fine] = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    # helpful defaults for missing labels
    for label in ["Epithelial", "Basal_Epi", "Luminal_Epi", "Basal", "LP", "ML"]:
        out.setdefault(label, "Epithelial")
    for label in ["TCell", "BCell", "Plasma", "Myeloid", "Mast", "NK", "Cycling", "Fibroblast", "Endothelial", "Fibro", "Endo", "Macro"]:
        out.setdefault(label, "Stromal_like")
    out.setdefault("Uncertain", "Uncertain")
    out.setdefault("Unknown", "Uncertain")

    return out


def build_coverage_report(tagged_by_pair: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fine_to_coarse = _build_fine_to_coarse_map(CONDITIONS)
    rows: list[dict[str, Any]] = []

    for pair, df in tagged_by_pair.items():
        total = len(df)
        assigned = int(df["cell_type"].notna().sum()) if "cell_type" in df.columns else 0

        hubs = _compute_hubs(pair, top_n=50)
        hub_set = set(hubs)
        if hubs and "approved_symbol" in df.columns:
            hub_df = df[df["approved_symbol"].astype(str).isin(hub_set)].copy()
            hub_assigned = int(hub_df["cell_type"].notna().sum())
            hub_total = len(hub_df)
        else:
            hub_assigned = 0
            hub_total = len(hubs)

        ct_switched_count = int(df["ct_switched"].sum()) if "ct_switched" in df.columns else 0

        if "cell_type" in df.columns and df["cell_type"].notna().any():
            most_common = str(df["cell_type"].dropna().mode().iloc[0])
        else:
            most_common = ""

        # agreement only where both are present
        agree_pct = np.nan
        if "cell_type" in df.columns and "major_compartment" in df.columns:
            tmp = df[["cell_type", "major_compartment"]].dropna().copy()
            if not tmp.empty:
                mapped = tmp["cell_type"].astype(str).map(lambda x: fine_to_coarse.get(x, ""))
                agree = (mapped.astype(str) == tmp["major_compartment"].astype(str)).sum()
                agree_pct = float(100.0 * agree / len(tmp))

        rows.append(
            {
                "pair": pair,
                "total_genes": total,
                "assigned_genes": assigned,
                "assigned_pct": float(100.0 * assigned / total) if total else 0.0,
                "hub_genes": hub_total,
                "hub_assigned": hub_assigned,
                "hub_assigned_pct": float(100.0 * hub_assigned / hub_total) if hub_total else 0.0,
                "ct_switched_count": ct_switched_count,
                "ct_switched_pct": float(100.0 * ct_switched_count / total) if total else 0.0,
                "most_common_cell_type": most_common,
                "fine_vs_coarse_agreement_pct": agree_pct,
            }
        )

    report = pd.DataFrame(rows).sort_values("pair")
    return report


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    ensure_output_dirs()

    pairs = find_pairs()
    logging.info("Pairs discovered: %d", len(pairs))

    network_genes_by_condition = collect_network_genes_by_condition(pairs)

    #step 0
    _, normalization_paths = run_preflight(pairs, network_genes_by_condition)

    if args.preflight_only:
        logging.info("Preflight-only mode enabled. Exiting after diagnostics.")
        return 0

    #step 1
    logging.info("=== Step 1: Build extended z-score matrices ===")
    fine_z_by_cond: dict[str, pd.DataFrame] = {}
    fine_counts_by_cond: dict[str, dict[str, int]] = {}
    coarse_z_by_cond: dict[str, pd.DataFrame] = {}
    coarse_counts_by_cond: dict[str, dict[str, int]] = {}

    for cond in tqdm(CONDITIONS, desc="Build z-score matrices"):
        h5ad_path = STAGE04_ANNOTATION_DIR / cond / f"{cond}_annotated.h5ad"
        genes = sorted(network_genes_by_condition.get(cond, set()))
        norm_path = normalization_paths.get(cond, "already_log_normalized_use_directly")

        fine_out = EXTENDED_ZSCORE_DIR / f"{cond}_extended_zscore_finegrained.csv"
        coarse_out = EXTENDED_ZSCORE_DIR / f"{cond}_extended_zscore_coarse.csv"

        logging.info("[%s] Building fine-grained z-score matrix", cond)
        fine_z_by_cond[cond], fine_counts_by_cond[cond] = build_extended_zscore(
            h5ad_path=h5ad_path,
            network_genes=genes,
            cell_type_col="cell_type_annot",
            normalization_path=norm_path,
            min_cells_per_type=args.min_cells_per_type,
            output_path=fine_out,
        )

        logging.info("[%s] Building coarse z-score matrix", cond)
        coarse_z_by_cond[cond], coarse_counts_by_cond[cond] = build_extended_zscore(
            h5ad_path=h5ad_path,
            network_genes=genes,
            cell_type_col="major_compartment",
            normalization_path=norm_path,
            min_cells_per_type=args.min_cells_per_type,
            output_path=coarse_out,
        )

    #step 2
    logging.info("=== Step 2: Assign gene labels from z-score matrices ===")
    fine_assignments: dict[str, pd.DataFrame] = {}
    coarse_assignments: dict[str, pd.DataFrame] = {}

    for cond in CONDITIONS:
        fine_assignments[cond] = assign_cell_type(
            zscore_matrix=fine_z_by_cond[cond],
            cell_counts=fine_counts_by_cond[cond],
            min_top_zscore=args.min_top_zscore,
            min_margin=args.min_margin,
            apply_confidence_penalty=True,
        )
        coarse_assignments[cond] = assign_cell_type(
            zscore_matrix=coarse_z_by_cond[cond],
            cell_counts=coarse_counts_by_cond[cond],
            min_top_zscore=args.min_top_zscore,
            min_margin=args.min_margin,
            apply_confidence_penalty=True,
        )

    #step 2b: Assignment Statistics
    logging.info("=== Step 2b: Cell-type assignment statistics ===")
    stats_records = []
    for cond in CONDITIONS:
        fine_assign = fine_assignments[cond]
        coarse_assign = coarse_assignments[cond]
        fine_counts = fine_counts_by_cond[cond]
        coarse_counts = coarse_counts_by_cond[cond]

        #fine-grained stats
        n_fine_assigned = fine_assign["cell_type"].notna().sum()
        fine_ct_dist = fine_assign[fine_assign["cell_type"].notna()]["cell_type"].value_counts().to_dict()
        if fine_counts:
            fine_median_count = float(np.median([int(c) for c in fine_counts.values()]))
            fine_cf_mean = fine_assign.loc[fine_assign["cell_type"].notna(), "confidence_factor"].mean()
        else:
            fine_median_count = 0.0
            fine_cf_mean = 1.0

        #coarse-grained stats
        n_coarse_assigned = coarse_assign["cell_type"].notna().sum()
        coarse_ct_dist = coarse_assign[coarse_assign["cell_type"].notna()]["cell_type"].value_counts().to_dict()
        if coarse_counts:
            coarse_median_count = float(np.median([int(c) for c in coarse_counts.values()]))
            coarse_cf_mean = coarse_assign.loc[coarse_assign["cell_type"].notna(), "confidence_factor"].mean()
        else:
            coarse_median_count = 0.0
            coarse_cf_mean = 1.0

        logging.info(
            "[%s] Fine-grained: %d/%d genes assigned (median cell count: %.0f, avg confidence: %.3f)",
            cond,
            n_fine_assigned,
            len(fine_assign),
            fine_median_count,
            fine_cf_mean,
        )
        logging.info("[%s] Fine distribution: %s", cond, fine_ct_dist)
        logging.info(
            "[%s] Coarse: %d/%d genes assigned (median cell count: %.0f, avg confidence: %.3f)",
            cond,
            n_coarse_assigned,
            len(coarse_assign),
            coarse_median_count,
            coarse_cf_mean,
        )
        logging.info("[%s] Coarse distribution: %s", cond, coarse_ct_dist)

        for ct_name, ct_count in fine_ct_dist.items():
            stats_records.append({
                "condition": cond,
                "granularity": "fine",
                "cell_type": ct_name,
                "n_genes_assigned": int(ct_count),
                "n_cells_in_type": int(fine_counts.get(str(ct_name), 0)),
            })
        for ct_name, ct_count in coarse_ct_dist.items():
            stats_records.append({
                "condition": cond,
                "granularity": "coarse",
                "cell_type": ct_name,
                "n_genes_assigned": int(ct_count),
                "n_cells_in_type": int(coarse_counts.get(str(ct_name), 0)),
            })

    stats_df = pd.DataFrame(stats_records)
    stats_path = OUTPUT_DIR / "assignment_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    logging.info("Assignment statistics saved: %s", stats_path)

    #step 3
    logging.info("=== Step 3: Resolve pair annotations ===")
    tagged_by_pair: dict[str, pd.DataFrame] = {}

    for pair in tqdm(pairs, desc="Resolve pairs"):
        master_path = get_master_annotation_path(pair)
        tagged = resolve_pair_annotation(
            pair_name=pair,
            fine_assignments=fine_assignments,
            coarse_assignments=coarse_assignments,
            master_annotation_path=master_path,
        )
        out_file = OUTPUT_DIR / f"{pair}_tagged.csv"
        tagged.to_csv(out_file, index=False)
        logging.info("[%s] Tagged output written: %s", pair, out_file)
        tagged_by_pair[pair] = tagged

    #step 4
    logging.info("=== Step 4: Coverage and hub diagnostics ===")
    report = build_coverage_report(tagged_by_pair)
    report_path = OUTPUT_DIR / "coverage_report.csv"
    report.to_csv(report_path, index=False)
    logging.info("Coverage report saved: %s", report_path)

    logging.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
