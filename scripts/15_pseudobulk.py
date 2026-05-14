#!/usr/bin/env python3
"""Build pseudobulk matrices using the decoupler pseudobulk workflow.

This script follows the decoupler tutorial logic and adapts it to BRANE:

Step 1) Load annotated single-cell data (per condition).
Step 2) Aggregate counts into pseudobulk profiles by sample and cell type.
Step 3) Visualize/filter low-quality pseudobulk profiles.
Step 4) Explore variability with PCA and metadata associations.
Step 5) Export counts, logCPM, metadata, and pseudobulk h5ad outputs.

The goal is to produce correlation-ready pseudobulk matrices while preserving
tutorial-style diagnostics and readability for thesis methods reporting.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_decoupler():
    try:
        return importlib.import_module("decoupler")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "decoupler is required for this script. "
            "Install with: pip install decoupler"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pseudobulk matrices from annotated h5ad files"
    )
    parser.add_argument(
        "--input-dir",
        default="results/04_annotation",
        help="Root annotation directory containing one folder per condition",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition folder name under input-dir, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition folders and exit",
    )
    parser.add_argument(
        "--output-dir",
        default="results/06_pseudobulk",
        help="Output folder for pseudobulk artifacts",
    )
    parser.add_argument(
        "--sample-col",
        default="SampleName",
        help="obs column used as biological replicate key",
    )
    parser.add_argument(
        "--group-col",
        default="cell_type_annot",
        help="obs column used as grouping label (e.g., cell type)",
    )
    parser.add_argument(
        "--condition-col",
        default="Condition",
        help="Optional obs condition column; falls back to folder name",
    )
    parser.add_argument(
        "--layer",
        default="counts",
        help="Expression layer to aggregate (default: counts)",
    )
    parser.add_argument(
        "--mode",
        default="sum",
        choices=["sum", "mean", "median"],
        help="Aggregation mode passed to decoupler.pp.pseudobulk",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=10,
        help="Minimum cells required for a pseudobulk profile",
    )
    parser.add_argument(
        "--min-total-counts",
        type=float,
        default=1000.0,
        help="Minimum summed counts required for a pseudobulk profile",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["auto", "fixed"],
        default="auto",
        help=(
            "Threshold strategy: 'auto' derives condition-specific thresholds "
            "from psbulk_cells and psbulk_counts; 'fixed' uses provided "
            "--min-cells and --min-total-counts directly"
        ),
    )
    parser.add_argument(
        "--auto-quantiles",
        default="0.10,0.20,0.25,0.30",
        help=(
            "Comma-separated quantiles used to build auto-threshold "
            "candidates"
        ),
    )
    parser.add_argument(
        "--auto-min-retained-frac",
        type=float,
        default=0.70,
        help=(
            "Minimum fraction of raw pseudobulk profiles that must remain "
            "after filtering in auto mode"
        ),
    )
    parser.add_argument(
        "--auto-min-profiles-per-celltype",
        type=int,
        default=5,
        help=(
            "Minimum retained pseudobulk profiles per cell type in auto mode"
        ),
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1_000_000.0,
        help="Target library size for CPM-like normalization",
    )
    parser.add_argument(
        "--pca-target-sum",
        type=float,
        default=10_000.0,
        help="Target sum used for pseudobulk PCA exploration",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip decoupler/scanpy QC and PCA plots",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def find_annotation_h5ad(condition_dir: Path) -> Path:
    preferred = condition_dir / f"{condition_dir.name}_annotated.h5ad"
    if preferred.exists():
        return preferred

    frozen = condition_dir / f"{condition_dir.name}_annotated_frozen.h5ad"
    if frozen.exists():
        return frozen

    hits = sorted(condition_dir.glob("*_annotated*.h5ad"))
    if not hits:
        raise FileNotFoundError(
            f"No annotated h5ad found in: {condition_dir}"
        )
    return hits[0]


def list_conditions(in_root: Path) -> list[str]:
    if not in_root.exists():
        return []
    names: list[str] = []
    for child in sorted(p for p in in_root.iterdir() if p.is_dir()):
        try:
            _ = find_annotation_h5ad(child)
        except FileNotFoundError:
            continue
        names.append(child.name)
    return names


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def resolve_conditions(in_root: Path, requested: str) -> list[str]:
    available = list_conditions(in_root)
    if not available:
        raise FileNotFoundError(
            f"No condition folders with annotated files found in: {in_root}"
        )
    if requested.strip().lower() == "all":
        return available
    if requested in available:
        return [requested]

    mapped = safe_name(requested)
    if mapped in available:
        return [mapped]

    near = difflib.get_close_matches(requested, available, n=5, cutoff=0.45)
    hints = [mapped, *near] if mapped != requested else near
    raise FileNotFoundError(
        "Condition not found: "
        f"'{requested}'.\n"
        "Available condition folders: "
        f"{', '.join(available)}\n"
        "Closest matches: "
        f"{', '.join(dict.fromkeys(hints)) if hints else 'none'}"
    )


def _set_x_from_layer(adata: ad.AnnData, layer: str) -> str:
    if layer in adata.layers:
        adata.X = adata.layers[layer].copy()
        return layer
    return "X"


def _to_dense_2d(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray())
    return np.asarray(x)


def _profile_metadata(
    pdata: ad.AnnData,
    condition_name: str,
    sample_col: str,
    group_col: str,
    condition_col: str,
    source_h5ad: Path,
    matrix_layer: str,
) -> pd.DataFrame:
    obs = pdata.obs.copy()
    for col in [sample_col, group_col]:
        if col not in obs.columns:
            raise ValueError(f"Expected pseudobulk obs column missing: {col}")
        obs[col] = obs[col].astype(str)

    if condition_col in obs.columns:
        condition_obs = obs[condition_col].astype(str)
    else:
        condition_obs = pd.Series(
            [condition_name] * pdata.n_obs,
            index=obs.index,
            dtype="string",
        )

    pseudobulk_id = (
        condition_name
        + "__"
        + obs[sample_col].astype(str)
        + "__"
        + obs[group_col].astype(str)
    )

    n_cells = obs.get(
        "psbulk_cells",
        pd.Series([np.nan] * pdata.n_obs, index=obs.index),
    )
    total_counts = obs.get(
        "psbulk_counts",
        pd.Series([np.nan] * pdata.n_obs, index=obs.index),
    )

    meta = pd.DataFrame(
        {
            "pseudobulk_id": pseudobulk_id.values,
            "condition": condition_name,
            "condition_obs": condition_obs.values,
            sample_col: obs[sample_col].values,
            group_col: obs[group_col].values,
            "n_cells": n_cells.values,
            "total_counts": total_counts.values,
            "source_h5ad": str(source_h5ad),
            "matrix_layer": matrix_layer,
        },
        index=obs.index,
    )
    return meta


def _plot_filter_samples(
    pdata_raw: ad.AnnData,
    fig_dir: Path,
    condition: str,
    condition_col: str,
    sample_col: str,
    group_col: str,
    min_cells: int,
    min_total_counts: float,
) -> None:
    dc = _import_decoupler()
    groupby = [
        c
        for c in [condition_col, sample_col, group_col]
        if c in pdata_raw.obs.columns
    ]
    if not groupby:
        groupby = [sample_col, group_col]

    dc.pl.filter_samples(
        adata=pdata_raw,
        groupby=groupby,
        min_cells=min_cells,
        min_counts=min_total_counts,
        figsize=(6, 8),
    )
    plt.savefig(
        fig_dir / f"{condition}_pseudobulk_filter_samples.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_obsbar(
    pdata: ad.AnnData,
    fig_dir: Path,
    condition: str,
    condition_col: str,
    group_col: str,
) -> None:
    dc = _import_decoupler()
    if group_col not in pdata.obs.columns:
        return
    hue = condition_col if condition_col in pdata.obs.columns else None
    dc.pl.obsbar(
        adata=pdata,
        y=group_col,
        hue=hue,
        figsize=(8, 4),
    )
    plt.savefig(
        fig_dir / f"{condition}_pseudobulk_obsbar.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")


def _plot_pca_variability(
    pdata: ad.AnnData,
    fig_dir: Path,
    condition: str,
    pca_target_sum: float,
    condition_col: str,
    sample_col: str,
    group_col: str,
) -> None:
    dc = _import_decoupler()
    if pdata.n_obs < 3:
        return

    counts_backup = _to_dense_2d(pdata.X)
    pdata.layers["counts"] = counts_backup.copy()
    sc.pp.normalize_total(pdata, target_sum=pca_target_sum)
    sc.pp.log1p(pdata)
    sc.pp.scale(pdata, max_value=10)
    sc.tl.pca(pdata)

    sc.pl.pca_variance_ratio(pdata, show=False)
    plt.savefig(
        fig_dir / f"{condition}_pseudobulk_pca_variance_ratio.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    color_vars = [
        c
        for c in [condition_col, sample_col, group_col]
        if c in pdata.obs.columns
    ]
    if color_vars:
        sc.pl.pca(
            pdata,
            color=color_vars,
            ncols=1,
            size=220,
            frameon=True,
            show=False,
        )
        plt.savefig(
            fig_dir / f"{condition}_pseudobulk_pca.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close("all")

    try:
        dc.tl.rankby_obsm(pdata, key="X_pca")
        fig = dc.pl.obsm(
            adata=pdata,
            return_fig=True,
            nvar=5,
            titles=["PC scores", "Adjusted p-values"],
            figsize=(10, 5),
        )
        if hasattr(fig, "savefig"):
            fig.savefig(
                fig_dir / f"{condition}_pseudobulk_pca_association.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.savefig(
                fig_dir / f"{condition}_pseudobulk_pca_association.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")
    except Exception:
        pass

    pdata.X = pdata.layers["counts"].copy()


def _run_qc_and_variability_plots(
    pdata_raw: ad.AnnData,
    pdata_filtered: ad.AnnData,
    fig_dir: Path,
    condition: str,
    sample_col: str,
    group_col: str,
    condition_col: str,
    min_cells: int,
    min_total_counts: float,
    pca_target_sum: float,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_filter_samples(
        pdata_raw=pdata_raw,
        fig_dir=fig_dir,
        condition=condition,
        condition_col=condition_col,
        sample_col=sample_col,
        group_col=group_col,
        min_cells=min_cells,
        min_total_counts=min_total_counts,
    )
    _plot_obsbar(
        pdata=pdata_filtered,
        fig_dir=fig_dir,
        condition=condition,
        condition_col=condition_col,
        group_col=group_col,
    )
    _plot_pca_variability(
        pdata=pdata_filtered,
        fig_dir=fig_dir,
        condition=condition,
        pca_target_sum=pca_target_sum,
        condition_col=condition_col,
        sample_col=sample_col,
        group_col=group_col,
    )


def _counts_table_from_pdata(
    pdata: ad.AnnData,
    pseudobulk_ids: list[str],
) -> pd.DataFrame:
    matrix = _to_dense_2d(pdata.X)
    counts_df = pd.DataFrame(
        matrix.T,
        index=pd.Index(map(str, pdata.var_names), name="gene"),
        columns=pseudobulk_ids,
    )
    counts_df.index.name = "gene"
    return counts_df


def _parse_quantiles(text: str) -> list[float]:
    values: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0.0 or value >= 1.0:
            continue
        values.append(value)
    if not values:
        return [0.10, 0.20, 0.25, 0.30]
    return sorted(set(values))


def _evaluate_threshold_pair(
    obs: pd.DataFrame,
    group_col: str,
    min_cells: int,
    min_counts: float,
) -> dict[str, float | int]:
    mask = (obs["psbulk_cells"] >= min_cells) & (
        obs["psbulk_counts"] >= min_counts
    )
    kept = obs.loc[mask].copy()
    total_profiles = int(obs.shape[0])
    kept_profiles = int(kept.shape[0])
    kept_frac = (kept_profiles / total_profiles) if total_profiles > 0 else 0.0
    total_celltypes = int(obs[group_col].astype(str).nunique())
    kept_celltypes = int(kept[group_col].astype(str).nunique())
    per_ct = kept[group_col].astype(str).value_counts()
    min_per_celltype = int(per_ct.min()) if not per_ct.empty else 0
    median_per_celltype = float(per_ct.median()) if not per_ct.empty else 0.0
    return {
        "min_cells": int(min_cells),
        "min_counts": float(min_counts),
        "kept_profiles": kept_profiles,
        "kept_fraction": float(kept_frac),
        "kept_celltypes": kept_celltypes,
        "total_celltypes": total_celltypes,
        "min_profiles_per_celltype": min_per_celltype,
        "median_profiles_per_celltype": median_per_celltype,
    }


def _select_auto_thresholds(
    pdata_raw: ad.AnnData,
    group_col: str,
    base_min_cells: int,
    base_min_counts: float,
    quantiles: list[float],
    auto_min_retained_frac: float,
    auto_min_profiles_per_celltype: int,
) -> tuple[int, float, pd.DataFrame, str]:
    obs = pdata_raw.obs.copy()
    if "psbulk_cells" not in obs.columns or "psbulk_counts" not in obs.columns:
        raise ValueError(
            "Auto threshold mode requires psbulk_cells and psbulk_counts "
            "in pseudobulk obs"
        )

    obs["psbulk_cells"] = obs["psbulk_cells"].astype(float)
    obs["psbulk_counts"] = obs["psbulk_counts"].astype(float)

    cell_candidates = {int(base_min_cells)}
    count_candidates = {float(base_min_counts)}
    for q in quantiles:
        cell_candidates.add(int(np.floor(obs["psbulk_cells"].quantile(q))))
        count_candidates.add(float(np.floor(obs["psbulk_counts"].quantile(q))))

    cell_candidates = sorted([max(1, c) for c in cell_candidates])
    count_candidates = sorted([max(1.0, c) for c in count_candidates])

    rows: list[dict[str, float | int]] = []
    for c in cell_candidates:
        for n in count_candidates:
            rows.append(
                _evaluate_threshold_pair(
                    obs=obs,
                    group_col=group_col,
                    min_cells=c,
                    min_counts=n,
                )
            )
    sweep = pd.DataFrame(rows)

    valid = sweep[
        (sweep["kept_fraction"] >= auto_min_retained_frac)
        & (sweep["kept_celltypes"] == sweep["total_celltypes"])
        & (
            sweep["min_profiles_per_celltype"]
            >= auto_min_profiles_per_celltype
        )
    ].copy()

    if not valid.empty:
        chosen = valid.sort_values(
            [
                "min_cells",
                "min_counts",
                "kept_profiles",
            ],
            ascending=[False, False, False],
        ).iloc[0]
        reason = "auto_valid_strictest"
        return (
            int(chosen["min_cells"]),
            float(chosen["min_counts"]),
            sweep,
            reason,
        )

    fallback = _evaluate_threshold_pair(
        obs=obs,
        group_col=group_col,
        min_cells=int(base_min_cells),
        min_counts=float(base_min_counts),
    )
    reason = "auto_fallback_to_base"
    return (
        int(fallback["min_cells"]),
        float(fallback["min_counts"]),
        sweep,
        reason,
    )


def _build_condition_pseudobulk(
    h5ad_file: Path,
    condition_name: str,
    sample_col: str,
    group_col: str,
    condition_col: str,
    layer: str,
    mode: str,
    min_cells: int,
    min_total_counts: float,
    threshold_mode: str,
    auto_quantiles: list[float],
    auto_min_retained_frac: float,
    auto_min_profiles_per_celltype: int,
) -> tuple[
    ad.AnnData,
    ad.AnnData,
    pd.DataFrame,
    pd.DataFrame,
    str,
    int,
    float,
    pd.DataFrame,
    str,
]:
    """Build one-condition pseudobulk, mirroring decoupler tutorial steps.

    Steps:
    1) Load annotated single-cell object.
    2) Move raw counts to X (if layer exists).
    3) Aggregate counts with decoupler pseudobulk.
    4) Filter low-quality pseudobulk profiles.
    5) Build export-ready metadata and count matrix.
    """
    dc = _import_decoupler()

    # Step 1: load condition-level annotated single-cell data.
    adata = ad.read_h5ad(h5ad_file)
    required = [sample_col, group_col]
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"Missing required obs columns in {h5ad_file.name}: {missing}"
        )

    # Step 2: ensure X contains the matrix used for aggregation.
    matrix_layer = _set_x_from_layer(adata, layer)

    # Step 3: tutorial core operation (sample x group pseudobulk).
    pdata_raw = dc.pp.pseudobulk(
        adata=adata,
        sample_col=sample_col,
        groups_col=group_col,
        mode=mode,
    )

    # Step 4: choose thresholds from psbulk metrics, then filter.
    if threshold_mode == "auto":
        chosen_cells, chosen_counts, sweep_df, threshold_reason = (
            _select_auto_thresholds(
                pdata_raw=pdata_raw,
                group_col=group_col,
                base_min_cells=min_cells,
                base_min_counts=min_total_counts,
                quantiles=auto_quantiles,
                auto_min_retained_frac=auto_min_retained_frac,
                auto_min_profiles_per_celltype=auto_min_profiles_per_celltype,
            )
        )
    else:
        chosen_cells = int(min_cells)
        chosen_counts = float(min_total_counts)
        sweep_df = pd.DataFrame([
            _evaluate_threshold_pair(
                obs=pdata_raw.obs.copy(),
                group_col=group_col,
                min_cells=chosen_cells,
                min_counts=chosen_counts,
            )
        ])
        threshold_reason = "fixed"

    pdata = pdata_raw.copy()
    dc.pp.filter_samples(
        adata=pdata,
        min_cells=chosen_cells,
        min_counts=chosen_counts,
    )

    if pdata.n_obs == 0:
        empty_counts = pd.DataFrame(
            index=pd.Index(map(str, adata.var_names), name="gene")
        )
        empty_counts.index.name = "gene"
        empty_meta = pd.DataFrame(columns=[
            "pseudobulk_id",
            "condition",
            "condition_obs",
            sample_col,
            group_col,
            "n_cells",
            "total_counts",
            "source_h5ad",
            "matrix_layer",
        ])
        return (
            pdata_raw,
            pdata,
            empty_counts,
            empty_meta,
            matrix_layer,
            chosen_cells,
            chosen_counts,
            sweep_df,
            threshold_reason,
        )

    if condition_col not in pdata.obs.columns:
        pdata.obs[condition_col] = condition_name

    # Step 5: build standardized metadata + expression tables.
    meta_df = _profile_metadata(
        pdata=pdata,
        condition_name=condition_name,
        sample_col=sample_col,
        group_col=group_col,
        condition_col=condition_col,
        source_h5ad=h5ad_file,
        matrix_layer=matrix_layer,
    )

    counts_df = _counts_table_from_pdata(
        pdata=pdata,
        pseudobulk_ids=meta_df["pseudobulk_id"].tolist(),
    )

    return (
        pdata_raw,
        pdata,
        counts_df,
        meta_df,
        matrix_layer,
        chosen_cells,
        chosen_counts,
        sweep_df,
        threshold_reason,
    )


def normalize_logcpm(
    counts_df: pd.DataFrame,
    target_sum: float,
) -> pd.DataFrame:
    if counts_df.empty:
        return counts_df.copy()
    lib_sizes = counts_df.sum(axis=0)
    safe_lib = lib_sizes.replace(0.0, np.nan)
    norm = counts_df.divide(safe_lib, axis=1) * float(target_sum)
    norm = norm.fillna(0.0)
    return np.log1p(norm)


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    auto_quantiles = _parse_quantiles(args.auto_quantiles)

    available = list_conditions(in_root)
    if args.list_conditions:
        if not available:
            print("No conditions found.")
        else:
            print("\n".join(available))
        return

    conditions = resolve_conditions(in_root, args.condition)
    out_root.mkdir(parents=True, exist_ok=True)

    counts_all: list[pd.DataFrame] = []
    meta_all: list[pd.DataFrame] = []

    for condition in conditions:
        # ---------------------------------------------------------------
        # Tutorial-style per-condition pseudobulk workflow:
        #   A) Load + aggregate (sample x cell type)
        #   B) Filter low-quality pseudobulk profiles
        #   C) Plot QC and variability diagnostics
        #   D) Export counts/logCPM/metadata/h5ad
        # ---------------------------------------------------------------
        source_h5ad = find_annotation_h5ad(in_root / condition)
        (
            pdata_raw,
            pdata_filtered,
            counts_df,
            meta_df,
            matrix_layer,
            selected_min_cells,
            selected_min_counts,
            threshold_sweep,
            threshold_reason,
        ) = _build_condition_pseudobulk(
            h5ad_file=source_h5ad,
            condition_name=condition,
            sample_col=args.sample_col,
            group_col=args.group_col,
            condition_col=args.condition_col,
            layer=args.layer,
            mode=args.mode,
            min_cells=args.min_cells,
            min_total_counts=args.min_total_counts,
            threshold_mode=args.threshold_mode,
            auto_quantiles=auto_quantiles,
            auto_min_retained_frac=args.auto_min_retained_frac,
            auto_min_profiles_per_celltype=(
                args.auto_min_profiles_per_celltype
            ),
        )
        if counts_df.empty:
            print(
                f"[{condition}] no pseudobulk profiles passed filters "
                "(selected min_cells="
                f"{selected_min_cells}, selected min_total_counts="
                f"{selected_min_counts})."
            )
            continue

        cond_dir = out_root / condition
        cond_dir.mkdir(parents=True, exist_ok=True)

        counts_file = cond_dir / f"{condition}_pseudobulk_counts.csv"
        meta_file = cond_dir / f"{condition}_pseudobulk_metadata.csv"
        logcpm_file = cond_dir / f"{condition}_pseudobulk_logcpm.csv"
        h5ad_file = cond_dir / f"{condition}_pseudobulk.h5ad"
        sweep_file = cond_dir / f"{condition}_threshold_sweep.csv"
        selected_file = cond_dir / f"{condition}_threshold_selected.csv"

        counts_df.to_csv(counts_file)
        meta_df.to_csv(meta_file, index=False)
        normalize_logcpm(counts_df, args.target_sum).to_csv(logcpm_file)
        threshold_sweep.to_csv(sweep_file, index=False)

        selected_df = pd.DataFrame(
            [
                {
                    "condition": condition,
                    "threshold_mode": args.threshold_mode,
                    "threshold_reason": threshold_reason,
                    "selected_min_cells": selected_min_cells,
                    "selected_min_total_counts": selected_min_counts,
                    "profiles_retained": int(meta_df.shape[0]),
                    "celltypes_retained": int(
                        meta_df[args.group_col].astype(str).nunique()
                    ),
                }
            ]
        )
        selected_df.to_csv(selected_file, index=False)

        if pdata_filtered.n_obs > 0:
            pdata_filtered.obs = pdata_filtered.obs.copy()
            pdata_filtered.obs["pseudobulk_id"] = meta_df[
                "pseudobulk_id"
            ].values
            pdata_filtered.write_h5ad(h5ad_file)

        if not args.skip_plots and pdata_filtered.n_obs > 0:
            _run_qc_and_variability_plots(
                pdata_raw=pdata_raw,
                pdata_filtered=pdata_filtered,
                fig_dir=cond_dir / "figures",
                condition=condition,
                sample_col=args.sample_col,
                group_col=args.group_col,
                condition_col=args.condition_col,
                min_cells=selected_min_cells,
                min_total_counts=selected_min_counts,
                pca_target_sum=args.pca_target_sum,
            )

        counts_all.append(counts_df)
        meta_all.append(meta_df)

        print(
            f"[{condition}] thresholds: min_cells={selected_min_cells}, "
            f"min_total_counts={selected_min_counts} ({threshold_reason}); "
            f"wrote {meta_df.shape[0]} pseudobulk profiles; "
            f"{counts_df.shape[0]} genes"
        )

    if counts_all:
        combined_counts = pd.concat(counts_all, axis=1)
        combined_counts = combined_counts.fillna(0.0)
        combined_meta = pd.concat(meta_all, axis=0, ignore_index=True)

        is_all_run = args.condition.strip().lower() == "all"
        if is_all_run:
            prefix = "all_conditions"
        else:
            prefix = f"run_scope_{safe_name(args.condition)}"

        combined_counts_file = (
            out_root / f"{prefix}_pseudobulk_counts.csv"
        )
        combined_meta_file = (
            out_root / f"{prefix}_pseudobulk_metadata.csv"
        )
        combined_logcpm_file = (
            out_root / f"{prefix}_pseudobulk_logcpm.csv"
        )

        combined_counts.to_csv(combined_counts_file)
        combined_meta.to_csv(combined_meta_file, index=False)
        normalize_logcpm(combined_counts, args.target_sum).to_csv(
            combined_logcpm_file
        )

        run_summary = pd.DataFrame(
            {
                "condition": combined_meta["condition"].value_counts().index,
                "n_pseudobulk_profiles": combined_meta["condition"]
                .value_counts()
                .values,
            }
        )
        run_summary["threshold_mode"] = args.threshold_mode
        if args.threshold_mode == "auto":
            run_summary["auto_quantiles"] = ",".join(
                f"{q:.2f}" for q in auto_quantiles
            )
            run_summary["auto_min_retained_frac"] = (
                args.auto_min_retained_frac
            )
            run_summary["auto_min_profiles_per_celltype"] = (
                args.auto_min_profiles_per_celltype
            )
        summary_file = out_root / f"{prefix}_pseudobulk_summary.csv"
        run_summary.to_csv(summary_file, index=False)

        # Keep backward compatibility for users expecting the old summary name.
        if is_all_run:
            run_summary.to_csv(
                out_root / "pseudobulk_summary.csv",
                index=False,
            )

        stage_dir = out_root.parent
        script_rel = str(Path(__file__).resolve().relative_to(REPO_ROOT))
        record = WarehouseRecord(
            input_file=str(in_root),
            output_file=str(combined_counts_file),
            script=script_rel,
            date_utc=utc_now_iso(),
            params_hash=params_hash(vars(args)),
            condition=args.condition,
            stage="07_network_pseudobulk",
        )
        append_warehouse(stage_dir, [record])

        print(
            f"Wrote combined outputs ({prefix}) to: {out_root}"
        )
    else:
        print(
            "No outputs generated. "
            "Check filters and input annotation files."
        )


if __name__ == "__main__":
    main()
