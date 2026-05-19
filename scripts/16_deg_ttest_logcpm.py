#!/usr/bin/env python3
"""Run thesis-style DEG analysis on pseudobulk logCPM matrices.

For each contrast (case vs control), this script computes:
1) mean expression per group,
2) log2 fold-change as mean(case) - mean(control),
3) independent t-test per gene,
4) Benjamini-Hochberg adjusted p-values (FDR),
5) DEG calls by FDR/log2FC thresholds.

It also writes per-condition DE allowlists compatible with post-correlation
network filtering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_ind

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

BRCA1_CASE = "Triple_negative_BRCA1_tumor"
BRCA1_CONTROL = "Normal_BRCA1_-_pre-neoplastic"

FULL_PSEUDOBULK_CSV = REPO_ROOT / "results/06_pseudobulk/all_conditions_pseudobulk_logcpm.csv"
FULL_PSEUDOBULK_META = REPO_ROOT / "results/06_pseudobulk/all_conditions_pseudobulk_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEG analysis from per-condition pseudobulk logCPM files"
    )
    parser.add_argument(
        "--input-dir",
        default="results/08_pre_correlation/per_condition",
        help="Directory containing *_pseudobulk_logcpm.h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/10_deg_ttest",
        help="Output root for contrast DEG tables and summaries",
    )
    parser.add_argument(
        "--use-full-pseudobulk",
        action="store_true",
        help=(
            "Read from the full pseudobulk CSV (06_pseudobulk/) instead of "
            "HVG-filtered h5ads. Covers all network genes, not just HVGs."
        ),
    )
    parser.add_argument(
        "--control-condition",
        default="Normal",
        help="Control condition for automatic case-vs-control contrasts",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="Custom contrast as 'case:control'. Repeat for multiple.",
    )
    parser.add_argument(
        "--include-brca1-pair",
        action="store_true",
        help=(
            "Include Triple_negative_BRCA1_tumor vs "
            "Normal_BRCA1_-_pre-neoplastic"
        ),
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold for DEG calls",
    )
    parser.add_argument(
        "--min-abs-log2fc",
        type=float,
        default=0.2,
        help="Minimum absolute log2 fold-change for DEG calls",
    )
    parser.add_argument(
        "--equal-var",
        action="store_true",
        help="Use Student t-test assumption of equal variance",
    )
    parser.add_argument(
        "--nan-policy",
        choices=["omit", "propagate", "raise"],
        default="omit",
        help="NaN handling policy in scipy.stats.ttest_ind",
    )
    parser.add_argument(
        "--allowlist-dir",
        default=(
            "results/10_deg_ttest/"
            "de_gene_filters"
        ),
        help=(
            "Output root for allowlists; writes per_condition and summary "
            "files for network filtering"
        ),
    )
    parser.add_argument(
        "--min-detected",
        type=int,
        default=10,
        help=(
            "Minimum number of non-NaN profiles required in EACH group for a "
            "gene to be tested. Applies only when --use-full-pseudobulk is set. "
            "Prevents spurious significance from sparsely detected genes."
        ),
    )
    return parser.parse_args()


def load_full_pseudobulk() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    #load full-gene pseudobulk csv; returns genes x profiles df and condition->cols map
    expr = pd.read_csv(FULL_PSEUDOBULK_CSV, index_col=0)  # genes × profiles
    meta = pd.read_csv(FULL_PSEUDOBULK_META)

    col_to_cond = meta.set_index("pseudobulk_id")["condition"].to_dict()
    cond_to_cols: dict[str, list[str]] = {}
    for col in expr.columns:
        cond = col_to_cond.get(col)
        if cond is not None:
            cond_to_cols.setdefault(cond, []).append(col)

    return expr, cond_to_cols


def run_one_contrast_from_csv(
    case_name: str,
    control_name: str,
    expr_df: pd.DataFrame,
    case_cols: list[str],
    ctrl_cols: list[str],
    out_root: Path,
    equal_var: bool,
    nan_policy: str,
    fdr_threshold: float,
    min_abs_log2fc: float,
    min_detected: int = 10,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    case_x = expr_df[case_cols].T.to_numpy(dtype=np.float64)  # profiles × genes
    ctrl_x = expr_df[ctrl_cols].T.to_numpy(dtype=np.float64)

    #filter genes with fewer than min_detected non-nan profiles per group
    case_detected = np.sum(~np.isnan(case_x), axis=0) >= min_detected
    ctrl_detected = np.sum(~np.isnan(ctrl_x), axis=0) >= min_detected
    keep = case_detected & ctrl_detected
    genes = np.array(expr_df.index.astype(str))[keep]
    case_x = case_x[:, keep]
    ctrl_x = ctrl_x[:, keep]

    mean_case = np.nanmean(case_x, axis=0)
    mean_ctrl = np.nanmean(ctrl_x, axis=0)
    log2fc = mean_case - mean_ctrl

    ttest_res = ttest_ind(
        case_x, ctrl_x, axis=0,
        equal_var=bool(equal_var), nan_policy=nan_policy,
    )
    t_stat = np.asarray(getattr(ttest_res, "statistic", ttest_res[0]), dtype=np.float64)
    p_vals = np.asarray(getattr(ttest_res, "pvalue", ttest_res[1]), dtype=np.float64)
    padj = bh_fdr(p_vals)

    res = pd.DataFrame({
        "gene": genes,
        "mean_control_expr": mean_ctrl,
        "mean_case_expr": mean_case,
        "lnFC": log2fc,
        "t_statistic": t_stat,
        "raw_pvalue": p_vals,
        "fdr": padj,
    })
    res["abs_lnFC"] = np.abs(res["lnFC"].to_numpy())
    res["is_significant_fdr"] = res["fdr"].le(float(fdr_threshold))
    res["is_deg"] = res["is_significant_fdr"] & res["abs_lnFC"].ge(float(min_abs_log2fc))
    res["direction"] = np.where(
        ~res["is_deg"], "not_deg",
        np.where(res["lnFC"] >= 0.0, "up", "down"),
    )
    res = res.sort_values("lnFC", ascending=False).reset_index(drop=True)

    contrast_name = f"{case_name}.vs.{control_name}"
    out_dir = out_root / contrast_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_file = out_dir / f"{contrast_name}_deg_stats.csv"
    res.to_csv(stats_file, index=False)

    up = res[(res["is_deg"]) & (res["lnFC"] >= 0.0)]["gene"].tolist()
    down = res[(res["is_deg"]) & (res["lnFC"] < 0.0)]["gene"].tolist()
    (out_dir / "up_degs.txt").write_text("\n".join(up) + ("\n" if up else ""), encoding="utf-8")
    (out_dir / "down_degs.txt").write_text("\n".join(down) + ("\n" if down else ""), encoding="utf-8")

    summary = {
        "contrast": contrast_name,
        "case": case_name,
        "control": control_name,
        "n_case_profiles": len(case_cols),
        "n_control_profiles": len(ctrl_cols),
        "n_genes_tested": int(keep.sum()),
        "n_genes_filtered_sparse": int((~keep).sum()),
        "n_deg_total": int(res["is_deg"].sum()),
        "n_up_deg": int(len(up)),
        "n_down_deg": int(len(down)),
        "fdr_threshold": float(fdr_threshold),
        "min_abs_log2fc": float(min_abs_log2fc),
        "min_detected": int(min_detected),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return res, summary, stats_file


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def condition_name_from_file(h5ad_file: Path) -> str:
    suffix = "_pseudobulk_logcpm.h5ad"
    name = h5ad_file.name
    return name[:-len(suffix)] if name.endswith(suffix) else h5ad_file.stem


def list_condition_files(in_root: Path) -> list[Path]:
    if not in_root.exists():
        return []
    return sorted(in_root.glob("*_pseudobulk_logcpm.h5ad"))


def to_dense_2d(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def parse_contrasts(
    all_conditions: list[str],
    user_contrasts: list[str],
    control: str,
    include_brca1_pair: bool,
) -> list[tuple[str, str]]:
    contrasts: list[tuple[str, str]] = []

    if user_contrasts:
        for text in user_contrasts:
            if ":" not in text:
                raise ValueError(
                    f"Invalid contrast '{text}'. Use format 'case:control'."
                )
            case, ctrl = [x.strip() for x in text.split(":", 1)]
            contrasts.append((case, ctrl))
    else:
        for cond in all_conditions:
            if cond == control:
                continue
            contrasts.append((cond, control))

    if include_brca1_pair:
        pair = (BRCA1_CASE, BRCA1_CONTROL)
        if pair[0] in all_conditions and pair[1] in all_conditions:
            if pair not in contrasts:
                contrasts.append(pair)

    valid: list[tuple[str, str]] = []
    for case, ctrl in contrasts:
        if case not in all_conditions or ctrl not in all_conditions:
            print(
                "[warn] skipping contrast "
                f"{case}:{ctrl} (missing in pseudobulk conditions)"
            )
            continue
        valid.append((case, ctrl))

    dedup = list(dict.fromkeys(valid))
    if not dedup:
        raise ValueError("No valid contrasts available to run.")
    return dedup


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan, dtype=np.float64)

    valid = np.isfinite(p)
    if not np.any(valid):
        return out

    pv = p[valid]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    scale = n / np.arange(1, n + 1, dtype=np.float64)
    q = ranked * scale
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    q_back = np.empty_like(q)
    q_back[order] = q
    out[valid] = q_back
    return out


def run_one_contrast(
    case_name: str,
    control_name: str,
    case_file: Path,
    control_file: Path,
    out_root: Path,
    equal_var: bool,
    nan_policy: str,
    fdr_threshold: float,
    min_abs_log2fc: float,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    #load expression data for case and control
    case_ad = ad.read_h5ad(case_file)
    ctrl_ad = ad.read_h5ad(control_file)

    #align to shared gene set
    case_genes = case_ad.var_names.astype(str)
    ctrl_genes = ctrl_ad.var_names.astype(str)
    shared = pd.Index(case_genes).intersection(pd.Index(ctrl_genes))
    if shared.size < 2:
        raise ValueError(f"{case_name}.vs.{control_name}: fewer than 2 shared genes")

    case_idx = pd.Index(case_genes).get_indexer(shared)
    ctrl_idx = pd.Index(ctrl_genes).get_indexer(shared)

    #build dense profiles x genes matrices
    case_x = to_dense_2d(case_ad.X)[:, case_idx].astype(np.float64)
    ctrl_x = to_dense_2d(ctrl_ad.X)[:, ctrl_idx].astype(np.float64)

    #compute mean expression and lnFC (natural log; column stored as lnFC)
    mean_case = np.nanmean(case_x, axis=0)
    mean_ctrl = np.nanmean(ctrl_x, axis=0)
    log2fc = mean_case - mean_ctrl  # lnFC; variable name kept for readability

    #run per-gene welch t-test
    ttest_res = ttest_ind(case_x, ctrl_x, axis=0, equal_var=bool(equal_var), nan_policy=nan_policy)
    t_stat = np.asarray(getattr(ttest_res, "statistic", ttest_res[0]), dtype=np.float64)
    p_vals = np.asarray(getattr(ttest_res, "pvalue", ttest_res[1]), dtype=np.float64)

    #apply benjamini-hochberg fdr correction
    padj = bh_fdr(p_vals)

    #assemble per-gene deg table
    res = pd.DataFrame(
        {
            "gene": shared.astype(str),
            "mean_control_expr": mean_ctrl,
            "mean_case_expr": mean_case,
            "lnFC": log2fc,
            "t_statistic": t_stat,
            "raw_pvalue": p_vals,
            "fdr": padj,
        }
    )
    res["abs_lnFC"] = np.abs(res["lnFC"].to_numpy())
    res["is_significant_fdr"] = res["fdr"].le(float(fdr_threshold))
    res["is_deg"] = (
        res["is_significant_fdr"]
        & res["abs_lnFC"].ge(float(min_abs_log2fc))
    )
    res["direction"] = np.where(
        ~res["is_deg"],
        "not_deg",
        np.where(res["lnFC"] >= 0.0, "up", "down"),
    )
    res = res.sort_values("lnFC", ascending=False).reset_index(drop=True)

    #write deg stats table and up/down gene lists
    contrast_name = f"{case_name}.vs.{control_name}"
    out_dir = out_root / contrast_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_file = out_dir / f"{contrast_name}_deg_stats.csv"
    res.to_csv(stats_file, index=False)

    up = res[(res["is_deg"]) & (res["lnFC"] >= 0.0)]["gene"].tolist()
    down = res[(res["is_deg"]) & (res["lnFC"] < 0.0)]["gene"].tolist()

    (out_dir / "up_degs.txt").write_text(
        "\n".join(up) + ("\n" if up else ""),
        encoding="utf-8",
    )
    (out_dir / "down_degs.txt").write_text(
        "\n".join(down) + ("\n" if down else ""),
        encoding="utf-8",
    )

    #save compact run summary json
    summary = {
        "contrast": contrast_name,
        "case": case_name,
        "control": control_name,
        "n_case_profiles": int(case_x.shape[0]),
        "n_control_profiles": int(ctrl_x.shape[0]),
        "n_shared_genes": int(shared.size),
        "n_deg_total": int(res["is_deg"].sum()),
        "n_up_deg": int(len(up)),
        "n_down_deg": int(len(down)),
        "fdr_threshold": float(fdr_threshold),
        "min_abs_log2fc": float(min_abs_log2fc),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return res, summary, stats_file


def build_condition_allowlists(
    contrast_results: list[pd.DataFrame],
    contrast_names: list[str],
    out_root: Path,
) -> None:
    #build condition-wise allowlists from all contrast deg calls
    cond_dir = out_root / "per_condition"
    cond_dir.mkdir(parents=True, exist_ok=True)

    by_condition: dict[str, list[pd.DataFrame]] = {}

    for df, cname in zip(contrast_results, contrast_names):
        #add deg table to both case and control condition bins
        case, control = cname.split(".vs.", 1)
        deg_df = df[df["is_deg"]].copy()
        if deg_df.empty:
            continue
        use_cols = ["gene", "lnFC", "fdr"]
        by_condition.setdefault(case, []).append(deg_df[use_cols].copy())
        by_condition.setdefault(control, []).append(deg_df[use_cols].copy())

    summary_rows: list[dict[str, object]] = []
    for condition, frames in sorted(by_condition.items()):
        #collapse repeated hits across contrasts into one allowlist row per gene
        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged["abs_lnFC"] = np.abs(merged["lnFC"].to_numpy())
        agg = (
            merged.groupby("gene", as_index=False)
            .agg(
                hit_count=("gene", "size"),
                max_abs_lnfc=("abs_lnFC", "max"),
                min_padj=("fdr", "min"),
            )
            .sort_values(
                ["hit_count", "max_abs_lnfc", "min_padj"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

        out_file = cond_dir / f"{condition}_de_gene_allowlist.csv"
        agg.to_csv(out_file, index=False)
        summary_rows.append(
            {
                "condition": condition,
                "n_genes_allowlisted": int(agg.shape[0]),
                "allowlist_file": str(out_file),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "de_gene_filter_summary.csv", index=False)


def main() -> int:
    args = parse_args()

    #resolve output paths
    out_root = resolve_base(args.output_dir)
    allowlist_root = resolve_base(args.allowlist_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    allowlist_root.mkdir(parents=True, exist_ok=True)

    contrast_rows: list[dict[str, object]] = []
    contrast_results: list[pd.DataFrame] = []
    contrast_names: list[str] = []
    warehouse_rows: list[WarehouseRecord] = []

    if args.use_full_pseudobulk:
        #full-pseudobulk path: all genes, no hvg filter
        print(f"[mode] full pseudobulk CSV  →  {FULL_PSEUDOBULK_CSV.name}")
        expr_df, cond_to_cols = load_full_pseudobulk()
        all_conditions = sorted(cond_to_cols.keys())

        contrasts = parse_contrasts(
            all_conditions=all_conditions,
            user_contrasts=args.contrast,
            control=args.control_condition,
            include_brca1_pair=args.include_brca1_pair,
        )

        #run deg stats per contrast from full csv
        for case, control in contrasts:
            res, summary, stats_file = run_one_contrast_from_csv(
                case_name=case, control_name=control, expr_df=expr_df,
                case_cols=cond_to_cols[case], ctrl_cols=cond_to_cols[control],
                out_root=out_root, equal_var=args.equal_var, nan_policy=args.nan_policy,
                fdr_threshold=float(args.fdr_threshold), min_abs_log2fc=float(args.min_abs_log2fc),
                min_detected=int(args.min_detected),
            )
            contrast_rows.append(summary)
            contrast_results.append(res)
            contrast_names.append(str(summary["contrast"]))
            warehouse_rows.append(WarehouseRecord(
                input_file=str(FULL_PSEUDOBULK_CSV),
                output_file=str(stats_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=f"{case}.vs.{control}",
                stage="deg_ttest_full",
            ))
            print(
                f"[{case}.vs.{control}] n_genes_tested={summary['n_genes_tested']} "
                f"filtered_sparse={summary['n_genes_filtered_sparse']} "
                f"deg={summary['n_deg_total']}"
            )

    else:
        #per-condition path: hvg-filtered h5ads from pre-correlation stage
        in_root = resolve_base(args.input_dir)
        files = list_condition_files(in_root)
        if not files:
            raise FileNotFoundError(f"No *_pseudobulk_logcpm.h5ad files found in {in_root}")

        file_by_condition = {condition_name_from_file(f): f for f in files}
        all_conditions = sorted(file_by_condition.keys())

        contrasts = parse_contrasts(
            all_conditions=all_conditions,
            user_contrasts=args.contrast,
            control=args.control_condition,
            include_brca1_pair=args.include_brca1_pair,
        )

        #run deg stats per contrast from h5ads
        for case, control in contrasts:
            case_file = file_by_condition[case]
            control_file = file_by_condition[control]

            res, summary, stats_file = run_one_contrast(
                case_name=case, control_name=control, case_file=case_file,
                control_file=control_file, out_root=out_root, equal_var=args.equal_var,
                nan_policy=args.nan_policy, fdr_threshold=float(args.fdr_threshold),
                min_abs_log2fc=float(args.min_abs_log2fc),
            )
            contrast_rows.append(summary)
            contrast_results.append(res)
            contrast_names.append(str(summary["contrast"]))
            warehouse_rows.append(WarehouseRecord(
                input_file=f"{case_file};{control_file}",
                output_file=str(stats_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=f"{case}.vs.{control}",
                stage="deg_ttest",
            ))
            print(f"[{case}.vs.{control}] shared_genes={summary['n_shared_genes']} deg={summary['n_deg_total']}")

    #write global contrast summary and per-condition allowlists
    pd.DataFrame(contrast_rows).to_csv(out_root / "deg_contrast_summary.csv", index=False)
    build_condition_allowlists(contrast_results=contrast_results, contrast_names=contrast_names, out_root=allowlist_root)

    #save run config json and warehouse provenance
    config = {
        "fdr_threshold": float(args.fdr_threshold),
        "min_abs_log2fc": float(args.min_abs_log2fc),
        "equal_var": bool(args.equal_var),
        "nan_policy": str(args.nan_policy),
        "control_condition": str(args.control_condition),
        "include_brca1_pair": bool(args.include_brca1_pair),
        "contrasts": [f"{case}:{control}" for case, control in contrasts],
        "summary_file": str(allowlist_root / "de_gene_filter_summary.csv"),
    }
    (allowlist_root / "de_gene_filter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    append_warehouse(out_root, warehouse_rows)

    print(f"Done. DEG outputs: {out_root}")
    print(f"Done. DE allowlists: {allowlist_root / 'per_condition'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
