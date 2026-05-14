#!/usr/bin/env python3
"""Tutorial-style downstream analysis on pseudobulk outputs.

This script is intentionally separate from `07_pseudobulk.py`.
It performs the next steps shown in the decoupler pseudobulk tutorial:

1) PCA variance and metadata association plots on pseudobulk profiles.
2) Per-cell-type feature filtering diagnostics:
   - decoupler.pl.filter_by_expr
   - decoupler.pl.filter_by_prop
   and corresponding filtering with decoupler.pp.*
3) Differential expression (DESeq2 / pydeseq2) for requested contrasts.
4) Volcano plot and stat matrix export.
5) Optional enrichment scoring with decoupler.mt.ulm on:
   - CollecTRI (TF activity)
   - PROGENy (pathway activity)
   - Hallmark (pathway activity)

Default contrast logic:
- compare each non-control condition vs control (control=Normal)
- add Normal_BRCA1_-_pre-neoplastic vs Triple_negative_BRCA1_tumor
"""

from __future__ import annotations

import argparse
import importlib
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
            "decoupler is required. Install with: pip install decoupler"
        ) from exc


def _import_pydeseq2():
    try:
        dds_mod = importlib.import_module("pydeseq2.dds")
        ds_mod = importlib.import_module("pydeseq2.ds")
        return (
            dds_mod.DeseqDataSet,
            dds_mod.DefaultInference,
            ds_mod.DeseqStats,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pydeseq2 is required for DE. Install with: pip install pydeseq2"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tutorial-style DE and enrichment on pseudobulk data"
    )
    parser.add_argument(
        "--input-dir",
        default="results/06_pseudobulk",
        help="Pseudobulk directory containing per-condition *_pseudobulk.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_network/06_downstream_decoupler",
        help="Output folder for DE/enrichment results",
    )
    parser.add_argument(
        "--condition-col",
        default="condition",
        help="Condition column in pseudobulk obs",
    )
    parser.add_argument(
        "--sample-col",
        default="SampleName",
        help="Sample column in pseudobulk obs",
    )
    parser.add_argument(
        "--group-col",
        default="cell_type_annot",
        help="Cell type/group column in pseudobulk obs",
    )
    parser.add_argument(
        "--control-condition",
        default="Normal",
        help="Control condition for automatic contrasts",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        help=(
            "Custom contrast as 'case:control'. Repeat flag for multiple "
            "contrasts. If omitted, defaults are generated automatically."
        ),
    )
    parser.add_argument(
        "--include-brca1-pair",
        action="store_true",
        help=(
            "Include Normal_BRCA1_-_pre-neoplastic vs "
            "Triple_negative_BRCA1_tumor if both exist"
        ),
    )
    parser.add_argument(
        "--min-replicates",
        type=int,
        default=2,
        help="Minimum pseudobulk profiles per condition for DE",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="filter_by_expr: min_count",
    )
    parser.add_argument(
        "--min-total-count",
        type=int,
        default=15,
        help="filter_by_expr: min_total_count",
    )
    parser.add_argument(
        "--large-n",
        type=int,
        default=10,
        help="filter_by_expr: large_n",
    )
    parser.add_argument(
        "--min-prop",
        type=float,
        default=0.1,
        help="filter_by_prop: min_prop",
    )
    parser.add_argument(
        "--min-smpls",
        type=int,
        default=2,
        help="filter_by_prop: min_smpls",
    )
    parser.add_argument(
        "--expr-min-prop",
        type=float,
        default=0.7,
        help="filter_by_expr: min_prop",
    )
    parser.add_argument(
        "--pvalue-threshold",
        type=float,
        default=0.05,
        help="Significance threshold for enrichment filtering",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=8,
        help="CPUs for pydeseq2 inference",
    )
    parser.add_argument(
        "--organism",
        default="human",
        help="Organism for decoupler resources (collectri/progeny/hallmark)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip decoupler ULM enrichment section",
    )
    parser.add_argument(
        "--skip-de",
        action="store_true",
        help="Skip DESeq2 and only run filtering/diagnostics",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return repo_candidate


def _to_dense_2d(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray())
    return np.asarray(x)


def list_condition_dirs(pb_root: Path) -> list[Path]:
    if not pb_root.exists():
        return []
    return sorted([p for p in pb_root.iterdir() if p.is_dir()])


def load_pseudobulk_h5ads(pb_root: Path) -> ad.AnnData:
    condition_dirs = list_condition_dirs(pb_root)
    adatas: list[ad.AnnData] = []

    for cdir in condition_dirs:
        h5 = cdir / f"{cdir.name}_pseudobulk.h5ad"
        if not h5.exists():
            continue
        pdata = ad.read_h5ad(h5)
        if "pseudobulk_id" not in pdata.obs.columns:
            pdata.obs = pdata.obs.copy()
            pdata.obs["pseudobulk_id"] = pdata.obs_names.astype(str)
        if "condition" not in pdata.obs.columns:
            pdata.obs["condition"] = cdir.name
        adatas.append(pdata)

    if not adatas:
        raise FileNotFoundError(
            "No per-condition pseudobulk h5ad files found. "
            "Run scripts/18_pseudobulk.py first."
        )

    merged = ad.concat(
        adatas,
        join="inner",
        merge="same",
        index_unique=None,
    )
    return merged


def run_global_pca_plots(
    pdata: ad.AnnData,
    out_dir: Path,
    condition_col: str,
    sample_col: str,
    group_col: str,
) -> None:
    """Generate tutorial-like PCA variance and metadata association plots."""
    dc = _import_decoupler()
    fig_dir = out_dir / "global_pca"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pdata_plot = pdata.copy()
    pdata_plot.layers["counts"] = _to_dense_2d(pdata_plot.X)

    sc.pp.normalize_total(pdata_plot, target_sum=1e4)
    sc.pp.log1p(pdata_plot)
    sc.pp.scale(pdata_plot, max_value=10)
    sc.tl.pca(pdata_plot)

    sc.pl.pca_variance_ratio(pdata_plot, show=False)
    plt.savefig(
        fig_dir / "global_pca_variance_ratio.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    color_vars = [
        c
        for c in [condition_col, sample_col, group_col]
        if c in pdata_plot.obs.columns
    ]
    if color_vars:
        sc.pl.pca(
            pdata_plot,
            color=color_vars,
            ncols=1,
            size=220,
            frameon=True,
            show=False,
        )
        plt.savefig(
            fig_dir / "global_pca_colored_metadata.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close("all")

    try:
        assoc_vars: list[str] = []

        if condition_col in pdata_plot.obs.columns:
            assoc_vars.append(condition_col)
        if group_col in pdata_plot.obs.columns:
            assoc_vars.append(group_col)

        if sample_col in pdata_plot.obs.columns:
            n_samples = int(pdata_plot.obs[sample_col].astype(str).nunique())
            if n_samples <= 15:
                assoc_vars.append(sample_col)
            else:
                print(
                    "[info] Skipping sample-level PCA association plot "
                    f"because {sample_col} has {n_samples} unique values "
                    "(too many to visualize clearly)."
                )

        if not assoc_vars:
            raise ValueError(
                "No suitable metadata variables for PCA association plotting."
            )

        pdata_assoc = pdata_plot.copy()
        pdata_assoc.obs = pdata_assoc.obs[assoc_vars].copy()

        dc.tl.rankby_obsm(pdata_assoc, key="X_pca")
        fig = dc.pl.obsm(
            adata=pdata_assoc,
            return_fig=True,
            nvar=min(10, len(assoc_vars)),
            titles=["PC scores", "Adjusted p-values"],
            figsize=(max(10, 3 + 2.0 * len(assoc_vars)), 6),
        )
        if hasattr(fig, "savefig"):
            fig.savefig(
                fig_dir / "global_pca_metadata_association.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.savefig(
                fig_dir / "global_pca_metadata_association.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")
    except Exception as exc:
        print(f"[warn] rankby_obsm plot skipped: {exc}")


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
        pair = (
            "Triple_negative_BRCA1_tumor",
            "Normal_BRCA1_-_pre-neoplastic",
        )
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


def _round_to_int_counts(adata_obj: ad.AnnData) -> None:
    x = _to_dense_2d(adata_obj.X)
    x = np.rint(np.clip(x, a_min=0, a_max=None)).astype(np.int64)
    adata_obj.X = x


def _filter_diagnostics_and_apply(
    tdata: ad.AnnData,
    out_dir: Path,
    condition_col: str,
    min_count: int,
    min_total_count: int,
    large_n: int,
    expr_min_prop: float,
    min_prop: float,
    min_smpls: int,
) -> ad.AnnData:
    dc = _import_decoupler()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dc.pl.filter_by_expr(
        adata=tdata,
        group=condition_col,
        min_count=min_count,
        min_total_count=min_total_count,
        large_n=large_n,
        min_prop=expr_min_prop,
    )
    plt.savefig(
        fig_dir / "filter_by_expr_diagnostic.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    dc.pl.filter_by_prop(
        adata=tdata,
        min_prop=min_prop,
        min_smpls=min_smpls,
    )
    plt.savefig(
        fig_dir / "filter_by_prop_diagnostic.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close("all")

    before_genes = int(tdata.n_vars)

    dc.pp.filter_by_expr(
        adata=tdata,
        group=condition_col,
        min_count=min_count,
        min_total_count=min_total_count,
        large_n=large_n,
        min_prop=expr_min_prop,
    )
    dc.pp.filter_by_prop(
        adata=tdata,
        min_prop=min_prop,
        min_smpls=min_smpls,
    )

    after_genes = int(tdata.n_vars)
    summary = pd.DataFrame(
        [
            {
                "n_samples": int(tdata.n_obs),
                "genes_before": before_genes,
                "genes_after": after_genes,
                "genes_removed": before_genes - after_genes,
                "min_count": min_count,
                "min_total_count": min_total_count,
                "large_n": large_n,
                "expr_min_prop": expr_min_prop,
                "min_prop": min_prop,
                "min_smpls": min_smpls,
            }
        ]
    )
    summary.to_csv(out_dir / "feature_filtering_summary.csv", index=False)
    return tdata


def run_deseq2(
    tdata: ad.AnnData,
    condition_col: str,
    case: str,
    control: str,
    n_cpus: int,
) -> pd.DataFrame:
    DeseqDataSet, DefaultInference, DeseqStats = _import_pydeseq2()

    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        adata=tdata,
        design_factors=[condition_col],
        refit_cooks=True,
        inference=inference,
    )

    dds.deseq2()
    stat_res = DeseqStats(
        dds,
        contrast=[condition_col, case, control],
        inference=inference,
    )
    stat_res.summary()
    return stat_res.results_df.copy()


def _safe_top_sources(score_df: pd.DataFrame, top_n: int = 4) -> list[str]:
    if score_df.empty:
        return []
    col = score_df.columns[0]
    ranked = score_df[col].sort_values(ascending=False)
    top = ranked.head(max(1, top_n // 2)).index.tolist()
    bottom = ranked.tail(max(1, top_n // 2)).index.tolist()
    names = top + bottom
    return list(dict.fromkeys([str(x) for x in names]))


def run_enrichment_plots(
    results_df: pd.DataFrame,
    contrast_name: str,
    out_dir: Path,
    pvalue_threshold: float,
    organism: str,
) -> None:
    dc = _import_decoupler()

    enr_dir = out_dir / "enrichment"
    fig_dir = enr_dir / "figures"
    enr_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if "stat" not in results_df.columns:
        print("[warn] stat column missing in DE results; skipping enrichment")
        return

    data = results_df[["stat"]].T.rename(index={"stat": contrast_name})
    data.to_csv(enr_dir / "contrast_stat_matrix.csv")

    try:
        collectri = dc.op.collectri(organism=organism)
        tf_acts, tf_padj = dc.mt.ulm(data=data, net=collectri)
        tf_acts.to_csv(enr_dir / "collectri_tf_acts_all.csv")
        tf_padj.to_csv(enr_dir / "collectri_tf_padj_all.csv")

        msk = (tf_padj.T < pvalue_threshold).iloc[:, 0]
        tf_sig = tf_acts.loc[:, msk]
        tf_sig.to_csv(enr_dir / "collectri_tf_acts_significant.csv")

        if not tf_sig.empty:
            dc.pl.barplot(
                data=tf_sig,
                name=contrast_name,
                figsize=(5, 3),
            )
            plt.savefig(
                fig_dir / "collectri_tf_barplot.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")

            sources = _safe_top_sources(tf_sig, top_n=4)
            if sources:
                try:
                    dc.pl.network(
                        net=collectri,
                        data=data,
                        score=tf_sig,
                        sources=sources,
                        targets=5,
                        figsize=(6, 6),
                        vcenter=True,
                        by_abs=True,
                        size_node=15,
                    )
                    plt.savefig(
                        fig_dir / "collectri_tf_network.png",
                        dpi=180,
                        bbox_inches="tight",
                    )
                    plt.close("all")
                except Exception as exc:
                    print(f"[warn] collectri network plot skipped: {exc}")
    except Exception as exc:
        print(f"[warn] CollecTRI enrichment skipped: {exc}")

    try:
        progeny = dc.op.progeny(organism=organism)
        pw_acts, pw_padj = dc.mt.ulm(data=data, net=progeny)
        pw_acts.to_csv(enr_dir / "progeny_acts_all.csv")
        pw_padj.to_csv(enr_dir / "progeny_padj_all.csv")

        msk = (pw_padj.T < pvalue_threshold).iloc[:, 0]
        pw_sig = pw_acts.loc[:, msk]
        pw_sig.to_csv(enr_dir / "progeny_acts_significant.csv")

        if not pw_sig.empty:
            dc.pl.barplot(
                data=pw_sig,
                name=contrast_name,
                figsize=(4, 3),
            )
            plt.savefig(
                fig_dir / "progeny_barplot.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")

            dot_df = pw_sig.melt(value_name="score").merge(
                pw_padj.melt(value_name="pvalue")
                .assign(logp=lambda x: x["pvalue"].clip(2.22e-16, 1.0))
                .assign(logp=lambda x: -np.log10(x["logp"]))
            )
            dc.pl.dotplot(
                df=dot_df,
                x="score",
                y="variable",
                s="logp",
                c="score",
                scale=1,
                figsize=(5, 4),
            )
            plt.savefig(
                fig_dir / "progeny_dotplot.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")
    except Exception as exc:
        print(f"[warn] PROGENy enrichment skipped: {exc}")

    try:
        hallmark = dc.op.hallmark(organism=organism)
        hm_acts, hm_padj = dc.mt.ulm(data=data, net=hallmark)
        hm_acts.to_csv(enr_dir / "hallmark_acts_all.csv")
        hm_padj.to_csv(enr_dir / "hallmark_padj_all.csv")

        msk = (hm_padj.T < pvalue_threshold).iloc[:, 0]
        hm_sig = hm_acts.loc[:, msk]
        hm_sig.to_csv(enr_dir / "hallmark_acts_significant.csv")

        if not hm_sig.empty:
            dc.pl.barplot(
                data=hm_sig,
                name=contrast_name,
                figsize=(7, 3),
            )
            plt.savefig(
                fig_dir / "hallmark_barplot.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")
    except Exception as exc:
        print(f"[warn] Hallmark enrichment skipped: {exc}")


def run_contrast_celltype(
    pdata: ad.AnnData,
    case: str,
    control: str,
    celltype: str,
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, object]:
    dc = _import_decoupler()

    contrast_name = f"{case}.vs.{control}"
    out_dir = out_root / contrast_name / celltype.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = (
        (pdata.obs[args.condition_col].astype(str).isin([case, control]))
        & (pdata.obs[args.group_col].astype(str) == celltype)
    )
    tdata = pdata[mask].copy()

    if tdata.n_obs == 0:
        return {
            "contrast": contrast_name,
            "cell_type": celltype,
            "status": "skipped_no_profiles",
        }

    rep_counts = tdata.obs[args.condition_col].astype(str).value_counts()
    n_case = int(rep_counts.get(case, 0))
    n_ctrl = int(rep_counts.get(control, 0))
    if n_case < args.min_replicates or n_ctrl < args.min_replicates:
        return {
            "contrast": contrast_name,
            "cell_type": celltype,
            "status": "skipped_low_replicates",
            "n_case": n_case,
            "n_control": n_ctrl,
        }

    tdata.X = _to_dense_2d(tdata.X)
    _round_to_int_counts(tdata)
    tdata.layers["counts"] = _to_dense_2d(tdata.X).copy()

    # Tutorial-like feature filtering diagnostics and filtering.
    tdata = _filter_diagnostics_and_apply(
        tdata=tdata,
        out_dir=out_dir,
        condition_col=args.condition_col,
        min_count=args.min_count,
        min_total_count=args.min_total_count,
        large_n=args.large_n,
        expr_min_prop=args.expr_min_prop,
        min_prop=args.min_prop,
        min_smpls=args.min_smpls,
    )

    # Keep counts matrix for DE.
    tdata.X = _to_dense_2d(tdata.X)
    _round_to_int_counts(tdata)

    if args.skip_de:
        return {
            "contrast": contrast_name,
            "cell_type": celltype,
            "status": "ok_filtered_only",
            "n_case": n_case,
            "n_control": n_ctrl,
            "n_genes_after_filter": int(tdata.n_vars),
        }

    try:
        results_df = run_deseq2(
            tdata=tdata,
            condition_col=args.condition_col,
            case=case,
            control=control,
            n_cpus=args.n_cpus,
        )
    except Exception as exc:
        return {
            "contrast": contrast_name,
            "cell_type": celltype,
            "status": f"de_failed: {exc}",
            "n_case": n_case,
            "n_control": n_ctrl,
            "n_genes_after_filter": int(tdata.n_vars),
        }

    de_dir = out_dir / "de"
    de_fig = de_dir / "figures"
    de_dir.mkdir(parents=True, exist_ok=True)
    de_fig.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(de_dir / "deseq2_results.csv")

    if (
        "log2FoldChange" in results_df.columns
        and "pvalue" in results_df.columns
    ):
        try:
            dc.pl.volcano(
                data=results_df,
                x="log2FoldChange",
                y="pvalue",
            )
            plt.savefig(
                de_fig / "volcano.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close("all")
        except Exception as exc:
            print(f"[warn] volcano plot skipped: {exc}")

    contrast_stat = results_df[["stat"]].T.rename(
        index={"stat": contrast_name}
    )
    contrast_stat.to_csv(de_dir / "contrast_stat_matrix.csv")

    if not args.skip_enrichment:
        run_enrichment_plots(
            results_df=results_df,
            contrast_name=contrast_name,
            out_dir=out_dir,
            pvalue_threshold=args.pvalue_threshold,
            organism=args.organism,
        )

    return {
        "contrast": contrast_name,
        "cell_type": celltype,
        "status": "ok",
        "n_case": n_case,
        "n_control": n_ctrl,
        "n_genes_after_filter": int(tdata.n_vars),
        "n_de_genes": int(results_df.shape[0]),
    }


def main() -> None:
    args = parse_args()
    pb_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pdata = load_pseudobulk_h5ads(pb_root)

    for col in [args.condition_col, args.sample_col, args.group_col]:
        if col not in pdata.obs.columns:
            raise ValueError(
                "Required column missing in pseudobulk obs: "
                f"{col}"
            )

    run_global_pca_plots(
        pdata=pdata,
        out_dir=out_root,
        condition_col=args.condition_col,
        sample_col=args.sample_col,
        group_col=args.group_col,
    )

    conditions = sorted(pdata.obs[args.condition_col].astype(str).unique())
    contrasts = parse_contrasts(
        all_conditions=conditions,
        user_contrasts=args.contrast,
        control=args.control_condition,
        include_brca1_pair=args.include_brca1_pair,
    )

    all_rows: list[dict[str, object]] = []
    all_celltypes = sorted(pdata.obs[args.group_col].astype(str).unique())

    for case, control in contrasts:
        print(f"\nRunning contrast: {case} vs {control}")
        for celltype in all_celltypes:
            row = run_contrast_celltype(
                pdata=pdata,
                case=case,
                control=control,
                celltype=celltype,
                args=args,
                out_root=out_root,
            )
            all_rows.append(row)
            print(
                f"  [{celltype}] {row.get('status')} "
                f"(case={row.get('n_case', 0)}, "
                f"control={row.get('n_control', 0)})"
            )

    summary = pd.DataFrame(all_rows)
    summary_file = out_root / "downstream_summary.csv"
    summary.to_csv(summary_file, index=False)

    record = WarehouseRecord(
        input_file=str(pb_root),
        output_file=str(summary_file),
        script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        date_utc=utc_now_iso(),
        params_hash=params_hash(vars(args)),
        condition="all",
        stage="07_network_downstream_decoupler",
    )
    append_warehouse(out_root.parent, [record])

    print(f"\nDone. Summary: {summary_file}")


if __name__ == "__main__":
    main()
