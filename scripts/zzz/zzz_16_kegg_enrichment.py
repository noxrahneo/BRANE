#!/usr/bin/env python3
"""Run KEGG enrichment on cluster marker genes.

R -> Python mapping used here:
- R: KEGG enrichment on DE outputs (kegga/topKEGG)
  Python: Enrichr KEGG gene-set enrichment via gseapy.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KEGG enrichment from cluster marker tables"
    )
    parser.add_argument(
        "--input-dir",
        default="results/04_annotation",
        help="Root annotation output directory",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition folder name, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition folders and exit",
    )
    parser.add_argument(
        "--gene-set",
        default="KEGG_2021_Human",
        help="Enrichr gene set library name",
    )
    parser.add_argument(
        "--pval-cutoff",
        type=float,
        default=0.05,
        help="Adjusted p-value cutoff for marker prefilter",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=10,
        help="Minimum genes required to run enrichment",
    )
    parser.add_argument(
        "--top-genes-fallback",
        type=int,
        default=150,
        help="Fallback top genes per cluster if strict filter is too small",
    )
    parser.add_argument(
        "--output-dir",
        default="results/zzz_06_kegg",
        help="Output directory for KEGG reports",
    )
    parser.add_argument(
        "--interpret-padj",
        type=float,
        default=0.05,
        help="Adjusted p-value threshold for interpretation export",
    )
    parser.add_argument(
        "--interpret-min-overlap",
        type=int,
        default=3,
        help="Minimum overlapping genes for interpretation export",
    )
    parser.add_argument(
        "--interpret-max-terms",
        type=int,
        default=15,
        help="Max interpreted terms per condition/cluster",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    names: list[str] = []
    for p in sorted(x for x in root.iterdir() if x.is_dir()):
        condition = p.name
        if (p / f"{condition}_cluster_markers_top.csv").exists():
            names.append(condition)
    return names


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if not available:
        raise FileNotFoundError(f"No condition folders found in: {root}")
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise FileNotFoundError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def pick_cluster_genes(
    cluster_df: pd.DataFrame,
    pval_cutoff: float,
    min_genes: int,
    top_genes_fallback: int,
) -> list[str]:
    # R logic equivalent:
    # prioritize significant up markers, then fallback to top ranked
    # markers when strict filtering yields too few genes.
    df = cluster_df.copy()
    if "pvals_adj" in df.columns:
        df = df[df["pvals_adj"].fillna(1.0) <= pval_cutoff]
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"].fillna(0.0) > 0]

    genes = df["names"].astype(str).tolist() if not df.empty else []
    genes = [g for g in genes if g]
    genes = list(dict.fromkeys(genes))
    if len(genes) >= min_genes:
        return genes

    fallback = cluster_df.copy()
    sort_cols: list[str] = []
    if "pvals_adj" in fallback.columns:
        sort_cols.append("pvals_adj")
    if "logfoldchanges" in fallback.columns:
        sort_cols.append("logfoldchanges")
    if sort_cols:
        asc = [True if c == "pvals_adj" else False for c in sort_cols]
        fallback = fallback.sort_values(sort_cols, ascending=asc)

    genes_fb = fallback["names"].astype(str).tolist()[:top_genes_fallback]
    genes_fb = [g for g in genes_fb if g]
    return list(dict.fromkeys(genes_fb))


def run_enrichr(
    genes: list[str],
    gene_set: str,
) -> pd.DataFrame:
    # R kegga/topKEGG analogue using Enrichr KEGG libraries.
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "gseapy is required for KEGG enrichment. "
            "Install with: pip install gseapy"
        ) from exc

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_set,
        organism="Human",
        outdir=None,
        cutoff=1.0,
    )
    return enr.results.copy()


def parse_overlap(value: str) -> tuple[int, int]:
    text = str(value).strip()
    if "/" not in text:
        return 0, 0
    left, right = text.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return 0, 0


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_interpretation_table(
    res_df: pd.DataFrame,
    condition: str,
    padj_threshold: float,
    min_overlap: int,
    max_terms: int,
) -> pd.DataFrame:
    if res_df.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "cluster",
                "Term",
                "Adjusted P-value",
                "P-value",
                "Overlap",
                "overlap_hits",
                "term_gene_count",
                "overlap_ratio",
                "Genes",
                "input_genes_n",
                "thesis_interpretation",
            ]
        )

    df = res_df.copy()
    padj_col = "Adjusted P-value"
    pval_col = "P-value"
    overlap_col = "Overlap"
    genes_col = "Genes"
    if padj_col not in df.columns:
        return pd.DataFrame()

    hits_and_size = df.get(overlap_col, pd.Series([""] * len(df))).map(
        parse_overlap
    )
    df["overlap_hits"] = [a for a, _ in hits_and_size]
    df["term_gene_count"] = [b for _, b in hits_and_size]
    df["overlap_ratio"] = (
        df["overlap_hits"] / df["term_gene_count"].replace(0, pd.NA)
    )

    df = df[df[padj_col].map(to_float) <= padj_threshold]
    df = df[df["overlap_hits"] >= min_overlap]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["cluster", padj_col, pval_col])
    df = df.groupby("cluster", as_index=False).head(max_terms)

    def make_sentence(row: pd.Series) -> str:
        term = str(row.get("Term", ""))
        cluster = str(row.get("cluster", ""))
        padj = to_float(row.get(padj_col))
        overlap = str(row.get(overlap_col, "0/0"))
        return (
            f"Cluster {cluster} is enriched for '{term}' "
            f"(FDR={padj:.2e}, overlap={overlap}). "
            "This indicates over-representation of this "
            "program among cluster markers, not direct causality."
        )

    df["condition"] = condition
    if genes_col not in df.columns:
        df[genes_col] = ""
    df["thesis_interpretation"] = df.apply(make_sentence, axis=1)

    keep_cols = [
        "condition",
        "cluster",
        "Term",
        "Adjusted P-value",
        "P-value",
        "Overlap",
        "overlap_hits",
        "term_gene_count",
        "overlap_ratio",
        "Genes",
        "input_genes_n",
        "thesis_interpretation",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].reset_index(drop=True)


def process_condition(
    input_root: Path,
    condition: str,
    gene_set: str,
    pval_cutoff: float,
    min_genes: int,
    top_genes_fallback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # One condition at a time, matching the structure used in
    # ER/HER2/TNBC R scripts.
    marker_file = (
        input_root / condition / f"{condition}_cluster_markers_top.csv"
    )
    if not marker_file.exists():
        raise FileNotFoundError(f"Missing marker file: {marker_file}")

    mk = pd.read_csv(marker_file)
    if not {"cluster", "names"}.issubset(set(mk.columns)):
        raise ValueError(
            f"Marker file missing required columns: {marker_file}"
        )

    all_rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []

    for cluster, sub in mk.groupby("cluster", observed=False):
        genes = pick_cluster_genes(
            cluster_df=sub,
            pval_cutoff=pval_cutoff,
            min_genes=min_genes,
            top_genes_fallback=top_genes_fallback,
        )
        if len(genes) < min_genes:
            skipped.append(
                {
                    "condition": condition,
                    "cluster": str(cluster),
                    "status": "too_few_genes",
                    "n_genes": len(genes),
                }
            )
            continue

        try:
            res = run_enrichr(genes=genes, gene_set=gene_set)
        except Exception as exc:
            skipped.append(
                {
                    "condition": condition,
                    "cluster": str(cluster),
                    "status": "enrichment_error",
                    "n_genes": len(genes),
                    "error": str(exc),
                }
            )
            continue

        if res.empty:
            skipped.append(
                {
                    "condition": condition,
                    "cluster": str(cluster),
                    "status": "no_results",
                    "n_genes": len(genes),
                }
            )
            continue

        res["condition"] = condition
        res["cluster"] = str(cluster)
        res["input_genes_n"] = len(genes)
        all_rows.append(res)

    out = (
        pd.concat(all_rows, axis=0, ignore_index=True)
        if all_rows
        else pd.DataFrame()
    )
    skip_df = pd.DataFrame(skipped)
    return out, skip_df


def main() -> int:
    args = parse_args()
    input_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    available = list_conditions(input_root)
    if args.list_conditions:
        if not available:
            print("No conditions found")
            return 0
        print("Available conditions:")
        for name in available:
            print(f"- {name}")
        return 0

    conditions = resolve_conditions(input_root, args.condition)
    arg_hash = params_hash(vars(args))
    now = utc_now_iso()
    all_res: list[pd.DataFrame] = []
    all_skip: list[pd.DataFrame] = []
    all_interp: list[pd.DataFrame] = []
    warehouse_rows: list[WarehouseRecord] = []

    for condition in conditions:
        res_df, skip_df = process_condition(
            input_root=input_root,
            condition=condition,
            gene_set=args.gene_set,
            pval_cutoff=args.pval_cutoff,
            min_genes=args.min_genes,
            top_genes_fallback=args.top_genes_fallback,
        )

        cond_dir = out_root / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        cond_file = cond_dir / f"{condition}_kegg_enrichment.csv"
        skip_file = cond_dir / f"{condition}_kegg_skipped.csv"
        interp_file = cond_dir / f"{condition}_kegg_interpretation.csv"

        if res_df.empty:
            pd.DataFrame(columns=["condition", "cluster"]).to_csv(
                cond_file, index=False
            )
        else:
            res_df = res_df.sort_values(
                ["cluster", "Adjusted P-value", "P-value"]
                if "Adjusted P-value" in res_df.columns
                else ["cluster"]
            )
            res_df.to_csv(cond_file, index=False)
            all_res.append(res_df)

        if skip_df.empty:
            pd.DataFrame(
                columns=["condition", "cluster", "status", "n_genes"]
            ).to_csv(skip_file, index=False)
        else:
            skip_df.to_csv(skip_file, index=False)
            all_skip.append(skip_df)

        interp_df = build_interpretation_table(
            res_df=res_df,
            condition=condition,
            padj_threshold=args.interpret_padj,
            min_overlap=args.interpret_min_overlap,
            max_terms=args.interpret_max_terms,
        )
        if interp_df.empty:
            pd.DataFrame(
                columns=[
                    "condition",
                    "cluster",
                    "Term",
                    "Adjusted P-value",
                    "P-value",
                    "Overlap",
                    "overlap_hits",
                    "term_gene_count",
                    "overlap_ratio",
                    "Genes",
                    "input_genes_n",
                    "thesis_interpretation",
                ]
            ).to_csv(interp_file, index=False)
        else:
            interp_df.to_csv(interp_file, index=False)
            all_interp.append(interp_df)

        print(f"KEGG complete: {condition}")
        warehouse_rows.append(
            WarehouseRecord(
                input_file=str(
                    input_root
                    / condition
                    / f"{condition}_cluster_markers_top.csv"
                ),
                output_file=str(cond_file),
                script="scripts/16_kegg_enrichment.py",
                date_utc=now,
                params_hash=arg_hash,
                condition=condition,
                stage="kegg",
            )
        )
        gc.collect()

    if all_res:
        pd.concat(all_res, axis=0, ignore_index=True).to_csv(
            out_root / "kegg_enrichment_all.csv", index=False
        )
    if all_skip:
        pd.concat(all_skip, axis=0, ignore_index=True).to_csv(
            out_root / "kegg_skipped_all.csv", index=False
        )
    if all_interp:
        pd.concat(all_interp, axis=0, ignore_index=True).to_csv(
            out_root / "kegg_interpretation_all.csv", index=False
        )

    warehouse_file = append_warehouse(out_root, warehouse_rows)

    print(f"Output directory: {out_root}")
    print(f"Warehouse log: {warehouse_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
