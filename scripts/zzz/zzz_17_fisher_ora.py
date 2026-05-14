#!/usr/bin/env python3
"""Run explicit Fisher/hypergeometric ORA with BH-FDR correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom

from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fisher/hypergeometric ORA from marker genes and user gene sets"
    )
    parser.add_argument(
        "--input-dir",
        default="results/04_annotation",
        help="Root with per-condition marker tables",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available conditions and exit",
    )
    parser.add_argument(
        "--query-file-name",
        default=None,
        help=(
            "Marker filename inside each condition folder. "
            "Defaults to <condition>_cluster_markers_top.csv"
        ),
    )
    parser.add_argument(
        "--group-col",
        default="cluster",
        help="Group/cluster column in marker table",
    )
    parser.add_argument(
        "--gene-col",
        default="names",
        help="Gene symbol column in marker table",
    )
    parser.add_argument(
        "--pval-col",
        default="pvals_adj",
        help="Adjusted p-value column for marker filtering",
    )
    parser.add_argument(
        "--logfc-col",
        default="logfoldchanges",
        help="Log fold-change column for marker filtering",
    )
    parser.add_argument(
        "--pval-cutoff",
        type=float,
        default=0.05,
        help="Adjusted p-value cutoff for marker query genes",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=10,
        help="Minimum genes required per group to run ORA",
    )
    parser.add_argument(
        "--top-genes-fallback",
        type=int,
        default=150,
        help="Fallback top genes per group when strict filter is too small",
    )
    parser.add_argument(
        "--gene-set-file",
        required=True,
        help="Gene-set definition file (.gmt or two-column csv/tsv)",
    )
    parser.add_argument(
        "--gene-set-term-col",
        default="term",
        help="Term column for two-column csv/tsv gene-set files",
    )
    parser.add_argument(
        "--gene-set-gene-col",
        default="gene",
        help="Gene column for two-column csv/tsv gene-set files",
    )
    parser.add_argument(
        "--universe-file",
        default=None,
        help="Optional text/csv file with one gene per line for universe",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=2,
        help="Minimum overlap genes to report a term",
    )
    parser.add_argument(
        "--output-dir",
        default="results/zzz_06_fisher_ora",
        help="Output directory for ORA tables",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    names: list[str] = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        default_marker = p / f"{p.name}_cluster_markers_top.csv"
        if default_marker.exists():
            names.append(p.name)
    return names


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if not available:
        raise FileNotFoundError(f"No condition folders with marker files in {root}")
    if requested.strip().lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise ValueError(f"Condition '{requested}' not found. Available: {available}")


def normalize_gene(gene: object) -> str:
    return str(gene).strip().upper()


def load_gene_sets(path: Path, term_col: str, gene_col: str) -> dict[str, set[str]]:
    if path.suffix.lower() == ".gmt":
        out: dict[str, set[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            genes = {normalize_gene(g) for g in parts[2:] if str(g).strip()}
            if term and genes:
                out[term] = genes
        return out

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if term_col not in df.columns or gene_col not in df.columns:
        raise ValueError(
            f"Gene-set file must contain columns '{term_col}' and '{gene_col}'"
        )

    out: dict[str, set[str]] = {}
    for term, sub in df.groupby(term_col, observed=False):
        genes = {normalize_gene(g) for g in sub[gene_col].tolist() if str(g).strip()}
        term_text = str(term).strip()
        if term_text and genes:
            out[term_text] = genes
    return out


def load_universe(path: Path | None, gene_sets: dict[str, set[str]]) -> set[str]:
    if path is None:
        all_set: set[str] = set()
        for genes in gene_sets.values():
            all_set.update(genes)
        return all_set

    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] < 1:
            return set()
        vals = df.iloc[:, 0].astype(str).tolist()
        return {normalize_gene(v) for v in vals if str(v).strip()}

    vals = path.read_text(encoding="utf-8").splitlines()
    return {normalize_gene(v) for v in vals if str(v).strip()}


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1, dtype=np.float64))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def pick_query_genes(
    group_df: pd.DataFrame,
    gene_col: str,
    pval_col: str,
    logfc_col: str,
    pval_cutoff: float,
    min_genes: int,
    top_genes_fallback: int,
) -> list[str]:
    df = group_df.copy()
    if pval_col in df.columns:
        df = df[df[pval_col].fillna(1.0) <= float(pval_cutoff)]
    if logfc_col in df.columns:
        df = df[df[logfc_col].fillna(0.0) > 0]

    genes = [normalize_gene(g) for g in df[gene_col].astype(str).tolist() if str(g).strip()]
    genes = list(dict.fromkeys(genes))
    if len(genes) >= int(min_genes):
        return genes

    fallback = group_df.copy()
    sort_cols: list[str] = []
    if pval_col in fallback.columns:
        sort_cols.append(pval_col)
    if logfc_col in fallback.columns:
        sort_cols.append(logfc_col)
    if sort_cols:
        asc = [True if c == pval_col else False for c in sort_cols]
        fallback = fallback.sort_values(sort_cols, ascending=asc)

    genes_fb = [normalize_gene(g) for g in fallback[gene_col].astype(str).tolist()[: int(top_genes_fallback)] if str(g).strip()]
    genes_fb = list(dict.fromkeys(genes_fb))
    return genes_fb


def ora_rows_for_query(
    query_genes: set[str],
    gene_sets: dict[str, set[str]],
    universe: set[str],
    min_overlap: int,
) -> list[dict[str, object]]:
    q = set(query_genes).intersection(universe)
    n = len(q)
    N = len(universe)
    rows: list[dict[str, object]] = []

    if n == 0 or N == 0:
        return rows

    for term, term_genes in gene_sets.items():
        t = term_genes.intersection(universe)
        K = len(t)
        if K == 0:
            continue

        overlap = sorted(q.intersection(t))
        k = len(overlap)
        if k < int(min_overlap):
            continue

        a = int(k)
        b = int(n - k)
        c = int(K - k)
        d = int(N - n - K + k)
        if min(a, b, c, d) < 0:
            continue

        contingency = np.array([[a, b], [c, d]], dtype=np.int64)
        _, fisher_p = fisher_exact(contingency, alternative="greater")
        hypergeom_p = hypergeom.sf(k - 1, N, K, n)

        rows.append(
            {
                "term": term,
                "N_universe": N,
                "n_query": n,
                "K_term": K,
                "k_overlap": k,
                "a_overlap": a,
                "b_query_not_term": b,
                "c_term_not_query": c,
                "d_neither": d,
                "fisher_p": float(fisher_p),
                "hypergeom_p": float(hypergeom_p),
                "overlap_genes": " | ".join(overlap),
            }
        )

    if not rows:
        return rows

    df = pd.DataFrame(rows)
    df["fisher_fdr"] = benjamini_hochberg(df["fisher_p"].to_numpy(dtype=np.float64))
    df["hypergeom_fdr"] = benjamini_hochberg(df["hypergeom_p"].to_numpy(dtype=np.float64))
    df = df.sort_values(["fisher_fdr", "fisher_p", "k_overlap"], ascending=[True, True, False])
    return df.to_dict(orient="records")


def marker_file_for_condition(root: Path, condition: str, query_file_name: str | None) -> Path:
    if query_file_name:
        return root / condition / query_file_name
    return root / condition / f"{condition}_cluster_markers_top.csv"


def main() -> int:
    args = parse_args()
    input_root = resolve_base(args.input_dir)
    output_root = resolve_base(args.output_dir)
    gene_set_file = resolve_base(args.gene_set_file)
    universe_file = resolve_base(args.universe_file) if args.universe_file else None

    output_root.mkdir(parents=True, exist_ok=True)

    available = list_conditions(input_root)
    if args.list_conditions:
        print("Available conditions:")
        for name in available:
            print(f"- {name}")
        return 0

    conditions = resolve_conditions(input_root, args.condition)

    gene_sets = load_gene_sets(
        path=gene_set_file,
        term_col=args.gene_set_term_col,
        gene_col=args.gene_set_gene_col,
    )
    if not gene_sets:
        raise ValueError(f"No usable gene sets loaded from {gene_set_file}")

    universe = load_universe(universe_file, gene_sets)
    if not universe:
        raise ValueError("Universe is empty after loading/filtering")

    all_rows: list[pd.DataFrame] = []
    all_skipped: list[pd.DataFrame] = []
    records: list[WarehouseRecord] = []

    for condition in conditions:
        marker_file = marker_file_for_condition(
            root=input_root,
            condition=condition,
            query_file_name=args.query_file_name,
        )
        if not marker_file.exists():
            raise FileNotFoundError(f"Missing marker file: {marker_file}")

        mk = pd.read_csv(marker_file)
        required = {args.group_col, args.gene_col}
        if not required.issubset(set(mk.columns)):
            raise ValueError(f"Missing required columns in {marker_file}: {required}")

        cond_out = output_root / condition
        cond_out.mkdir(parents=True, exist_ok=True)

        cond_rows: list[dict[str, object]] = []
        cond_skipped: list[dict[str, object]] = []

        for group_name, sub in mk.groupby(args.group_col, observed=False):
            query = pick_query_genes(
                group_df=sub,
                gene_col=args.gene_col,
                pval_col=args.pval_col,
                logfc_col=args.logfc_col,
                pval_cutoff=float(args.pval_cutoff),
                min_genes=int(args.min_genes),
                top_genes_fallback=int(args.top_genes_fallback),
            )

            if len(query) < int(args.min_genes):
                cond_skipped.append(
                    {
                        "condition": condition,
                        "group": str(group_name),
                        "status": "too_few_genes",
                        "n_genes": int(len(query)),
                    }
                )
                continue

            rows = ora_rows_for_query(
                query_genes=set(query),
                gene_sets=gene_sets,
                universe=universe,
                min_overlap=int(args.min_overlap),
            )
            if not rows:
                cond_skipped.append(
                    {
                        "condition": condition,
                        "group": str(group_name),
                        "status": "no_enriched_terms",
                        "n_genes": int(len(query)),
                    }
                )
                continue

            for row in rows:
                row["condition"] = condition
                row["group"] = str(group_name)
                row["query_genes"] = " | ".join(sorted(set(query).intersection(universe)))
                row["n_query_input"] = int(len(query))
            cond_rows.extend(rows)

        cond_df = pd.DataFrame(cond_rows)
        skip_df = pd.DataFrame(cond_skipped)

        cond_file = cond_out / f"{condition}_fisher_ora.csv"
        skip_file = cond_out / f"{condition}_fisher_ora_skipped.csv"
        summary_file = cond_out / f"{condition}_fisher_ora_summary.csv"

        if cond_df.empty:
            pd.DataFrame(
                columns=[
                    "condition",
                    "group",
                    "term",
                    "N_universe",
                    "n_query",
                    "K_term",
                    "k_overlap",
                    "a_overlap",
                    "b_query_not_term",
                    "c_term_not_query",
                    "d_neither",
                    "fisher_p",
                    "hypergeom_p",
                    "fisher_fdr",
                    "hypergeom_fdr",
                    "overlap_genes",
                    "query_genes",
                    "n_query_input",
                ]
            ).to_csv(cond_file, index=False)
        else:
            cond_df.to_csv(cond_file, index=False)
            all_rows.append(cond_df)

        if skip_df.empty:
            pd.DataFrame(
                columns=["condition", "group", "status", "n_genes"]
            ).to_csv(skip_file, index=False)
        else:
            skip_df.to_csv(skip_file, index=False)
            all_skipped.append(skip_df)

        summary = {
            "condition": condition,
            "input_marker_file": str(marker_file),
            "gene_set_file": str(gene_set_file),
            "universe_file": str(universe_file) if universe_file else "",
            "N_universe": int(len(universe)),
            "n_gene_sets": int(len(gene_sets)),
            "n_groups_total": int(mk[args.group_col].nunique(dropna=False)),
            "n_groups_with_results": int(cond_df["group"].nunique()) if not cond_df.empty else 0,
            "n_rows": int(cond_df.shape[0]),
            "min_overlap": int(args.min_overlap),
            "pval_cutoff": float(args.pval_cutoff),
            "min_genes": int(args.min_genes),
            "top_genes_fallback": int(args.top_genes_fallback),
            "ora_file": str(cond_file),
            "skipped_file": str(skip_file),
        }
        pd.DataFrame([summary]).to_csv(summary_file, index=False)

        records.append(
            WarehouseRecord(
                input_file=str(marker_file),
                output_file=str(summary_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=condition,
                stage="06h_fisher_ora",
            )
        )

        print(
            f"[{condition}] groups_with_results={summary['n_groups_with_results']}, "
            f"rows={summary['n_rows']}"
        )

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            output_root / "fisher_ora_all.csv",
            index=False,
        )
    if all_skipped:
        pd.concat(all_skipped, ignore_index=True).to_csv(
            output_root / "fisher_ora_skipped_all.csv",
            index=False,
        )

    append_warehouse(output_root, records)
    print(f"Done. Fisher ORA outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
