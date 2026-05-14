#!/usr/bin/env python3
"""Estimate C/S/D thresholds from permutation null and apply to real data.

Workflow:
1) Load case/control pseudobulk matrices (h5ad) and real correlations.
2) Build null C/S/D distributions by gene-wise sample shuffling per condition.
3) Derive type-specific thresholds (e.g., mean 99th percentile across permutations).
4) Apply thresholds to real C/S/D edges and export filtered network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutation-based C/S/D threshold estimation"
    )
    parser.add_argument(
        "--expr-dir",
        default="results/08_pre_correlation/per_condition",
        help="Directory with <condition>_pseudobulk_logcpm.h5ad",
    )
    parser.add_argument(
        "--corr-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition *_pearson_corr.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_network/zzz_15_csd_permutation",
        help="Output root",
    )
    parser.add_argument(
        "--pair",
        required=True,
        help="Pair as case:control",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=50,
        help="Number of permutations",
    )
    parser.add_argument(
        "--edge-sample-size",
        type=int,
        default=200000,
        help="Random edge sample size per permutation (<=0 uses all)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Null quantile for threshold (0-1)",
    )
    parser.add_argument(
        "--null-stat",
        choices=["quantile", "max", "both"],
        default="quantile",
        help=(
            "How to derive null thresholds from permutation runs: "
            "quantile=mean per-run quantiles, max=mean per-run maxima, "
            "both=compute and export both"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-edge-exports",
        action="store_true",
        help=(
            "Skip writing thresholded edge CSV files and only export "
            "permutation summaries (useful for fast batched runs)"
        ),
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


def parse_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError("--pair must be in format case:control")
    case, ctrl = [x.strip() for x in text.split(":", 1)]
    if not case or not ctrl:
        raise ValueError("--pair requires both case and control")
    return case, ctrl


def load_h5ad(expr_dir: Path, condition: str) -> tuple[np.ndarray, np.ndarray]:
    f = expr_dir / f"{condition}_pseudobulk_logcpm.h5ad"
    if not f.exists():
        raise FileNotFoundError(f"Missing expression file: {f}")
    adata = ad.read_h5ad(f)
    x = np.asarray(adata.X, dtype=np.float64)
    genes = adata.var_names.astype(str).to_numpy()
    return x, genes


def load_corr(corr_dir: Path, condition: str) -> tuple[np.ndarray, np.ndarray, Path]:
    cdir = corr_dir / condition
    matches = sorted(cdir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cdir}")
    f = matches[0]
    d = np.load(f, allow_pickle=True)
    return np.asarray(d["corr"], dtype=np.float64), d["genes"].astype(str), f


def align_by_genes(
    x_case: np.ndarray,
    genes_case: np.ndarray,
    x_ctrl: np.ndarray,
    genes_ctrl: np.ndarray,
    corr_genes_case: np.ndarray,
    corr_genes_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = sorted(
        set(map(str, genes_case))
        .intersection(map(str, genes_ctrl))
        .intersection(map(str, corr_genes_case))
        .intersection(map(str, corr_genes_ctrl))
    )
    if not shared:
        raise ValueError("No shared genes between expression and correlation inputs")

    idx_case_expr = {str(g): i for i, g in enumerate(genes_case)}
    idx_ctrl_expr = {str(g): i for i, g in enumerate(genes_ctrl)}

    ic = np.array([idx_case_expr[g] for g in shared], dtype=int)
    it = np.array([idx_ctrl_expr[g] for g in shared], dtype=int)

    return x_case[:, ic], x_ctrl[:, it], np.array(shared, dtype=str)


def corr_from_expr(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 3:
        raise ValueError("Need >=3 profiles per condition for correlation")
    c = np.corrcoef(x, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def permute_expr_gene_wise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_samples, n_genes = x.shape
    out = np.empty_like(x)
    for j in range(n_genes):
        out[:, j] = x[rng.permutation(n_samples), j]
    return out


def csd_scores(r_case: np.ndarray, r_ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.abs(r_case + r_ctrl)
    s = np.abs(np.abs(r_case) - np.abs(r_ctrl))
    d = np.abs(r_case) + np.abs(r_ctrl) - np.abs(r_case + r_ctrl)
    return c, s, d


def sample_upper_tri(
    m_case: np.ndarray,
    m_ctrl: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(m_case.shape[0], k=1)
    if sample_size > 0 and sample_size < ii.shape[0]:
        take = rng.choice(ii.shape[0], size=int(sample_size), replace=False)
        ii = ii[take]
        jj = jj[take]
    return m_case[ii, jj], m_ctrl[ii, jj]


def build_thresholded_edges(
    shared_genes: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    r_case: np.ndarray,
    r_ctrl: np.ndarray,
    c_real: np.ndarray,
    s_real: np.ndarray,
    d_real: np.ndarray,
    tC: float,
    tS: float,
    tD: float,
) -> pd.DataFrame:
    keep = (c_real >= tC) | (s_real >= tS) | (d_real >= tD)
    if np.any(keep):
        c_k = c_real[keep]
        s_k = s_real[keep]
        d_k = d_real[keep]
        idx = np.argmax(np.vstack([c_k, s_k, d_k]), axis=0)
        labels = np.array(["C", "S", "D"], dtype=object)[idx]

        return pd.DataFrame(
            {
                "gene_a": shared_genes[ii[keep]],
                "gene_b": shared_genes[jj[keep]],
                "rho_case": r_case[keep],
                "rho_control": r_ctrl[keep],
                "C": c_k,
                "S": s_k,
                "D": d_k,
                "link_type": labels,
                "passes_C": c_k >= tC,
                "passes_S": s_k >= tS,
                "passes_D": d_k >= tD,
            }
        ).sort_values(["C", "S", "D"], ascending=False)

    return pd.DataFrame(
        columns=[
            "gene_a",
            "gene_b",
            "rho_case",
            "rho_control",
            "C",
            "S",
            "D",
            "link_type",
            "passes_C",
            "passes_S",
            "passes_D",
        ]
    )


def count_thresholded_edges(
    c_real: np.ndarray,
    s_real: np.ndarray,
    d_real: np.ndarray,
    tC: float,
    tS: float,
    tD: float,
) -> int:
    keep = (c_real >= tC) | (s_real >= tS) | (d_real >= tD)
    return int(np.sum(keep))


def main() -> None:
    args = parse_args()
    case, ctrl = parse_pair(args.pair)

    expr_root = resolve_base(args.expr_dir)
    corr_root = resolve_base(args.corr_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pair_name = f"{case}__vs__{ctrl}"
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    x_case, g_case = load_h5ad(expr_root, case)
    x_ctrl, g_ctrl = load_h5ad(expr_root, ctrl)

    real_case_corr, real_case_genes, real_case_file = load_corr(corr_root, case)
    real_ctrl_corr, real_ctrl_genes, real_ctrl_file = load_corr(corr_root, ctrl)

    x_case_aligned, x_ctrl_aligned, shared_genes = align_by_genes(
        x_case,
        g_case,
        x_ctrl,
        g_ctrl,
        real_case_genes,
        real_ctrl_genes,
    )

    idx_case_real = {str(g): i for i, g in enumerate(real_case_genes)}
    idx_ctrl_real = {str(g): i for i, g in enumerate(real_ctrl_genes)}
    ir_case = np.array([idx_case_real[g] for g in shared_genes], dtype=int)
    ir_ctrl = np.array([idx_ctrl_real[g] for g in shared_genes], dtype=int)
    real_case = real_case_corr[np.ix_(ir_case, ir_case)]
    real_ctrl = real_ctrl_corr[np.ix_(ir_ctrl, ir_ctrl)]

    rng = np.random.default_rng(int(args.seed))

    q_rows: list[dict[str, float | int]] = []
    q = float(args.quantile)

    for p in range(1, int(args.n_permutations) + 1):
        xp_case = permute_expr_gene_wise(x_case_aligned, rng)
        xp_ctrl = permute_expr_gene_wise(x_ctrl_aligned, rng)

        corr_case = corr_from_expr(xp_case)
        corr_ctrl = corr_from_expr(xp_ctrl)

        r_case, r_ctrl = sample_upper_tri(
            corr_case,
            corr_ctrl,
            sample_size=int(args.edge_sample_size),
            rng=rng,
        )
        c, s, d = csd_scores(r_case, r_ctrl)

        q_rows.append(
            {
                "permutation": int(p),
                "qC": float(np.quantile(c, q)),
                "qS": float(np.quantile(s, q)),
                "qD": float(np.quantile(d, q)),
                "maxC": float(np.max(c)),
                "maxS": float(np.max(s)),
                "maxD": float(np.max(d)),
                "meanC": float(np.mean(c)),
                "meanS": float(np.mean(s)),
                "meanD": float(np.mean(d)),
            }
        )

        if p == 1 or p % 10 == 0 or p == int(args.n_permutations):
            print(f"[{pair_name}] permutation {p}/{args.n_permutations}")

    q_df = pd.DataFrame(q_rows)
    q_file = pair_out / f"{pair_name}_permutation_quantiles.csv"
    q_df.to_csv(q_file, index=False)

    tC_quant = float(q_df["qC"].mean())
    tS_quant = float(q_df["qS"].mean())
    tD_quant = float(q_df["qD"].mean())

    tC_max = float(q_df["maxC"].mean())
    tS_max = float(q_df["maxS"].mean())
    tD_max = float(q_df["maxD"].mean())

    ii, jj = np.triu_indices(real_case.shape[0], k=1)
    r_case = real_case[ii, jj]
    r_ctrl = real_ctrl[ii, jj]
    c_real, s_real, d_real = csd_scores(r_case, r_ctrl)

    n_edges_quant = count_thresholded_edges(
        c_real,
        s_real,
        d_real,
        tC_quant,
        tS_quant,
        tD_quant,
    )
    n_edges_max = count_thresholded_edges(
        c_real,
        s_real,
        d_real,
        tC_max,
        tS_max,
        tD_max,
    )

    edges_file_quant = ""
    edges_file_max = ""
    edges_file = ""

    if not args.skip_edge_exports:
        edges_quant = build_thresholded_edges(
            shared_genes,
            ii,
            jj,
            r_case,
            r_ctrl,
            c_real,
            s_real,
            d_real,
            tC_quant,
            tS_quant,
            tD_quant,
        )
        edges_max = build_thresholded_edges(
            shared_genes,
            ii,
            jj,
            r_case,
            r_ctrl,
            c_real,
            s_real,
            d_real,
            tC_max,
            tS_max,
            tD_max,
        )

        edges_file_quant_path = (
            pair_out / f"{pair_name}_thresholded_edges_quantile.csv"
        )
        edges_file_max_path = (
            pair_out / f"{pair_name}_thresholded_edges_max.csv"
        )
        edges_quant.to_csv(edges_file_quant_path, index=False)
        edges_max.to_csv(edges_file_max_path, index=False)

        # Backward-compatible default edge file used by downstream scripts.
        if args.null_stat == "quantile":
            edges_default = edges_quant
        elif args.null_stat == "max":
            edges_default = edges_max
        else:
            edges_default = edges_max
        edges_file_path = pair_out / f"{pair_name}_thresholded_edges.csv"
        edges_default.to_csv(edges_file_path, index=False)

        edges_file_quant = str(edges_file_quant_path)
        edges_file_max = str(edges_file_max_path)
        edges_file = str(edges_file_path)

    summary = {
        "case": case,
        "control": ctrl,
        "n_shared_genes": int(shared_genes.shape[0]),
        "n_permutations": int(args.n_permutations),
        "edge_sample_size": int(args.edge_sample_size),
        "quantile": float(args.quantile),
        "null_stat": str(args.null_stat),
        "threshold_C_quantile": tC_quant,
        "threshold_S_quantile": tS_quant,
        "threshold_D_quantile": tD_quant,
        "threshold_C_max": tC_max,
        "threshold_S_max": tS_max,
        "threshold_D_max": tD_max,
        "n_edges_thresholded_quantile": n_edges_quant,
        "n_edges_thresholded_max": n_edges_max,
        "threshold_C": (
            tC_max if args.null_stat in {"max", "both"} else tC_quant
        ),
        "threshold_S": (
            tS_max if args.null_stat in {"max", "both"} else tS_quant
        ),
        "threshold_D": (
            tD_max if args.null_stat in {"max", "both"} else tD_quant
        ),
        "n_edges_thresholded": (
            n_edges_max if args.null_stat in {"max", "both"} else n_edges_quant
        ),
        "permutation_quantiles_file": str(q_file),
        "thresholded_edges_file": edges_file,
        "thresholded_edges_file_quantile": edges_file_quant,
        "thresholded_edges_file_max": edges_file_max,
        "skip_edge_exports": bool(args.skip_edge_exports),
        "real_case_corr_file": str(real_case_file),
        "real_control_corr_file": str(real_ctrl_file),
    }
    summary_file = pair_out / f"{pair_name}_permutation_threshold_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    append_warehouse(
        out_root,
        [
            WarehouseRecord(
                input_file=str(real_case_file),
                output_file=str(summary_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=pair_name,
                stage="08f_csd_permutation_threshold",
            )
        ],
    )

    print(
        "[{}] quantile-thresholds C={:.4f} S={:.4f} D={:.4f}; "
        "kept_edges_quantile={}".format(
            pair_name,
            tC_quant,
            tS_quant,
            tD_quant,
            n_edges_quant,
        )
    )
    print(
        "[{}] max-thresholds C={:.4f} S={:.4f} D={:.4f}; "
        "kept_edges_max={}".format(
            pair_name,
            tC_max,
            tS_max,
            tD_max,
            n_edges_max,
        )
    )
    print(f"Done. Permutation threshold outputs: {pair_out}")


if __name__ == "__main__":
    main()
