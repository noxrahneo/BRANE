#!/usr/bin/env python3
"""Full-parity permutation null pipeline.

Per permutation:
1) Shuffle each gene independently across samples in per-condition h5ad files.
2) Run DEG on randomized expression (script 19b).
3) Recompute correlations with DEG filtering (script 23).
4) Compute C/S/D for requested pairs (script 35).
5) Record per-pair max C/S/D null scores.

This mirrors the real differential workflow logic while using randomized input.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PRENEO = "Normal_BRCA1_-_pre-neoplastic"
TN_BRCA1 = "Triple_negative_BRCA1_tumor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-parity permutation null")
    parser.add_argument(
        "--expr-dir",
        default="results/08_pre_correlation/per_condition",
        help="Input directory with *_pseudobulk_logcpm.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        default="results/08_permutation/permutations/full_parity",
        help="Output root for permutation artifacts",
    )
    parser.add_argument(
        "--control-condition",
        default="Normal",
        help="Control condition used to form default pairs",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional pair case:control. Repeat for multiple.",
    )
    parser.add_argument(
        "--include-brca1-pair",
        action="store_true",
        help="Include Triple_negative_BRCA1_tumor vs Normal_BRCA1_-_pre-neoplastic",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10,
        help="Number of permutations to run in this invocation",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start index for naming permutations (1-based)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed")
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold passed to DEG script",
    )
    parser.add_argument(
        "--min-abs-log2fc",
        type=float,
        default=0.2,
        help="|log2FC| threshold passed to DEG script",
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


def condition_name_from_file(h5ad_file: Path) -> str:
    suffix = "_pseudobulk_logcpm.h5ad"
    name = h5ad_file.name
    return name[: -len(suffix)] if name.endswith(suffix) else h5ad_file.stem


def list_condition_files(expr_dir: Path) -> list[Path]:
    return sorted(expr_dir.glob("*_pseudobulk_logcpm.h5ad"))


def parse_pair(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"Invalid pair '{text}', expected case:control")
    case, ctrl = [x.strip() for x in text.split(":", 1)]
    if not case or not ctrl:
        raise ValueError(f"Invalid pair '{text}', expected case:control")
    return case, ctrl


def build_pairs(
    requested_pairs: list[str],
    conditions: list[str],
    control: str,
    include_brca1_pair: bool,
) -> list[tuple[str, str]]:
    if requested_pairs:
        pairs = [parse_pair(p) for p in requested_pairs]
    else:
        pairs = [(c, control) for c in conditions if c != control]

    if include_brca1_pair:
        pairs.append((TN_BRCA1, PRENEO))

    valid = []
    cond_set = set(conditions)
    for case, ctrl in pairs:
        if case in cond_set and ctrl in cond_set:
            valid.append((case, ctrl))

    return list(dict.fromkeys(valid))


def shuffle_gene_wise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_samples, n_genes = x.shape
    out = np.empty_like(x)
    for j in range(n_genes):
        out[:, j] = x[rng.permutation(n_samples), j]
    return out


def write_randomized_h5ad(
    src_h5ad: Path,
    dst_h5ad: Path,
    rng: np.random.Generator,
) -> None:
    adata = ad.read_h5ad(src_h5ad)
    x = np.asarray(adata.X, dtype=np.float64)
    x_rand = shuffle_gene_wise(x, rng)

    out = ad.AnnData(X=x_rand, obs=adata.obs.copy(), var=adata.var.copy())
    out.obs_names = adata.obs_names.copy()
    out.var_names = adata.var_names.copy()
    dst_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(dst_h5ad)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + proc.stdout
            + "\nSTDERR:\n"
            + proc.stderr
        )


def summarize_perm(
    perm_dir: Path,
    pairs: list[tuple[str, str]],
    perm_idx: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    csd_root = perm_dir / "csd"
    for case, ctrl in pairs:
        pair_name = f"{case}__vs__{ctrl}"
        edges_file = csd_root / pair_name / f"{pair_name}_csd_edges.csv"
        if not edges_file.exists():
            rows.append(
                {
                    "permutation": perm_idx,
                    "pair": pair_name,
                    "maxC": float("nan"),
                    "maxS": float("nan"),
                    "maxD": float("nan"),
                    "n_edges": 0,
                    "n_nodes": 0,
                }
            )
            continue

        edges = pd.read_csv(edges_file)
        if edges.empty:
            rows.append(
                {
                    "permutation": perm_idx,
                    "pair": pair_name,
                    "maxC": float("nan"),
                    "maxS": float("nan"),
                    "maxD": float("nan"),
                    "n_edges": 0,
                    "n_nodes": 0,
                }
            )
            continue

        nodes = set(edges["gene_a"].astype(str)).union(set(edges["gene_b"].astype(str)))
        rows.append(
            {
                "permutation": perm_idx,
                "pair": pair_name,
                "maxC": float(pd.to_numeric(edges["C"], errors="coerce").max()),
                "maxS": float(pd.to_numeric(edges["S"], errors="coerce").max()),
                "maxD": float(pd.to_numeric(edges["D"], errors="coerce").max()),
                "n_edges": int(edges.shape[0]),
                "n_nodes": int(len(nodes)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    expr_dir = resolve_base(args.expr_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cond_files = list_condition_files(expr_dir)
    if not cond_files:
        raise FileNotFoundError(f"No *_pseudobulk_logcpm.h5ad in {expr_dir}")

    cond_names = [condition_name_from_file(f) for f in cond_files]
    pairs = build_pairs(
        args.pair,
        cond_names,
        args.control_condition,
        bool(args.include_brca1_pair),
    )
    if not pairs:
        raise ValueError("No valid pairs found")

    all_rows: list[dict[str, object]] = []

    for i in range(int(args.n_permutations)):
        perm_idx = int(args.start_index) + i
        perm_seed = int(args.seed) + perm_idx
        rng = np.random.default_rng(perm_seed)

        perm_dir = out_root / f"perm_{perm_idx:03d}"
        expr_rand_dir = perm_dir / "pre_correlation" / "per_condition"
        deg_dir = perm_dir / "downstream_deg_ttest"
        allowlist_dir = deg_dir / "de_gene_filters"
        corr_dir = perm_dir / "correlation" / "pearson"
        csd_dir = perm_dir / "csd"
        perm_dir.mkdir(parents=True, exist_ok=True)

        done_flag = perm_dir / "_done.json"
        if done_flag.exists():
            print(f"[skip] perm_{perm_idx:03d} already complete")
            rows = summarize_perm(perm_dir, pairs, perm_idx)
            all_rows.extend(rows)
            continue

        print(f"[run] perm_{perm_idx:03d}")
        # 1) Randomized expression matrices
        for src in cond_files:
            dst = expr_rand_dir / src.name
            write_randomized_h5ad(src, dst, rng)

        # 2) DEG on randomized expression
        cmd_19b = [
            sys.executable,
            "scripts/19b_deg_ttest_logcpm.py",
            "--input-dir",
            str(expr_rand_dir),
            "--output-dir",
            str(deg_dir),
            "--allowlist-dir",
            str(allowlist_dir),
            "--control-condition",
            str(args.control_condition),
            "--fdr-threshold",
            str(args.fdr_threshold),
            "--min-abs-log2fc",
            str(args.min_abs_log2fc),
        ]
        if bool(args.include_brca1_pair):
            cmd_19b.append("--include-brca1-pair")
        for pair in args.pair:
            cmd_19b.extend(["--contrast", pair])
        run_cmd(cmd_19b, REPO_ROOT)

        # 3) Correlations with DEG filtering from randomized DEG allowlists
        cmd_23 = [
            sys.executable,
            "scripts/23_compute_correlations.py",
            "--input-dir",
            str(expr_rand_dir),
            "--output-dir",
            str(corr_dir),
            "--condition",
            "all",
            "--use-de-filter",
            "--de-filter-dir",
            str(allowlist_dir / "per_condition"),
            "--require-de-filter",
        ]
        run_cmd(cmd_23, REPO_ROOT)

        # 4) C/S/D on randomized, DEG-filtered correlations
        for case, ctrl in pairs:
            cmd_35 = [
                sys.executable,
                "scripts/35_csd_scores_homogeneity.py",
                "--input-dir",
                str(corr_dir),
                "--output-dir",
                str(csd_dir),
                "--pair",
                f"{case}:{ctrl}",
                "--min-abs-corr",
                "0.0",
                "--min-score",
                "0.0",
                "--max-edges",
                "200000",
            ]
            run_cmd(cmd_35, REPO_ROOT)

        done_flag.write_text(
            json.dumps({"perm": perm_idx, "seed": perm_seed}, indent=2),
            encoding="utf-8",
        )

        rows = summarize_perm(perm_dir, pairs, perm_idx)
        all_rows.extend(rows)

    # 5) Batch summary outputs
    max_file = out_root / f"maxima_{int(args.start_index):03d}_{int(args.start_index) + int(args.n_permutations) - 1:03d}.csv"
    max_df = pd.DataFrame(all_rows).sort_values(["pair", "permutation"]).reset_index(drop=True)
    max_df.to_csv(max_file, index=False, quoting=csv.QUOTE_MINIMAL)

    if not max_df.empty:
        thr = (
            max_df.groupby("pair", as_index=False)[["maxC", "maxS", "maxD"]]
            .mean()
            .rename(
                columns={
                    "maxC": "threshold_C_max_mean",
                    "maxS": "threshold_S_max_mean",
                    "maxD": "threshold_D_max_mean",
                }
            )
        )
        thr_file = out_root / (
            f"thresholds_max_mean_{int(args.start_index):03d}_"
            f"{int(args.start_index) + int(args.n_permutations) - 1:03d}.csv"
        )
        thr.to_csv(thr_file, index=False)
        print(f"Wrote: {max_file}")
        print(f"Wrote: {thr_file}")
    else:
        print(f"Wrote (empty): {max_file}")


if __name__ == "__main__":
    main()
