#!/usr/bin/env python3
"""Compute C/S/D edge scores and node homogeneity from condition correlations.

Given two condition correlation matrices (case vs control), compute for each
shared gene pair:
- Conserved score: C = |rho_case + rho_control|
- Specific score:  S = ||rho_case| - |rho_control||
- Differentiated:  D = |rho_case| + |rho_control| - |rho_case + rho_control|

Then label each retained edge by argmax(C, S, D) and summarize per-node
homogeneity:
H_i = (kC_i / k_i)^2 + (kS_i / k_i)^2 + (kD_i / k_i)^2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute C/S/D scores and node homogeneity"
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition *_pearson_corr.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_network/zzz_11_csd",
        help="Output root for C/S/D edge and node summaries",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Condition pair in format case:control; repeat for multiple",
    )
    parser.add_argument(
        "--control",
        default="Normal",
        help="Control condition when --pair is omitted",
    )
    parser.add_argument(
        "--min-abs-corr",
        type=float,
        default=0.0,
        help="Retain edge if max(|rho_case|, |rho_control|) >= this",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Retain edge if max(C, S, D) >= this",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=200000,
        help="Maximum retained edges per pair by max(C,S,D); <=0 keeps all",
    )
    parser.add_argument(
        "--keep-self-edges",
        action="store_true",
        help="Include diagonal gene==gene edges (default: drop)",
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
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def parse_pairs(raw_pairs: list[str], all_conditions: list[str], control: str) -> list[tuple[str, str]]:
    if raw_pairs:
        out: list[tuple[str, str]] = []
        for text in raw_pairs:
            if ":" not in text:
                raise ValueError(f"Invalid --pair '{text}', expected case:control")
            case, ctrl = [x.strip() for x in text.split(":", 1)]
            out.append((case, ctrl))
    else:
        out = [(c, control) for c in all_conditions if c != control]

    valid: list[tuple[str, str]] = []
    for case, ctrl in out:
        if case not in all_conditions or ctrl not in all_conditions:
            print(f"[warn] skipping pair {case}:{ctrl} (condition missing)")
            continue
        valid.append((case, ctrl))

    dedup = list(dict.fromkeys(valid))
    if not dedup:
        raise ValueError("No valid condition pairs found")
    return dedup


def load_corr_payload(root: Path, condition: str) -> tuple[np.ndarray, np.ndarray, Path]:
    cond_dir = root / condition
    matches = sorted(cond_dir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cond_dir}")
    npz_file = matches[0]
    payload = np.load(npz_file, allow_pickle=True)
    corr = np.asarray(payload["corr"], dtype=np.float64)
    genes = payload["genes"].astype(str)
    return corr, genes, npz_file


def align_corrs(
    corr_case: np.ndarray,
    genes_case: np.ndarray,
    corr_ctrl: np.ndarray,
    genes_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_case = {str(g): i for i, g in enumerate(genes_case)}
    idx_ctrl = {str(g): i for i, g in enumerate(genes_ctrl)}
    common = sorted(set(idx_case).intersection(idx_ctrl))
    if not common:
        raise ValueError("No shared genes between case/control")

    ic = np.array([idx_case[g] for g in common], dtype=int)
    it = np.array([idx_ctrl[g] for g in common], dtype=int)
    return corr_case[np.ix_(ic, ic)], corr_ctrl[np.ix_(it, it)], np.array(common, dtype=str)


def classify_edges(
    corr_case: np.ndarray,
    corr_ctrl: np.ndarray,
    genes: np.ndarray,
    min_abs_corr: float,
    min_score: float,
    max_edges: int,
    keep_self_edges: bool,
) -> pd.DataFrame:
    if keep_self_edges:
        ii, jj = np.triu_indices(corr_case.shape[0], k=0)
    else:
        ii, jj = np.triu_indices(corr_case.shape[0], k=1)

    r_case = corr_case[ii, jj]
    r_ctrl = corr_ctrl[ii, jj]

    c_val = np.abs(r_case + r_ctrl)
    s_val = np.abs(np.abs(r_case) - np.abs(r_ctrl))
    d_val = np.abs(r_case) + np.abs(r_ctrl) - np.abs(r_case + r_ctrl)

    max_abs_corr = np.maximum(np.abs(r_case), np.abs(r_ctrl))
    max_score = np.maximum(c_val, np.maximum(s_val, d_val))

    keep = (max_abs_corr >= float(min_abs_corr)) & (max_score >= float(min_score))
    if not np.any(keep):
        return pd.DataFrame(
            columns=[
                "gene_a",
                "gene_b",
                "rho_case",
                "rho_control",
                "C",
                "S",
                "D",
                "max_abs_corr",
                "max_score",
                "link_type",
            ]
        )

    c_k = c_val[keep]
    s_k = s_val[keep]
    d_k = d_val[keep]
    type_idx = np.argmax(np.vstack([c_k, s_k, d_k]), axis=0)
    labels = np.array(["C", "S", "D"], dtype=object)[type_idx]

    df = pd.DataFrame(
        {
            "gene_a": genes[ii[keep]],
            "gene_b": genes[jj[keep]],
            "rho_case": r_case[keep],
            "rho_control": r_ctrl[keep],
            "C": c_k,
            "S": s_k,
            "D": d_k,
            "max_abs_corr": max_abs_corr[keep],
            "max_score": max_score[keep],
            "link_type": labels,
        }
    ).sort_values("max_score", ascending=False)

    if int(max_edges) > 0 and df.shape[0] > int(max_edges):
        df = df.head(int(max_edges)).copy()

    return df.reset_index(drop=True)


def node_homogeneity(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "degree",
                "kC",
                "kS",
                "kD",
                "pC",
                "pS",
                "pD",
                "homogeneity",
                "entropy_norm",
            ]
        )

    long = pd.concat(
        [
            edges[["gene_a", "link_type"]].rename(columns={"gene_a": "gene"}),
            edges[["gene_b", "link_type"]].rename(columns={"gene_b": "gene"}),
        ],
        axis=0,
        ignore_index=True,
    )

    counts = (
        long.groupby(["gene", "link_type"], as_index=False)
        .size()
        .pivot(index="gene", columns="link_type", values="size")
        .fillna(0)
    )

    for col in ["C", "S", "D"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[["C", "S", "D"]].astype(int)
    degree = counts.sum(axis=1).astype(int)

    p = counts.div(degree.replace(0, np.nan), axis=0).fillna(0.0)
    h = (p["C"] ** 2 + p["S"] ** 2 + p["D"] ** 2).astype(float)

    # Normalized entropy complement can be used alongside H.
    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    entropy_norm = 1.0 - (entropy / np.log(3.0))

    out = pd.DataFrame(
        {
            "gene": counts.index.astype(str),
            "degree": degree.to_numpy(),
            "kC": counts["C"].to_numpy(),
            "kS": counts["S"].to_numpy(),
            "kD": counts["D"].to_numpy(),
            "pC": p["C"].to_numpy(),
            "pS": p["S"].to_numpy(),
            "pD": p["D"].to_numpy(),
            "homogeneity": h.to_numpy(),
            "entropy_norm": entropy_norm.to_numpy(),
        }
    )

    return out.sort_values(["degree", "homogeneity"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = list_conditions(in_root)
    pairs = parse_pairs(args.pair, conditions, args.control)

    records: list[WarehouseRecord] = []

    for case, ctrl in pairs:
        pair_name = f"{case}__vs__{ctrl}"
        pair_out = out_root / pair_name
        pair_out.mkdir(parents=True, exist_ok=True)

        corr_case, genes_case, file_case = load_corr_payload(in_root, case)
        corr_ctrl, genes_ctrl, file_ctrl = load_corr_payload(in_root, ctrl)

        sub_case, sub_ctrl, genes = align_corrs(
            corr_case,
            genes_case,
            corr_ctrl,
            genes_ctrl,
        )

        edges = classify_edges(
            corr_case=sub_case,
            corr_ctrl=sub_ctrl,
            genes=genes,
            min_abs_corr=float(args.min_abs_corr),
            min_score=float(args.min_score),
            max_edges=int(args.max_edges),
            keep_self_edges=bool(args.keep_self_edges),
        )

        nodes = node_homogeneity(edges)

        edges_file = pair_out / f"{pair_name}_csd_edges.csv"
        nodes_file = pair_out / f"{pair_name}_node_homogeneity.csv"
        summary_file = pair_out / f"{pair_name}_csd_summary.json"

        edges.to_csv(edges_file, index=False)
        nodes.to_csv(nodes_file, index=False)

        link_counts = edges["link_type"].value_counts().to_dict() if not edges.empty else {}
        summary = {
            "case": case,
            "control": ctrl,
            "n_shared_genes": int(genes.shape[0]),
            "n_edges_retained": int(edges.shape[0]),
            "n_nodes_with_edges": int(nodes.shape[0]),
            "link_type_counts": {k: int(v) for k, v in link_counts.items()},
            "min_abs_corr": float(args.min_abs_corr),
            "min_score": float(args.min_score),
            "max_edges": int(args.max_edges),
            "keep_self_edges": bool(args.keep_self_edges),
            "edges_file": str(edges_file),
            "nodes_file": str(nodes_file),
            "case_corr_file": str(file_case),
            "control_corr_file": str(file_ctrl),
        }
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        records.append(
            WarehouseRecord(
                input_file=str(file_case),
                output_file=str(summary_file),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=pair_name,
                stage="08e_csd_scores_homogeneity",
            )
        )

        print(
            f"[{pair_name}] edges={summary['n_edges_retained']} "
            f"nodes={summary['n_nodes_with_edges']} "
            f"types={summary['link_type_counts']}"
        )

    append_warehouse(out_root, records)
    print(f"Done. C/S/D outputs: {out_root}")


if __name__ == "__main__":
    main()
