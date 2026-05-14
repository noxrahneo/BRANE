#!/usr/bin/env python3
"""Differential scale-free CSD network workflow.

Implements a non-permutation differential network approach with:
1) Fixed/default pair definitions,
2) DEG-based gene filtering,
3) WGCNA-style soft-power scan on differential correlation,
4) C/S/D score computation with denominator,
5) Paper-style score-type-specific importance thresholds,
6) Consolidated C/S/D network exports, homogeneity, modules and degree plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities

from utils.h5ad_compat import read_h5ad_compat
from utils.warehouse import WarehouseRecord, append_warehouse, params_hash, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]
NORMAL = "Normal"
PRENEO = "Normal_BRCA1_-_pre-neoplastic"
TN_BRCA1 = "Triple_negative_BRCA1_tumor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Differential network inference with scale-free power scan "
            "and importance-thresholded C/S/D classification"
        )
    )
    parser.add_argument(
        "--input-dir",
        default="results/09_correlation/pearson",
        help="Root with per-condition *_pearson_corr.npz",
    )
    parser.add_argument(
        "--expr-dir",
        default="results/08_pre_correlation/per_condition",
        help="Root with <condition>_pseudobulk_logcpm.h5ad",
    )
    parser.add_argument(
        "--deg-dir",
        default="results/10_deg_ttest",
        help="Root with contrast DEG tables",
    )
    parser.add_argument(
        "--output-dir",
        default="results/07_network/zzz_12_differential_scalefree",
        help="Output root for pairwise differential results",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Optional pair case:control. Repeat for multiple",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="DEG threshold on FDR",
    )
    parser.add_argument(
        "--min-abs-log2fc",
        type=float,
        default=0.2,
        help="DEG threshold on |log2FC|",
    )
    parser.add_argument(
        "--use-degs",
        action="store_true",
        help="Filter network genes to contrast DEGs",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned"],
        default="signed",
        help="Adjacency transform type for differential matrix",
    )
    parser.add_argument(
        "--delta-scale",
        choices=["half", "none"],
        default="half",
        help="Scale delta by 1/2 before adjacency",
    )
    parser.add_argument(
        "--powers",
        default="1,2,3,4,5,6,7,8,9,10,12,14,16,18,20",
        help="Comma-separated soft-threshold powers",
    )
    parser.add_argument(
        "--target-signed-r2",
        type=float,
        default=0.70,
        help="Target signed R^2 for power selection",
    )
    parser.add_argument(
        "--min-mean-connectivity",
        type=float,
        default=5.0,
        help="Minimum mean connectivity",
    )
    parser.add_argument(
        "--r2-plateau-delta",
        type=float,
        default=0.03,
        help="Plateau tolerance for beta selection",
    )
    parser.add_argument(
        "--r2-near-best-delta",
        type=float,
        default=0.02,
        help="Fallback tolerance near best R^2",
    )
    parser.add_argument(
        "--degree-bins",
        type=int,
        default=20,
        help="Histogram bins for scale-free fit",
    )
    parser.add_argument(
        "--force-power",
        type=int,
        default=None,
        help="Force beta power",
    )
    parser.add_argument(
        "--candidate-min-weight",
        type=float,
        default=0.0,
        help=(
            "Adjacency pre-filter before C/S/D thresholding. "
            "0.0 means use all candidate edges"
        ),
    )
    parser.add_argument(
        "--csd-threshold-mode",
        choices=["importance", "none"],
        default="none",
        help=(
            "How to convert C/S/D scores to edges: "
            "importance=use score-specific thresholds, "
            "none=keep all candidate edges and label by argmax(C,S,D)"
        ),
    )
    parser.add_argument(
        "--variance-mode",
        choices=["none", "fisher"],
        default="fisher",
        help="Denominator mode for C/S/D",
    )
    parser.add_argument(
        "--importance-L",
        type=int,
        default=5000,
        help="Sample size L for importance threshold estimation",
    )
    parser.add_argument(
        "--importance-m",
        type=int,
        default=1000,
        help="Number of samples m for importance thresholds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=50,
        help="Top nodes by weighted degree to export",
    )
    parser.add_argument(
        "--skip-modules",
        action="store_true",
        help="Skip Louvain module detection",
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


def parse_powers(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v > 0:
            values.append(v)
    if not values:
        raise ValueError("No valid powers parsed from --powers")
    return sorted(set(values))


def default_pairs(conditions: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for cond in conditions:
        if cond == NORMAL:
            continue
        pairs.append((cond, NORMAL))
    if PRENEO in conditions and TN_BRCA1 in conditions:
        pairs.append((TN_BRCA1, PRENEO))
    return pairs


def parse_pairs(raw_pairs: list[str], conditions: list[str]) -> list[tuple[str, str]]:
    if raw_pairs:
        pairs: list[tuple[str, str]] = []
        for text in raw_pairs:
            if ":" not in text:
                raise ValueError(f"Invalid --pair '{text}', expected case:control")
            case, ctrl = [x.strip() for x in text.split(":", 1)]
            pairs.append((case, ctrl))
    else:
        pairs = default_pairs(conditions)

    valid: list[tuple[str, str]] = []
    for case, ctrl in pairs:
        if case not in conditions or ctrl not in conditions:
            print(f"[warn] skipping pair {case}:{ctrl} (missing condition)")
            continue
        valid.append((case, ctrl))

    dedup = list(dict.fromkeys(valid))
    if not dedup:
        raise ValueError("No valid condition pairs available")
    return dedup


def load_corr_payload(root: Path, condition: str) -> tuple[np.ndarray, np.ndarray, Path]:
    cdir = root / condition
    matches = sorted(cdir.glob("*_pearson_corr.npz"))
    if not matches:
        raise FileNotFoundError(f"No *_pearson_corr.npz in {cdir}")
    f = matches[0]
    payload = np.load(f, allow_pickle=True)
    corr = np.asarray(payload["corr"], dtype=np.float64)
    genes = payload["genes"].astype(str)
    return corr, genes, f


def load_n_profiles(expr_root: Path, condition: str) -> int:
    h5ad_file = expr_root / f"{condition}_pseudobulk_logcpm.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(f"Missing expression file: {h5ad_file}")
    adata = read_h5ad_compat(h5ad_file)
    return int(adata.n_obs)


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
    genes = np.array(common, dtype=str)
    return corr_case[np.ix_(ic, ic)], corr_ctrl[np.ix_(it, it)], genes


def load_deg_stats(deg_root: Path, case: str, ctrl: str) -> pd.DataFrame:
    contrast = f"{case}.vs.{ctrl}"
    stats_file = deg_root / contrast / f"{contrast}_deg_stats.csv"
    if not stats_file.exists():
        raise FileNotFoundError(f"Missing DEG stats file: {stats_file}")
    df = pd.read_csv(stats_file)
    need = {"gene", "log2FC", "fdr"}
    if not need.issubset(df.columns):
        raise ValueError(f"DEG stats missing {need}: {stats_file}")
    return df


def pick_degs(
    deg_df: pd.DataFrame,
    fdr_threshold: float,
    min_abs_log2fc: float,
) -> tuple[set[str], list[str], list[str]]:
    work = deg_df.copy()
    work["gene"] = work["gene"].astype(str)
    mask = (
        pd.to_numeric(work["fdr"], errors="coerce").le(float(fdr_threshold))
        & pd.to_numeric(work["log2FC"], errors="coerce")
        .abs()
        .ge(float(min_abs_log2fc))
    )
    work = work[mask].copy()
    up = work[work["log2FC"] >= 0.0]["gene"].tolist()
    down = work[work["log2FC"] < 0.0]["gene"].tolist()
    all_degs = set(work["gene"].tolist())
    return all_degs, up, down


def build_gene_change_table(
    deg_df: pd.DataFrame,
    genes_keep: np.ndarray,
    fdr_threshold: float,
    min_abs_log2fc: float,
) -> pd.DataFrame:
    work = deg_df.copy()
    work["gene"] = work["gene"].astype(str)
    work["log2FC"] = pd.to_numeric(work["log2FC"], errors="coerce")
    work["fdr"] = pd.to_numeric(work["fdr"], errors="coerce")

    keep_set = set(np.asarray(genes_keep, dtype=str).tolist())
    work = work[work["gene"].isin(keep_set)].copy()

    sig = (
        work["fdr"].le(float(fdr_threshold))
        & work["log2FC"].abs().ge(float(min_abs_log2fc))
    )
    direction = np.where(
        sig & (work["log2FC"] > 0),
        "up",
        np.where(sig & (work["log2FC"] < 0), "down", "unchanged"),
    )
    work["deg_direction"] = direction
    work["is_deg"] = sig.astype(bool)

    out = work[["gene", "log2FC", "fdr", "is_deg", "deg_direction"]].copy()
    out = out.drop_duplicates(subset=["gene"]).sort_values("gene").reset_index(drop=True)
    return out


def adjacency_from_delta(delta: np.ndarray, power: int, network_type: str) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64)
    d = np.clip(d, -1.0, 1.0)
    if network_type == "signed":
        sim = (1.0 + d) / 2.0
    else:
        sim = np.abs(d)
    adj = np.power(sim, int(power), dtype=np.float64)
    np.fill_diagonal(adj, 0.0)
    return adj


def scale_free_fit_from_connectivity(k: np.ndarray, bins: int) -> dict[str, float]:
    k = np.asarray(k, dtype=np.float64)
    k = k[np.isfinite(k)]
    k = k[k > 0]
    if k.size < 5:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "signed_r2": float("nan"),
            "n_bins_used": 0,
        }

    counts, edges = np.histogram(k, bins=int(max(5, bins)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = counts / max(1, counts.sum())

    mask = (counts > 0) & (centers > 0) & (probs > 0)
    if int(mask.sum()) < 3:
        return {
            "slope": float("nan"),
            "r2": float("nan"),
            "signed_r2": float("nan"),
            "n_bins_used": int(mask.sum()),
        }

    x = np.log10(centers[mask])
    y = np.log10(probs[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    signed_r2 = float(np.sign(-slope) * r2) if np.isfinite(r2) else float("nan")

    return {
        "slope": float(slope),
        "r2": float(r2),
        "signed_r2": signed_r2,
        "n_bins_used": int(mask.sum()),
    }


def choose_power(
    scan_df: pd.DataFrame,
    target_signed_r2: float,
    min_mean_k: float,
    r2_plateau_delta: float,
    r2_near_best_delta: float,
) -> tuple[int, str]:
    ordered = scan_df.sort_values("power").reset_index(drop=True)
    eligible = ordered[
        (ordered["signed_r2"] >= float(target_signed_r2))
        & (ordered["mean_connectivity"] >= float(min_mean_k))
    ].copy()

    if not eligible.empty:
        plateau_rows: list[pd.Series] = []
        for row in eligible.itertuples(index=False):
            cur_p = int(getattr(row, "power"))
            cur_r2 = float(getattr(row, "signed_r2"))
            future = ordered[ordered["power"] >= cur_p]
            future_r2 = pd.to_numeric(future["signed_r2"], errors="coerce")
            future_r2 = future_r2[np.isfinite(future_r2)]
            if future_r2.empty:
                continue
            gain = float(future_r2.max()) - cur_r2
            if gain <= float(r2_plateau_delta):
                plateau_rows.append(ordered[ordered["power"] == cur_p].iloc[0])

        if plateau_rows:
            chosen = pd.DataFrame(plateau_rows).sort_values("power").iloc[0]
            return int(chosen["power"]), "lowest eligible power at plateau"

        p = int(eligible.sort_values("power").iloc[0]["power"])
        return p, "lowest power meeting signed R2 and mean connectivity"

    pool = ordered[ordered["mean_connectivity"] >= float(min_mean_k)].copy()
    if pool.empty:
        pool = ordered.copy()

    r2_vals = pd.to_numeric(pool["signed_r2"], errors="coerce")
    r2_vals = r2_vals[np.isfinite(r2_vals)]
    if r2_vals.empty:
        p = int(pool.sort_values("power").iloc[0]["power"])
        return p, "fallback smallest power"

    best_r2 = float(r2_vals.max())
    near = pool[pool["signed_r2"] >= best_r2 - float(r2_near_best_delta)]
    near = near.sort_values("power")
    return int(near.iloc[0]["power"]), "fallback near-best signed R2"


def plot_soft_threshold(scan_df: pd.DataFrame, target_r2: float, out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    x = scan_df["power"].to_numpy()
    y1 = scan_df["signed_r2"].to_numpy()
    y2 = scan_df["mean_connectivity"].to_numpy()

    axes[0].plot(x, y1, marker="o")
    axes[0].axhline(float(target_r2), color="red", linestyle="--", linewidth=1.3)
    axes[0].set_xlabel("Power")
    axes[0].set_ylabel("Scale-free fit (signed R^2)")
    axes[0].set_title(f"{title}: scale-free fit vs power")

    axes[1].plot(x, y2, marker="o")
    axes[1].set_xlabel("Power")
    axes[1].set_ylabel("Mean connectivity")
    axes[1].set_title(f"{title}: mean connectivity vs power")

    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def top_edges_from_adjacency(
    adjacency: np.ndarray,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(adjacency, dtype=np.float64)
    tri_i, tri_j = np.triu_indices(a.shape[0], k=1)
    vals = a[tri_i, tri_j]

    keep = np.where(vals >= float(min_weight))[0]
    if keep.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)

    tri_i = tri_i[keep]
    tri_j = tri_j[keep]
    vals = vals[keep]
    return tri_i, tri_j, vals


def sigma_sq_from_rho(rho: np.ndarray, n_profiles: int, mode: str) -> np.ndarray:
    if mode == "none":
        return np.zeros_like(rho, dtype=np.float64)

    n_eff = max(int(n_profiles) - 3, 1)
    rho = np.clip(np.asarray(rho, dtype=np.float64), -0.999999, 0.999999)
    return np.square(1.0 - np.square(rho)) / float(n_eff)


def csd_scores(
    rho_case: np.ndarray,
    rho_ctrl: np.ndarray,
    n_case: int,
    n_ctrl: int,
    variance_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r1 = np.asarray(rho_case, dtype=np.float64)
    r2 = np.asarray(rho_ctrl, dtype=np.float64)

    sigma_sq_1 = sigma_sq_from_rho(r1, n_case, variance_mode)
    sigma_sq_2 = sigma_sq_from_rho(r2, n_ctrl, variance_mode)
    denom = np.sqrt(np.maximum(sigma_sq_1 + sigma_sq_2, 1e-12))

    if variance_mode == "none":
        denom = np.ones_like(denom, dtype=np.float64)

    c_val = np.abs(r1 + r2) / denom
    s_val = np.abs(np.abs(r1) - np.abs(r2)) / denom
    d_val = (np.abs(r1) + np.abs(r2) - np.abs(r1 + r2)) / denom
    return c_val, s_val, d_val, denom


def importance_threshold(
    values: np.ndarray,
    sample_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")

    l_eff = max(1, int(sample_size))
    m_eff = max(1, int(n_samples))
    n = vals.size

    maxima = np.empty(m_eff, dtype=np.float64)
    for i in range(m_eff):
        idx = rng.integers(0, n, size=l_eff)
        maxima[i] = float(np.max(vals[idx]))
    return float(np.mean(maxima))


def classify_edges_by_thresholds(
    c_val: np.ndarray,
    s_val: np.ndarray,
    d_val: np.ndarray,
    k_c: float,
    k_s: float,
    k_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c_val, dtype=np.float64)
    s = np.asarray(s_val, dtype=np.float64)
    d = np.asarray(d_val, dtype=np.float64)

    pass_c = c > float(k_c)
    pass_s = s > float(k_s)
    pass_d = d > float(k_d)
    pass_any = pass_c | pass_s | pass_d

    labels = np.full(c.shape[0], "", dtype=object)
    selected_value = np.full(c.shape[0], np.nan, dtype=np.float64)
    if not np.any(pass_any):
        return labels, selected_value

    nc = c / max(float(k_c), 1e-12)
    ns = s / max(float(k_s), 1e-12)
    nd = d / max(float(k_d), 1e-12)
    stack = np.vstack([nc, ns, nd])
    idx = np.argmax(stack, axis=0)
    type_arr = np.array(["C", "S", "D"], dtype=object)

    labels[pass_any] = type_arr[idx[pass_any]]
    selected_value[pass_any] = np.where(
        idx[pass_any] == 0,
        c[pass_any],
        np.where(idx[pass_any] == 1, s[pass_any], d[pass_any]),
    )
    return labels, selected_value


def classify_edges_no_threshold(
    c_val: np.ndarray,
    s_val: np.ndarray,
    d_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c_val, dtype=np.float64)
    s = np.asarray(s_val, dtype=np.float64)
    d = np.asarray(d_val, dtype=np.float64)

    stack = np.vstack([c, s, d])
    idx = np.argmax(stack, axis=0)
    type_arr = np.array(["C", "S", "D"], dtype=object)
    labels = type_arr[idx]

    selected_value = np.where(
        idx == 0,
        c,
        np.where(idx == 1, s, d),
    )
    return labels, selected_value


def node_homogeneity(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "degree",
                "weighted_degree",
                "kC",
                "kS",
                "kD",
                "homogeneity",
            ]
        )

    long = pd.concat(
        [
            edges_df[["gene_a", "link_type", "weight"]].rename(
                columns={"gene_a": "gene"}
            ),
            edges_df[["gene_b", "link_type", "weight"]].rename(
                columns={"gene_b": "gene"}
            ),
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
    degree = counts.sum(axis=1)

    p = counts.div(degree.replace(0, np.nan), axis=0).fillna(0.0)
    h = (p["C"] ** 2 + p["S"] ** 2 + p["D"] ** 2).astype(float)

    weighted_degree = long.groupby("gene", as_index=True)["weight"].sum()

    out = pd.DataFrame(
        {
            "gene": counts.index.astype(str),
            "degree": degree.to_numpy(dtype=int),
            "weighted_degree": weighted_degree.reindex(counts.index)
            .fillna(0.0)
            .to_numpy(),
            "kC": counts["C"].to_numpy(dtype=int),
            "kS": counts["S"].to_numpy(dtype=int),
            "kD": counts["D"].to_numpy(dtype=int),
            "homogeneity": h.to_numpy(dtype=float),
        }
    )

    return out.sort_values(["weighted_degree", "degree"], ascending=False).reset_index(drop=True)


def attach_gene_change(nodes_df: pd.DataFrame, gene_change_df: pd.DataFrame) -> pd.DataFrame:
    if nodes_df.empty:
        out = pd.DataFrame(columns=list(nodes_df.columns) + ["log2FC", "fdr", "is_deg", "deg_direction"])
        return out

    if gene_change_df.empty:
        out = nodes_df.copy()
        out["log2FC"] = np.nan
        out["fdr"] = np.nan
        out["is_deg"] = False
        out["deg_direction"] = "unknown"
        return out

    merged = nodes_df.merge(gene_change_df, how="left", left_on="gene", right_on="gene")
    merged["is_deg"] = merged["is_deg"].fillna(False).astype(bool)
    merged["deg_direction"] = merged["deg_direction"].fillna("unknown")
    return merged


def build_graph(edges_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for row in edges_df.to_dict(orient="records"):
        g.add_edge(
            str(row["gene_a"]),
            str(row["gene_b"]),
            weight=float(row["weight"]),
            link_type=str(row["link_type"]),
            C=float(row["C"]),
            S=float(row["S"]),
            D=float(row["D"]),
        )
    return g


def export_degree_distribution(g: nx.Graph, out_dir: Path, pair_name: str) -> tuple[Path, Path]:
    degrees = np.array([d for _, d in g.degree()], dtype=int)
    if degrees.size == 0:
        df = pd.DataFrame(columns=["degree", "n_nodes"])
    else:
        uniq, counts = np.unique(degrees, return_counts=True)
        df = pd.DataFrame({"degree": uniq, "n_nodes": counts})

    tsv_file = out_dir / f"{pair_name}_degree_distribution.tsv"
    df.to_csv(tsv_file, sep="\t", index=False)

    png_file = out_dir / f"{pair_name}_degree_distribution_loglog.png"
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if not df.empty:
        x = df["degree"].to_numpy(dtype=float)
        y = df["n_nodes"].to_numpy(dtype=float)
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        ax.loglog(x, y, "o", color="black", label="Observed")
        if x.size >= 3:
            lx = np.log10(x)
            ly = np.log10(y)
            slope, intercept = np.polyfit(lx, ly, 1)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            y_fit = 10 ** intercept * (x_fit ** slope)
            ax.loglog(x_fit, y_fit, "r--", label=f"Fit: y~x^{slope:.3f}")
            ax.legend(frameon=False)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"{pair_name}: degree distribution")
    fig.savefig(png_file, dpi=190, bbox_inches="tight")
    plt.close(fig)

    return tsv_file, png_file


def export_modules(g: nx.Graph, out_dir: Path, pair_name: str) -> tuple[Path, float, str]:
    out_file = out_dir / f"{pair_name}_louvain_modules.tsv"
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        pd.DataFrame(columns=["gene", "module"]).to_csv(
            out_file,
            sep="\t",
            index=False,
        )
        return out_file, float("nan"), "empty_graph"

    method = "networkx_louvain"
    try:
        communities = nx.community.louvain_communities(
            g,
            weight="weight",
            seed=7,
            resolution=1.0,
        )
    except Exception:
        communities = list(greedy_modularity_communities(g, weight="weight"))
        method = "greedy_modularity"

    rows: list[dict[str, object]] = []
    for idx, comm in enumerate(communities):
        for gene in sorted(comm):
            rows.append({"gene": str(gene), "module": int(idx)})

    pd.DataFrame(rows).to_csv(out_file, sep="\t", index=False)
    modularity = float(nx.community.modularity(g, communities, weight="weight"))
    return out_file, modularity, method


def export_csd_files(edges_df: pd.DataFrame, out_dir: Path, pair_name: str) -> dict[str, str]:
    agg_file = out_dir / f"{pair_name}_CSDSelection.txt"
    detail_file = out_dir / f"{pair_name}_CSDSelectionDetailed.txt"
    c_file = out_dir / f"{pair_name}_CNetwork.txt"
    s_file = out_dir / f"{pair_name}_SNetwork.txt"
    d_file = out_dir / f"{pair_name}_DNetwork.txt"

    if edges_df.empty:
        for f in [agg_file, detail_file, c_file, s_file, d_file]:
            pd.DataFrame().to_csv(f, sep="\t", index=False)
        return {
            "c_network": str(c_file),
            "s_network": str(s_file),
            "d_network": str(d_file),
            "aggregate": str(agg_file),
            "detailed": str(detail_file),
        }

    c_df = edges_df[edges_df["link_type"] == "C"][["gene_a", "gene_b", "C"]].copy()
    s_df = edges_df[edges_df["link_type"] == "S"][["gene_a", "gene_b", "S"]].copy()
    d_df = edges_df[edges_df["link_type"] == "D"][["gene_a", "gene_b", "D"]].copy()

    c_df["type"] = "C"
    s_df["type"] = "S"
    d_df["type"] = "D"

    c_df.columns = ["gene_a", "gene_b", "value", "type"]
    s_df.columns = ["gene_a", "gene_b", "value", "type"]
    d_df.columns = ["gene_a", "gene_b", "value", "type"]

    agg = pd.concat([c_df, s_df, d_df], axis=0, ignore_index=True)
    detail = edges_df[
        ["gene_a", "gene_b", "rho_case", "rho_control", "C", "S", "D", "link_type"]
    ].copy()

    c_df.to_csv(c_file, sep="\t", index=False)
    s_df.to_csv(s_file, sep="\t", index=False)
    d_df.to_csv(d_file, sep="\t", index=False)
    agg.to_csv(agg_file, sep="\t", index=False)
    detail.to_csv(detail_file, sep="\t", index=False)

    return {
        "c_network": str(c_file),
        "s_network": str(s_file),
        "d_network": str(d_file),
        "aggregate": str(agg_file),
        "detailed": str(detail_file),
    }


def run_pair(
    case: str,
    ctrl: str,
    in_root: Path,
    expr_root: Path,
    deg_root: Path,
    out_root: Path,
    powers: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, object]:
    pair_name = f"{case}__vs__{ctrl}"
    pair_out = out_root / pair_name
    pair_out.mkdir(parents=True, exist_ok=True)

    corr_case, genes_case, case_file = load_corr_payload(in_root, case)
    corr_ctrl, genes_ctrl, ctrl_file = load_corr_payload(in_root, ctrl)
    corr_case, corr_ctrl, genes_shared = align_corrs(
        corr_case,
        genes_case,
        corr_ctrl,
        genes_ctrl,
    )

    n_case_profiles = load_n_profiles(expr_root, case)
    n_ctrl_profiles = load_n_profiles(expr_root, ctrl)

    deg_df = load_deg_stats(deg_root, case, ctrl)
    deg_all, up_degs, down_degs = pick_degs(
        deg_df,
        fdr_threshold=float(args.fdr_threshold),
        min_abs_log2fc=float(args.min_abs_log2fc),
    )

    (pair_out / "up_degs.txt").write_text(
        "\n".join(up_degs) + ("\n" if up_degs else ""),
        encoding="utf-8",
    )
    (pair_out / "down_degs.txt").write_text(
        "\n".join(down_degs) + ("\n" if down_degs else ""),
        encoding="utf-8",
    )

    if args.use_degs:
        genes_keep = np.array([g for g in genes_shared if g in deg_all], dtype=str)
        if genes_keep.size < 3:
            print(
                f"[warn] {pair_name}: too few DEGs ({genes_keep.size}), "
                "fallback to shared genes"
            )
            genes_keep = genes_shared.copy()
    else:
        genes_keep = genes_shared.copy()

    gene_change_df = build_gene_change_table(
        deg_df=deg_df,
        genes_keep=genes_keep,
        fdr_threshold=float(args.fdr_threshold),
        min_abs_log2fc=float(args.min_abs_log2fc),
    )
    gene_change_file = pair_out / f"{pair_name}_gene_expression_change.csv"
    gene_change_df.to_csv(gene_change_file, index=False)

    idx_map = {g: i for i, g in enumerate(genes_shared)}
    ig = np.array([idx_map[g] for g in genes_keep], dtype=int)

    r_case = corr_case[np.ix_(ig, ig)]
    r_ctrl = corr_ctrl[np.ix_(ig, ig)]

    delta = r_case - r_ctrl
    if args.delta_scale == "half":
        delta = delta / 2.0

    rows: list[dict[str, object]] = []
    for p in powers:
        adj_p = adjacency_from_delta(delta, p, args.network_type)
        k = adj_p.sum(axis=1)
        sf = scale_free_fit_from_connectivity(k, bins=int(args.degree_bins))
        rows.append(
            {
                "pair": pair_name,
                "power": int(p),
                "n_genes": int(genes_keep.size),
                "mean_connectivity": float(np.mean(k)),
                "median_connectivity": float(np.median(k)),
                "max_connectivity": float(np.max(k)),
                **sf,
            }
        )

    scan_df = pd.DataFrame(rows).sort_values("power").reset_index(drop=True)
    scan_csv = pair_out / f"{pair_name}_soft_threshold_scan.csv"
    scan_df.to_csv(scan_csv, index=False)

    scan_png = pair_out / f"{pair_name}_soft_threshold_scan.png"
    plot_soft_threshold(
        scan_df,
        target_r2=float(args.target_signed_r2),
        out_png=scan_png,
        title=f"{pair_name} [{args.network_type}]",
    )

    if args.force_power is not None:
        selected_power = int(args.force_power)
        reason = "forced by --force-power"
    else:
        selected_power, reason = choose_power(
            scan_df,
            target_signed_r2=float(args.target_signed_r2),
            min_mean_k=float(args.min_mean_connectivity),
            r2_plateau_delta=float(args.r2_plateau_delta),
            r2_near_best_delta=float(args.r2_near_best_delta),
        )

    adj = adjacency_from_delta(delta, selected_power, args.network_type)
    adj_file = pair_out / (
        f"{pair_name}_diff_adjacency_beta{selected_power}_{args.network_type}.npz"
    )
    np.savez_compressed(
        adj_file,
        adjacency=np.asarray(adj, dtype=np.float32),
        genes=genes_keep,
    )

    ei, ej, ew = top_edges_from_adjacency(
        adjacency=adj,
        min_weight=float(args.candidate_min_weight),
    )

    if ei.size > 0:
        rv_case = r_case[ei, ej]
        rv_ctrl = r_ctrl[ei, ej]
        c_val, s_val, d_val, denom = csd_scores(
            rv_case,
            rv_ctrl,
            n_case=n_case_profiles,
            n_ctrl=n_ctrl_profiles,
            variance_mode=args.variance_mode,
        )

        all_values = pd.DataFrame(
            {
                "gene_a": genes_keep[ei],
                "gene_b": genes_keep[ej],
                "rho_case": rv_case,
                "rho_control": rv_ctrl,
                "adj_weight": ew,
                "denominator": denom,
                "C": c_val,
                "S": s_val,
                "D": d_val,
            }
        )
    else:
        all_values = pd.DataFrame(
            columns=[
                "gene_a",
                "gene_b",
                "rho_case",
                "rho_control",
                "adj_weight",
                "denominator",
                "C",
                "S",
                "D",
            ]
        )

    all_values_file = pair_out / f"{pair_name}_AllValues.tsv"
    all_values.to_csv(all_values_file, sep="\t", index=False)

    if all_values.empty:
        k_c = float("nan")
        k_s = float("nan")
        k_d = float("nan")
        labels = np.array([], dtype=object)
        sel_value = np.array([], dtype=np.float64)
        edges_df = pd.DataFrame(
            columns=[
                "gene_a",
                "gene_b",
                "weight",
                "rho_case",
                "rho_control",
                "delta_r",
                "C",
                "S",
                "D",
                "link_type",
                "selected_value",
            ]
        )
    else:
        if args.csd_threshold_mode == "importance":
            k_c = importance_threshold(
                all_values["C"].to_numpy(),
                sample_size=int(args.importance_L),
                n_samples=int(args.importance_m),
                rng=rng,
            )
            k_s = importance_threshold(
                all_values["S"].to_numpy(),
                sample_size=int(args.importance_L),
                n_samples=int(args.importance_m),
                rng=rng,
            )
            k_d = importance_threshold(
                all_values["D"].to_numpy(),
                sample_size=int(args.importance_L),
                n_samples=int(args.importance_m),
                rng=rng,
            )

            labels, sel_value = classify_edges_by_thresholds(
                all_values["C"].to_numpy(),
                all_values["S"].to_numpy(),
                all_values["D"].to_numpy(),
                k_c=k_c,
                k_s=k_s,
                k_d=k_d,
            )
            keep = labels != ""
        else:
            k_c = float("nan")
            k_s = float("nan")
            k_d = float("nan")
            labels, sel_value = classify_edges_no_threshold(
                all_values["C"].to_numpy(),
                all_values["S"].to_numpy(),
                all_values["D"].to_numpy(),
            )
            keep = np.ones(all_values.shape[0], dtype=bool)

        if np.any(keep):
            tmp = all_values.loc[keep].copy().reset_index(drop=True)
            tmp["link_type"] = labels[keep]
            tmp["selected_value"] = sel_value[keep]
            tmp["weight"] = tmp["adj_weight"]
            tmp["delta_r"] = tmp["rho_case"] - tmp["rho_control"]
            edges_df = tmp[
                [
                    "gene_a",
                    "gene_b",
                    "weight",
                    "rho_case",
                    "rho_control",
                    "delta_r",
                    "C",
                    "S",
                    "D",
                    "link_type",
                    "selected_value",
                ]
            ].copy()
            edges_df = edges_df.sort_values(
                ["selected_value", "weight"],
                ascending=False,
            ).reset_index(drop=True)
        else:
            edges_df = pd.DataFrame(
                columns=[
                    "gene_a",
                    "gene_b",
                    "weight",
                    "rho_case",
                    "rho_control",
                    "delta_r",
                    "C",
                    "S",
                    "D",
                    "link_type",
                    "selected_value",
                ]
            )

    thresholds = {
        "csd_threshold_mode": str(args.csd_threshold_mode),
        "importance_L": int(args.importance_L),
        "importance_m": int(args.importance_m),
        "k_C": k_c,
        "k_S": k_s,
        "k_D": k_d,
        "p_estimate": 1.0 / float(max(int(args.importance_L), 1)),
    }
    thresholds_file = pair_out / f"{pair_name}_importance_thresholds.json"
    thresholds_file.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    edges_file = pair_out / f"{pair_name}_differential_edges_scalefree.csv"
    edges_df.to_csv(edges_file, index=False)

    nodes_df = node_homogeneity(edges_df)
    nodes_df = attach_gene_change(nodes_df, gene_change_df)
    nodes_file = pair_out / f"{pair_name}_node_homogeneity_scalefree.csv"
    nodes_df.to_csv(nodes_file, index=False)

    hubs_df = nodes_df.head(int(args.top_hubs)).copy()
    hubs_file = pair_out / f"{pair_name}_top_hubs_scalefree.csv"
    hubs_df.to_csv(hubs_file, index=False)

    csd_files = export_csd_files(edges_df, pair_out, pair_name)

    g = build_graph(edges_df)
    degree_tsv, degree_png = export_degree_distribution(g, pair_out, pair_name)

    if args.skip_modules:
        modules_file = pair_out / f"{pair_name}_louvain_modules.tsv"
        pd.DataFrame(columns=["gene", "module"]).to_csv(
            modules_file,
            sep="\t",
            index=False,
        )
        modularity = float("nan")
        module_method = "skipped"
    else:
        modules_file, modularity, module_method = export_modules(g, pair_out, pair_name)

    summary = {
        "pair": pair_name,
        "case": case,
        "control": ctrl,
        "use_degs": bool(args.use_degs),
        "fdr_threshold": float(args.fdr_threshold),
        "min_abs_log2fc": float(args.min_abs_log2fc),
        "n_degs_total": int(len(deg_all)),
        "n_up_degs": int(len(up_degs)),
        "n_down_degs": int(len(down_degs)),
        "network_type": args.network_type,
        "delta_scale": args.delta_scale,
        "variance_mode": args.variance_mode,
        "csd_threshold_mode": args.csd_threshold_mode,
        "n_case_profiles": int(n_case_profiles),
        "n_control_profiles": int(n_ctrl_profiles),
        "n_genes_shared": int(genes_shared.size),
        "n_genes_used": int(genes_keep.size),
        "selected_power": int(selected_power),
        "selection_reason": reason,
        "candidate_min_weight": float(args.candidate_min_weight),
        "n_candidate_edges": int(all_values.shape[0]),
        "n_edges_exported": int(edges_df.shape[0]),
        "n_nodes_exported": int(g.number_of_nodes()),
        "modularity": modularity,
        "community_method": module_method,
        "source_case_corr": str(case_file),
        "source_control_corr": str(ctrl_file),
        "scan_csv": str(scan_csv),
        "scan_png": str(scan_png),
        "adjacency_file": str(adj_file),
        "all_values_file": str(all_values_file),
        "thresholds_file": str(thresholds_file),
        "edges_file": str(edges_file),
        "nodes_file": str(nodes_file),
        "gene_change_file": str(gene_change_file),
        "hubs_file": str(hubs_file),
        "degree_distribution_tsv": str(degree_tsv),
        "degree_distribution_png": str(degree_png),
        "modules_file": str(modules_file),
        **csd_files,
    }

    summary_file = pair_out / f"{pair_name}_summary_scalefree.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[{pair_name}] genes={summary['n_genes_used']} "
        f"beta={summary['selected_power']} edges={summary['n_edges_exported']}"
    )
    return summary


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    expr_root = resolve_base(args.expr_dir)
    deg_root = resolve_base(args.deg_dir)
    out_root = resolve_base(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    conditions = list_conditions(in_root)
    pairs = parse_pairs(args.pair, conditions)
    powers = parse_powers(args.powers)

    rng = np.random.default_rng(int(args.seed))

    summaries: list[dict[str, object]] = []
    records: list[WarehouseRecord] = []

    for case, ctrl in pairs:
        summary = run_pair(
            case=case,
            ctrl=ctrl,
            in_root=in_root,
            expr_root=expr_root,
            deg_root=deg_root,
            out_root=out_root,
            powers=powers,
            args=args,
            rng=rng,
        )
        summaries.append(summary)
        records.append(
            WarehouseRecord(
                input_file=str(summary["source_case_corr"]),
                output_file=str(summary["edges_file"]),
                script=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                date_utc=utc_now_iso(),
                params_hash=params_hash(vars(args)),
                condition=str(summary["pair"]),
                stage="08j_differential_scalefree_csd",
            )
        )

    summary_csv = out_root / "differential_scalefree_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    append_warehouse(out_root, records)
    print(f"Done. Differential scale-free outputs: {out_root}")


if __name__ == "__main__":
    main()
