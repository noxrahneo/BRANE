#!/usr/bin/env python3
# flake8: noqa: E501
"""Shared utilities for stage-09 differential network scripts.

This module intentionally reuses ideas from stage-07 scripts while keeping
all stage-09 logic isolated from existing workflows.
"""

from __future__ import annotations

import json
from pathlib import Path

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .h5ad_compat import read_h5ad_compat

REPO_ROOT = Path(__file__).resolve().parents[1]
NORMAL = "Normal"
PRENEO = "Normal_BRCA1_-_pre-neoplastic"
TN_BRCA1 = "Triple_negative_BRCA1_tumor"


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


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
    need = {"gene", "lnFC", "fdr"}
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
        & pd.to_numeric(work["lnFC"], errors="coerce").abs().ge(float(min_abs_log2fc))
    )
    work = work[mask].copy()
    up = work[work["lnFC"] >= 0.0]["gene"].tolist()
    down = work[work["lnFC"] < 0.0]["gene"].tolist()
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
    work["lnFC"] = pd.to_numeric(work["lnFC"], errors="coerce")
    work["fdr"] = pd.to_numeric(work["fdr"], errors="coerce")

    keep_set = set(np.asarray(genes_keep, dtype=str).tolist())
    work = work[work["gene"].isin(keep_set)].copy()

    sig = work["fdr"].le(float(fdr_threshold)) & work["lnFC"].abs().ge(float(min_abs_log2fc))
    direction = np.where(
        sig & (work["lnFC"] > 0),
        "up",
        np.where(sig & (work["lnFC"] < 0), "down", "unchanged"),
    )
    work["deg_direction"] = direction
    work["is_deg"] = sig.astype(bool)

    out = work[["gene", "lnFC", "fdr", "is_deg", "deg_direction"]].copy()
    out = out.drop_duplicates(subset=["gene"]).sort_values("gene").reset_index(drop=True)
    return out


def load_n_profiles(expr_root: Path, condition: str) -> int:
    h5ad_file = expr_root / f"{condition}_pseudobulk_logcpm.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(f"Missing expression file: {h5ad_file}")
    adata = read_h5ad_compat(h5ad_file)
    return int(adata.n_obs)


def compute_allvalues(corr_case: np.ndarray, corr_ctrl: np.ndarray, genes_keep: np.ndarray) -> pd.DataFrame:
    ii, jj = np.triu_indices(corr_case.shape[0], k=1)
    rv_case = corr_case[ii, jj]
    rv_ctrl = corr_ctrl[ii, jj]

    # Stage-09 design: denominator fixed to one for both branches.
    c_val = np.abs(rv_case + rv_ctrl)
    s_val = np.abs(np.abs(rv_case) - np.abs(rv_ctrl))
    d_val = np.abs(rv_case) + np.abs(rv_ctrl) - np.abs(rv_case + rv_ctrl)

    return pd.DataFrame(
        {
            "gene_a": genes_keep[ii],
            "gene_b": genes_keep[jj],
            "rho_case": rv_case,
            "rho_control": rv_ctrl,
            "denominator": np.ones(ii.shape[0], dtype=np.float64),
            "C": c_val,
            "S": s_val,
            "D": d_val,
        }
    )


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
        plateau_candidates: list[int] = []
        eligible_powers = eligible["power"].astype(int).tolist()
        for power in eligible_powers:
            row_match = eligible[eligible["power"] == power]
            if row_match.empty:
                continue
            row_signed_r2 = float(row_match.iloc[0]["signed_r2"])
            later = eligible[eligible["power"] >= power]
            if later.empty:
                continue
            later_max_r2 = float(later["signed_r2"].max())
            if later_max_r2 <= row_signed_r2 + float(r2_plateau_delta):
                plateau_candidates.append(power)

        if plateau_candidates:
            p = int(min(plateau_candidates))
            return p, "lowest eligible power in plateau"

        p = int(eligible.sort_values("power").iloc[0]["power"])
        return p, "lowest power meeting signed R2 and mean connectivity"

    pool = ordered[ordered["mean_connectivity"] >= float(min_mean_k)].copy()
    if pool.empty:
        pool = ordered.copy()

    r2_vals = pd.to_numeric(pool["signed_r2"], errors="coerce")
    r2_vals = r2_vals[np.isfinite(r2_vals)]
    if r2_vals.empty:
        p = int(pool.sort_values("power").iloc[0]["power"])
        return p, "fallback lowest power"

    best_r2 = float(r2_vals.max())
    near = pool[pool["signed_r2"] >= best_r2 - float(r2_near_best_delta)]
    near = near.sort_values("power")
    return int(near.iloc[0]["power"]), "fallback near-best signed R2"


def top_edges_from_adjacency(adjacency: np.ndarray, min_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(adjacency, dtype=np.float64)
    tri_i, tri_j = np.triu_indices(a.shape[0], k=1)
    vals = a[tri_i, tri_j]
    keep = np.where(vals >= float(min_weight))[0]
    if keep.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    return tri_i[keep], tri_j[keep], vals[keep]


def classify_edges_no_threshold(c_val: np.ndarray, s_val: np.ndarray, d_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c_val, dtype=np.float64)
    s = np.asarray(s_val, dtype=np.float64)
    d = np.asarray(d_val, dtype=np.float64)

    stack = np.vstack([c, s, d])
    idx = np.argmax(stack, axis=0)
    type_arr = np.array(["C", "S", "D"], dtype=object)
    labels = type_arr[idx]

    selected_value = np.where(idx == 0, c, np.where(idx == 1, s, d))
    return labels, selected_value


def attach_gene_change(nodes_df: pd.DataFrame, gene_change_df: pd.DataFrame) -> pd.DataFrame:
    if nodes_df.empty:
        out = pd.DataFrame(columns=list(nodes_df.columns) + ["lnFC", "fdr", "is_deg", "deg_direction"])
        return out

    if gene_change_df.empty:
        out = nodes_df.copy()
        out["lnFC"] = np.nan
        out["fdr"] = np.nan
        out["is_deg"] = False
        out["deg_direction"] = "unknown"
        return out

    merged = nodes_df.merge(gene_change_df, how="left", left_on="gene", right_on="gene")
    merged["is_deg"] = merged["is_deg"].fillna(False).astype(bool)
    merged["deg_direction"] = merged["deg_direction"].fillna("unknown")
    return merged


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
    detail = edges_df[["gene_a", "gene_b", "rho_case", "rho_control", "C", "S", "D", "link_type"]].copy()

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
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title(f"{pair_name}: degree distribution")
    fig.savefig(png_file, dpi=190, bbox_inches="tight")
    plt.close(fig)

    return tsv_file, png_file


def safe_knn_exponent(g: nx.Graph) -> float:
    if g.number_of_nodes() < 3 or g.number_of_edges() < 2:
        return float("nan")
    try:
        knn = nx.average_degree_connectivity(g)
    except Exception:
        return float("nan")
    if not knn:
        return float("nan")

    k = np.array(sorted(knn.keys()), dtype=float)
    knn_vals = np.array([knn[int(x)] for x in k], dtype=float)
    mask = (k > 0) & (knn_vals > 0) & np.isfinite(knn_vals)
    if np.sum(mask) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log10(k[mask]), np.log10(knn_vals[mask]), 1)
    return float(slope)


def _node_avg_wto_from_edges(edges_df: pd.DataFrame) -> dict[str, float]:
    if edges_df.empty or "wTO" not in edges_df.columns:
        return {}
    a = edges_df[["gene_a", "wTO"]].rename(columns={"gene_a": "gene"})
    b = edges_df[["gene_b", "wTO"]].rename(columns={"gene_b": "gene"})
    long_df = pd.concat([a, b], axis=0, ignore_index=True)
    long_df["gene"] = long_df["gene"].astype(str)
    long_df["wTO"] = pd.to_numeric(long_df["wTO"], errors="coerce")
    long_df = long_df[np.isfinite(long_df["wTO"])].copy()
    if long_df.empty:
        return {}
    means = long_df.groupby("gene", as_index=False)["wTO"].mean()
    out: dict[str, float] = {}
    for row in means.itertuples(index=False):
        out[str(row.gene)] = float(row.wTO)
    return out


def augment_edges_with_wto(edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Append weighted topological overlap (wTO) to edge list.

    wTO_{ij} = (sum_u a_iu a_uj + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    where k_i = sum_u a_iu for weighted adjacency A.
    """
    if edges_df.empty:
        out = edges_df.copy()
        out["wTO"] = pd.Series(dtype=float)
        wto_edges = pd.DataFrame(columns=["gene_a", "gene_b", "weight", "wTO", "link_type", "C", "S", "D"])
        return out, wto_edges, {}

    need = {"gene_a", "gene_b", "weight"}
    if not need.issubset(edges_df.columns):
        raise ValueError(f"edges_df must include {need} to compute wTO")

    work = edges_df.copy()
    work["gene_a"] = work["gene_a"].astype(str)
    work["gene_b"] = work["gene_b"].astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)

    genes = sorted(set(work["gene_a"]).union(set(work["gene_b"])))
    idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    ia = work["gene_a"].map(idx).to_numpy(dtype=int)
    ib = work["gene_b"].map(idx).to_numpy(dtype=int)
    w = work["weight"].to_numpy(dtype=float)
    w = np.maximum(w, 0.0)

    rows = np.concatenate([ia, ib])
    cols = np.concatenate([ib, ia])
    vals = np.concatenate([w, w])
    a_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    k = np.asarray(a_mat.sum(axis=1), dtype=np.float64).ravel()
    aa = sparse.csr_matrix(a_mat @ a_mat, dtype=np.float64)

    shared = np.asarray(aa[ia, ib]).ravel()
    aij = w
    denom = np.minimum(k[ia], k[ib]) + 1.0 - aij
    wto = np.full(work.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    wto[valid] = (shared[valid] + aij[valid]) / denom[valid]

    work["wTO"] = wto

    keep_cols = [c for c in ["gene_a", "gene_b", "weight", "wTO", "link_type", "C", "S", "D"] if c in work.columns]
    wto_edges = work[keep_cols].copy().sort_values("wTO", ascending=False, na_position="last").reset_index(drop=True)

    node_avg = _node_avg_wto_from_edges(work)
    return work, wto_edges, node_avg


def compute_node_and_network_metrics(
    edges_df: pd.DataFrame,
    gene_change_df: pd.DataFrame,
    pair_name: str,
    out_dir: Path,
    top_hubs: int,
    node_avg_wto: dict[str, float] | None = None,
) -> dict[str, object]:
    g = nx.Graph()
    for row in edges_df.to_dict(orient="records"):
        g.add_edge(
            str(row["gene_a"]),
            str(row["gene_b"]),
            weight=float(row.get("weight", 1.0)),
            link_type=str(row.get("link_type", "C")),
            C=float(row.get("C", np.nan)),
            S=float(row.get("S", np.nan)),
            D=float(row.get("D", np.nan)),
        )

    weighted_degree = dict(g.degree(weight="weight"))
    degree = dict(g.degree())

    if g.number_of_nodes() > 0:
        clustering_raw = nx.clustering(g, weight="weight")
        if isinstance(clustering_raw, dict):
            clustering: dict[str, float] = {
                str(k): float(v)
                for k, v in clustering_raw.items()
            }
        else:
            clustering = {}
        closeness: dict[str, float] = {
            str(k): float(v)
            for k, v in nx.closeness_centrality(g).items()
        }
        inv_weight_graph = g.copy()
        for u, v, d in inv_weight_graph.edges(data=True):
            w = float(d.get("weight", 0.0))
            d["distance"] = 1.0 / max(w, 1e-12)
        betweenness: dict[str, float] = {
            str(k): float(v)
            for k, v in nx.betweenness_centrality(
                inv_weight_graph,
                weight="distance",
            ).items()
        }
    else:
        clustering = {}
        closeness = {}
        betweenness = {}

    nodes_df = pd.DataFrame(
        {
            "gene": list(weighted_degree.keys()),
            "degree": [int(degree[gname]) for gname in weighted_degree.keys()],
            "weighted_degree": [float(weighted_degree[gname]) for gname in weighted_degree.keys()],
            "clustering_coefficient": [float(clustering.get(gname, np.nan)) for gname in weighted_degree.keys()],
            "closeness_centrality": [float(closeness.get(gname, np.nan)) for gname in weighted_degree.keys()],
            "betweenness_centrality": [float(betweenness.get(gname, np.nan)) for gname in weighted_degree.keys()],
        }
    )

    if g.number_of_nodes() > 0 and g.number_of_edges() > 0:
        #build igraph from networkx for leiden
        nodes = list(g.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        ig_g = ig.Graph(
            n=len(nodes),
            edges=[(node_idx[u], node_idx[v]) for u, v in g.edges()],
            edge_attrs={"weight": [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]},
        )
        partition = leidenalg.find_partition(
            ig_g, leidenalg.ModularityVertexPartition, weights="weight", seed=7
        )
        #sort by size descending so module 1 is the largest
        communities = sorted(
            [frozenset(nodes[i] for i in c) for c in partition], key=len, reverse=True
        )
        module_method = "leiden"

        module_rows: list[dict[str, object]] = []
        module_map: dict[str, int] = {}
        for mod_id, comm in enumerate(communities, start=1):
            for gene in sorted(comm):
                module_rows.append({"gene": str(gene), "module": int(mod_id)})
                module_map[str(gene)] = int(mod_id)
        modules_df = pd.DataFrame(module_rows)
        modularity_q = float(nx.community.modularity(g, communities, weight="weight"))
    else:
        module_method = "none"
        modularity_q = float("nan")
        modules_df = pd.DataFrame(columns=["gene", "module"])
        module_map = {}

    if not nodes_df.empty:
        nodes_df["module"] = nodes_df["gene"].map(module_map).fillna(-1).astype(int)

    nodes_df = attach_gene_change(nodes_df, gene_change_df)
    if node_avg_wto is None:
        node_avg_wto = _node_avg_wto_from_edges(edges_df)
    nodes_df["avg_wTO"] = nodes_df["gene"].map(node_avg_wto).astype(float)
    nodes_df = nodes_df.sort_values(["weighted_degree", "degree"], ascending=False).reset_index(drop=True)

    top_hubs_df = nodes_df.head(int(top_hubs)).copy()

    degree_tsv, degree_png = export_degree_distribution(g, out_dir, pair_name)

    if g.number_of_nodes() > 0 and g.number_of_edges() > 0:
        assortativity = float(nx.degree_assortativity_coefficient(g))
        avg_clustering = float(nx.average_clustering(g, weight="weight"))
    else:
        assortativity = float("nan")
        avg_clustering = float("nan")

    knn_exponent = safe_knn_exponent(g)
    if np.isfinite(knn_exponent):
        if knn_exponent > 0.05:
            knn_pattern = "assortative"
        elif knn_exponent < -0.05:
            knn_pattern = "disassortative"
        else:
            knn_pattern = "neutral"
    else:
        knn_pattern = "undetermined"

    network_metrics = {
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "assortativity_coefficient": assortativity,
        "knn_exponent": float(knn_exponent),
        "knn_pattern": knn_pattern,
        "global_modularity_q": modularity_q,
        "average_clustering_coefficient": avg_clustering,
        "mean_edge_wTO": float(pd.to_numeric(edges_df.get("wTO", pd.Series(dtype=float)), errors="coerce").mean()),
        "median_edge_wTO": float(pd.to_numeric(edges_df.get("wTO", pd.Series(dtype=float)), errors="coerce").median()),
        "module_method": module_method,
        "degree_distribution_tsv": str(degree_tsv),
        "degree_distribution_png": str(degree_png),
    }

    return {
        "graph": g,
        "nodes_df": nodes_df,
        "top_hubs_df": top_hubs_df,
        "modules_df": modules_df,
        "network_metrics": network_metrics,
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
