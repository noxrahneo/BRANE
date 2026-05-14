#!/usr/bin/env python3
"""Detect TOM-based modules and compute eigengene/trait associations.

This script consumes TOM matrices produced by 23_network_power_tom_prep.py
and performs a WGCNA-style module workflow:
1) TOM distance = 1 - TOM,
2) hierarchical clustering on genes,
3) module assignment by dynamic-style distance cut,
4) module eigengene computation,
5) eigengene-based module merging,
6) module-trait association from pseudobulk metadata.

Note: this is a Python approximation of WGCNA/cutreeDynamic behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TOM-based module detection and eigengene analysis"
    )
    parser.add_argument(
        "--input-dir",
        default="results/11_coexpression_prep/single",
        help=(
            "Root with condition TOM outputs from "
            "23_network_power_tom_prep.py"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/12_network_modules/single",
        help="Output root for module assignments and eigengene results",
    )
    parser.add_argument(
        "--expr-dir",
        default="results/08_pre_correlation/per_condition",
        help="Directory with <condition>_pseudobulk_logcpm.h5ad files",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned", "both"],
        default="signed",
        help="Network type folder layout used in input/output",
    )
    parser.add_argument(
        "--cut-height",
        type=float,
        default=0.95,
        help=(
            "Static distance cut for fcluster "
            "(used when dynamic cut is disabled)"
        ),
    )
    parser.add_argument(
        "--disable-dynamic-cut",
        action="store_true",
        help="Disable dynamic-style cut search and use --cut-height directly",
    )
    parser.add_argument(
        "--deep-split",
        type=int,
        default=2,
        help=(
            "Dynamic cut sensitivity (0..4, "
            "higher tends to produce more modules)"
        ),
    )
    parser.add_argument(
        "--merge-cut-height",
        type=float,
        default=0.25,
        help="Eigengene merge dissimilarity threshold (0.25 ~ corr 0.75)",
    )
    parser.add_argument(
        "--max-largest-module-fraction",
        type=float,
        default=0.35,
        help=(
            "For dynamic cut selection, prefer cuts where the largest "
            "module is <= this fraction of non-grey genes"
        ),
    )
    parser.add_argument(
        "--disable-module-merge",
        action="store_true",
        help="Disable eigengene-based module merging",
    )
    parser.add_argument(
        "--min-module-size",
        type=int,
        default=30,
        help="Minimum genes per module; smaller clusters become module 0",
    )
    parser.add_argument(
        "--max-categorical-levels",
        type=int,
        default=8,
        help="Max categories for one-hot trait expansion",
    )
    parser.add_argument(
        "--skip-trait-association",
        action="store_true",
        help="Skip module-trait association outputs",
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


def list_condition_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def resolve_conditions(root: Path, requested: str) -> list[Path]:
    dirs = list_condition_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No condition directories in {root}")
    if requested.strip().lower() == "all":
        return dirs
    match = [d for d in dirs if d.name == requested]
    if not match:
        raise ValueError(
            f"Condition '{requested}' not found; "
            f"available={[d.name for d in dirs]}"
        )
    return match


def find_tom_npz(cond_dir: Path, condition: str, network_type: str) -> Path:
    preferred = sorted(
        cond_dir.glob(f"{condition}_tom_beta*_{network_type}.npz")
    )
    if preferred:
        return preferred[0]
    fallback = sorted(cond_dir.glob(f"{condition}_tom_beta*.npz"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"No TOM npz found in {cond_dir}")


def load_tom_payload(tom_file: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(tom_file, allow_pickle=True)
    tom = np.asarray(payload["tom"], dtype=np.float64)
    genes = payload["genes"].astype(str)
    if tom.shape[0] != tom.shape[1] or tom.shape[0] != genes.shape[0]:
        raise ValueError(f"Invalid TOM dimensions in {tom_file}")
    tom = np.clip(tom, 0.0, 1.0)
    np.fill_diagonal(tom, 1.0)
    return tom, genes


def build_linkage_from_tom(tom: np.ndarray) -> np.ndarray:
    dist = 1.0 - np.asarray(tom, dtype=np.float64)
    dist = np.clip(dist, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method="average")


def candidate_cut_heights(link: np.ndarray, deep_split: int) -> np.ndarray:
    ds = int(np.clip(int(deep_split), 0, 4))
    if link.size == 0:
        return np.array([0.95], dtype=float)

    distances = np.asarray(link[:, 2], dtype=float)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return np.array([0.95], dtype=float)

    # Higher deepSplit explores finer cuts (lower dendrogram heights).
    quantile_ranges: dict[int, tuple[float, float]] = {
        0: (0.85, 0.995),
        1: (0.75, 0.99),
        2: (0.65, 0.98),
        3: (0.55, 0.97),
        4: (0.45, 0.96),
    }
    q_lo, q_hi = quantile_ranges[ds]
    qs = np.linspace(q_lo, q_hi, 14)
    heights = np.quantile(distances, qs)
    heights = np.unique(
        np.clip(
            heights,
            float(distances.min()),
            float(distances.max()),
        )
    )
    return heights.astype(float)


def select_modules_dynamic(
    link: np.ndarray,
    min_size: int,
    deep_split: int,
    max_largest_module_fraction: float,
) -> tuple[np.ndarray, float]:
    candidates: list[tuple[np.ndarray, float, int, int, float]] = []

    for h in candidate_cut_heights(link, deep_split):
        raw = fcluster(link, t=float(h), criterion="distance")
        labels = relabel_modules(raw, min_size=min_size)
        non_grey_mask = labels > 0
        non_grey_genes = int(np.sum(non_grey_mask))
        n_modules = int(len([x for x in np.unique(labels) if int(x) > 0]))
        if non_grey_genes > 0:
            _, counts = np.unique(labels[non_grey_mask], return_counts=True)
            largest = int(counts.max())
            largest_frac = float(largest / non_grey_genes)
        else:
            largest_frac = 1.0
        candidates.append(
            (labels, float(h), non_grey_genes, n_modules, largest_frac)
        )

    if not candidates:
        raw = fcluster(link, t=0.95, criterion="distance")
        return relabel_modules(raw, min_size=min_size), 0.95

    feasible = [
        c
        for c in candidates
        if c[2] > 0
        and c[3] >= 2
        and c[4] <= float(max_largest_module_fraction)
    ]

    if feasible:
        best_labels, best_height, _, _, _ = max(
            feasible,
            key=lambda c: (c[2], c[3], -c[1]),
        )
        return best_labels, float(best_height)

    best_labels, best_height, _, _, _ = max(
        candidates,
        key=lambda c: (c[2], c[3], -c[1]),
    )

    return best_labels, float(best_height)


def relabel_modules(raw_labels: np.ndarray, min_size: int) -> np.ndarray:
    labels = np.asarray(raw_labels, dtype=int).copy()
    uniq, cnts = np.unique(labels, return_counts=True)

    # Assign clusters smaller than min_size to grey module 0.
    for lab, n in zip(uniq.tolist(), cnts.tolist()):
        if int(n) < int(min_size):
            labels[labels == int(lab)] = 0

    nonzero = sorted([x for x in np.unique(labels) if int(x) > 0])
    mapping = {old: new for new, old in enumerate(nonzero, start=1)}
    relabeled = np.array([mapping.get(int(x), 0) for x in labels], dtype=int)
    return relabeled


def module_color(module_id: int) -> str:
    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]
    if int(module_id) == 0:
        return "#9CA3AF"
    return palette[(int(module_id) - 1) % len(palette)]


def compute_module_eigengenes(
    expr: np.ndarray,
    module_ids: np.ndarray,
    profile_ids: np.ndarray,
) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {}
    for module_id in sorted(np.unique(module_ids)):
        if int(module_id) == 0:
            continue
        idx = np.where(module_ids == int(module_id))[0]
        if idx.size == 0:
            continue

        x = np.asarray(expr[:, idx], dtype=np.float64)
        x = x - np.nanmean(x, axis=0, keepdims=True)
        std = np.nanstd(x, axis=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        x = x / std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.shape[1] == 1:
            eig = x[:, 0]
        else:
            u, _, _ = np.linalg.svd(x, full_matrices=False)
            eig = u[:, 0]

        mean_signal = np.nanmean(expr[:, idx], axis=1)
        if np.isfinite(mean_signal).sum() >= 3:
            corr = np.corrcoef(eig, mean_signal)[0, 1]
            if np.isfinite(corr) and corr < 0:
                eig = -eig

        rows[f"ME_M{int(module_id):03d}"] = eig

    if not rows:
        return pd.DataFrame(index=profile_ids.astype(str))

    df = pd.DataFrame(rows, index=profile_ids.astype(str))
    df.index.name = "profile_id"
    return df


def module_id_from_me_col(col: str) -> int:
    token = str(col).replace("ME_M", "")
    return int(token)


def merge_modules_by_eigengene(
    expr: np.ndarray,
    module_ids: np.ndarray,
    profile_ids: np.ndarray,
    merge_cut_height: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = np.asarray(module_ids, dtype=int).copy()
    non_grey = sorted([int(x) for x in np.unique(labels) if int(x) > 0])
    if len(non_grey) < 2:
        merge_map = pd.DataFrame(
            {
                "module_premerge": non_grey,
                "module_postmerge": non_grey,
            }
        )
        return labels, merge_map

    eig_df = compute_module_eigengenes(expr, labels, profile_ids)
    if eig_df.shape[1] < 2:
        merge_map = pd.DataFrame(
            {
                "module_premerge": non_grey,
                "module_postmerge": non_grey,
            }
        )
        return labels, merge_map

    corr = eig_df.corr(method="pearson").to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    diss = 1.0 - corr
    diss = np.clip(diss, 0.0, 2.0)
    np.fill_diagonal(diss, 0.0)

    me_link = linkage(squareform(diss, checks=False), method="average")
    me_clusters = fcluster(
        me_link,
        t=float(merge_cut_height),
        criterion="distance",
    )

    me_cols = eig_df.columns.tolist()
    old_module_ids = [module_id_from_me_col(c) for c in me_cols]
    old_to_cluster = {
        int(old): int(clu)
        for old, clu in zip(old_module_ids, me_clusters.tolist())
    }

    cluster_ids = sorted(set(old_to_cluster.values()))
    cluster_to_new = {
        int(c): int(i)
        for i, c in enumerate(cluster_ids, start=1)
    }
    old_to_new = {
        int(old): int(cluster_to_new[int(clu)])
        for old, clu in old_to_cluster.items()
    }

    merged = np.array([old_to_new.get(int(x), 0) for x in labels], dtype=int)
    merge_map = pd.DataFrame(
        {
            "module_premerge": sorted(old_to_new.keys()),
            "module_postmerge": [
                old_to_new[k]
                for k in sorted(old_to_new.keys())
            ],
        }
    )
    return merged, merge_map


def largest_module_fraction(module_ids: np.ndarray) -> float:
    labels = np.asarray(module_ids, dtype=int)
    non_grey = labels[labels > 0]
    if non_grey.size == 0:
        return 1.0
    _, counts = np.unique(non_grey, return_counts=True)
    return float(int(counts.max()) / int(non_grey.size))


def build_trait_matrix(obs: pd.DataFrame, max_levels: int) -> pd.DataFrame:
    traits: list[pd.DataFrame] = []
    for col in obs.columns:
        s = obs[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() >= max(3, int(0.5 * len(s))):
            if num.nunique(dropna=True) > 1:
                traits.append(pd.DataFrame({str(col): num}))
            continue

        cat = s.astype(str).fillna("")
        n_levels = cat.nunique(dropna=True)
        if 1 < n_levels <= int(max_levels):
            dummies = pd.get_dummies(cat, prefix=str(col), dtype=float)
            traits.append(dummies)

    if not traits:
        return pd.DataFrame(index=obs.index)

    trait_df = pd.concat(traits, axis=1)
    trait_df = trait_df.loc[:, ~trait_df.columns.duplicated()].copy()
    trait_df.index = obs.index
    return trait_df


def module_trait_association(
    eig_df: pd.DataFrame,
    trait_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if eig_df.empty or trait_df.empty:
        return pd.DataFrame(
            columns=["module", "trait", "pearson_r", "p_value", "n"]
        )

    aligned_traits = trait_df.reindex(eig_df.index)
    for me_col in eig_df.columns:
        x = pd.to_numeric(eig_df[me_col], errors="coerce")
        for tr_col in aligned_traits.columns:
            y = pd.to_numeric(aligned_traits[tr_col], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < 3:
                continue
            try:
                r, p = pearsonr(x[mask].to_numpy(), y[mask].to_numpy())
            except Exception:
                continue
            r_val = float(np.asarray(r).item())
            p_val = float(np.asarray(p).item())
            rows.append(
                {
                    "module": str(me_col),
                    "trait": str(tr_col),
                    "pearson_r": r_val,
                    "p_value": p_val,
                    "n": n,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["p_value", "trait"]).reset_index(drop=True)


def plot_dendrogram(
    link: np.ndarray,
    out_file: Path,
    title: str,
    module_ids: np.ndarray | None = None,
    cut_height: float | None = None,
) -> None:
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 0.5],
        hspace=0.05,
    )
    ax_tree = fig.add_subplot(gs[0])
    ax_colors = fig.add_subplot(gs[1])

    # Use default black dendrogram; don't color threshold
    dend = dendrogram(
        link,
        no_labels=True,
        ax=ax_tree,
    )
    ax_tree.set_title(title)
    ax_tree.set_xlabel("Genes")
    ax_tree.set_ylabel("TOM distance")

    if module_ids is not None:
        from matplotlib.patches import Rectangle

        module_ids_arr = np.asarray(module_ids, dtype=int)
        leaf_order = dend["leaves"]
        n_genes = len(leaf_order)
        for idx, gene_idx in enumerate(leaf_order):
            mod_id = int(module_ids_arr[gene_idx])
            col = module_color(mod_id)
            rect = Rectangle(
                (idx, 0),
                1,
                1,
                facecolor=col,
                edgecolor="none",
            )
            ax_colors.add_patch(rect)
        ax_colors.set_xlim(0, n_genes)
        ax_colors.set_ylim(0, 1)
        ax_colors.set_yticks([])
        ax_colors.set_xticks([])
        ax_colors.set_ylabel("Dynamic Tree Cut", fontsize=10, fontweight="bold")
    else:
        ax_colors.axis("off")

    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.95)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_module_sizes(
    module_size_df: pd.DataFrame,
    out_file: Path,
    title: str,
) -> None:
    df = module_size_df[module_size_df["module_id"] != 0].copy()
    if df.empty:
        return
    x = np.arange(df.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, df["n_genes"].to_numpy(), color=df["module_color"].tolist())
    ax.set_xticks(x)
    ax.set_xticklabels(df["module_label"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Genes")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_module_trait_heatmap(
    assoc_df: pd.DataFrame,
    out_file: Path,
    title: str,
) -> None:
    if assoc_df.empty:
        return
    pivot = assoc_df.pivot_table(
        index="module",
        columns="trait",
        values="pearson_r",
        aggfunc="mean",
    )
    if pivot.empty:
        return

    arr = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.25 * arr.shape[1]), max(3, 0.4 * arr.shape[0]))
    )
    im = ax.imshow(arr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(
        pivot.columns.tolist(),
        rotation=70,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_expression_for_condition(
    expr_dir: Path,
    condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    h5ad_file = expr_dir / f"{condition}_pseudobulk_logcpm.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(f"Missing expression file: {h5ad_file}")
    adata = ad.read_h5ad(h5ad_file)
    x = np.asarray(adata.X, dtype=np.float64)
    genes = adata.var_names.astype(str).to_numpy()
    profiles = adata.obs_names.astype(str).to_numpy()
    obs = cast(pd.DataFrame, adata.obs.copy())
    obs.index = profiles
    return x, genes, profiles, obs


def main() -> None:
    args = parse_args()
    in_root = resolve_base(args.input_dir)
    out_root = resolve_base(args.output_dir)
    expr_root = resolve_base(args.expr_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    network_types = (
        ["signed", "unsigned"]
        if args.network_type == "both"
        else [args.network_type]
    )
    records: list[WarehouseRecord] = []

    for network_type in network_types:
        if args.network_type == "both":
            type_in_root = in_root / network_type
        else:
            nested = in_root / network_type
            type_in_root = nested if nested.exists() else in_root
        type_out_root = (
            out_root / network_type
            if args.network_type == "both"
            else out_root
        )
        type_out_root.mkdir(parents=True, exist_ok=True)

        cond_dirs = resolve_conditions(type_in_root, args.condition)
        for cdir in cond_dirs:
            condition = cdir.name
            tom_file = find_tom_npz(cdir, condition, network_type)
            tom, tom_genes = load_tom_payload(tom_file)

            link = build_linkage_from_tom(tom)
            if args.disable_dynamic_cut:
                used_cut_height = float(args.cut_height)
                raw_labels = fcluster(
                    link,
                    t=used_cut_height,
                    criterion="distance",
                )
                module_ids_pre = relabel_modules(
                    raw_labels,
                    min_size=int(args.min_module_size),
                )
            else:
                module_ids_pre, used_cut_height = select_modules_dynamic(
                    link,
                    min_size=int(args.min_module_size),
                    deep_split=int(args.deep_split),
                    max_largest_module_fraction=float(
                        args.max_largest_module_fraction
                    ),
                )

            cond_out = type_out_root / condition
            fig_out = cond_out / "figures"
            cond_out.mkdir(parents=True, exist_ok=True)
            fig_out.mkdir(parents=True, exist_ok=True)

            module_df = pd.DataFrame({"gene": tom_genes})
            module_df["module_id_premerge"] = module_ids_pre
            module_df["module_label_premerge"] = module_df[
                "module_id_premerge"
            ].map(lambda x: f"M{int(x):03d}" if int(x) > 0 else "M000_GREY")

            module_df["module_id"] = module_ids_pre
            module_df["module_label"] = module_df["module_id"].map(
                lambda x: f"M{int(x):03d}" if int(x) > 0 else "M000_GREY"
            )
            module_df["module_color"] = module_df["module_id"].map(
                module_color
            )

            x, expr_genes, profiles, obs = load_expression_for_condition(
                expr_root, condition
            )
            gene_index = pd.Index(expr_genes)
            idx = gene_index.get_indexer(pd.Index(tom_genes))
            valid = idx >= 0
            if not np.all(valid):
                missing = int((~valid).sum())
                print(
                    f"[warn] {condition}: {missing} TOM genes "
                    "missing in expression"
                )

            expr_sub = x[:, idx[valid]]
            module_sub_pre = module_ids_pre[valid]
            tom_genes_sub = tom_genes[valid]

            merge_map_df = pd.DataFrame(
                columns=["module_premerge", "module_postmerge"]
            )
            module_sub_post = module_sub_pre
            used_merge_cut_height = float(args.merge_cut_height)
            if (not args.disable_module_merge) and np.any(module_sub_pre > 0):
                module_sub_post, merge_map_df = merge_modules_by_eigengene(
                    expr=expr_sub,
                    module_ids=module_sub_pre,
                    profile_ids=profiles,
                    merge_cut_height=used_merge_cut_height,
                )

                # If merging creates a dominant module, reduce merge height.
                max_frac = float(args.max_largest_module_fraction)
                cur_frac = largest_module_fraction(module_sub_post)
                if cur_frac > max_frac:
                    trial_heights = np.arange(
                        max(used_merge_cut_height - 0.05, 0.05),
                        0.049,
                        -0.05,
                    )
                    best_labels = module_sub_post
                    best_map = merge_map_df
                    best_h = used_merge_cut_height
                    best_frac = cur_frac

                    for mh in trial_heights.tolist():
                        trial_labels, trial_map = merge_modules_by_eigengene(
                            expr=expr_sub,
                            module_ids=module_sub_pre,
                            profile_ids=profiles,
                            merge_cut_height=float(mh),
                        )
                        trial_frac = largest_module_fraction(trial_labels)
                        if trial_frac < best_frac:
                            best_labels = trial_labels
                            best_map = trial_map
                            best_h = float(mh)
                            best_frac = trial_frac
                        if trial_frac <= max_frac:
                            best_labels = trial_labels
                            best_map = trial_map
                            best_h = float(mh)
                            best_frac = trial_frac
                            break

                    module_sub_post = best_labels
                    merge_map_df = best_map
                    used_merge_cut_height = best_h

            module_ids_final = module_ids_pre.copy()
            module_ids_final[valid] = module_sub_post
            module_ids_final[~valid] = 0
            module_df["module_id"] = module_ids_final
            module_df["module_label"] = module_df["module_id"].map(
                lambda x: f"M{int(x):03d}" if int(x) > 0 else "M000_GREY"
            )
            module_df["module_color"] = module_df["module_id"].map(
                module_color
            )

            gene_to_module_file = cond_out / f"{condition}_gene_to_module.csv"
            module_df.to_csv(gene_to_module_file, index=False)

            module_size_pre_df = (
                module_df.groupby(
                    ["module_id_premerge", "module_label_premerge"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "n_genes"})
                .sort_values(["module_id_premerge"])
                .reset_index(drop=True)
            )
            module_size_pre_df["module_color"] = module_size_pre_df[
                "module_id_premerge"
            ].map(module_color)
            module_sizes_pre_file = (
                cond_out / f"{condition}_module_sizes_premerge.csv"
            )
            module_size_pre_df.to_csv(module_sizes_pre_file, index=False)

            module_size_df = (
                module_df.groupby(
                    ["module_id", "module_label", "module_color"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "n_genes"})
                .sort_values(["module_id"])
                .reset_index(drop=True)
            )
            module_sizes_file = cond_out / f"{condition}_module_sizes.csv"
            module_size_df.to_csv(module_sizes_file, index=False)

            merge_map_file = cond_out / f"{condition}_module_merge_map.csv"
            merge_map_df.to_csv(merge_map_file, index=False)

            eig_df = compute_module_eigengenes(
                expr=expr_sub,
                module_ids=module_sub_post,
                profile_ids=profiles,
            )
            eig_file = cond_out / f"{condition}_module_eigengenes.csv"
            eig_df.to_csv(eig_file)

            assoc_file = cond_out / f"{condition}_module_trait_association.csv"
            assoc_df = pd.DataFrame()
            if not args.skip_trait_association:
                trait_df = build_trait_matrix(
                    obs,
                    max_levels=int(args.max_categorical_levels),
                )
                assoc_df = module_trait_association(eig_df, trait_df)
                assoc_df.to_csv(assoc_file, index=False)

            dendro_file = fig_out / f"{condition}_tom_dendrogram.png"
            plot_dendrogram(
                link,
                out_file=dendro_file,
                title=f"{condition} [{network_type}] TOM dendrogram",
                module_ids=module_ids_final,
                cut_height=used_cut_height,
            )
            size_fig = fig_out / f"{condition}_module_sizes.png"
            plot_module_sizes(
                module_size_df,
                out_file=size_fig,
                title=f"{condition} [{network_type}] module sizes",
            )
            heatmap_file = fig_out / f"{condition}_module_trait_heatmap.png"
            if not assoc_df.empty:
                plot_module_trait_heatmap(
                    assoc_df,
                    out_file=heatmap_file,
                    title=f"{condition} [{network_type}] ME-trait correlation",
                )

            summary = {
                "condition": condition,
                "network_type": network_type,
                "n_genes_tom": int(tom_genes.shape[0]),
                "n_genes_used_for_modules": int(tom_genes_sub.shape[0]),
                "dynamic_cut_enabled": bool(not args.disable_dynamic_cut),
                "deep_split": int(args.deep_split),
                "max_largest_module_fraction": float(
                    args.max_largest_module_fraction
                ),
                "cut_height": float(used_cut_height),
                "min_module_size": int(args.min_module_size),
                "module_merge_enabled": bool(not args.disable_module_merge),
                "merge_cut_height": float(used_merge_cut_height),
                "n_modules_non_grey_premerge": int(
                    (module_size_pre_df["module_id_premerge"] > 0).sum()
                ),
                "n_modules_non_grey": int(
                    (module_size_df["module_id"] > 0).sum()
                ),
                "largest_module_size": (
                    int(
                        module_size_df.loc[
                            module_size_df["module_id"] > 0,
                            "n_genes",
                        ].max()
                    )
                    if (module_size_df["module_id"] > 0).any()
                    else 0
                ),
                "gene_to_module_file": str(gene_to_module_file),
                "module_sizes_premerge_file": str(module_sizes_pre_file),
                "module_sizes_file": str(module_sizes_file),
                "module_merge_map_file": str(merge_map_file),
                "module_eigengenes_file": str(eig_file),
                "module_trait_association_file": (
                    str(assoc_file)
                    if not args.skip_trait_association
                    else ""
                ),
                "dendrogram_file": str(dendro_file),
            }
            summary_file = cond_out / f"{condition}_module_summary.json"
            summary_file.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

            records.append(
                WarehouseRecord(
                    input_file=str(tom_file),
                    output_file=str(summary_file),
                    script=str(
                        Path(__file__).resolve().relative_to(REPO_ROOT)
                    ),
                    date_utc=utc_now_iso(),
                    params_hash=params_hash(vars(args)),
                    condition=f"{condition}::{network_type}",
                    stage="08d_tom_module_detection",
                )
            )

            print(
                f"[{condition}::{network_type}] modules_non_grey="
                f"{summary['n_modules_non_grey']}"
            )

    append_warehouse(out_root, records)
    print(f"Done. TOM module outputs: {out_root}")


if __name__ == "__main__":
    main()
