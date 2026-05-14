#!/usr/bin/env python3
"""Build WGCNA-style multi-panel figures from TOM module outputs.

Panels:
A) Gene dendrogram + module color bar
B) Module eigengene dendrogram + adjacency heatmap
C) Module-trait relationship heatmap (r with p-value)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WGCNA-style overview figures per condition"
    )
    parser.add_argument(
        "--network-prep-dir",
        default="results/11_coexpression_prep/single",
        help="Root with TOM files from 23_network_power_tom_prep.py",
    )
    parser.add_argument(
        "--module-dir",
        default="results/12_network_modules/single",
        help="Root with module outputs from 24_tom_module_detection.py",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition name or 'all'",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned"],
        default="signed",
        help="Network type for TOM files",
    )
    parser.add_argument(
        "--max-traits",
        type=int,
        default=12,
        help="Maximum number of traits shown in panel C",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI",
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


def list_condition_dirs(root: Path, requested: str) -> list[Path]:
    if not root.exists():
        return []
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if requested.strip().lower() == "all":
        return dirs
    out = [d for d in dirs if d.name == requested]
    if not out:
        raise ValueError(
            f"Condition '{requested}' not found; "
            f"available={[d.name for d in dirs]}"
        )
    return out


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


def find_tom_file(network_root: Path, condition: str, network_type: str) -> Path:
    candidate_dirs = [
        network_root / network_type / condition,
        network_root / condition,
    ]
    for cond_dir in candidate_dirs:
        if not cond_dir.exists():
            continue
        matches = sorted(
            cond_dir.glob(f"{condition}_tom_beta*_{network_type}.npz")
        )
        if matches:
            return matches[0]
        fallback = sorted(cond_dir.glob(f"{condition}_tom_beta*.npz"))
        if fallback:
            return fallback[0]
    raise FileNotFoundError(
        f"No TOM file found for {condition} in {candidate_dirs}"
    )


def load_gene_modules_in_tom_order(
    condition_dir: Path,
    tom_genes: np.ndarray,
) -> np.ndarray:
    g2m = pd.read_csv(condition_dir / f"{condition_dir.name}_gene_to_module.csv")
    m = g2m.set_index("gene")["module_id"].astype(int)
    module_ids = m.reindex(pd.Index(tom_genes)).fillna(0).astype(int).to_numpy()
    return module_ids


def build_panel_a(
    ax_tree: plt.Axes,
    ax_bar: plt.Axes,
    tom: np.ndarray,
    module_ids: np.ndarray,
) -> list[int]:
    dist = 1.0 - np.asarray(tom, dtype=float)
    dist = np.clip(dist, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")

    dend = dendrogram(link, no_labels=True, ax=ax_tree)
    ax_tree.set_ylabel("Height")
    ax_tree.set_xticks([])

    leaves = dend["leaves"]
    n_genes = len(leaves)
    for idx, gene_idx in enumerate(leaves):
        col = module_color(int(module_ids[int(gene_idx)]))
        ax_bar.add_patch(
            Rectangle((idx, 0), 1, 1, facecolor=col, edgecolor="none")
        )
    ax_bar.set_xlim(0, n_genes)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("Module\ncolors", rotation=0, labelpad=28, va="center")
    return leaves


def eigengene_corr_order(eig: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    corr = eig.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    diss = 1.0 - corr
    np.fill_diagonal(diss, 0.0)
    link = linkage(squareform(diss, checks=False), method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    return corr, order


def build_panel_b(
    ax_dend: plt.Axes,
    ax_heat: plt.Axes,
    eig_df: pd.DataFrame,
) -> list[str]:
    corr, order = eigengene_corr_order(eig_df)
    diss = 1.0 - corr
    np.fill_diagonal(diss, 0.0)
    link = linkage(squareform(diss, checks=False), method="average")
    dendrogram(link, no_labels=True, ax=ax_dend)
    ax_dend.set_xticks([])
    ax_dend.set_ylabel("Height")

    corr_ord = corr[np.ix_(order, order)]
    cols = eig_df.columns.tolist()
    ordered_cols = [cols[i] for i in order]

    im = ax_heat.imshow(corr_ord, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(np.arange(len(ordered_cols)))
    ax_heat.set_yticks(np.arange(len(ordered_cols)))
    ax_heat.set_xticklabels(ordered_cols, rotation=90, fontsize=7)
    ax_heat.set_yticklabels(ordered_cols, fontsize=7)
    ax_heat.figure.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
    return ordered_cols


def select_trait_columns(assoc: pd.DataFrame, max_traits: int) -> list[str]:
    if assoc.empty:
        return []
    agg = assoc.groupby("trait", as_index=False)["p_value"].min()
    agg = agg.sort_values("p_value").head(max(1, int(max_traits)))
    return agg["trait"].astype(str).tolist()


def build_panel_c(
    ax: plt.Axes,
    assoc_df: pd.DataFrame,
    ordered_modules: list[str],
    max_traits: int,
) -> None:
    if assoc_df.empty:
        ax.axis("off")
        ax.set_title("No module-trait associations")
        return

    traits = select_trait_columns(assoc_df, max_traits=max_traits)
    data = assoc_df[assoc_df["trait"].isin(traits)].copy()

    r_pivot = data.pivot_table(
        index="module",
        columns="trait",
        values="pearson_r",
        aggfunc="mean",
    )
    p_pivot = data.pivot_table(
        index="module",
        columns="trait",
        values="p_value",
        aggfunc="min",
    )

    keep_mods = [m for m in ordered_modules if m in r_pivot.index]
    if keep_mods:
        r_pivot = r_pivot.reindex(keep_mods)
        p_pivot = p_pivot.reindex(keep_mods)

    arr = r_pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(r_pivot.shape[1]))
    ax.set_yticks(np.arange(r_pivot.shape[0]))
    ax.set_xticklabels(r_pivot.columns.tolist(), rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(r_pivot.index.tolist(), fontsize=8)

    for i in range(r_pivot.shape[0]):
        for j in range(r_pivot.shape[1]):
            r_val = r_pivot.iat[i, j]
            p_val = p_pivot.iat[i, j]
            if np.isfinite(r_val) and np.isfinite(p_val):
                txt = f"{r_val:.2f}\n({p_val:.1e})"
            else:
                txt = ""
            ax.text(j, i, txt, ha="center", va="center", fontsize=6)

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def make_figure(
    condition: str,
    tom: np.ndarray,
    module_ids: np.ndarray,
    eig_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    out_file: Path,
    max_traits: int,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.2, 1.2, 1.25],
        height_ratios=[1, 1],
        wspace=0.30,
        hspace=0.25,
    )

    gs_a = gs[0, 0].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.03)
    ax_a_tree = fig.add_subplot(gs_a[0])
    ax_a_bar = fig.add_subplot(gs_a[1])
    build_panel_a(ax_a_tree, ax_a_bar, tom, module_ids)
    ax_a_tree.set_title("A  Gene dendrogram and module colors", loc="left")

    gs_b = gs[0, 1].subgridspec(2, 1, height_ratios=[2, 4], hspace=0.15)
    ax_b_d = fig.add_subplot(gs_b[0])
    ax_b_h = fig.add_subplot(gs_b[1])
    ordered_modules = build_panel_b(ax_b_d, ax_b_h, eig_df)
    ax_b_d.set_title("B  Eigengene adjacency clustering", loc="left")

    ax_c = fig.add_subplot(gs[:, 2])
    build_panel_c(ax_c, assoc_df, ordered_modules, max_traits=max_traits)
    ax_c.set_title("C  Module-trait relationships", loc="left")

    fig.suptitle(f"{condition} WGCNA-style module summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_file, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    network_root = resolve_base(args.network_prep_dir)
    module_root = resolve_base(args.module_dir)

    cond_dirs = list_condition_dirs(module_root, args.condition)
    if not cond_dirs:
        raise FileNotFoundError(f"No condition directories found in {module_root}")

    for cdir in cond_dirs:
        condition = cdir.name

        tom_file = find_tom_file(network_root, condition, args.network_type)
        payload = np.load(tom_file, allow_pickle=True)
        tom = np.asarray(payload["tom"], dtype=float)
        tom_genes = payload["genes"].astype(str)

        module_ids = load_gene_modules_in_tom_order(cdir, tom_genes)
        eig_df = pd.read_csv(
            cdir / f"{condition}_module_eigengenes.csv",
            index_col=0,
        )
        assoc_path = cdir / f"{condition}_module_trait_association.csv"
        assoc_df = pd.read_csv(assoc_path) if assoc_path.exists() else pd.DataFrame()

        fig_out = cdir / "figures"
        fig_out.mkdir(parents=True, exist_ok=True)
        out_file = fig_out / f"{condition}_wgcna_style_overview.png"

        make_figure(
            condition=condition,
            tom=tom,
            module_ids=module_ids,
            eig_df=eig_df,
            assoc_df=assoc_df,
            out_file=out_file,
            max_traits=int(args.max_traits),
            dpi=int(args.dpi),
        )

        print(f"[{condition}] wrote {out_file}")


if __name__ == "__main__":
    main()
