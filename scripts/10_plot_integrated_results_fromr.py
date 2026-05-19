#!/usr/bin/env python3
"""Create per-condition embedding plots from integrated objects.

R_TRANSLATED: yes
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
from pathlib import Path

import scanpy as sc
from matplotlib.lines import Line2D

from utils.h5ad_compat import read_h5ad_compat


REPO_ROOT = Path(__file__).resolve().parents[1]
COLOR_KEYS = (("leiden", "leiden"), ("SampleName", "sample"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create post-integration visualization plots"
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
        "--input-dir",
        default="results/03_integration/integrated",
        help="Root directory with integrated condition outputs",
    )
    parser.add_argument(
        "--fig-subdir",
        default="figures",
        help="Subfolder inside each condition folder for figures",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument(
        "--compute-tsne",
        action="store_true",
        help="Compute t-SNE if missing and save t-SNE plots",
    )
    parser.add_argument(
        "--legend-loc",
        default="right margin",
        help="Legend location for Scanpy plots",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=int,
        default=8,
        help="Legend font size",
    )
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "unknown"


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]

    mapped = safe_name(requested)
    if mapped in available:
        return [mapped]

    near = difflib.get_close_matches(requested, available, n=5, cutoff=0.45)
    hints = [mapped, *near] if mapped != requested else near
    msg = [
        f"Condition not found: '{requested}'",
        "Available condition folders:",
        *[f"- {name}" for name in available],
    ]
    if hints:
        msg += [
            "Closest matches:",
            *[f"- {name}" for name in dict.fromkeys(hints)],
        ]
    raise ValueError("\n".join(msg))


def find_integrated_file(condition_dir: Path) -> Path | None:
    preferred = condition_dir / f"{condition_dir.name}_integrated.h5ad"
    if preferred.exists():
        return preferred
    matches = sorted(condition_dir.glob("*_integrated.h5ad"))
    return matches[0] if matches else None


def save_embedding_plots(
    adata: sc.AnnData,
    plotter,
    emb_label: str,
    condition: str,
    out_prefix: Path,
    dpi: int,
    legend_loc: str,
    legend_fontsize: int,
) -> None:
    for obs_col, suffix in COLOR_KEYS:
        if obs_col not in adata.obs.columns:
            continue
        current_legend_loc = legend_loc
        if obs_col == "SampleName":
            #sample legends go below the panel to avoid right-margin width inflation
            current_legend_loc = "none"
        fig = plotter(
            adata,
            color=obs_col,
            show=False,
            frameon=False,
            return_fig=True,
            title=f"{condition} | {emb_label}: {obs_col}",
            legend_loc=current_legend_loc,
            legend_fontsize=legend_fontsize,
        )
        if obs_col == "SampleName":
            categories = [str(v) for v in adata.obs[obs_col].astype("category").cat.categories]
            palette = adata.uns.get(f"{obs_col}_colors", [])
            handles = []
            for idx, label in enumerate(categories):
                color = palette[idx] if idx < len(palette) else "#666666"
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markersize=4,
                        label=label,
                    )
                )
            if handles:
                legend_cols = 4
                legend_rows = math.ceil(len(handles) / legend_cols)
                fig.subplots_adjust(bottom=min(0.30, 0.11 + 0.03 * legend_rows))
                fig.legend(
                    handles=handles,
                    loc="lower center",
                    ncol=legend_cols,
                    frameon=False,
                    fontsize=max(6, legend_fontsize - 1),
                    bbox_to_anchor=(0.5, 0.01),
                )
        out_file = out_prefix.with_name(
            f"{out_prefix.name}_{emb_label.lower()}_{suffix}.png"
        )
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight", pad_inches=0.04)


def plot_condition(
    condition: str,
    root: Path,
    fig_subdir: str,
    dpi: int,
    compute_tsne: bool,
    legend_loc: str,
    legend_fontsize: int,
) -> bool:
    #locate integrated h5ad and verify umap exists
    condition_dir = root / condition
    integrated_file = find_integrated_file(condition_dir)
    if integrated_file is None:
        print(f"Skipping {condition}: no *_integrated.h5ad found")
        return False

    adata = read_h5ad_compat(integrated_file)
    if "X_umap" not in adata.obsm:
        print(f"Skipping {condition}: X_umap missing")
        return False

    #create output dir and save umap plots coloured by leiden and sample
    fig_dir = condition_dir / fig_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = fig_dir / condition
    save_embedding_plots(adata, sc.pl.umap, "UMAP", condition, out_prefix, dpi, legend_loc, legend_fontsize)

    #optionally compute and plot tsne
    if compute_tsne:
        if "X_tsne" not in adata.obsm:
            sc.tl.tsne(adata, use_rep="X_pca")
        save_embedding_plots(adata, sc.pl.tsne, "TSNE", condition, out_prefix, dpi, legend_loc, legend_fontsize)

    print(f"[{condition}] saved figures in {fig_dir}")
    return True


def main() -> int:
    args = parse_args()

    #resolve input root
    root = resolve_base(args.input_dir)

    #list available conditions and exit if requested
    if args.list_conditions:
        names = list_conditions(root)
        if not names:
            print(f"No condition folders found in: {root}")
            return 1
        print("Available condition folders:")
        for name in names:
            print(f"- {name}")
        return 0

    #resolve target conditions with fuzzy matching on failure
    try:
        targets = resolve_conditions(root, args.condition)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not targets:
        print(f"ERROR: No condition folders found in {root}")
        return 1

    #plot each condition
    done = 0
    for condition in targets:
        done += int(plot_condition(
            condition=condition,
            root=root,
            fig_subdir=args.fig_subdir,
            dpi=args.dpi,
            compute_tsne=args.compute_tsne,
            legend_loc=args.legend_loc,
            legend_fontsize=args.legend_fontsize,
        ))

    if done == 0:
        print("ERROR: No figures were generated.")
        return 1

    print(f"\nPlotting complete. Conditions plotted: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
