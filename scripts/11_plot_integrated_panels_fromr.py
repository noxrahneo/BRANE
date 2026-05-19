#!/usr/bin/env python3
"""Build publication-style UMAP panel figures across conditions.

R_TRANSLATED: yes

Input:
- results/03_integration/integrated/
    <Condition>/<Condition>_integrated.h5ad

Output:
- results/03_integration/integrated/panels/umap_panel_<color_key>.png
"""

from __future__ import annotations

import argparse
import csv
import difflib
import math
import re
from pathlib import Path

import matplotlib.cm as cm
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-style UMAP panel figures"
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
        "--out-dir",
        default="results/03_integration/integrated/panels",
        help="Output directory for panel figures",
    )
    parser.add_argument(
        "--color-keys",
        default="leiden,SampleName",
        help="Comma-separated obs keys to plot",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=3,
        help="Number of panel columns",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size for scatter plots",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Point alpha",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Figure file format",
    )
    parser.add_argument(
        "--legend-max-items",
        type=int,
        default=999,
        help="Max number of legend items to render in legend figure",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=3,
        help="Number of columns in legend figure",
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
    files = sorted(condition_dir.glob("*_integrated.h5ad"))
    return files[0] if files else None


def load_integrated_objects(
    root: Path,
    conditions: list[str],
) -> dict[str, sc.AnnData]:
    #load each integrated h5ad; skip if missing or has no umap
    out: dict[str, sc.AnnData] = {}
    for condition in conditions:
        cond_dir = root / condition
        h5ad = find_integrated_file(cond_dir)
        if h5ad is None:
            print(f"Skipping {condition}: no *_integrated.h5ad found")
            continue
        adata = sc.read_h5ad(h5ad)
        if "X_umap" not in adata.obsm:
            print(f"Skipping {condition}: X_umap missing")
            continue
        out[condition] = adata
    return out


def build_palette(
    categories: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    uniq = list(dict.fromkeys(categories))
    if not uniq:
        return {}

    cmap = cm.get_cmap("tab20")
    if len(uniq) <= 20:
        colors = [cmap(i) for i in range(len(uniq))]
    else:
        hsv = cm.get_cmap("hsv")
        colors = [hsv(i / len(uniq)) for i in range(len(uniq))]
    return {cat: color for cat, color in zip(uniq, colors)}


def collect_categories(
    adatas: dict[str, sc.AnnData],
    color_key: str,
) -> list[str]:
    values: list[str] = []
    for adata in adatas.values():
        if color_key not in adata.obs.columns:
            continue
        vals = adata.obs[color_key].astype(str).tolist()
        values.extend(vals)
    return list(dict.fromkeys(values))


def panel_shape(n_panels: int, n_cols: int) -> tuple[int, int]:
    cols = max(1, n_cols)
    rows = int(math.ceil(n_panels / cols))
    return rows, cols


def plot_panel(
    adatas: dict[str, sc.AnnData],
    color_key: str,
    out_file: Path,
    n_cols: int,
    point_size: float,
    alpha: float,
    dpi: int,
    max_items: int,
    legend_cols: int,
) -> tuple[list[str], dict[str, tuple[float, float, float, float]]]:
    conditions = list(adatas.keys())
    rows, cols = panel_shape(len(conditions), n_cols)

    #size figure to accommodate panel grid plus right-margin legend
    legend_width = max(3.5, 1.35 * max(1, legend_cols) + 1.0)
    fig_w = (5 * cols) + legend_width
    fig_h = 4.5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    axes_array = np.array(axes).reshape(-1)
    categories = collect_categories(adatas, color_key)
    palette = build_palette(categories)

    #draw one umap scatter per condition panel
    for idx, condition in enumerate(conditions):
        ax = axes_array[idx]
        adata = adatas[condition]
        umap = adata.obsm["X_umap"]

        if color_key not in adata.obs.columns:
            ax.scatter(umap[:, 0], umap[:, 1], s=point_size, alpha=alpha)
            ax.set_title(f"{condition}\n(no {color_key})")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        vals = adata.obs[color_key].astype(str).values
        for cat in dict.fromkeys(vals):
            mask = vals == cat
            ax.scatter(
                umap[mask, 0], umap[mask, 1],
                s=point_size, alpha=alpha,
                c=[palette.get(cat, (0.5, 0.5, 0.5, 1.0))],
                linewidths=0, rasterized=True,
            )
        ax.set_title(f"{condition} (n={adata.n_obs})")
        ax.set_xticks([])
        ax.set_yticks([])

    #hide unused axes
    for idx in range(len(conditions), len(axes_array)):
        axes_array[idx].axis("off")

    #build shared legend; truncate to max_items and note overflow
    shown = categories[: max(1, max_items)]
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[cat][:3], markersize=6, label=cat)
        for cat in shown
    ]
    if handles:
        fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.992, 0.5),
                   frameon=False, ncol=max(1, legend_cols), fontsize=8, title=f"Legend: {color_key}")

    if len(categories) > len(shown):
        fig.text(0.5, 0.02,
                 f"Showing first {len(shown)} of {len(categories)} legend items. See CSV for full mapping.",
                 ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"UMAP panels by {color_key}", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.04, 0.77, 0.96))
    fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return categories, palette


def save_legend_assets(
    color_key: str,
    categories: list[str],
    palette: dict[str, tuple[float, float, float, float]],
    out_dir: Path,
) -> None:
    safe_key = safe_name(color_key)

    csv_path = out_dir / f"umap_panel_{safe_key}_legend.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "color_hex"])
        for cat in categories:
            writer.writerow([cat, to_hex(palette[cat][:3])])


def main() -> int:
    args = parse_args()

    #resolve paths and create output dir
    root = resolve_base(args.input_dir)
    out_dir = resolve_base(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    #load all integrated objects
    adatas = load_integrated_objects(root, targets)
    if not adatas:
        print("ERROR: No integrated condition objects available for plotting.")
        return 1

    #parse color keys from comma-separated arg
    color_keys = [key.strip() for key in args.color_keys.split(",") if key.strip()]
    if not color_keys:
        print("ERROR: No valid --color-keys provided.")
        return 1

    #generate one panel figure and legend csv per color key
    for color_key in color_keys:
        out_file = out_dir / f"umap_panel_{safe_name(color_key)}.{args.format}"
        categories, palette = plot_panel(
            adatas=adatas, color_key=color_key, out_file=out_file,
            n_cols=args.n_cols, point_size=args.point_size, alpha=args.alpha,
            dpi=args.dpi, max_items=args.legend_max_items, legend_cols=args.legend_cols,
        )
        save_legend_assets(color_key=color_key, categories=categories, palette=palette, out_dir=out_dir)
        print(f"Saved: {out_file}")

    print(f"\nPanel plotting complete. Conditions plotted: {len(adatas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
