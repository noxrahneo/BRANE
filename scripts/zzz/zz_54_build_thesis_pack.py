#!/usr/bin/env python3
"""Build a compact thesis figure/table pack from analysis outputs.

This script reads existing outputs (composition, heatmap summary, KEGG
interpretation) and writes publication-friendly CSV/PNG artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact thesis tables/figures from result files"
    )
    parser.add_argument(
        "--input-root",
        default="results",
        help="Root folder with staged outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/90_reports/thesis_pack",
        help="Output directory for thesis-ready tables and figures",
    )
    parser.add_argument(
        "--top-shifts",
        type=int,
        default=10,
        help="Top composition shifts to include",
    )
    parser.add_argument(
        "--top-kegg-per-condition",
        type=int,
        default=3,
        help="Top KEGG terms per condition by adjusted p-value",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_composition_top_shifts(
    prop_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    id_cols = {"condition", "sample_id"}
    groups = [c for c in prop_df.columns if c not in id_cols]
    mean_prop = prop_df.groupby("condition", observed=False)[groups].mean()

    rows: list[dict[str, object]] = []
    for group in groups:
        series = mean_prop[group]
        cond_max = str(series.idxmax())
        cond_min = str(series.idxmin())
        max_val = float(series.loc[cond_max])
        min_val = float(series.loc[cond_min])
        rows.append(
            {
                "cell_group": group,
                "range": max_val - min_val,
                "condition_max": cond_max,
                "value_max": max_val,
                "condition_min": cond_min,
                "value_min": min_val,
            }
        )

    out = pd.DataFrame(rows).sort_values("range", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def build_kegg_top_terms(
    kegg_df: pd.DataFrame,
    top_per_condition: int,
) -> pd.DataFrame:
    if kegg_df.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "cluster",
                "Term",
                "Adjusted P-value",
                "P-value",
                "Overlap",
                "Genes",
            ]
        )

    df = kegg_df.copy()
    df["Adjusted P-value"] = pd.to_numeric(
        df["Adjusted P-value"], errors="coerce"
    )
    df = df.sort_values(["condition", "Adjusted P-value"])
    df = df.groupby("condition", as_index=False).head(top_per_condition)
    keep = [
        "condition",
        "cluster",
        "Term",
        "Adjusted P-value",
        "P-value",
        "Overlap",
        "Genes",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


def plot_top_shifts(
    top_shifts: pd.DataFrame,
    out_file: Path,
    dpi: int,
) -> None:
    if top_shifts.empty:
        return
    labels = top_shifts["cell_group"].astype(str).tolist()[::-1]
    values = top_shifts["range"].astype(float).tolist()[::-1]

    fig_h = max(4.2, 0.45 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    bars = ax.barh(labels, values, color="#4C78A8")
    ax.set_xlabel("Range of mean proportion across conditions")
    ax.set_title("Top composition shifts by cell group")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_kegg_strength(
    kegg_top: pd.DataFrame,
    out_file: Path,
    dpi: int,
) -> None:
    if kegg_top.empty:
        return

    df = kegg_top.copy()
    df["Adjusted P-value"] = pd.to_numeric(
        df["Adjusted P-value"], errors="coerce"
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Adjusted P-value"]
    )
    if df.empty:
        return
    df["score"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
    df["label"] = (
        df["condition"].astype(str)
        + " | Clst "
        + df["cluster"].astype(str)
        + " | "
        + df["Term"].astype(str)
    )

    df = df.sort_values("score", ascending=True)
    labels = df["label"].tolist()
    values = df["score"].tolist()

    fig_h = max(5.0, 0.3 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.barh(labels, values, color="#F58518")
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.set_title("Top KEGG enrichments (strength)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def write_pack_readme(
    out_dir: Path,
    comp_global: pd.DataFrame,
) -> None:
    row = comp_global.iloc[0].to_dict() if not comp_global.empty else {}
    content = [
        "# Thesis Pack",
        "",
        "Generated compact tables/figures for thesis results chapter.",
        "",
        "## Files",
        "",
        "- `table_01_composition_global.csv`",
        "- `table_02_composition_top_shifts.csv`",
        "- `table_03_heatmap_coverage.csv`",
        "- `table_04_kegg_top_terms.csv`",
        "- `fig_01_composition_top_shifts.png`",
        "- `fig_02_kegg_top_terms_strength.png`",
        "",
        "## Composition global test",
        "",
        f"- test: {row.get('test', 'NA')}",
        f"- LR stat: {row.get('lr_stat', 'NA')}",
        f"- df: {row.get('df', 'NA')}",
        f"- p-value: {row.get('p_value', 'NA')}",
    ]
    (out_dir / "README.md").write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = resolve_base(args.input_root)
    out_dir = resolve_base(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arg_hash = params_hash(vars(args))
    now = utc_now_iso()

    comp_global = read_required_csv(
        input_root / "05_composition" / "composition_glm_global_test.csv"
    )
    comp_prop = read_required_csv(
        input_root
        / "05_composition"
        / "composition_proportions_by_sample.csv"
    )
    heatmap_summary = read_required_csv(
        input_root / "04_annotation" / "marker_heatmap_summary.csv"
    )
    kegg_interp = read_required_csv(
        input_root / "zzz_06_kegg" / "kegg_interpretation_all.csv"
    )

    top_shifts = build_composition_top_shifts(
        prop_df=comp_prop,
        top_n=args.top_shifts,
    )
    kegg_top = build_kegg_top_terms(
        kegg_df=kegg_interp,
        top_per_condition=args.top_kegg_per_condition,
    )

    comp_global.to_csv(
        out_dir / "table_01_composition_global.csv", index=False
    )
    top_shifts.to_csv(
        out_dir / "table_02_composition_top_shifts.csv", index=False
    )
    heatmap_summary.to_csv(
        out_dir / "table_03_heatmap_coverage.csv", index=False
    )
    kegg_top.to_csv(out_dir / "table_04_kegg_top_terms.csv", index=False)

    plot_top_shifts(
        top_shifts=top_shifts,
        out_file=out_dir / "fig_01_composition_top_shifts.png",
        dpi=args.dpi,
    )
    plot_kegg_strength(
        kegg_top=kegg_top,
        out_file=out_dir / "fig_02_kegg_top_terms_strength.png",
        dpi=args.dpi,
    )

    write_pack_readme(out_dir=out_dir, comp_global=comp_global)

    warehouse_file = append_warehouse(
        out_dir,
        [
            WarehouseRecord(
                input_file=str(input_root),
                output_file=str(out_dir / "table_04_kegg_top_terms.csv"),
                script="scripts/54_build_thesis_pack.py",
                date_utc=now,
                params_hash=arg_hash,
                condition="all",
                stage="thesis_pack",
            )
        ],
    )

    print("Thesis pack generated")
    print(f"Output directory: {out_dir}")
    print(f"Warehouse log: {warehouse_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
