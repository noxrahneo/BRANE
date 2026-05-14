#!/usr/bin/env python3
"""Compare baseline network outputs vs a sensitivity rerun.

Produces per-condition deltas for profiles, selected power, node/edge counts,
network density, and hub-gene overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


CONDITIONS = [
    "ER_tumor",
    "HER2_tumor",
    "Normal",
    "Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_BRCA1_tumor",
    "Triple_negative_tumor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs sensitivity network outputs"
    )
    parser.add_argument(
        "--baseline-corr-dir",
        default="results/09_correlation/pearson",
        help="Baseline correlation root",
    )
    parser.add_argument(
        "--baseline-prep-dir",
        default="results/11_coexpression_prep/single",
        help="Baseline network prep root",
    )
    parser.add_argument(
        "--baseline-viz-dir",
        default="results/13_coexpression_viz/single",
        help="Baseline network viz root",
    )
    parser.add_argument(
        "--sensitivity-corr-dir",
        required=True,
        help="Sensitivity correlation root",
    )
    parser.add_argument(
        "--sensitivity-prep-dir",
        required=True,
        help="Sensitivity network prep root",
    )
    parser.add_argument(
        "--sensitivity-viz-dir",
        required=True,
        help="Sensitivity network viz root",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=20,
        help="Top-N hubs (weighted_degree) used for overlap",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write comparison outputs",
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


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_corr_metrics(corr_root: Path, condition: str) -> dict:
    f = corr_root / condition / f"{condition}_corr_summary.csv"
    df = read_csv_optional(f)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "n_profiles": int(row.get("n_profiles", 0)),
        "n_genes_used": int(row.get("n_genes_used", 0)),
        "offdiag_abs_q95": float(row.get("offdiag_abs_q95", float("nan"))),
    }


def get_selected_power(prep_root: Path, condition: str) -> int | None:
    f = prep_root / condition / f"{condition}_selected_power.json"
    meta = read_json_optional(f)
    value = meta.get("selected_power")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def get_viz_metrics(viz_root: Path, condition: str) -> dict:
    summary_file = (
        viz_root / condition / f"{condition}_network_viz_summary.json"
    )
    summary = read_json_optional(summary_file)
    n_nodes = int(summary.get("n_nodes", 0) or 0)
    n_edges = int(summary.get("n_edges", 0) or 0)
    density = (
        (2.0 * n_edges / (n_nodes * (n_nodes - 1)))
        if n_nodes > 1
        else 0.0
    )
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(density),
    }


def top_hubs(viz_root: Path, condition: str, n_top: int) -> list[str]:
    f = viz_root / condition / f"{condition}_network_weighted_degree.csv"
    df = read_csv_optional(f)
    if df.empty:
        return []
    if "gene" not in df.columns:
        return []
    col = "weighted_degree" if "weighted_degree" in df.columns else None
    if col is None:
        ranked = df.copy()
    else:
        ranked = df.sort_values(col, ascending=False)
    return ranked["gene"].astype(str).head(int(n_top)).tolist()


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    denom = len(sa | sb)
    if denom == 0:
        return 0.0
    return float(len(sa & sb) / denom)


def format_summary_md(df: pd.DataFrame, out_file: Path) -> None:
    if df.empty:
        out_file.write_text(
            "# Sensitivity comparison\n\nNo comparable rows found.\n",
            encoding="utf-8",
        )
        return

    cols = [
        "condition",
        "profiles_base",
        "profiles_sens",
        "d_profiles",
        "power_base",
        "power_sens",
        "nodes_base",
        "nodes_sens",
        "edges_base",
        "edges_sens",
        "hub_jaccard_top20",
    ]
    rows = []
    for _, row in df[cols].iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    text = [
        "# Sensitivity comparison: baseline vs exclude-flagged",
        "",
        header,
        sep,
        *rows,
        "",
    ]
    out_file.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()

    baseline_corr = resolve_base(args.baseline_corr_dir)
    baseline_prep = resolve_base(args.baseline_prep_dir)
    baseline_viz = resolve_base(args.baseline_viz_dir)

    sensitivity_corr = resolve_base(args.sensitivity_corr_dir)
    sensitivity_prep = resolve_base(args.sensitivity_prep_dir)
    sensitivity_viz = resolve_base(args.sensitivity_viz_dir)

    out_dir = resolve_base(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for condition in CONDITIONS:
        corr_base = get_corr_metrics(baseline_corr, condition)
        corr_sens = get_corr_metrics(sensitivity_corr, condition)
        viz_base = get_viz_metrics(baseline_viz, condition)
        viz_sens = get_viz_metrics(sensitivity_viz, condition)
        power_base = get_selected_power(baseline_prep, condition)
        power_sens = get_selected_power(sensitivity_prep, condition)

        hubs_base = top_hubs(baseline_viz, condition, args.top_hubs)
        hubs_sens = top_hubs(sensitivity_viz, condition, args.top_hubs)

        if not corr_base and not corr_sens and not viz_base and not viz_sens:
            continue

        records.append(
            {
                "condition": condition,
                "profiles_base": int(corr_base.get("n_profiles", 0)),
                "profiles_sens": int(corr_sens.get("n_profiles", 0)),
                "d_profiles": (
                    int(corr_sens.get("n_profiles", 0))
                    - int(corr_base.get("n_profiles", 0))
                ),
                "genes_base": int(corr_base.get("n_genes_used", 0)),
                "genes_sens": int(corr_sens.get("n_genes_used", 0)),
                "power_base": power_base,
                "power_sens": power_sens,
                "nodes_base": int(viz_base.get("n_nodes", 0)),
                "nodes_sens": int(viz_sens.get("n_nodes", 0)),
                "d_nodes": (
                    int(viz_sens.get("n_nodes", 0))
                    - int(viz_base.get("n_nodes", 0))
                ),
                "edges_base": int(viz_base.get("n_edges", 0)),
                "edges_sens": int(viz_sens.get("n_edges", 0)),
                "d_edges": (
                    int(viz_sens.get("n_edges", 0))
                    - int(viz_base.get("n_edges", 0))
                ),
                "density_base": float(viz_base.get("density", 0.0)),
                "density_sens": float(viz_sens.get("density", 0.0)),
                "d_density": (
                    float(viz_sens.get("density", 0.0))
                    - float(viz_base.get("density", 0.0))
                ),
                "q95absr_base": float(
                    corr_base.get("offdiag_abs_q95", float("nan"))
                ),
                "q95absr_sens": float(
                    corr_sens.get("offdiag_abs_q95", float("nan"))
                ),
                "d_q95absr": (
                    float(corr_sens.get("offdiag_abs_q95", float("nan")))
                    - float(corr_base.get("offdiag_abs_q95", float("nan")))
                ),
                "hub_jaccard_top20": float(jaccard(hubs_base, hubs_sens)),
                "hub_overlap_top20": " | ".join(
                    sorted(set(hubs_base) & set(hubs_sens))
                ),
            }
        )

    comp = pd.DataFrame(records).sort_values("condition")
    comp_csv = out_dir / "baseline_vs_exclude_flagged_comparison.csv"
    comp.to_csv(comp_csv, index=False)

    comp_md = out_dir / "baseline_vs_exclude_flagged_comparison.md"
    format_summary_md(comp, comp_md)

    print(f"Wrote: {comp_csv}")
    print(f"Wrote: {comp_md}")


if __name__ == "__main__":
    main()
