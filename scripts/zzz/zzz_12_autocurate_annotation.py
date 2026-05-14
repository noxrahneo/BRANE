#!/usr/bin/env python3
"""Auto-curate cluster-to-cell-type mappings for annotation freeze.

This script creates:
- <Condition>_cluster_to_celltype_mapping_curated.csv

It keeps predicted labels by default, adds confidence columns, and marks
low-confidence clusters for optional manual review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_ROOT = "results/04_annotation"
DEFAULT_INTEGRATED_ROOT = "results/03_integration/integrated"


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-curate mapping tables with confidence flags"
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Condition folder name, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available conditions and exit",
    )
    parser.add_argument(
        "--annotation-dir",
        default=DEFAULT_ANNOTATION_ROOT,
        help="Annotation root with mapping tables",
    )
    parser.add_argument(
        "--integration-dir",
        default=DEFAULT_INTEGRATED_ROOT,
        help="Integrated root used only for thesis-source guard",
    )
    parser.add_argument(
        "--curated-label-col",
        default="curated_cell_type",
        help="Output column name for curated labels",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.05,
        help="Minimum score_margin for confident assignment",
    )
    parser.add_argument(
        "--min-marker-overlap",
        type=int,
        default=1,
        help="Minimum marker_overlap_n for confident assignment",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing curated mapping files",
    )
    parser.add_argument(
        "--no-review-flag",
        action="store_true",
        help="Do not flag low-confidence rows for manual review",
    )
    return parser.parse_args()


def assert_thesis_integrated_root(path: Path) -> None:
    norm = str(path).replace("\\", "/")
    if "integrated_fixed" in norm:
        raise ValueError(
            "integration-dir points to integrated_fixed, excluded for thesis"
        )


def list_conditions(annotation_root: Path) -> list[str]:
    if not annotation_root.exists():
        return []

    out: list[str] = []
    for cond_dir in sorted(p for p in annotation_root.iterdir() if p.is_dir()):
        condition = cond_dir.name
        mapping = cond_dir / f"{condition}_cluster_to_celltype_mapping.csv"
        if mapping.exists():
            out.append(condition)
    return out


def resolve_conditions(annotation_root: Path, requested: str) -> list[str]:
    available = list_conditions(annotation_root)
    if not available:
        raise FileNotFoundError("No mapping tables found in annotation dir")
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise FileNotFoundError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def safe_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(default)
    return vals.astype(float)


def confidence_label(
    has_margin: pd.Series,
    has_overlap: pd.Series,
) -> pd.Series:
    score = has_margin.astype(int) + has_overlap.astype(int)
    return score.map({2: "high", 1: "medium", 0: "low"}).fillna("low")


def build_curated_table(
    df: pd.DataFrame,
    curated_label_col: str,
    min_score_margin: float,
    min_marker_overlap: int,
    review_low: bool,
) -> pd.DataFrame:
    if "predicted_cell_type" not in df.columns:
        raise ValueError("predicted_cell_type column missing")

    out = df.copy()
    out[curated_label_col] = out["predicted_cell_type"].astype(str)

    score_margin = safe_num(
        out.get("score_margin", pd.Series(index=out.index))
    )
    marker_overlap = safe_num(
        out.get("marker_overlap_n", pd.Series(index=out.index)),
        default=0.0,
    )

    has_margin = score_margin >= float(min_score_margin)
    has_overlap = marker_overlap >= int(min_marker_overlap)
    conf = confidence_label(has_margin, has_overlap)

    out["auto_confidence"] = conf
    out["auto_has_margin"] = has_margin.values
    out["auto_has_marker_overlap"] = has_overlap.values
    out["needs_manual_review"] = (conf == "low") if review_low else False

    reason = []
    for idx in out.index:
        c = conf.loc[idx]
        if c == "high":
            reason.append("auto_high: margin+marker overlap")
        elif c == "medium":
            reason.append("auto_medium: one confidence signal")
        else:
            reason.append("auto_low: weak signals")

    if "curation_note" in out.columns:
        old = out["curation_note"].fillna("").astype(str).str.strip()
        merged = []
        for i, txt in enumerate(reason):
            prev = old.iloc[i]
            merged.append(txt if not prev else f"{txt}; {prev}")
        out["curation_note"] = merged
    else:
        out["curation_note"] = reason

    preferred = [
        c
        for c in [
            "leiden",
            "predicted_cell_type",
            curated_label_col,
            "auto_confidence",
            "needs_manual_review",
            "curation_note",
        ]
        if c in out.columns
    ]
    rest = [c for c in out.columns if c not in preferred]
    return out[preferred + rest]


def process_condition(
    condition: str,
    annotation_root: Path,
    curated_label_col: str,
    min_score_margin: float,
    min_marker_overlap: int,
    overwrite: bool,
    review_low: bool,
) -> dict[str, object]:
    cond_dir = annotation_root / condition
    in_file = cond_dir / f"{condition}_cluster_to_celltype_mapping.csv"
    out_file = (
        cond_dir / f"{condition}_cluster_to_celltype_mapping_curated.csv"
    )

    if out_file.exists() and not overwrite:
        print(f"[{condition}] curated exists, skip: {out_file}")
        return {
            "condition": condition,
            "status": "skipped_exists",
            "curated_file": str(out_file),
            "n_low_conf": None,
        }

    df = pd.read_csv(in_file)
    curated = build_curated_table(
        df=df,
        curated_label_col=curated_label_col,
        min_score_margin=min_score_margin,
        min_marker_overlap=min_marker_overlap,
        review_low=review_low,
    )
    curated.to_csv(out_file, index=False)

    n_low = int((curated["auto_confidence"] == "low").sum())
    print(f"[{condition}] wrote curated table: {out_file}")
    print(f"[{condition}] low-confidence clusters: {n_low}")

    return {
        "condition": condition,
        "status": "written",
        "curated_file": str(out_file),
        "n_low_conf": n_low,
    }


def main() -> int:
    args = parse_args()
    annotation_root = resolve_base(args.annotation_dir)
    integration_root = resolve_base(args.integration_dir)
    assert_thesis_integrated_root(integration_root)

    if args.list_conditions:
        names = list_conditions(annotation_root)
        if not names:
            print("No conditions found")
            return 1
        print("Available conditions:")
        for name in names:
            print(f"- {name}")
        return 0

    conditions = resolve_conditions(annotation_root, args.condition)
    rows = []
    for condition in conditions:
        row = process_condition(
            condition=condition,
            annotation_root=annotation_root,
            curated_label_col=args.curated_label_col,
            min_score_margin=args.min_score_margin,
            min_marker_overlap=args.min_marker_overlap,
            overwrite=args.overwrite,
            review_low=not args.no_review_flag,
        )
        rows.append(row)

    summary_file = annotation_root / "annotation_autocuration_summary.csv"
    pd.DataFrame(rows).to_csv(summary_file, index=False)
    print(f"Autocuration summary: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
