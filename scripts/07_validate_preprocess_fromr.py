#!/usr/bin/env python3
"""Validate per-sample preprocessing outputs.

R_TRANSLATED: yes

Source stage script: 06_per_sample_preprocess_fromr.py.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate per-sample preprocessing outputs"
    )
    parser.add_argument(
        "--condition",
        default="Normal",
        help="Condition folder name, or 'all'",
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List available condition folders and exit",
    )
    parser.add_argument(
        "--filtered-dir",
        default="results/02_preprocess/01_filtered",
        help="Root directory with filtered sample .h5ad files",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="results/02_preprocess/02_preprocessed",
        help="Root directory with preprocessed sample .h5ad files",
    )
    parser.add_argument(
        "--out-dir",
        default="results/02_preprocess/03_preprocess_validation",
        help="Root directory for validation reports",
    )
    parser.add_argument(
        "--expected-hvg", type=int, default=3000, help="Expected top HVG count"
    )
    parser.add_argument(
        "--hvg-mode",
        choices=["strict", "relaxed"],
        default="strict",
        help="HVG validation mode",
    )
    parser.add_argument(
        "--hvg-tolerance",
        type=float,
        default=0.05,
        help="Relative HVG tolerance used in relaxed mode",
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
    hint = [mapped, *near] if mapped != requested else near
    msg = [
        f"Condition not found: '{requested}'",
        "Available condition folders:",
        *[f"- {x}" for x in available],
    ]
    if hint:
        msg += ["Closest matches:", *[f"- {x}" for x in dict.fromkeys(hint)]]
    raise ValueError("\n".join(msg))


def sample_from_filtered(path: Path) -> str:
    if path.name.endswith("_filtered.h5ad"):
        return path.name[:-14]
    return path.stem


def sample_from_preprocessed(path: Path) -> str:
    if path.name.endswith("_preprocessed.h5ad"):
        return path.name[:-18]
    return path.stem


def is_finite_matrix(matrix) -> bool:
    if sparse.issparse(matrix):
        return bool(np.isfinite(matrix.data).all())
    return bool(np.isfinite(np.asarray(matrix)).all())


def hvg_issue(
    hvg_count: int,
    target: int,
    n_vars: int,
    mode: str,
    tol: float,
) -> str | None:
    if hvg_count <= 0:
        return "zero_hvg"
    if mode == "strict":
        return None if hvg_count == target else "unexpected_hvg_count"

    tol = max(0.0, float(tol))
    low = max(1, int(np.floor(target * (1.0 - tol))))
    high = min(n_vars, max(1, int(np.ceil(target * (1.0 + tol)))))
    return None if low <= hvg_count <= high else "unexpected_hvg_count_relaxed"


def validate_one(
    sample: str,
    filtered_path: Path,
    preprocessed_path: Path,
    args: argparse.Namespace,
) -> dict:
    row = {
        "SampleName": sample,
        "filtered_file": str(filtered_path),
        "preprocessed_file": str(preprocessed_path),
        "exists_preprocessed": preprocessed_path.exists(),
        "expected_hvg": args.expected_hvg,
        "hvg_mode": args.hvg_mode,
        "hvg_tolerance": args.hvg_tolerance,
    }

    if not preprocessed_path.exists():
        row.update(
            {
                "cells_filtered": np.nan,
                "cells_preprocessed": np.nan,
                "genes_filtered": np.nan,
                "genes_preprocessed": np.nan,
                "has_counts_layer": False,
                "has_hvg_column": False,
                "hvg_count": np.nan,
                "hvg_target": np.nan,
                "x_is_finite": False,
                "status": "FAIL",
                "issues": "missing_preprocessed_file",
            }
        )
        return row

    adata_f = sc.read_h5ad(filtered_path)
    adata_p = sc.read_h5ad(preprocessed_path)

    issues: list[str] = []
    if adata_f.n_obs != adata_p.n_obs:
        issues.append("cell_count_mismatch")
    if adata_f.n_vars != adata_p.n_vars:
        issues.append("gene_count_mismatch")

    has_counts = "counts" in adata_p.layers
    if not has_counts:
        issues.append("missing_counts_layer")

    has_hvg = "highly_variable" in adata_p.var.columns
    hvg_target = min(args.expected_hvg, adata_p.n_vars)
    if has_hvg:
        hvg_count = int(adata_p.var["highly_variable"].sum())
        issue = hvg_issue(
            hvg_count,
            hvg_target,
            adata_p.n_vars,
            args.hvg_mode,
            args.hvg_tolerance,
        )
        if issue:
            issues.append(issue)
    else:
        hvg_count = np.nan
        issues.append("missing_highly_variable_column")

    finite_x = is_finite_matrix(adata_p.X)
    if not finite_x:
        issues.append("non_finite_values_in_X")

    row.update(
        {
            "cells_filtered": adata_f.n_obs,
            "cells_preprocessed": adata_p.n_obs,
            "genes_filtered": adata_f.n_vars,
            "genes_preprocessed": adata_p.n_vars,
            "has_counts_layer": has_counts,
            "has_hvg_column": has_hvg,
            "hvg_count": hvg_count,
            "hvg_target": hvg_target,
            "x_is_finite": finite_x,
            "status": "PASS" if not issues else "FAIL",
            "issues": ";".join(issues),
        }
    )
    return row


def validate_condition(
    condition: str,
    filtered_base: Path,
    preprocessed_base: Path,
    args: argparse.Namespace,
) -> list[dict]:
    filtered_dir = filtered_base / condition
    preprocessed_dir = preprocessed_base / condition

    #find filtered reference files for this condition
    filtered_files = sorted(filtered_dir.glob("*_filtered.h5ad"))
    if not filtered_files:
        print(f"Skipping {condition}: no *_filtered.h5ad files found")
        return []

    print(f"\nCondition: {condition} | Samples: {len(filtered_files)}")

    #index preprocessed files by sample name for fast lookup
    pre_index = {
        sample_from_preprocessed(p): p
        for p in preprocessed_dir.glob("*_preprocessed.h5ad")
    }

    #validate each sample and collect rows
    rows: list[dict] = []
    for file_path in filtered_files:
        sample = sample_from_filtered(file_path)
        pre_file = pre_index.get(sample, preprocessed_dir / f"{sample}_preprocessed.h5ad")
        row = validate_one(sample, file_path, pre_file, args)
        row["Condition"] = condition
        rows.append(row)
        print(f"[{row['status']}] {sample} - {row['issues'] or 'ok'}")
    return rows


def main() -> int:
    args = parse_args()

    #resolve paths and create output dir
    filtered_base = resolve_base(args.filtered_dir)
    preprocessed_base = resolve_base(args.preprocessed_dir)
    out_dir = resolve_base(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #list available conditions and exit if requested
    if args.list_conditions:
        conditions = list_conditions(filtered_base)
        if not conditions:
            print(f"No condition folders found in: {filtered_base}")
            return 1
        print("Available condition folders:")
        for name in conditions:
            print(f"- {name}")
        return 0

    #resolve target conditions with fuzzy matching on failure
    try:
        targets = resolve_conditions(filtered_base, args.condition)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not targets:
        print(f"ERROR: No condition folders found in {filtered_base}")
        return 1

    #validate each condition and write per-condition csv
    all_rows: list[dict] = []
    processed_conditions = 0
    for condition in targets:
        rows = validate_condition(condition, filtered_base, preprocessed_base, args)
        if not rows:
            continue
        processed_conditions += 1
        all_rows.extend(rows)
        pd.DataFrame(rows).sort_values(["status", "SampleName"]).to_csv(
            out_dir / f"preprocess_validation_{condition}.csv", index=False
        )

    if not all_rows:
        print("ERROR: No samples were validated.")
        return 1

    #write combined report csv and json summary
    report = pd.DataFrame(all_rows).sort_values(["Condition", "status", "SampleName"]).reset_index(drop=True)
    csv_path = out_dir / f"preprocess_validation_{args.condition}.csv"
    report.to_csv(csv_path, index=False)

    failed = int((report["status"] == "FAIL").sum())
    summary = {
        "condition": args.condition,
        "conditions_processed": processed_conditions,
        "filtered_dir": str(filtered_base),
        "preprocessed_dir": str(preprocessed_base),
        "hvg_mode": args.hvg_mode,
        "hvg_tolerance": args.hvg_tolerance,
        "expected_hvg": args.expected_hvg,
        "samples_total": int(len(report)),
        "samples_passed": int(len(report) - failed),
        "samples_failed": failed,
        "report_csv": str(csv_path),
    }

    summary_path = out_dir / f"preprocess_validation_{args.condition}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nValidation complete")
    print(
        f"Total: {summary['samples_total']}"
        f" | Passed: {summary['samples_passed']}"
        f" | Failed: {summary['samples_failed']}"
    )
    print(f"CSV report: {csv_path}")
    print(f"JSON summary: {summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
