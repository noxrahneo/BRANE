#!/usr/bin/env python3
"""Wrapper to render combined-condition NetworkX visualizations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run combined-condition visualization using 08d_networkx_visualization.py"
    )
    parser.add_argument(
        "--input-dir",
        default="results/11_coexpression_prep/zzz_combined",
        help="Combined network prep root",
    )
    parser.add_argument(
        "--output-dir",
        default="results/13_coexpression_viz/zzz_combined",
        help="Combined network viz root",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="Combined condition name or 'all'",
    )
    parser.add_argument(
        "--network-type",
        choices=["signed", "unsigned"],
        default="signed",
        help="Adjacency network type",
    )
    parser.add_argument("--power", type=int, default=None, help="Optional fixed power")
    parser.add_argument("--max-edges", type=int, default=2500)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument(
        "--min-degree",
        type=int,
        default=1,
        help="Minimum node degree to keep in visualization graph",
    )
    parser.add_argument("--global-max-edges", type=int, default=800)
    parser.add_argument("--global-min-weight", type=float, default=0.20)
    parser.add_argument("--layout", choices=["spring", "kamada"], default="spring")
    parser.add_argument("--interactive-html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interactive-physics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--interactive-label-mode",
        choices=["hubs", "all", "none"],
        default="hubs",
    )
    parser.add_argument(
        "--interactive-celltype-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Combined outputs default to no annotation overlay",
    )
    parser.add_argument(
        "--annotation-dir",
        default="results/04_annotation",
        help="Passed through if overlay is enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_08d = REPO_ROOT / "scripts" / "26_networkx_visualization.py"

    cmd = [
        sys.executable,
        str(script_08d),
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--condition",
        args.condition,
        "--network-type",
        args.network_type,
        "--max-edges",
        str(args.max_edges),
        "--min-weight",
        str(args.min_weight),
        "--min-degree",
        str(args.min_degree),
        "--global-max-edges",
        str(args.global_max_edges),
        "--global-min-weight",
        str(args.global_min_weight),
        "--layout",
        args.layout,
        "--annotation-dir",
        args.annotation_dir,
        "--interactive-label-mode",
        args.interactive_label_mode,
    ]

    if args.power is not None:
        cmd.extend(["--power", str(args.power)])

    cmd.append("--interactive-html" if args.interactive_html else "--no-interactive-html")
    cmd.append("--interactive-physics" if args.interactive_physics else "--no-interactive-physics")
    cmd.append(
        "--interactive-celltype-overlay"
        if args.interactive_celltype_overlay
        else "--no-interactive-celltype-overlay"
    )

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
