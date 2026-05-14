#!/usr/bin/env python3
"""Replot permutation threshold convergence figures from saved maxima CSVs.

Reads *_permutation_maxima.csv files already in results/13_csd_thresholds/
and regenerates the convergence PNGs with the canonical CSD colour scheme
(C=blue, S=teal-green, D=red). Does not re-run any permutations.

Run from BRANE/:
    conda run -n breast_cancer_scrnaseq python3 scripts/replot_convergence_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_DIR = REPO_ROOT / "results/13_csd_thresholds"
FIGURES_DIR = REPO_ROOT / "../overleaf/Thesis/figures/permutation"

EDGE_COLORS = {"C": "#2b6fb0", "S": "#2a9d8f", "D": "#e63946"}

LABEL_MAP = {
    "C": "Conserved (C)",
    "S": "Specific (S)",
    "D": "Differentiated (D)",
}


def plot_convergence(maxima_df: pd.DataFrame, pair_name: str, out_png: Path) -> None:
    run = maxima_df["permutation"].to_numpy(dtype=int)
    c_run = maxima_df["max_C"].expanding().mean().to_numpy(dtype=float)
    s_run = maxima_df["max_S"].expanding().mean().to_numpy(dtype=float)
    d_run = maxima_df["max_D"].expanding().mean().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    ax.plot(run, c_run, label=LABEL_MAP["C"], linewidth=1.8, color=EDGE_COLORS["C"])
    ax.plot(run, s_run, label=LABEL_MAP["S"], linewidth=1.8, color=EDGE_COLORS["S"])
    ax.plot(run, d_run, label=LABEL_MAP["D"], linewidth=1.8, color=EDGE_COLORS["D"])
    ax.set_xlabel("Permutation")
    ax.set_ylabel("Running mean of permutation maxima")
    ax.set_title(f"{pair_name.replace('__vs__', ' vs ')}: threshold convergence")
    ax.legend(frameon=False)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_png}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pair_dirs = sorted(
        p for p in THRESHOLDS_DIR.iterdir()
        if p.is_dir() and "__vs__" in p.name
    )

    if not pair_dirs:
        raise RuntimeError(f"No pair directories found in {THRESHOLDS_DIR}")

    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        maxima_csv = pair_dir / f"{pair_name}_permutation_maxima.csv"

        if not maxima_csv.exists():
            print(f"  [skip] no maxima CSV for {pair_name}")
            continue

        maxima_df = pd.read_csv(maxima_csv)

        # overwrite in results dir
        results_png = pair_dir / f"{pair_name}_permutation_threshold_convergence.png"
        plot_convergence(maxima_df, pair_name, results_png)

        # copy to thesis figures dir
        thesis_png = FIGURES_DIR / f"{pair_name}_permutation_threshold_convergence.png"
        import shutil
        shutil.copy2(results_png, thesis_png)
        print(f"  copied → {thesis_png}")

    print(f"\nDone. {len(pair_dirs)} contrasts processed.")


if __name__ == "__main__":
    main()
