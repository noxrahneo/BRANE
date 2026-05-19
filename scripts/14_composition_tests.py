#!/usr/bin/env python3
"""Run global composition tests across conditions.

R -> Python mapping used here:
- R: quasi-Poisson GLM on cluster/cell-type count tables.
  Python: statsmodels GLM(Poisson, scale='X2') with interaction tests.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from scipy.stats import f as f_dist

from utils.warehouse import (
    WarehouseRecord,
    append_warehouse,
    params_hash,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global composition tests across conditions"
    )
    parser.add_argument(
        "--input-dir",
        default="auto",
        help=(
            "Input annotation root with per-condition *_annotated.h5ad, or 'auto' "
            "to use canonical 04_annotation"
        ),
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
        "--group-col",
        default="cell_type_annot",
        help="obs column for composition units",
    )
    parser.add_argument(
        "--sample-col",
        default="SampleName",
        help="obs column for sample identity",
    )
    parser.add_argument(
        "--min-cells-per-sample",
        type=int,
        default=100,
        help="Minimum cells per sample to include",
    )
    parser.add_argument(
        "--output-dir",
        default="results/05_composition",
        help="Output directory for composition test outputs",
    )
    parser.add_argument("--dpi", type=int, default=170, help="Figure DPI")
    return parser.parse_args()


def resolve_base(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_annotation_root(input_dir_arg: str) -> Path:
    if input_dir_arg.strip().lower() != "auto":
        return resolve_base(input_dir_arg)

    candidates = [REPO_ROOT / "results/04_annotation"]
    for root in candidates:
        if root.exists() and list_conditions(root):
            return root

    return candidates[0]


def list_conditions(root: Path) -> list[str]:
    if not root.exists():
        return []
    names: list[str] = []
    for p in sorted(x for x in root.iterdir() if x.is_dir()):
        condition = p.name
        if (p / f"{condition}_annotated.h5ad").exists():
            names.append(condition)
    return names


def resolve_conditions(root: Path, requested: str) -> list[str]:
    available = list_conditions(root)
    if not available:
        raise FileNotFoundError(f"No condition folders found in: {root}")
    if requested.lower() == "all":
        return available
    if requested in available:
        return [requested]
    raise FileNotFoundError(
        f"Condition '{requested}' not found. Available: {', '.join(available)}"
    )


def load_obs_tables(
    root: Path,
    conditions: list[str],
    group_col: str,
    sample_col: str,
    min_cells_per_sample: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for condition in conditions:
        #load obs table and filter to samples meeting minimum cell count
        h5ad = root / condition / f"{condition}_annotated.h5ad"
        if not h5ad.exists():
            print(f"Skipping {condition}: missing {h5ad.name}")
            continue
        adata = sc.read_h5ad(h5ad)
        need = {group_col, sample_col}
        if not need.issubset(set(adata.obs.columns)):
            print(f"Skipping {condition}: missing required obs columns")
            continue

        obs = adata.obs[[group_col, sample_col]].copy()
        obs = obs.rename(
            columns={group_col: "cell_group", sample_col: "sample"}
        )
        obs["condition"] = condition
        obs["sample"] = obs["sample"].astype(str)
        obs["sample_id"] = obs["condition"].astype(str) + "__" + obs["sample"]
        obs["cell_group"] = obs["cell_group"].astype(str)

        sample_sizes = obs["sample_id"].value_counts()
        keep_samples = sample_sizes[sample_sizes >= min_cells_per_sample].index
        obs = obs[obs["sample_id"].isin(keep_samples)]
        if obs.empty:
            print(
                f"Skipping {condition}: no sample passes min_cells_per_sample="
                f"{min_cells_per_sample}"
            )
            continue
        chunks.append(obs)
        del adata
        gc.collect()

    if not chunks:
        return pd.DataFrame(columns=["condition", "sample_id", "cell_group"])
    return pd.concat(chunks, axis=0, ignore_index=True)


def build_count_table(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        obs.groupby(["condition", "sample_id", "cell_group"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = counts.pivot_table(
        index=["condition", "sample_id"],
        columns="cell_group",
        values="count",
        fill_value=0,
    )
    prop = pivot.div(pivot.sum(axis=1), axis=0)
    prop = prop.reset_index()
    return counts, prop


def fit_composition_glm(
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    #quasi-poisson glm: count ~ cell_group + sample + condition:cell_group
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for composition tests. "
            "Install with: pip install statsmodels"
        ) from exc

    full_formula = (
        "count ~ C(cell_group) + C(sample_id) + "
        "C(condition):C(cell_group)"
    )
    reduced_formula = "count ~ C(cell_group) + C(sample_id)"

    full = smf.glm(
        formula=full_formula,
        data=counts,
        family=sm.families.Poisson(),
    ).fit(scale="X2")
    reduced = smf.glm(
        formula=reduced_formula,
        data=counts,
        family=sm.families.Poisson(),
    ).fit(scale="X2")

    lr = 2.0 * (full.llf - reduced.llf)
    df_diff = max(int(full.df_model - reduced.df_model), 1)
    df_resid = int(full.df_resid)
    scale = float(full.scale)
    #quasi-poisson f-test: F = (LR / df_diff) / scale vs F(df_diff, df_resid)
    f_stat = (lr / df_diff) / scale
    pval = float(f_dist.sf(f_stat, df_diff, df_resid))

    global_df = pd.DataFrame(
        [
            {
                "test": "condition_x_cell_group",
                "lr_stat": float(lr),
                "f_stat": f_stat,
                "df_num": df_diff,
                "df_denom": df_resid,
                "p_value_f": pval,
                "scale_full": scale,
                "n_obs": int(full.nobs),
            }
        ]
    )

    coef = full.summary2().tables[1].reset_index()
    coef = coef.rename(columns={"index": "term"})
    coef["is_interaction"] = coef["term"].str.contains(
        r"C\(condition\).*:C\(cell_group\)", regex=True
    )
    return global_df, coef


CELL_TYPE_COLORS: dict[str, str] = {
    "BCell":          "#2C3591",
    "Basal_Epi":      "#E53935",
    "Cycling":        "#A4B815",
    "Endothelial":    "#2D5A27",
    "Epithelial":     "#FB8C00",
    "Fibroblast":     "#F06292",
    "Luminal_Epi":    "#F5D142",
    "Lymphoid_mixed": "#20857C",
    "Myeloid":        "#BCB1E6",
    "NK":             "#AD7C75",
    "Plasma":         "#00ACC1",
    "Stroma":         "#6D4C41",
    "TCell":          "#9E02EC",
}


def plot_condition_composition(
    prop: pd.DataFrame,
    out_file: Path,
    dpi: int,
) -> None:
    if prop.empty:
        return
    groups = [
        c for c in prop.columns if c not in {"condition", "sample_id"}
    ]
    mean_prop = prop.groupby("condition", observed=False)[groups].mean()
    colors = [CELL_TYPE_COLORS.get(g, "#cccccc") for g in groups]
    ax = mean_prop.plot(kind="bar", stacked=True, figsize=(10, 6), width=0.8, color=colors)
    ax.set_ylabel("Mean proportion")
    ax.set_xlabel("Condition")
    ax.set_title("Cell composition by condition (sample-mean proportions)")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title="Cell group",
    )
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main() -> int:
    args = parse_args()

    #resolve input annotation root and output dir
    root = resolve_annotation_root(args.input_dir)
    out_dir = resolve_base(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #list available conditions and exit if requested
    available = list_conditions(root)
    if args.list_conditions:
        if not available:
            print("No conditions found")
            return 0
        print("Available conditions:")
        for name in available:
            print(f"- {name}")
        return 0

    #resolve target conditions and load obs tables
    conditions = resolve_conditions(root, args.condition)
    arg_hash = params_hash(vars(args))
    now = utc_now_iso()
    obs = load_obs_tables(
        root=root,
        conditions=conditions,
        group_col=args.group_col,
        sample_col=args.sample_col,
        min_cells_per_sample=args.min_cells_per_sample,
    )
    if obs.empty:
        raise ValueError("No valid annotated cells found for composition test")

    #build count and proportion tables, then fit glm
    counts, prop = build_count_table(obs)
    #fit global glm: does composition differ between conditions?
    global_df, coef_df = fit_composition_glm(counts)

    #save all outputs
    counts_file = out_dir / "composition_counts_long.csv"
    prop_file = out_dir / "composition_proportions_by_sample.csv"
    global_file = out_dir / "composition_glm_global_test.csv"
    coef_file = out_dir / "composition_glm_coefficients.csv"
    fig_file = out_dir / "composition_condition_stacked_bar.png"

    counts.to_csv(counts_file, index=False)
    prop.to_csv(prop_file, index=False)
    global_df.to_csv(global_file, index=False)
    coef_df.to_csv(coef_file, index=False)
    plot_condition_composition(prop=prop, out_file=fig_file, dpi=args.dpi)

    #append provenance to warehouse log
    records = [
        WarehouseRecord(
            input_file=str(root / condition / f"{condition}_annotated.h5ad"),
            output_file=str(global_file),
            script="scripts/14_composition_tests.py",
            date_utc=now,
            params_hash=arg_hash,
            condition=condition,
            stage="composition",
        )
        for condition in conditions
    ]
    warehouse_file = append_warehouse(out_dir, records)

    print("Composition test complete")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Global test: {global_file}")
    print(f"Warehouse log: {warehouse_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
