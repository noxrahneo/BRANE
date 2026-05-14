from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def select_samples(cohort: pd.DataFrame, condition: str) -> pd.DataFrame:
    if condition.lower() == "all":
        return cohort.copy()
    return cohort[cohort["Condition"].astype(str) == condition].copy()


def safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def save_per_cell_table(
    per_cell_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> Path:
    parquet_file = out_dir / f"{prefix}_per_cell.parquet"
    csv_file = out_dir / f"{prefix}_per_cell.csv"
    try:
        per_cell_df.to_parquet(parquet_file, index=False)
        return parquet_file
    except Exception:
        per_cell_df.to_csv(csv_file, index=False)
        return csv_file


def load_per_cell_table(path: Path) -> pd.DataFrame | None:
    if path.suffix == ".parquet":
        csv_fallback = path.with_suffix(".csv")
    else:
        csv_fallback = path

    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback)
    return None


def plot_sample_boxpanels(
    per_cell_df: pd.DataFrame,
    out_file: Path,
    title_suffix: str,
) -> None:
    metrics = [
        ("n_counts", f"Library size per cell ({title_suffix})", True),
        ("n_genes", f"Detected genes per cell ({title_suffix})", True),
        ("pct_mito", f"Mitochondrial % per cell ({title_suffix})", False),
    ]

    sample_order = (
        per_cell_df.groupby("SampleName", observed=False)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for ax, (metric, title, use_log) in zip(axes, metrics):
        grouped = []
        labels = []
        for sample_name in sample_order:
            values = pd.Series(
                pd.to_numeric(
                    per_cell_df.loc[
                        per_cell_df["SampleName"] == sample_name,
                        metric,
                    ],
                    errors="coerce",
                )
            ).dropna()
            if values.empty:
                continue
            grouped.append(values.to_numpy())
            labels.append(sample_name)

        ax.boxplot(grouped, labels=labels, showfliers=False)
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(str(out_file), dpi=150)
    plt.close(fig)


def plot_sample_violinpanels(
    per_cell_df: pd.DataFrame,
    out_file: Path,
    title_suffix: str,
) -> None:
    metrics = [
        ("n_counts", f"Library size per cell ({title_suffix})", True),
        ("n_genes", f"Detected genes per cell ({title_suffix})", True),
        ("pct_mito", f"Mitochondrial % per cell ({title_suffix})", False),
    ]

    if "Condition" in per_cell_df.columns:
        group_order = (
            per_cell_df.groupby("Condition", observed=False)
            .size()
            .sort_values(ascending=False)
            .index.tolist()
        )
        group_col = "Condition"
    else:
        group_order = (
            per_cell_df.groupby("SampleName", observed=False)
            .size()
            .sort_values(ascending=False)
            .index.tolist()
        )
        group_col = "SampleName"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, (metric, title, use_log) in zip(axes, metrics):
        grouped = []
        labels = []
        for group_name in group_order:
            values = pd.Series(
                pd.to_numeric(
                    per_cell_df.loc[
                        per_cell_df[group_col] == group_name,
                        metric,
                    ],
                    errors="coerce",
                )
            ).dropna()
            if values.empty:
                continue
            grouped.append(values.to_numpy())
            labels.append(group_name)

        violin = ax.violinplot(grouped, showmedians=True)
        for body in violin["bodies"]:
            body.set_alpha(0.5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(str(out_file), dpi=150)
    plt.close(fig)


def plot_pre_post_violinpanels(
    per_cell_df: pd.DataFrame,
    out_file: Path,
) -> None:
    metrics = [
        ("n_counts", "Library size per cell"),
        ("n_genes", "Detected genes per cell"),
        ("pct_mito", "Mitochondrial % per cell"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, title) in zip(axes, metrics):
        pre_values = pd.Series(
            pd.to_numeric(
                per_cell_df.loc[per_cell_df["stage"] == "pre", metric],
                errors="coerce",
            )
        ).dropna()
        post_values = pd.Series(
            pd.to_numeric(
                per_cell_df.loc[per_cell_df["stage"] == "post", metric],
                errors="coerce",
            )
        ).dropna()

        data = [pre_values.to_numpy(), post_values.to_numpy()]
        violin = ax.violinplot(data, showmedians=True)
        for body in violin["bodies"]:
            body.set_alpha(0.5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre", "Post"])
        if metric in {"n_counts", "n_genes"}:
            ax.set_yscale("log")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(str(out_file), dpi=150)
    plt.close(fig)


def plot_pre_post_scatter(compare_df: pd.DataFrame, out_file: Path) -> None:
    metrics = [
        ("n_cells_pre", "n_cells_post", "Cells per sample"),
        (
            "median_genes_per_cell_pre",
            "median_genes_per_cell_post",
            "Median genes per cell",
        ),
        (
            "pct_mito_median_pre",
            "pct_mito_median_post",
            "Median mitochondrial %",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (x_col, y_col, title) in zip(axes, metrics):
        sub = compare_df[[x_col, y_col]].copy()
        sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
        sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        x_vals = sub[x_col].to_numpy(dtype=float)
        y_vals = sub[y_col].to_numpy(dtype=float)
        lo = min(x_vals.min(), y_vals.min())
        hi = max(x_vals.max(), y_vals.max())

        ax.scatter(x_vals, y_vals, s=25, alpha=0.8)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlabel("Pre")
        ax.set_ylabel("Post")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(str(out_file), dpi=150)
    plt.close(fig)
