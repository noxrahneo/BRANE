"""KM curves, log-rank test, and univariate Cox HR for top hub genes in TCGA-BRCA (DSS and OS)."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_ANNOT_DIR = REPO_ROOT / "results/20_node_annotation"
OUTPUT_ROOT = REPO_ROOT / "results/23_survival"
CACHE_DIR = OUTPUT_ROOT / "tcga_cache"

TIERS = ["D", "S_case", "S_ctrl"]

PAIR_SHORT = {
    "ER_tumor__vs__Normal": "ER",
    "HER2_tumor__vs__Normal": "HER2",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": "NormalBRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal": "TNBC_BRCA1",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": "TNBC_BRCA1_vs_NormalBRCA1",
    "Triple_negative_tumor__vs__Normal": "TNBC",
}

N_HUBS = 50
MIN_EVENTS = 3
FDR_THRESH = 0.1

#uCSC Xena hubs — confirmed working endpoints
XENA_TCGA_HUB = "https://tcga.xenahubs.net"
XENA_PANCAN_HUB = "https://pancanatlas.xenahubs.net"

#expression: TCGA BRCA hub (log2 normalised counts)
EXPR_DATASET = "TCGA.BRCA.sampleMap/HiSeqV2"

#survival: Liu et al. 2018 pan-cancer curated endpoints (OS, DSS, DFI)
SURV_DATASET = "Survival_SupplementalTable_S1_20171025_xena_sp"

#phenotype: ER/PR/HER2 status for subtype matching
PHENO_DATASET = "TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"

#phenotype field names (decoded values: Positive/Negative/Indeterminate)
PHENO_ER_FIELD = "breast_carcinoma_estrogen_receptor_status"
PHENO_PR_FIELD = "breast_carcinoma_progesterone_receptor_status"
PHENO_HER2_FIELD = "HER2_Final_Status_nature2012"

PAIR_NAMES = [
    "ER_tumor__vs__Normal",
    "HER2_tumor__vs__Normal",
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal",
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic",
    "Triple_negative_tumor__vs__Normal",
]

#subtype filters applied to TCGA phenotype columns.
#none → use all BRCA patients (fallback for BRCA1 pairs without matched cohort).
PAIR_SUBTYPE: dict[str, Optional[dict[str, list[str]]]] = {
    "ER_tumor__vs__Normal": {
        PHENO_ER_FIELD: ["Positive"],
    },
    "HER2_tumor__vs__Normal": {
        PHENO_HER2_FIELD: ["Positive"],
    },
    "Triple_negative_tumor__vs__Normal": {
        PHENO_ER_FIELD: ["Negative"],
        PHENO_PR_FIELD: ["Negative"],
        PHENO_HER2_FIELD: ["Negative"],
    },
    "Triple_negative_BRCA1_tumor__vs__Normal": {
        PHENO_ER_FIELD: ["Negative"],
        PHENO_PR_FIELD: ["Negative"],
        PHENO_HER2_FIELD: ["Negative"],
    },
    #no matched TCGA cohort for BRCA1 germline carriers → all BRCA fallback
    "Triple_negative_BRCA1_tumor__vs__Normal_BRCA1_-_pre-neoplastic": None,
    "Normal_BRCA1_-_pre-neoplastic__vs__Normal": None,
}

DIRECTION_COLOURS = {
    "up": "#d62728",
    "down": "#1f77b4",
    "stable": "#7f7f7f",
    "unchanged": "#7f7f7f",
}

_ENDPOINT_LABEL = {
    "DSS": "Disease-Specific Survival",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)



def load_tier_hubs(pair_name: str, tier: str) -> pd.DataFrame:
    #load pre-computed hub gene list for a given tier from node_annotation outputs
    prefix = PAIR_SHORT.get(pair_name, pair_name)
    hub_file = NODE_ANNOT_DIR / pair_name / f"{prefix}_hubs_{tier}.csv"
    if not hub_file.exists():
        raise FileNotFoundError(f"Hub file not found: {hub_file}")
    df = pd.read_csv(hub_file, low_memory=False)
    #normalise column names: tier_degree → weighted_degree, lnFC → lfc
    if "tier_degree" in df.columns and "weighted_degree" not in df.columns:
        df = df.rename(columns={"tier_degree": "weighted_degree"})
    if "lnFC" in df.columns and "lfc" not in df.columns:
        df = df.rename(columns={"lnFC": "lfc"})
    if "deg_direction" not in df.columns and "direction" in df.columns:
        df = df.rename(columns={"direction": "deg_direction"})
    df["deg_direction"] = df.get("deg_direction", pd.Series("unknown", index=df.index)).fillna("unknown").str.lower()
    return df



def _xena_field_values(
    hub: str, dataset: str, fields: list[str], samples: list[str]
) -> pd.DataFrame:
    #fetch fields from a Xena clinical/phenotype dataset, decoding categorical codes
    import xenaPython as xena
    rows: dict[str, list] = {"sample": samples}
    for field in fields:
        try:
            _, [vals] = xena.dataset_probe_values(hub, dataset, samples, [field])
            codes = xena.field_codes(hub, dataset, [field])
            if codes and codes[0].get("code"):
                #categorical field: decode integer index → string label
                code_list = codes[0]["code"].split("\t")
                decoded = []
                for v in vals:
                    try:
                        if str(v) == "NaN" or np.isnan(float(v)):
                            decoded.append(None)
                        else:
                            decoded.append(code_list[int(float(v))])
                    except (ValueError, TypeError):
                        decoded.append(None)
                rows[field] = decoded
            else:
                #numeric field: coerce to float (replaces 'NaN' strings)
                rows[field] = pd.to_numeric(vals, errors="coerce").tolist()
        except Exception as exc:
            log.warning("Could not fetch field %s: %s", field, exc)
            rows[field] = [None] * len(samples)
    return pd.DataFrame(rows)


def _xena_gene_expression(
    hub: str, dataset: str, samples: list[str], genes: list[str],
    batch_size: int = 20,
) -> pd.DataFrame:
    #return (samples × genes) DataFrame from a Xena gene expression dataset
    import xenaPython as xena

    all_expr: dict[str, list] = {}
    n_found = 0

    for i in range(0, len(genes), batch_size):
        batch = genes[i : i + batch_size]
        try:
            result = xena.dataset_gene_probe_avg(hub, dataset, samples, batch)
            for entry in result:
                gene_name = entry["gene"]
                scores = entry["scores"]
                if not scores or not scores[0]:
                    continue
                if len(scores) == 1:
                    vals = [float(v) if str(v) != "NaN" else float("nan") for v in scores[0]]
                else:
                    arr = np.array(
                        [[float(v) if str(v) != "NaN" else float("nan") for v in row] for row in scores]
                    )
                    vals = np.nanmean(arr, axis=0).tolist()
                #only keep if length matches sample count
                if len(vals) != len(samples):
                    continue
                all_expr[gene_name] = vals
                n_found += 1
        except Exception as exc:
            log.warning("Expression batch %d–%d failed: %s", i, i + batch_size, exc)

    if not all_expr:
        log.error("No gene expression data retrieved for any gene in the batch.")
        return pd.DataFrame(index=samples)

    log.info("Expression: %d/%d genes retrieved from Xena", n_found, len(genes))
    return pd.DataFrame(all_expr, index=samples)


def load_tcga(genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    #load TCGA-BRCA expression + survival + phenotype (Xena, cached)
    import xenaPython as xena

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expr_cache = CACHE_DIR / "tcga_brca_expression.parquet"
    clin_cache = CACHE_DIR / "tcga_brca_clinical.parquet"

    if clin_cache.exists():
        log.info("Loading clinical data from cache")
        clin_df = pd.read_parquet(clin_cache)
    else:
        log.info("Downloading TCGA-BRCA clinical data from Xena ...")

        #bRCA sample IDs come from the expression dataset (BRCA-specific hub)
        brca_samples = xena.dataset_samples(XENA_TCGA_HUB, EXPR_DATASET, None)
        log.info("  %d BRCA samples in expression dataset", len(brca_samples))

        #survival: Liu et al. pan-cancer curated (OS, DSS, DFI) from pancanAtlas
        surv_df = _xena_field_values(
            XENA_PANCAN_HUB, SURV_DATASET,
            ["DSS", "DSS.time"],
            brca_samples,
        )

        #phenotype: ER/PR/HER2 status from BRCA clinical matrix
        pheno_df = _xena_field_values(
            XENA_TCGA_HUB, PHENO_DATASET,
            [PHENO_ER_FIELD, PHENO_PR_FIELD, PHENO_HER2_FIELD],
            brca_samples,
        )

        clin_df = surv_df.merge(
            pheno_df.drop(columns=["sample"], errors="ignore"),
            left_index=True, right_index=True, how="left",
        )
        clin_df.to_parquet(clin_cache, index=False)
        log.info("Clinical cache saved (%d samples)", len(clin_df))

    cached_genes: set[str] = set()
    if expr_cache.exists():
        log.info("Loading expression data from cache")
        expr_df = pd.read_parquet(expr_cache)
        cached_genes = set(expr_df.columns)
    else:
        expr_df = pd.DataFrame()

    missing = [g for g in genes if g not in cached_genes]
    if missing:
        log.info("Fetching %d new genes from Xena expression dataset ...", len(missing))
        brca_samples = (
            list(clin_df["sample"])
            if "sample" in clin_df.columns
            else xena.dataset_samples(XENA_TCGA_HUB, EXPR_DATASET, None)
        )
        new_expr = _xena_gene_expression(XENA_TCGA_HUB, EXPR_DATASET, brca_samples, missing)
        if not new_expr.empty:
            expr_df = pd.concat([expr_df, new_expr], axis=1)
            expr_df.to_parquet(expr_cache)
            log.info("Expression cache updated (%d genes total)", expr_df.shape[1])

    return expr_df, clin_df



def filter_subtype(
    clin_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    pair_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    #return (filtered_clin, filtered_expr, cohort_note)
    subtype_filter = PAIR_SUBTYPE.get(pair_name)
    if subtype_filter is None:
        cohort_note = "all_BRCA_fallback"
        mask = pd.Series(True, index=clin_df.index)
    else:
        cohort_note = "subtype_matched"
        mask = pd.Series(True, index=clin_df.index)
        for col, allowed in subtype_filter.items():
            if col in clin_df.columns:
                mask &= clin_df[col].isin(allowed)
            else:
                log.warning("Phenotype column %s not found in clinical data", col)

    clin_sub = clin_df[mask].copy()
    sample_col = "sample" if "sample" in clin_sub.columns else None
    if sample_col:
        expr_sub = expr_df.loc[expr_df.index.isin(clin_sub[sample_col])]
    else:
        expr_sub = expr_df

    log.info("%s: %d patients (%s)", pair_name, len(clin_sub), cohort_note)
    return clin_sub, expr_sub, cohort_note



def _prepare_km_data(
    gene: str,
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
    endpoint: str,
) -> Optional[tuple]:
    #merge expression + survival, median split. Returns None if skipped
    time_col = f"{endpoint}.time"
    event_col = endpoint

    if gene not in expr_df.columns:
        return None
    if time_col not in clin_df.columns or event_col not in clin_df.columns:
        return None

    sample_col = "sample" if "sample" in clin_df.columns else None
    clin_idx = clin_df.set_index(sample_col) if sample_col else clin_df.copy()

    merged = pd.DataFrame({
        "expr": expr_df[gene].reindex(clin_idx.index),
        "T": pd.to_numeric(clin_idx[time_col], errors="coerce"),
        "E": pd.to_numeric(clin_idx[event_col], errors="coerce"),
    }).dropna()
    merged = merged[merged["T"] > 0]

    if len(merged) < 20:
        return None

    median_expr = merged["expr"].median()
    high = merged[merged["expr"] >= median_expr]
    low = merged[merged["expr"] < median_expr]

    if min(len(high), len(low)) < 5:
        return None
    if high["E"].sum() < MIN_EVENTS or low["E"].sum() < MIN_EVENTS:
        return None

    return (
        high["T"].values, high["E"].values,
        low["T"].values, low["E"].values,
        len(high), len(low),
    )


def _run_logrank_cox(T_high, E_high, T_low, E_low) -> dict:
    from lifelines.statistics import logrank_test
    from lifelines import CoxPHFitter

    lr = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)

    cox_data = pd.DataFrame({
        "T": np.concatenate([T_high, T_low]),
        "E": np.concatenate([E_high, E_low]),
        "group": np.concatenate([np.ones(len(T_high)), np.zeros(len(T_low))]),
    })
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="T", event_col="E")
    s = cph.summary

    return {
        "p_logrank": lr.p_value,
        "hr": float(np.exp(s.loc["group", "coef"])),
        "hr_lower": float(np.exp(s.loc["group", "coef lower 95%"])),
        "hr_upper": float(np.exp(s.loc["group", "coef upper 95%"])),
    }



def plot_km_curve(
    T_high, E_high, T_low, E_low,
    gene: str, pair_name: str, endpoint: str,
    p_logrank: float, hr: float, n_high: int, n_low: int,
    out_path: Path,
) -> None:
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_high = KaplanMeierFitter().fit(T_high, E_high, label=f"High expression (n={n_high})")
    kmf_low  = KaplanMeierFitter().fit(T_low,  E_low,  label=f"Low expression (n={n_low})")
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#d62728")
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")

    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=7)

    ax.set_title(f"{gene}  |  {pair_name.replace('__vs__', ' vs ')}", fontsize=9)
    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel(f"{_ENDPOINT_LABEL.get(endpoint, endpoint)} probability", fontsize=9)
    ax.text(
        0.97, 0.97, f"Log-rank p={p_logrank:.3g}\nHR={hr:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top5_km(
    results: list[dict],
    hubs_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
    pair_name: str,
    tier: str,
    endpoint: str,
    out_path: Path,
    n_top: int = 6,
    panel_kind: str = "top_hubs",  # "top_hubs" or "sig_genes"
) -> None:
    #multi-panel KM figure; panel_kind='top_hubs' or 'sig_genes'
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from lifelines.plotting import add_at_risk_counts

    if direction_filter is not None:
        subset_genes = hubs.loc[
            hubs["deg_direction"].str.lower() == direction_filter, "gene"
        ].tolist()
    else:
        subset_genes = hubs["gene"].tolist()

    available = [g for g in subset_genes if g in expr_df.columns]
    if len(available) < 3:
        log.warning(
            "%s: %d genes available for %s risk score — skipping",
            pair_name, len(available), direction_filter or "all",
        )
        return None

    time_col = f"{endpoint}.time"
    event_col = endpoint
    sample_col = "sample" if "sample" in clin_df.columns else None
    clin_idx = clin_df.set_index(sample_col) if sample_col else clin_df.copy()

    expr_sub = expr_df[available].reindex(clin_idx.index)
    z = (expr_sub - expr_sub.mean()) / (expr_sub.std() + 1e-9)
    risk_score = z.mean(axis=1)

    merged = pd.DataFrame({
        "risk": risk_score,
        "T": pd.to_numeric(clin_idx[time_col], errors="coerce"),
        "E": pd.to_numeric(clin_idx[event_col], errors="coerce"),
    }).dropna()
    merged = merged[merged["T"] > 0]

    if len(merged) < 20:
        return None

    median_risk = merged["risk"].median()
    high = merged[merged["risk"] >= median_risk]
    low = merged[merged["risk"] < median_risk]

    if high["E"].sum() < MIN_EVENTS or low["E"].sum() < MIN_EVENTS:
        return None

    lr = logrank_test(high["T"], low["T"], event_observed_A=high["E"], event_observed_B=low["E"])
    p = lr.p_value

    #median follow-up: median observed time across all patients (Kaplan-Meier estimator of follow-up)
    median_followup = int(merged["T"].median())

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_high = KaplanMeierFitter().fit(high["T"], high["E"], label=f"High risk (n={len(high)})")
    kmf_low  = KaplanMeierFitter().fit(low["T"],  low["E"],  label=f"Low risk (n={len(low)})")
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#d62728")
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")

    add_at_risk_counts(kmf_high, kmf_low, ax=ax, fontsize=7)

    dir_label = {None: "all hubs", "up": "UP hubs", "down": "DOWN hubs"}.get(direction_filter, direction_filter)
    ax.set_title(
        f"Hub risk score ({dir_label})  |  {pair_name.replace('__vs__', ' vs ')}\n"
        f"{cohort_note}  |  n_genes={len(available)}  |  median follow-up={median_followup} days",
        fontsize=9,
    )
    ax.set_xlabel("Time (days)", fontsize=9)
    ax.set_ylabel(f"{_ENDPOINT_LABEL.get(endpoint, endpoint)} probability", fontsize=9)
    ax.text(
        0.97, 0.97, f"Log-rank p={p:.3g}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"n_genes": len(available), "p_logrank": p, "direction_filter": direction_filter or "all",
            "median_followup_days": median_followup}



def run_pair(
    pair_name: str,
    tier: str,
    expr_df: pd.DataFrame,
    clin_df: pd.DataFrame,
) -> list[dict]:
    out_dir = OUTPUT_ROOT / pair_name / tier
    out_dir.mkdir(parents=True, exist_ok=True)

    hubs = load_tier_hubs(pair_name, tier)
    log.info("%s [%s]: %d hub genes", pair_name, tier, len(hubs))

    clin_sub, expr_sub, cohort_note = filter_subtype(clin_df, expr_df, pair_name)
    n_patients = len(clin_sub)

    if n_patients < 200:
        log.warning(
            "%s: only %d patients — individual gene FDR results are exploratory "
            "(low power). Interpret absence of significance cautiously.",
            pair_name, n_patients,
        )

    all_results: list[dict] = []

    for endpoint in ["DSS"]:
        log.info("%s: running %s ...", pair_name, endpoint)
        per_gene_rows: list[dict] = []

        for _, hub_row in hubs.iterrows():
            gene = hub_row["gene"]
            base = {
                "pair_name": pair_name,
                "tier": tier,
                "gene": gene,
                "endpoint": endpoint,
                "deg_direction": hub_row.get("deg_direction", "unknown"),
                "lfc": hub_row.get("lfc", float("nan")),
                "weighted_degree": hub_row.get("weighted_degree", float("nan")),
                "cohort_note": cohort_note,
                "n_patients": n_patients,
            }

            km_data = _prepare_km_data(gene, expr_sub, clin_sub, endpoint)
            if km_data is None:
                per_gene_rows.append({**base, "skipped": True, "skip_reason": "insufficient data/events"})
                continue

            T_high, E_high, T_low, E_low, n_high, n_low = km_data
            try:
                stats = _run_logrank_cox(T_high, E_high, T_low, E_low)
            except Exception as exc:
                log.debug("%s %s Cox failed: %s", pair_name, gene, exc)
                per_gene_rows.append({**base, "skipped": True, "skip_reason": "Cox fit failed"})
                continue

            per_gene_rows.append({
                **base,
                "n_high": n_high, "n_low": n_low,
                "n_events_high": int(E_high.sum()), "n_events_low": int(E_low.sum()),
                "p_logrank": stats["p_logrank"],
                "hr": stats["hr"], "hr_lower": stats["hr_lower"], "hr_upper": stats["hr_upper"],
                "skipped": False,
            })

        #bH-FDR correction across all tested genes
        tested = [r for r in per_gene_rows if not r.get("skipped", True)]
        if tested:
            from statsmodels.stats.multitest import multipletests
            _, padj, _, _ = multipletests([r["p_logrank"] for r in tested], method="fdr_bh")
            for r, pa in zip(tested, padj):
                r["p_adj"] = pa

        results_df = pd.DataFrame(per_gene_rows)

        #directionality consistency flag:
        # "consistent"   = high expression → worse survival AND gene is UP in tumor
        #                   OR high expression → better survival AND gene is DOWN in tumor
        # "paradoxical"  = the survival direction conflicts with tumor expression direction
        #                   e.g. gene is DOWN in tumor but high expression → worse survival (like EMP1)
        # "not_assessed" = gene was skipped or direction is unknown
        if "hr" in results_df.columns and "deg_direction" in results_df.columns:
            def _consistency(row):
                if row.get("skipped", True) or pd.isna(row.get("hr")):
                    return "not_assessed"
                d = str(row.get("deg_direction", "")).lower()
                if d not in ("up", "down"):
                    return "not_assessed"
                hr_high = row["hr"] > 1  #true = high expression → worse survival
                if (hr_high and d == "up") or (not hr_high and d == "down"):
                    return "consistent"
                return "paradoxical"
            results_df["direction_consistency"] = results_df.apply(_consistency, axis=1)

        prefix = PAIR_SHORT.get(pair_name, pair_name)
        results_df.to_csv(out_dir / f"{prefix}_{tier}_{endpoint}_km_results.csv", index=False)
        all_results.extend(results_df.to_dict(orient="records"))

        #kM plots for significant genes
        sig_genes = [r for r in tested if r.get("p_adj", 1) < FDR_THRESH]
        MAX_PANEL = 18  # beyond this a grid becomes unreadable

        if sig_genes:
            sig_dir = out_dir / f"{endpoint}_km_plots"
            for r in sig_genes:
                km_data = _prepare_km_data(r["gene"], expr_sub, clin_sub, endpoint)
                if km_data:
                    T_h, E_h, T_l, E_l, n_h, n_l = km_data
                    plot_km_curve(
                        T_h, E_h, T_l, E_l,
                        gene=r["gene"], pair_name=pair_name, endpoint=endpoint,
                        p_logrank=r["p_logrank"], hr=r["hr"],
                        n_high=n_h, n_low=n_l,
                        out_path=sig_dir / f"{r['gene']}_{endpoint}_km.png",
                    )

            #combined panel when number of significant genes is manageable
            if len(sig_genes) <= MAX_PANEL:
                #sort by p_adj ascending so most significant genes come first
                sig_sorted = sorted(sig_genes, key=lambda r: r.get("p_adj", 1))
                plot_top5_km(
                    results=sig_sorted,
                    hubs_df=pd.DataFrame(
                        [{"gene": r["gene"],
                          "weighted_degree": len(sig_sorted) - i}  # rank preserves p_adj order
                         for i, r in enumerate(sig_sorted)]
                    ),
                    expr_df=expr_sub,
                    clin_df=clin_sub,
                    pair_name=pair_name,
                    tier=tier,
                    endpoint=endpoint,
                    out_path=out_dir / f"{prefix}_{tier}_{endpoint}_sig_km_panel.png",
                    n_top=min(len(sig_sorted), 6),
                    panel_kind="sig_genes",
                )

        log.info("%s %s: %d FDR<%.1f genes, plots saved", pair_name, endpoint, len(sig_genes), FDR_THRESH)

        #forest plot
        tested_df = results_df[~results_df["skipped"].fillna(True)]
        if not tested_df.empty:
            plot_forest(
                tested_df, pair_name=pair_name, endpoint=endpoint,
                out_path=out_dir / f"{prefix}_{tier}_{endpoint}_forest_plot.png",
            )

        #top-5 hub KM panel — always generated regardless of FDR
        plot_top5_km(
            results=tested,
            hubs_df=hubs,
            expr_df=expr_sub,
            clin_df=clin_sub,
            pair_name=pair_name,
            tier=tier,
            endpoint=endpoint,
            out_path=out_dir / f"{prefix}_{tier}_{endpoint}_top6_km.png",
        )

    return all_results



def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    #collect all hub genes across all pairs × tiers for a single TCGA load
    log.info("Loading hub gene lists for all pairs × tiers ...")
    run_plan: list[tuple[str, str, pd.DataFrame]] = []  # (pair, tier, hubs)
    all_genes: set[str] = set()
    for pair in PAIR_NAMES:
        for tier in TIERS:
            try:
                hubs = load_tier_hubs(pair, tier)
                run_plan.append((pair, tier, hubs))
                all_genes.update(hubs["gene"].dropna().tolist())
            except FileNotFoundError as exc:
                log.warning("Skipping %s / %s: %s", pair, tier, exc)

    log.info("Total unique hub genes across all pairs × tiers: %d", len(all_genes))

    log.info("Loading TCGA-BRCA data ...")
    expr_df, clin_df = load_tcga(list(all_genes))

    if expr_df.empty or clin_df.empty:
        log.error(
            "TCGA data could not be loaded. Check network connectivity to Xena "
            "or inspect %s for cached data.", CACHE_DIR,
        )
        return 1

    all_results: list[dict] = []
    for pair, tier, _ in tqdm(run_plan, desc="Pair/tier", unit="run"):
        try:
            all_results.extend(run_pair(pair, tier, expr_df, clin_df))
        except Exception as exc:
            log.error("Pair %s / tier %s failed: %s", pair, tier, exc, exc_info=True)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(OUTPUT_ROOT / "survival_summary.csv", index=False)
        log.info("Survival summary: %d rows", len(summary_df))

        top_df = (
            summary_df[
                (~summary_df["skipped"].fillna(True))
                & (summary_df["p_adj"].fillna(1) < FDR_THRESH)
            ]
            .sort_values("hr", ascending=False, key=abs)
            .drop_duplicates(subset=["gene", "pair_name", "tier"])
        )
        top_df.to_csv(OUTPUT_ROOT / "top_prognostic_genes.csv", index=False)
        log.info("Top prognostic genes (FDR<%.1f): %d", FDR_THRESH, len(top_df))

    log.info("Done. Results in %s", OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
