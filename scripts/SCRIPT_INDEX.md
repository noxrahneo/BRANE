# BRANE Pipeline Script Index

**47 scripts** numbered 01–47, fully sequential. Scripts 03–07 and 08–11 carry the `_fromr` suffix, indicating they are adapted from the original Pal et al. R analysis code, reimplemented in Python/Scanpy.

---

## Data ingestion

| # | Script | Description |
|---|---|---|
| 01 | `01_create_bigboss.py` | Build the BigBoss sample inventory from SampleStats.txt, Table EV4 metadata and raw GEO filenames |
| 02 | `02_chopper.py` | Filter to the 49-sample analysis cohort (female, total-cell captures, six conditions) |

---

## Quality control

| # | Script | Description |
|---|---|---|
| 03 | `03_pre_qc_fromr.py` | Per-sample QC metrics on raw count matrices before cell filtering |
| 04 | `04_filtering_cells_fromr.py` | Apply per-sample QC thresholds to filter low-quality cells |
| 05 | `05_post_qc_fromr.py` | Post-filter QC and pre/post comparison |

---

## Preprocessing

| # | Script | Description |
|---|---|---|
| 06 | `06_per_sample_preprocess_fromr.py` | Per-sample library-size normalisation, log-transformation, HVG identification |
| 07 | `07_validate_preprocess_fromr.py` | Validate per-sample preprocessing outputs |

---

## Integration

| # | Script | Description |
|---|---|---|
| 08 | `08_integrate_per_condition_fromr.py` | Per-condition sample concatenation, re-normalisation, HVG selection, ComBat batch correction, PCA, UMAP, Leiden clustering |
| 09 | `09_hvg_diagnostics.py` | HVG and PCA diagnostics per integrated condition |
| 10 | `10_plot_integrated_results_fromr.py` | Per-condition UMAP and embedding plots |
| 11 | `11_plot_integrated_panels_fromr.py` | Multi-panel UMAP figures across all conditions |

---

## Cell type annotation

| # | Script | Description |
|---|---|---|
| 12 | `12_annotate_clusters_fromr.py` | Signature-score-based cell-type label assignment to Leiden clusters |
| 13 | `13_marker_heatmaps.py` | Marker gene heatmaps confirming cluster-level cell-type identity |

---

## Cell type composition

| # | Script | Description |
|---|---|---|
| 14 | `14_composition_tests.py` | Quasi-Poisson GLM composition tests across conditions; stacked bar visualisation |

---

## Pseudobulk aggregation and QC

| # | Script | Description |
|---|---|---|
| 15 | `15_pseudobulk.py` | Build pseudobulk count and logCPM matrices per condition using decoupler |
| 16 | `16_deg_ttest_logcpm.py` | DEG analysis on pseudobulk logCPM via Welch's t-test; BH–FDR correction |
| 17 | `17_deg_ttest_plots.py` | Volcano, MA and summary plots from DEG results |
| 18 | `18_pseudobulk_sanity_check.py` | Library-size outlier flags, low-cell flags, sample-dominance checks |

---

## Pre-correlation and correlation

| # | Script | Description |
|---|---|---|
| 19 | `19_prepare_pre_correlation_pack.py` | Assemble pre-correlation input pack; generate QC inclusion masks |
| 20 | `20_prepare_condition_logcpm_h5ad.py` | Export per-condition filtered logCPM h5ad files for network input |
| 21 | `21_compute_correlations.py` | Per-condition gene–gene Pearson correlation matrices from pseudobulk logCPM |
| 22 | `22_correlation_qc_plots.py` | QC and diagnostic plots for correlation matrices |

---

## Differential network construction

| # | Script | Description |
|---|---|---|
| 23 | `23_csd_scoring.py` | CSD decomposition: compute C, S, D scores for all gene pairs per condition pair |
| 24 | `24_csd_permutation_thresholds.py` | Permutation-based edge-weight thresholds (500 permutations per condition pair) |
| 25 | `25_csd_network_builder.py` | Build union differential network; Leiden community detection; node topology metrics |
| 26 | `26_csd_visualization.py` | Interactive HTML and static PNG visualisations per differential network |
| 27 | `27_csd_panels.py` | Per-module panel visualisations combining full network layout with community structure |

---

## Node homogeneity

| # | Script | Description |
|---|---|---|
| 28 | `28_node_homogeneity.py` | Per-gene edge-type concentration score H_i; identifies nodes that specialise in one CSD type |

---

## Gene and cancer annotation

| # | Script | Description |
|---|---|---|
| 29 | `29_hgnc_normalization.py` | Normalize gene symbols to approved HGNC nomenclature via REST API |
| 30 | `30_ncbi_gene_fetch.py` | Fetch NCBI Gene metadata: biotype, summary, GO terms, OMIM disease identifiers |
| 31 | `31_annotation_bridge.py` | Consolidate HGNC/NCBI tables into annotation bridge inputs for downstream scripts |
| 32 | `32_cancer_gene_annotation.py` | Assign cancer gene confidence tiers from eight curated databases (COSMIC, IntOGen, OncoKB, NCG, OncoVar, CancerMine, UniProt, Open Targets) |
| 33 | `33_cell_type_assignment.py` | Assign primary cell-type label per network gene via z-score across single-cell expression data |

---

## Node annotation assembly

| # | Script | Description |
|---|---|---|
| 34 | `34_node_annotation_assembly.py` | Consolidate all annotation layers into master node table; update network HTML metadata |
| 35 | `35_lfc_annotation.py` | Add condition-level log-fold-change and directional category (UP/DOWN/STABLE) to node tables |
| 36 | `36_prepare_visualization_assets.py` | Assemble fully-annotated node tables, per-subnetwork hub rankings, module summaries, and annotated network HTML and PNG outputs |

---

## Pathway enrichment

| # | Script | Description |
|---|---|---|
| 37 | `37_go_pathway_enrichment.py` | GO/KEGG/Reactome enrichment via g:Profiler REST API for hub gene sets, all-tier gene sets and top Leiden modules |

---

## Network homophily

| # | Script | Description |
|---|---|---|
| 38 | `38_tag_permutation.py` | Weighted neighbourhood homophily; 1,000 Fisher–Yates label-preserving permutation test |

---

## Survival analysis

| # | Script | Description |
|---|---|---|
| 39 | `39_survival_analysis.py` | KM curves, log-rank test, univariate Cox HR for top hub genes in TCGA-BRCA (DSS and OS); BH–FDR correction |

---

## Drug-target integration

| # | Script | Description |
|---|---|---|
| 40 | `40_drug_target_candidates.py` | DGIdb GraphQL query for hub gene–drug interactions; synonym expansion via ChEMBL/RxNorm; composite candidate scoring |
| 41 | `41_visualize_drug_candidates.py` | Visualisation plots for drug candidate scores and interaction networks |
| 42 | `42_drug_candidates_per_pair.py` | Per-network drug candidate rankings and summary tables |
| 43 | `43_drug_survival_km.py` | Drug-annotated KM curves for druggable hub genes; HR and FDR annotation |

---

## Validation and cross-network analysis

| # | Script | Description |
|---|---|---|
| 44 | `44_cross_network_overlap.py` | Cross-network hub gene overlap: Jaccard similarity, presence/absence heatmaps, pan-subtype recurring gene identification |
| 45 | `45_threshold_quantiles.py` | 95th-percentile CSD and correlation thresholds; per-edge empirical FWER p-values from stored permutation maxima |
| 46 | `46_topology_null.py` | Degree-preserving configuration-model null tests for modularity, assortativity and clustering |
| 47 | `47_annotated_network_assembly.py` | Earlier implementation of node annotation assembly; superseded by Script 36 |

---

## Utilities (`scripts/utils/`)

| Module | Description |
|---|---|
| `warehouse.py` | Provenance logging: `WarehouseRecord`, `append_warehouse`, `params_hash` |
| `qc_common.py` | Shared QC plot functions (boxpanels, violin panels, per-cell table saving) |
| `h5ad_compat.py` | AnnData compatibility shim: `read_h5ad_compat` |
| `network_utils.py` | Shared utilities for differential network scripts |
| `annotation_signature_utils.py` | Cell-type signature scoring utilities for the annotation stage |
