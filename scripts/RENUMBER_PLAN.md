# Script Renumbering Plan

**Status**: EXECUTED — renaming complete, registry updated  
**Reference**: `SCRIPT_NAMES_BEFORE_RENUMBER.csv` has the full pre-change snapshot

---

## Summary

- Active pipeline scripts: **55** (numbered 00–54, no gaps, no sub-letters)
- Scripts moving to `zzz/`: **17** (their result folders are already archived)
- Non-pipeline files unchanged: `README.md`, `network_interactive_template.html`, `00_setup_environment.py`

---

## Proposed mapping: ACTIVE pipeline scripts

| New # | New name | Current name | Stage | Notes |
|---|---|---|---|---|
| 00 | `00_setup_environment.py` | `00_setup_environment.py` | pre-pipeline | unchanged — dev tool |
| 01 | `01_create_bigboss.py` | `01_create_bigboss.py` | 00_data_ingestion | unchanged |
| 02 | `02_chopper.py` | `02_chopper.py` | 00_data_ingestion | unchanged |
| 03 | `03_pre_qc_fromr.py` | `03_pre_qc_fromr.py` | 01_qc | unchanged |
| 04 | `04_filtering_cells_fromr.py` | `04_filtering_cells_fromr.py` | 02_preprocess | unchanged |
| 05 | `05_post_qc_fromr.py` | `05_post_qc_fromr.py` | 01_qc | unchanged |
| 06 | `06_per_sample_preprocess_fromr.py` | `06_per_sample_preprocess_fromr.py` | 02_preprocess | unchanged |
| 07 | `07_validate_preprocess_fromr.py` | `07_validate_preprocess_fromr.py` | 02_preprocess | unchanged |
| 08 | `08_integrate_per_condition_fromr.py` | `08_integrate_per_condition_fromr.py` | 03_integration | unchanged |
| 09 | `09_hvg_diagnostics.py` | `08b_hvg_diagnostics.py` | 03_integration | was 08b |
| 10 | `10_plot_integrated_results_fromr.py` | `09_plot_integrated_results_fromr.py` | 03_integration | was 09 |
| 11 | `11_plot_integrated_panels_fromr.py` | `10_plot_integrated_panels_fromr.py` | 03_integration | was 10 |
| 12 | `12_annotate_clusters_fromr.py` | `11_annotate_clusters_fromr.py` | 04_annotation | was 11 |
| 13 | `13_marker_heatmaps.py` | `14_marker_heatmaps.py` | 04_annotation | was 14, gap from archived 12+13 |
| 14 | `14_composition_tests.py` | `15_composition_tests.py` | 05_composition | was 15 |
| 15 | `15_pseudobulk.py` | `18_pseudobulk.py` | 07_network/02 | was 18, gap from archived 16+17+19 |
| 16 | `16_deg_ttest_logcpm.py` | `19b_deg_ttest_logcpm.py` | 07_network/07_deg | was 19b |
| 17 | `17_deg_ttest_plots.py` | `19c_deg_ttest_plots.py` | 07_network/07_deg | was 19c |
| 18 | `18_pseudobulk_sanity_check.py` | `20_pseudobulk_sanity_check.py` | 07_network/03 | was 20 |
| 19 | `19_prepare_pre_correlation_pack.py` | `21_prepare_pre_correlation_pack.py` | 07_network/04 | was 21 |
| 20 | `20_prepare_condition_logcpm_h5ad.py` | `22_prepare_condition_logcpm_h5ad.py` | 07_network/04 | was 22 |
| 21 | `21_compute_correlations.py` | `23_compute_correlations.py` | 07_network/05 | was 23 |
| 22 | `22_correlation_qc_plots.py` | `24_correlation_qc_plots.py` | 07_network/05 | was 24 |
| 23 | `23_network_power_tom_prep.py` | `25_network_power_tom_prep.py` | 07_network/08 | was 25 |
| 24 | `24_tom_module_detection.py` | `25b_tom_module_detection.py` | 07_network/09 | was 25b |
| 25 | `25_wgcna_style_plots.py` | `25c_wgcna_style_plots.py` | 07_network/09 | was 25c |
| 26 | `26_networkx_visualization.py` | `26_networkx_visualization.py` | 07_network/14 | unchanged |
| 27 | `27_network_export_indexes.py` | `29_network_export_indexes.py` | 07_network/14 | was 29, gap from archived 27+28 |
| 28 | `28_stage09_shared_upstream.py` | `40_stage09_shared_upstream.py` | 09/01 | was 40, large gap from archived 30-38 |
| 29 | `29_stage09_scalefree_branch.py` | `41_stage09_scalefree_branch.py` | 09/02 | was 41 |
| 30 | `30_stage09_permutation_thresholds.py` | `42_stage09_permutation_thresholds.py` | 09/03 | was 42 |
| 31 | `31_stage09_permutation_network_builder.py` | `43_stage09_permutation_network_builder.py` | 09/04 | was 43 |
| 32 | `32_stage09_network_visualization.py` | `45_stage09_network_visualization.py` | 09/06 | was 45, 44 → zzz |
| 33 | `33_stage09_persistent_overlap_visualization.py` | `46_stage09_persistent_overlap_visualization.py` | 09/08 | was 46 |
| 34 | `34_hgnc_normalisation.py` | `47_hgnc_normalisation.py` | 09/09 | was 47 |
| 35 | `35_ncbi_gene_fetch.py` | `48_ncbi_gene_fetch.py` | 09/10 | was 48 |
| 36 | `36_bootstrap_inputs.py` | `49a_bootstrap_inputs.py` | 09/11 prep | was 49a — runs before 49 |
| 37 | `37_cancer_gene_annotation.py` | `49_cancer_gene_annotation.py` | 09/11 | was 49 |
| 38 | `38_extended_zscore_tagging.py` | `49e_extended_zscore_tagging.py` | 09/12 | was 49e |
| 39 | `39_cell_type_master_annotation_and_viz.py` | `49b_cell_type_master_annotation_and_viz.py` | 09/12 | was 49b |
| 40 | `40_regenerate_interactive_networks.py` | `49c_regenerate_interactive_networks.py` | 09/08+12 | was 49c |
| 41 | `41_update_persistent_html_metadata.py` | `49d_update_persistent_html_metadata.py` | 09/08+12 | was 49d |
| 42 | `42_add_condition_lfc_to_tagged.py` | `50_add_condition_lfc_to_tagged.py` | 09/12 | was 50 |
| 43 | `43_generate_final_networks_with_lfc.py` | `51_generate_final_networks_with_lfc.py` | 09/12 | was 51 |
| 44 | `44_prepare_visualization_assets.py` | `58_prepare_visualization_assets.py` | 09/12 | was 58, gap from archived 52-57 |
| 45 | `45_regenerate_stable_pngs.py` | `59_regenerate_stable_pngs_from_html_style.py` | 09/12 | was 59 |
| 46 | `46_go_pathway_enrichment.py` | `60c_go_pathway_enrichment_clean.py` | 09/13 | was 60c — canonical clean version |
| 47 | `47_tag_permutation.py` | `61_tag_permutation.py` | 09/14 | was 61 |
| 48 | `48_survival_analysis.py` | `62_survival_analysis.py` | 09/15 | was 62 |
| 49 | `49_drug_target_candidates.py` | `63_drug_target_candidates.py` | 09/16 | was 63 |
| 50 | `50_gm1_connection.py` | `63b_gm1_connection.py` | 09/16 | was 63b |
| 51 | `51_visualize_drug_candidates.py` | `64_visualize_drug_candidates.py` | 09/16 | was 64 |
| 52 | `52_drug_candidates_per_pair.py` | `65_drug_candidates_per_pair.py` | 09/16 | was 65 |
| 53 | `53_drug_survival_km.py` | `66_drug_survival_km.py` | 09/16 | was 66 |
| 54 | `54_build_thesis_pack.py` | `34_build_thesis_pack.py` | 90_reports | was 34 — moved to END (reads all stages) |

---

## Proposed: MOVE TO zzz/ (their result folders are already archived)

| Current name | Result folder | Reason |
|---|---|---|
| `16_kegg_enrichment.py` | `zzz_06_kegg/` | archived branch |
| `17_fisher_ora.py` | `zzz_06_fisher_ora/` | archived branch |
| `19_pseudobulk_decoupler_downstream.py` | `zzz_06_downstream_decoupler/` | archived branch |
| `24b_filter_correlations_by_de.py` | `zzz_pearson_de_filtered/` | archived branch |
| `27_combined_network_power_tom_prep.py` | `zzz_combined/` | archived branch |
| `28_combined_networkx_visualization.py` | `zzz_combined/` | archived branch |
| `30_differential_coexpression.py` | `zzz_10/` | archived branch |
| `31_mutual_rank_export.py` | `zzz_13/` | archived branch |
| `32_compare_network_sensitivity.py` | `zzz_16/` | archived branch |
| `33_build_sensitivity_thesis_figures.py` | `zzz_16/` | archived branch |
| `35_csd_scores_homogeneity.py` | `zzz_11/` | archived branch |
| `36_csd_permutation_threshold.py` | `zzz_15/` | archived branch |
| `37_differential_scalefree_csd.py` | `zzz_12/` | archived branch |
| `38_differential_network_visualization.py` | `zzz_12/` | archived branch |
| `44_stage09_compare_branches.py` | `zzz_05_comparison/` | archived branch |
| `60_go_pathway_enrichment.py` | `13_enrichment/` | superseded by 60c |
| `60b_go_pathway_enrichment_revised.py` | `13_enrichment/` | superseded by 60c |

---

## Questions for your review

1. **`34_build_thesis_pack.py` → moved to position 54 (end)**: it reads from all previous stages so it runs last. Does this make sense?
2. **17 scripts moving to `zzz/`**: do you agree all their result folders are archived and these can go to `zzz/`? Or do you want to keep any in the main sequence?
3. **`49a_bootstrap_inputs.py` → position 36** (before `49_cancer_gene_annotation` → position 37): does this execution order match how you ran them?
4. **`60c_go_pathway_enrichment_clean.py` becomes canonical `46_go_pathway_enrichment.py`**: do you confirm 60c is the final version you used for results?
5. **`00_setup_environment.py`**: keep at 00 (pre-pipeline dev tool) or move to `utils/`?
