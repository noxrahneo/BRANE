# Scripts Overview

This repository now uses a single flat `scripts/` folder.

## Core Ordered Pipeline

From project setup to final reports:

1. `00_setup_environment.py`
2. `01_create_bigboss.py`
3. `02_chopper.py`
4. `03_pre_qc_fromr.py`
5. `04_filtering_cells_fromr.py`
6. `05_post_qc_fromr.py`
7. `06_per_sample_preprocess_fromr.py`
8. `07_validate_preprocess_fromr.py`
9. `08_integrate_per_condition_fromr.py`
10. `09_plot_integrated_results_fromr.py`
11. `10_plot_integrated_panels_fromr.py`
12. `11_annotate_clusters_fromr.py`
13. `12_autocurate_annotation.py`
14. `13_freeze_annotation.py`
15. `14_marker_heatmaps.py`
16. `15_composition_tests.py`
17. `16_kegg_enrichment.py`
18. `17_fisher_ora.py`
19. `18_pseudobulk.py`
20. `19_pseudobulk_decoupler_downstream.py`
21. `20_pseudobulk_sanity_check.py`
22. `21_prepare_pre_correlation_pack.py`
23. `22_prepare_condition_logcpm_h5ad.py`
24. `23_compute_correlations.py`
25. `24_correlation_qc_plots.py`
26. `25_network_power_tom_prep.py`
27. `26_networkx_visualization.py`
28. `27_combined_network_power_tom_prep.py`
29. `28_combined_networkx_visualization.py`
30. `29_network_export_indexes.py`
31. `30_differential_coexpression.py`
32. `31_mutual_rank_export.py`
33. `32_compare_network_sensitivity.py`
34. `33_build_sensitivity_thesis_figures.py`
35. `34_build_thesis_pack.py`
36. `35_csd_scores_homogeneity.py`
37. `36_csd_permutation_threshold.py`
38. `37_differential_scalefree_csd.py`
39. `38_differential_network_visualization.py`

## Helper Modules

These are imported by pipeline scripts and should stay in `scripts/`:

- `annotation_signature_utils.py`
- `h5ad_compat.py`
- `qc_common.py`
- `warehouse.py`
- `network_interactive_template.html`

## Minimal Example Run Block

```bash
python3 scripts/18_pseudobulk.py --condition all --threshold-mode auto
python3 scripts/19_pseudobulk_decoupler_downstream.py
python3 scripts/19b_deg_ttest_logcpm.py --control-condition Normal --include-brca1-pair
python3 scripts/19c_deg_ttest_plots.py
python3 scripts/20_pseudobulk_sanity_check.py
python3 scripts/21_prepare_pre_correlation_pack.py
python3 scripts/22_prepare_condition_logcpm_h5ad.py
python3 scripts/23_compute_correlations.py --condition all
python3 scripts/24b_filter_correlations_by_de.py --condition all
python3 scripts/24_correlation_qc_plots.py --condition all
python3 scripts/25_network_power_tom_prep.py --condition all
python3 scripts/25b_tom_module_detection.py --condition all
python3 scripts/08b_hvg_diagnostics.py --condition all
python3 scripts/26_networkx_visualization.py --condition all
python3 scripts/27_combined_network_power_tom_prep.py --condition all
python3 scripts/28_combined_networkx_visualization.py --condition all
python3 scripts/35_csd_scores_homogeneity.py --control Normal --min-abs-corr 0.3 --max-edges 200000
python3 scripts/36_csd_permutation_threshold.py --pair ER_tumor:Normal --n-permutations 50 --quantile 0.99
python3 scripts/37_differential_scalefree_csd.py --use-degs --network-type signed
python3 scripts/38_differential_network_visualization.py --pair all --max-edges 1200
```
