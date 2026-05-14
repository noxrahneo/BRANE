#!/usr/bin/env bash
# Sequential runner for scripts 47–56 with per-script and total timing.
# Run AFTER scripts 44, 45, 46 are complete.
set -e

PYTHON=/triumvirate/home/alexarol/.conda/envs/breast_cancer_scrnaseq/bin/python
SCRIPTS=/triumvirate/home/alexarol/breast_cancer_analysis/BRANE/scripts
LOGS=/triumvirate/home/alexarol/breast_cancer_analysis/BRANE/logs
TIMING_LOG=$LOGS/pipeline_47_56_timing.log

cd /triumvirate/home/alexarol/breast_cancer_analysis/BRANE

PIPELINE_START=$(date +%s)
echo "Pipeline start: $(date)" | tee "$TIMING_LOG"

run_script() {
    local num=$1; local name=$2; shift 2
    echo ""
    echo "================================================================"
    echo "  [$num] $name  —  started $(date +%H:%M:%S)"
    echo "================================================================"
    local t0=$(date +%s)
    $PYTHON "$SCRIPTS/${num}_${name}.py" "$@" 2>&1 | tee "$LOGS/run_${num}_${name}.log"
    local t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local mm=$(( elapsed / 60 ))
    local ss=$(( elapsed % 60 ))
    echo "  ✓ $num done in ${mm}m ${ss}s"
    echo "$num $name: ${mm}m ${ss}s  ($(date +%H:%M:%S))" | tee -a "$TIMING_LOG"
}

run_script 47 prepare_visualization_assets
run_script 48 go_pathway_enrichment
run_script 49 tag_permutation
run_script 50 survival_analysis
run_script 51 drug_target_candidates
run_script 52 visualize_drug_candidates
run_script 53 drug_candidates_per_pair
run_script 54 drug_survival_km
run_script 55 threshold_quantiles
# 56: skip single-condition networks (removed from pipeline), differential only
run_script 56 topology_null --family differential

PIPELINE_END=$(date +%s)
TOTAL=$(( PIPELINE_END - PIPELINE_START ))
TOTAL_MM=$(( TOTAL / 60 ))
TOTAL_SS=$(( TOTAL % 60 ))

echo ""
echo "================================================================"
echo "  ALL DONE — pipeline 47–56 complete"
echo "  Total time: ${TOTAL_MM}m ${TOTAL_SS}s"
echo "================================================================"
echo "" | tee -a "$TIMING_LOG"
echo "TOTAL: ${TOTAL_MM}m ${TOTAL_SS}s" | tee -a "$TIMING_LOG"
echo "Pipeline end: $(date)" | tee -a "$TIMING_LOG"
