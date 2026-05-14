#!/usr/bin/env python3
"""
Fix the corrupted _summary_row function.
This script replaces the malformed function with a clean version.
"""

import re

with open("scripts/script_60b_go_pathway_enrichment_revised.py", "r") as f:
    content = f.read()

# Find and replace the broken _summary_row function
broken_pattern = r'def _summary_row\(result: AnalysisResult\).*?(?=def generate_summary)'

fixed_function = '''def _summary_row(result: AnalysisResult) -> dict[str, Any]:
    if result.status != "completed" or result.full_df.empty:
        return {
            "pair_name": result.pair_name,
            "analysis_type": result.analysis_type,
            "status": result.status,
            "skip_reason": result.skip_reason,
            "n_genes_queried": result.n_genes_queried,
            "n_background_genes": result.n_background_genes,
            "background_source": result.background_source,
            "threshold_fdr": result.threshold_used,
            "is_sensitivity_analysis": result.is_sensitivity_analysis,
            "n_significant_terms": 0,
            "n_GO_BP": 0,
            "n_GO_MF": 0,
            "n_GO_CC": 0,
            "n_KEGG": 0,
            "n_REAC": 0,
            "top_term_name": "",
            "top_term_source": "",
            "top_term_adjusted_p": None,
        }

    sig_df = result.full_df[result.full_df["significant"] == True].copy()

    if sig_df.empty:
        return {
            "pair_name": result.pair_name,
            "analysis_type": result.analysis_type,
            "status": "completed",
            "skip_reason": "",
            "n_genes_queried": result.n_genes_queried,
            "n_background_genes": result.n_background_genes,
            "background_source": result.background_source,
            "threshold_fdr": result.threshold_used,
            "is_sensitivity_analysis": result.is_sensitivity_analysis,
            "n_significant_terms": 0,
            "n_GO_BP": 0,
            "n_GO_MF": 0,
            "n_GO_CC": 0,
            "n_KEGG": 0,
            "n_REAC": 0,
            "top_term_name": "",
            "top_term_source": "",
            "top_term_adjusted_p": None,
        }

    top = sig_df.nsmallest(1, "adjusted_p").iloc[0]
    return {
        "pair_name": result.pair_name,
        "analysis_type": result.analysis_type,
        "status": "completed",
        "skip_reason": "",
        "n_genes_queried": result.n_genes_queried,
        "n_background_genes": result.n_background_genes,
        "background_source": result.background_source,
        "threshold_fdr": result.threshold_used,
        "is_sensitivity_analysis": result.is_sensitivity_analysis,
        "n_significant_terms": int(len(sig_df)),
        "n_GO_BP": int((sig_df["source"] == "GO:BP").sum()),
        "n_GO_MF": int((sig_df["source"] == "GO:MF").sum()),
        "n_GO_CC": int((sig_df["source"] == "GO:CC").sum()),
        "n_KEGG": int((sig_df["source"] == "KEGG").sum()),
        "n_REAC": int((sig_df["source"] == "REAC").sum()),
        "top_term_name": str(top["term_name"]),
        "top_term_source": str(top["source"]),
        "top_term_adjusted_p": float(top["adjusted_p"]),
    }


'''

content = re.sub(broken_pattern, fixed_function, content, flags=re.DOTALL)

with open("scripts/script_60b_go_pathway_enrichment_revised.py", "w") as f:
    f.write(content)

print("✓ Fixed _summary_row function")
