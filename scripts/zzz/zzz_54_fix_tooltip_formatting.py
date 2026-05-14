#!/usr/bin/env python3
"""
Script 54: Fix tooltip newline formatting in HTMLs

The issue: _buildEnhancedNodeTitle is using '\\\\n' (4 backslashes + n)
which renders as literal text instead of newlines.

Solution: Replace with proper '\n' escape sequences.
"""

from __future__ import annotations

import logging
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_NETWORKS_DIR = (
    REPO_ROOT
    / "results"
    / "stages"
    / "09_differential_restructured"
    / "12_tagging"
    / "final_networks_with_lfc"
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def fix_tooltip_formatting(html_file: Path) -> bool:
    """Fix newline escaping in tooltip building functions."""
    logging.info(f"Fixing tooltips: {html_file.name}")
    
    try:
        html_text = html_file.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read HTML: {e}")
        return False
    
    # Replace incorrect escape sequences
    original_count = html_text.count("'\\\\\\\\n'")
    
    # Fix lines.join('\\\\n') -> lines.join('\n')
    html_text = html_text.replace("lines.join('\\\\\\\\n')", "lines.join('\\n')")
    
    # Fix base + '\\\\n' + lines -> base + '\n' + lines
    html_text = html_text.replace("base + '\\\\\\\\n'", "base + '\\n'")
    
    # Also fix any remaining instances with proper context
    html_text = html_text.replace("return base + '\\n' + lines.join('\\n');", 
                                  "return base + '\\n' + lines.join('\\n');")
    
    if original_count > 0:
        logging.info(f"  Fixed {original_count} newline sequences")
    
    try:
        html_file.write_text(html_text, encoding="utf-8")
        logging.info(f"✓ Tooltip formatting fixed: {html_file.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to write HTML: {e}")
        return False


def main() -> None:
    configure_logging()
    
    logging.info("Script 54: Fix tooltip newline formatting")
    logging.info("=" * 80)
    
    html_files = sorted(FINAL_NETWORKS_DIR.glob("*/*_network_interactive.html"))
    if not html_files:
        logging.error(f"No HTML files found in {FINAL_NETWORKS_DIR}")
        return
    
    success_count = 0
    for html_file in html_files:
        if fix_tooltip_formatting(html_file):
            success_count += 1
    
    logging.info("=" * 80)
    logging.info(f"COMPLETE: {success_count}/{len(html_files)} networks fixed")


if __name__ == "__main__":
    main()
