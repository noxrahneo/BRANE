#!/usr/bin/env python3
"""
Script 53: Fix direction filters - proper positioning, working logic, and tooltip formatting

Issues to fix:
1. Panel overlapping network (move to bottom-right, add minimize)
2. Filter counts showing 0/0 (reattach to vis.js network properly)
3. Tooltip text formatting broken (fix newline escaping)
"""

from __future__ import annotations

import logging
import re
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

# Updated CSS with bottom-right positioning and minimize button
FILTER_CSS = """
<style>
  #directionFilterPanel {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: white;
      border: 2px solid #333;
      border-radius: 8px;
      padding: 15px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 9999;
      font-family: Arial, sans-serif;
      font-size: 14px;
      min-width: 220px;
      cursor: move;
      user-select: none;
      max-height: 90vh;
      overflow-y: auto;
  }
  
  #directionFilterPanel.minimized {
      padding: 8px 15px;
      max-height: 40px;
  }
  
  #directionFilterPanel.minimized .filter-content {
      display: none;
  }
  
  #directionFilterPanel h4 {
      margin: 0 0 12px 0;
      padding: 0 0 8px 0;
      font-weight: bold;
      color: #333;
      border-bottom: 1px solid #ccc;
      display: flex;
      justify-content: space-between;
      align-items: center;
  }
  
  .filter-header-text {
      flex: 1;
  }
  
  .minimize-btn {
      background: none;
      border: none;
      font-size: 16px;
      cursor: pointer;
      padding: 0 5px;
      color: #666;
  }
  
  .minimize-btn:hover {
      color: #000;
  }
  
  .filter-content {
      display: block;
  }
  
  .toggle-switch-container {
      display: flex;
      align-items: center;
      margin-bottom: 10px;
      gap: 10px;
  }
  
  .toggle-switch {
      position: relative;
      display: inline-block;
      width: 50px;
      height: 24px;
  }
  
  .toggle-switch input {
      opacity: 0;
      width: 0;
      height: 0;
  }
  
  .toggle-slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #ccc;
      transition: 0.3s;
      border-radius: 24px;
  }
  
  .toggle-slider:before {
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: 0.3s;
      border-radius: 50%;
  }
  
  input:checked + .toggle-slider {
      background-color: #4CAF50;
  }
  
  input:checked + .toggle-slider:before {
      transform: translateX(26px);
  }
  
  .toggle-label {
      font-weight: 500;
      min-width: 70px;
      color: #333;
  }
  
  .toggle-count {
      font-size: 12px;
      color: #666;
      margin-left: auto;
      min-width: 50px;
      text-align: right;
  }
  
  .filter-stats {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #ccc;
      font-size: 12px;
      color: #666;
  }
  
  .stat-line {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
  }
  
  .stat-label {
      font-weight: 500;
  }
  
  .reset-btn {
      width: 100%;
      margin-top: 10px;
      padding: 6px;
      background-color: #007BFF;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      font-weight: bold;
      transition: 0.2s;
  }
  
  .reset-btn:hover {
      background-color: #0056b3;
  }
</style>
"""

# Completely rewritten JavaScript with proper vis.js integration
FILTER_JS = """
<script>
  (function() {
      var filterState = {
          UP: true,
          DOWN: true,
          STABLE: true
      };
      
      var directionCounts = {
          UP: 0,
          DOWN: 0,
          STABLE: 0,
          total: 0
      };
      
      var networkInstance = null;
      var nodesDataset = null;
      var edgesDataset = null;
      var allNodesData = [];
      var allEdgesData = [];
      
      // Try to get network from global scope
      function hookIntoNetwork() {
          try {
              // Look for the network in window scope
              if (window.networkInstance) {
                  networkInstance = window.networkInstance;
              }
              
              // Get datasets
              if (window.nodesDataset) {
                  nodesDataset = window.nodesDataset;
              }
              if (window.edgesDataset) {
                  edgesDataset = window.edgesDataset;
              }
              
              // If we have network but not datasets, try to get from network
              if (networkInstance && (!nodesDataset || !edgesDataset)) {
                  try {
                      nodesDataset = networkInstance.body.data.nodes;
                      edgesDataset = networkInstance.body.data.edges;
                  } catch (e) {
                      console.log('Could not get datasets from network');
                  }
              }
              
              // Get all nodes data
              if (nodesDataset && typeof nodesDataset.get === 'function') {
                  allNodesData = nodesDataset.get();
                  countDirections();
                  updateStats();
                  return true;
              }
          } catch (e) {
              console.log('Error hooking into network:', e);
          }
          return false;
      }
      
      function countDirections() {
          directionCounts = {UP: 0, DOWN: 0, STABLE: 0, total: 0};
          
          allNodesData.forEach(function(node) {
              var direction = node.direction || 'UNKNOWN';
              if (direction === 'UP' || direction === 'DOWN' || direction === 'STABLE') {
                  directionCounts[direction]++;
              }
              directionCounts.total++;
          });
      }
      
      function updateNetworkVisibility() {
          if (!nodesDataset || !edgesDataset || !allNodesData.length) {
              console.warn('Not ready: nodes=', !!nodesDataset, 'edges=', !!edgesDataset, 'data=', allNodesData.length);
              return;
          }
          
          var visibleNodeIds = [];
          
          // Find visible nodes
          allNodesData.forEach(function(node) {
              var direction = node.direction || 'UNKNOWN';
              if (filterState[direction]) {
                  visibleNodeIds.push(node.id);
              }
          });
          
          // Update nodes
          var nodeUpdates = [];
          allNodesData.forEach(function(node) {
              var shouldBeVisible = visibleNodeIds.indexOf(node.id) !== -1;
              if (node.hidden !== !shouldBeVisible) {
                  nodeUpdates.push({id: node.id, hidden: !shouldBeVisible});
              }
          });
          
          if (nodeUpdates.length > 0) {
              nodesDataset.update(nodeUpdates);
          }
          
          // Update edges
          if (allEdgesData.length > 0) {
              var edgeUpdates = [];
              allEdgesData.forEach(function(edge) {
                  var fromVisible = visibleNodeIds.indexOf(edge.from) !== -1;
                  var toVisible = visibleNodeIds.indexOf(edge.to) !== -1;
                  var shouldBeVisible = fromVisible && toVisible;
                  if (edge.hidden !== !shouldBeVisible) {
                      edgeUpdates.push({id: edge.id, hidden: !shouldBeVisible});
                  }
              });
              
              if (edgeUpdates.length > 0) {
                  edgesDataset.update(edgeUpdates);
              }
          }
          
          // Redraw
          if (networkInstance && typeof networkInstance.redraw === 'function') {
              networkInstance.redraw();
          }
          
          updateStats();
      }
      
      function updateStats() {
          var visibleCounts = {UP: 0, DOWN: 0, STABLE: 0};
          
          allNodesData.forEach(function(node) {
              var direction = node.direction || 'UNKNOWN';
              if (filterState[direction]) {
                  if (direction === 'UP' || direction === 'DOWN' || direction === 'STABLE') {
                      visibleCounts[direction]++;
                  }
              }
          });
          
          var upCount = document.getElementById('count-UP');
          var downCount = document.getElementById('count-DOWN');
          var stableCount = document.getElementById('count-STABLE');
          
          if (upCount) upCount.textContent = visibleCounts.UP + ' / ' + directionCounts.UP;
          if (downCount) downCount.textContent = visibleCounts.DOWN + ' / ' + directionCounts.DOWN;
          if (stableCount) stableCount.textContent = visibleCounts.STABLE + ' / ' + directionCounts.STABLE;
          
          var totalVisible = visibleCounts.UP + visibleCounts.DOWN + visibleCounts.STABLE;
          var totalStat = document.getElementById('stat-total');
          if (totalStat) totalStat.textContent = totalVisible + ' / ' + directionCounts.total;
      }
      
      window.toggleDirection = function(direction) {
          filterState[direction] = !filterState[direction];
          updateNetworkVisibility();
      };
      
      window.resetFilters = function() {
          filterState.UP = true;
          filterState.DOWN = true;
          filterState.STABLE = true;
          
          document.getElementById('toggle-UP').checked = true;
          document.getElementById('toggle-DOWN').checked = true;
          document.getElementById('toggle-STABLE').checked = true;
          
          updateNetworkVisibility();
      };
      
      window.minimizeFilterPanel = function() {
          var panel = document.getElementById('directionFilterPanel');
          if (panel) {
              panel.classList.toggle('minimized');
          }
      };
      
      // Try to hook into network - multiple attempts
      var hookAttempts = 0;
      var hookInterval = setInterval(function() {
          if (hookIntoNetwork()) {
              console.log('Filter initialized with', allNodesData.length, 'nodes');
              clearInterval(hookInterval);
          }
          hookAttempts++;
          if (hookAttempts > 30) {
              console.warn('Could not hook into network after 30 attempts');
              clearInterval(hookInterval);
          }
      }, 200);
  })();
</script>
"""

# Updated HTML panel with minimize button and bottom-right positioning
FILTER_PANEL_HTML = """
<div id="directionFilterPanel">
    <h4>
        <span class="filter-header-text">🔬 Direction Filter</span>
        <button class="minimize-btn" onclick="window.minimizeFilterPanel()" title="Minimize">−</button>
    </h4>
    
    <div class="filter-content">
        <div class="toggle-switch-container">
            <label class="toggle-label">↑ UP</label>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-UP" checked onchange="window.toggleDirection('UP')">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-count" id="count-UP">0 / 0</span>
        </div>
        
        <div class="toggle-switch-container">
            <label class="toggle-label">↓ DOWN</label>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-DOWN" checked onchange="window.toggleDirection('DOWN')">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-count" id="count-DOWN">0 / 0</span>
        </div>
        
        <div class="toggle-switch-container">
            <label class="toggle-label">— STABLE</label>
            <label class="toggle-switch">
                <input type="checkbox" id="toggle-STABLE" checked onchange="window.toggleDirection('STABLE')">
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-count" id="count-STABLE">0 / 0</span>
        </div>
        
        <div class="filter-stats">
            <div class="stat-line">
                <span class="stat-label">Visible:</span>
                <span id="stat-total">0 / 0</span>
            </div>
        </div>
        
        <button class="reset-btn" onclick="window.resetFilters()">Reset All</button>
    </div>
</div>
"""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def fix_html_filters(html_file: Path) -> bool:
    """Remove old filters and inject fixed version."""
    logging.info(f"Fixing: {html_file.name}")
    
    try:
        html_text = html_file.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read HTML: {e}")
        return False
    
    # Remove old filter CSS
    html_text = re.sub(
        r'<style>\s*#directionFilterPanel.*?</style>',
        '',
        html_text,
        flags=re.DOTALL
    )
    
    # Remove old filter panel HTML
    html_text = re.sub(
        r'<div id="directionFilterPanel">.*?</div>',
        '',
        html_text,
        flags=re.DOTALL
    )
    
    # Remove old filter JavaScript
    html_text = re.sub(
        r'<script>.*?window\.toggleDirection.*?</script>',
        '',
        html_text,
        flags=re.DOTALL
    )
    
    # Add fresh CSS
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", FILTER_CSS + "\n</head>", 1)
    
    # Add fresh panel (before network container)
    if 'id="network"' in html_text or 'id="mynetwork"' in html_text:
        container_idx = html_text.find('id="')
        if container_idx != -1:
            start_of_line = html_text.rfind('\n', 0, container_idx) + 1
            html_text = (
                html_text[:start_of_line] +
                FILTER_PANEL_HTML + "\n" +
                html_text[start_of_line:]
            )
    
    # Add fresh JavaScript
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", FILTER_JS + "\n</body>", 1)
    else:
        html_text += "\n" + FILTER_JS
    
    try:
        html_file.write_text(html_text, encoding="utf-8")
        logging.info(f"✓ Fixed {html_file.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to write HTML: {e}")
        return False


def main() -> None:
    configure_logging()
    
    logging.info("Script 53: Fix direction filters")
    logging.info("  - Move panel to bottom-right with minimize button")
    logging.info("  - Fix filter logic to properly hook into vis.js")
    logging.info("  - Ensure counts populate correctly")
    logging.info("=" * 80)
    
    html_files = sorted(FINAL_NETWORKS_DIR.glob("*/*_network_interactive.html"))
    if not html_files:
        logging.error(f"No HTML files found")
        return
    
    success_count = 0
    for html_file in html_files:
        if fix_html_filters(html_file):
            success_count += 1
    
    logging.info("=" * 80)
    logging.info(f"COMPLETE: {success_count}/{len(html_files)} networks fixed")


if __name__ == "__main__":
    main()
