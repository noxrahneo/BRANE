#!/usr/bin/env python3
"""
Script 52: Add interactive direction filters (toggles) to final network HTMLs.

Adds a filter control panel with toggle switches that allow users to:
  - Show/hide UP regulated genes
  - Show/hide DOWN regulated genes  
  - Show/hide STABLE genes
  - Dynamically update network visibility

Injects CSS + JavaScript into HTML to create:
  1. Filter control panel (top-left corner, draggable)
  2. Toggle switches for each direction category
  3. Real-time node/edge visibility updates
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

# CSS for filter panel and toggle switches
FILTER_CSS = """
<style>
  #directionFilterPanel {
      position: absolute;
      top: 20px;
      left: 20px;
      background: white;
      border: 2px solid #333;
      border-radius: 8px;
      padding: 15px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 9999;
      font-family: Arial, sans-serif;
      font-size: 14px;
      min-width: 200px;
      cursor: move;
      user-select: none;
  }
  
  #directionFilterPanel h4 {
      margin: 0 0 12px 0;
      padding: 0;
      font-weight: bold;
      color: #333;
      border-bottom: 1px solid #ccc;
      padding-bottom: 8px;
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
      min-width: 60px;
      color: #333;
  }
  
  .toggle-count {
      font-size: 12px;
      color: #666;
      margin-left: auto;
      min-width: 40px;
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

# JavaScript for filter logic
FILTER_JS = """
<script>
  (function() {
      // Initialize filter state and counts
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
      
      var network = null;
      var allNodes = null;
      var allEdges = null;
      
      // Wait for vis.js network to be initialized
      function initializeFilters() {
          // Try to get the network object
          if (typeof network === 'undefined' || network === null) {
              // Try to find it in the global scope or the page
              try {
                  // The network should be created in the main page script
                  // We'll set up an observer pattern
                  setTimeout(initializeFilters, 100);
                  return;
              } catch (e) {
                  console.log('Waiting for network initialization...');
                  setTimeout(initializeFilters, 100);
                  return;
              }
          }
          
          // Count nodes by direction
          if (allNodes) {
              allNodes.forEach(function(node) {
                  var direction = node.direction || 'UNKNOWN';
                  if (direction === 'UP' || direction === 'DOWN' || direction === 'STABLE') {
                      directionCounts[direction]++;
                  }
                  directionCounts.total++;
              });
              updateStats();
          }
      }
      
      function getMetadata(nodeId) {
          try {
              var el = document.getElementById('cell-type-metadata');
              if (!el) return null;
              var raw = (el.textContent || el.innerText || '').trim();
              if (!raw) return null;
              var metadata = JSON.parse(raw);
              return metadata[nodeId] || null;
          } catch (e) {
              return null;
          }
      }
      
      function updateNetworkVisibility() {
          if (!allNodes) {
              console.warn('allNodes not initialized');
              return;
          }
          
          var visibleNodeIds = [];
          
          // Determine which nodes should be visible based on filter state
          allNodes.forEach(function(node) {
              var meta = getMetadata(node.id || node.label);
              var direction = (meta && meta.direction) ? meta.direction : 'UNKNOWN';
              
              // Show node if its direction is enabled in filterState
              if (filterState[direction]) {
                  visibleNodeIds.push(node.id);
              }
          });
          
          // Hide/show edges based on their node visibility
          var visibleEdgeIds = [];
          if (allEdges) {
              allEdges.forEach(function(edge) {
                  if (visibleNodeIds.indexOf(edge.from) !== -1 && visibleNodeIds.indexOf(edge.to) !== -1) {
                      visibleEdgeIds.push(edge.id);
                  }
              });
          }
          
          // Update vis.js datasets
          if (window.nodesDataset) {
              var updates = [];
              allNodes.forEach(function(node) {
                  var isVisible = visibleNodeIds.indexOf(node.id) !== -1;
                  if (node.hidden !== !isVisible) {
                      updates.push({id: node.id, hidden: !isVisible});
                  }
              });
              if (updates.length > 0) {
                  window.nodesDataset.update(updates);
              }
          }
          
          if (window.edgesDataset) {
              var edgeUpdates = [];
              allEdges.forEach(function(edge) {
                  var isVisible = visibleEdgeIds.indexOf(edge.id) !== -1;
                  if (edge.hidden !== !isVisible) {
                      edgeUpdates.push({id: edge.id, hidden: !isVisible});
                  }
              });
              if (edgeUpdates.length > 0) {
                  window.edgesDataset.update(edgeUpdates);
              }
          }
          
          // Refresh the network visualization
          if (window.networkInstance) {
              window.networkInstance.redraw();
          }
          
          updateStats();
      }
      
      function updateStats() {
          var visibleCounts = {UP: 0, DOWN: 0, STABLE: 0};
          
          if (allNodes) {
              allNodes.forEach(function(node) {
                  var meta = getMetadata(node.id || node.label);
                  var direction = (meta && meta.direction) ? meta.direction : 'UNKNOWN';
                  var isVisible = filterState[direction];
                  if (isVisible && (direction === 'UP' || direction === 'DOWN' || direction === 'STABLE')) {
                      visibleCounts[direction]++;
                  }
              });
          }
          
          // Update UI counts
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
      
      function toggleDirection(direction) {
          filterState[direction] = !filterState[direction];
          updateNetworkVisibility();
      }
      
      function resetFilters() {
          filterState.UP = true;
          filterState.DOWN = true;
          filterState.STABLE = true;
          
          // Update toggle switches
          document.getElementById('toggle-UP').checked = true;
          document.getElementById('toggle-DOWN').checked = true;
          document.getElementById('toggle-STABLE').checked = true;
          
          updateNetworkVisibility();
      }
      
      // Make functions globally accessible
      window.toggleDirection = toggleDirection;
      window.resetFilters = resetFilters;
      window.initializeDirectionFilters = function(nodesData, edgesData, networkInstance) {
          allNodes = nodesData;
          allEdges = edgesData;
          window.networkInstance = networkInstance;
          window.nodesDataset = nodesData;
          window.edgesDataset = edgesData;
          initializeFilters();
      };
      
      // Try to initialize when DOM is ready
      if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', function() {
              setTimeout(function() {
                  // Give the page time to set up network
                  initializeFilters();
              }, 500);
          });
      } else {
          setTimeout(function() {
              initializeFilters();
          }, 500);
      }
  })();
</script>
"""

# HTML for filter panel UI
FILTER_PANEL_HTML = """
<div id="directionFilterPanel">
    <h4>🔬 Direction Filter</h4>
    
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
"""


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def add_filters_to_html(html_file: Path) -> bool:
    """Add filter controls to an HTML file."""
    logging.info(f"Processing: {html_file.name}")
    
    try:
        html_text = html_file.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Failed to read HTML: {e}")
        return False
    
    # Add CSS just before </head>
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", FILTER_CSS + "\n</head>", 1)
    else:
        logging.warning("Could not find </head> tag")
    
    # Add filter panel HTML before <body> content (or before vis.js container)
    # Look for the network container div
    if 'id="network"' in html_text or 'id="mynetwork"' in html_text:
        # Find the container and insert panel before it
        container_pattern = '<div id="'
        idx = html_text.find(container_pattern)
        if idx != -1:
            # Find the opening <div for network
            start_of_line = html_text.rfind('\n', 0, idx) + 1
            html_text = (
                html_text[:start_of_line] +
                FILTER_PANEL_HTML + "\n" +
                html_text[start_of_line:]
            )
    else:
        logging.warning("Could not find network container")
    
    # Add filter JavaScript before </body>
    if "</body>" in html_text:
        html_text = html_text.replace("</body>", FILTER_JS + "\n</body>", 1)
    else:
        html_text += "\n" + FILTER_JS
    
    try:
        html_file.write_text(html_text, encoding="utf-8")
        logging.info(f"✓ Added filter controls to {html_file.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to write HTML: {e}")
        return False


def main() -> None:
    configure_logging()
    
    logging.info("Script 52: Add interactive direction filters to networks")
    logging.info(f"Target folder: {FINAL_NETWORKS_DIR}")
    
    # Find all HTML files
    html_files = sorted(FINAL_NETWORKS_DIR.glob("*/*_network_interactive.html"))
    if not html_files:
        logging.error(f"No HTML files found in {FINAL_NETWORKS_DIR}")
        return
    
    logging.info("=" * 80)
    logging.info(f"Found {len(html_files)} network HTML files")
    logging.info("=" * 80)
    
    success_count = 0
    for html_file in html_files:
        if add_filters_to_html(html_file):
            success_count += 1
    
    logging.info("\n" + "=" * 80)
    logging.info(f"COMPLETE: {success_count}/{len(html_files)} networks updated with filters")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
