#!/usr/bin/env python3
"""
Script 55: Fix direction filters - bottom-left positioning and proper data access

Issues to fix:
1. Panel should be bottom-LEFT (not right)
2. Counts showing 0/0 (nodes data not being accessed)
3. Need to embed direction data DIRECTLY in node objects, not just metadata Map
"""

from __future__ import annotations

import json
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


FILTER_CSS = """
<style>
  #directionFilterPanel {
      position: fixed;
      bottom: 20px;
      left: 20px;
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
      
      function initializeFilter() {
          console.log('[Filter] Initializing...');
          
          // Try to find the network and datasets
          if (typeof network !== 'undefined') {
              networkInstance = network;
              console.log('[Filter] Found global network object');
          }
          if (typeof nodesDataset !== 'undefined' || typeof nodes !== 'undefined') {
              nodesDataset = (typeof nodesDataset !== 'undefined') ? nodesDataset : nodes;
              console.log('[Filter] Found global nodesDataset');
          }
          if (typeof edgesDataset !== 'undefined' || typeof edges !== 'undefined') {
              edgesDataset = (typeof edgesDataset !== 'undefined') ? edgesDataset : edges;
              console.log('[Filter] Found global edgesDataset');
          }
          
          // If we have datasets, try to get all nodes
          if (nodesDataset && typeof nodesDataset.get === 'function') {
              try {
                  allNodesData = nodesDataset.get();
                  console.log('[Filter] Got', allNodesData.length, 'nodes from dataset');
                  
                  // Log first node to see if direction is present
                  if (allNodesData.length > 0) {
                      console.log('[Filter] First node:', JSON.stringify(allNodesData[0], null, 2).substring(0, 200));
                  }
                  
                  countDirections();
                  updateStats();
              } catch (e) {
                  console.error('[Filter] Error getting nodes:', e);
              }
          }
          
          if (edgesDataset && typeof edgesDataset.get === 'function') {
              try {
                  allEdgesData = edgesDataset.get();
                  console.log('[Filter] Got', allEdgesData.length, 'edges from dataset');
              } catch (e) {
                  console.error('[Filter] Error getting edges:', e);
              }
          }
          
          console.log('[Filter] Initialization complete. Ready for filtering:', !!allNodesData.length);
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
          
          console.log('[Filter] Direction counts:', directionCounts);
      }
      
      function updateNetworkVisibility() {
          if (!nodesDataset || !edgesDataset || !allNodesData.length) {
              console.warn('[Filter] Not ready for visibility update');
              return;
          }
          
          console.log('[Filter] Updating visibility, state:', filterState);
          
          var visibleNodeIds = [];
          
          // Find visible nodes
          allNodesData.forEach(function(node) {
              var direction = node.direction || 'UNKNOWN';
              if (filterState[direction]) {
                  visibleNodeIds.push(node.id);
              }
          });
          
          console.log('[Filter] Visible nodes:', visibleNodeIds.length);
          
          // Update nodes
          var nodeUpdates = [];
          allNodesData.forEach(function(node) {
              var shouldBeVisible = visibleNodeIds.indexOf(node.id) !== -1;
              if (node.hidden !== !shouldBeVisible) {
                  nodeUpdates.push({id: node.id, hidden: !shouldBeVisible});
              }
          });
          
          if (nodeUpdates.length > 0) {
              console.log('[Filter] Updating', nodeUpdates.length, 'nodes');
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
                  console.log('[Filter] Updating', edgeUpdates.length, 'edges');
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
          
          console.log('[Filter] Stats updated:', visibleCounts);
      }
      
      window.toggleDirection = function(direction) {
          filterState[direction] = !filterState[direction];
          console.log('[Filter] Toggled', direction, '- new state:', filterState);
          updateNetworkVisibility();
      };
      
      window.resetFilters = function() {
          filterState.UP = true;
          filterState.DOWN = true;
          filterState.STABLE = true;
          
          document.getElementById('toggle-UP').checked = true;
          document.getElementById('toggle-DOWN').checked = true;
          document.getElementById('toggle-STABLE').checked = true;
          
          console.log('[Filter] Reset all filters');
          updateNetworkVisibility();
      };
      
      window.minimizeFilterPanel = function() {
          var panel = document.getElementById('directionFilterPanel');
          if (panel) {
              panel.classList.toggle('minimized');
          }
      };
      
      // Initialize after a short delay to ensure network is loaded
      // Try multiple times with increasing delays
      var attempts = 0;
      var maxAttempts = 50;
      
      function tryInitialize() {
          attempts++;
          console.log('[Filter] Initialize attempt', attempts);
          
          if (attempts > 5 && allNodesData.length > 0) {
              console.log('[Filter] ✓ Successfully initialized');
              return;
          }
          
          initializeFilter();
          
          if (allNodesData.length === 0 && attempts < maxAttempts) {
              setTimeout(tryInitialize, 200);
          }
      }
      
      // Start initialization immediately
      tryInitialize();
  })();
</script>
"""

PANEL_HTML = """
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


def inject_direction_into_nodes(html_text: str) -> str:
    """
    Parse the metadata JSON block and inject direction data directly into nodes.
    This ensures vis.js has the direction property on each node object.
    """
    # Find the metadata JSON block
    meta_match = re.search(
        r'var\s+geneMetadata\s*=\s*(\{.*?\});',
        html_text,
        re.DOTALL
    )
    
    if not meta_match:
        logging.warning("Could not find metadata block to extract directions")
        return html_text
    
    try:
        meta_json_str = meta_match.group(1)
        metadata = json.loads(meta_json_str)
        logging.info(f"Parsed metadata with {len(metadata)} genes")
    except Exception as e:
        logging.warning(f"Could not parse metadata: {e}")
        return html_text
    
    # Find the nodes initialization code
    # Look for: var nodes = new vis.DataSet([...]);
    nodes_init_pattern = r'(var\s+nodes\s*=\s*new\s+vis\.DataSet\(\[)(.*?)(\]\);)'
    nodes_match = re.search(nodes_init_pattern, html_text, re.DOTALL)
    
    if not nodes_match:
        logging.warning("Could not find nodes initialization")
        return html_text
    
    nodes_prefix = nodes_match.group(1)
    nodes_array_str = nodes_match.group(2)
    nodes_suffix = nodes_match.group(3)
    
    # Parse individual node objects and add direction property
    # Split by }, { pattern to separate nodes
    nodes_updated = []
    try:
        # Extract individual node JSON strings
        depth = 0
        current_obj = ""
        for char in nodes_array_str:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            
            current_obj += char
            
            if depth == 0 and current_obj.strip():
                try:
                    node = json.loads(current_obj.rstrip(',').strip())
                    
                    # Add direction from metadata if available
                    gene_id = node.get('id') or node.get('label')
                    if gene_id and gene_id in metadata:
                        direction = metadata[gene_id].get('direction', 'UNKNOWN')
                        node['direction'] = direction
                    else:
                        node['direction'] = 'UNKNOWN'
                    
                    nodes_updated.append(json.dumps(node))
                    current_obj = ""
                except Exception as e:
                    pass
        
        if nodes_updated:
            new_nodes_str = ",".join(nodes_updated)
            html_text = html_text.replace(
                nodes_prefix + nodes_array_str + nodes_suffix,
                nodes_prefix + new_nodes_str + nodes_suffix,
                1
            )
            logging.info(f"Injected direction into {len(nodes_updated)} nodes")
    except Exception as e:
        logging.warning(f"Error injecting directions: {e}")
    
    return html_text


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
    
    # Inject direction data into node objects
    html_text = inject_direction_into_nodes(html_text)
    
    # Add fresh CSS
    if "</head>" in html_text:
        html_text = html_text.replace("</head>", FILTER_CSS + "\n</head>", 1)
    
    # Add fresh panel (before network container)
    if 'id="' in html_text:
        # Find first HTML element ID
        container_idx = html_text.find('id="')
        if container_idx != -1:
            start_of_line = html_text.rfind('\n', 0, container_idx)
            if start_of_line == -1:
                start_of_line = 0
            else:
                start_of_line += 1
            html_text = (
                html_text[:start_of_line] +
                PANEL_HTML + "\n" +
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
    
    logging.info("Script 55: Fix direction filters v2")
    logging.info("  - Move panel to BOTTOM-LEFT")
    logging.info("  - Inject direction data directly into nodes")
    logging.info("  - Add debug logging to diagnose population issue")
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
    logging.info("\nNOTE: Check browser console (F12) for [Filter] debug messages")


if __name__ == "__main__":
    main()
