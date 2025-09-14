#!/usr/bin/env python3
"""
Results Visualization Script
============================

This script reads optimization results from the results/ folder and creates
graphical representations without modifying any existing project files.

Features:
- Algorithm performance comparison
- Cost distribution analysis
- Flow network visualization
- Execution time analysis
- Results summary dashboard
"""

import json
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsVisualizer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results_data = []
        self.load_all_results()
    
    def load_all_results(self):
        """Load all JSON result files from the results directory."""
        # Look for multiple patterns to catch all result files
        patterns = [
            os.path.join(self.results_dir, "cagliari_*.json"),
            os.path.join(self.results_dir, "solution_*.json"),
            os.path.join(self.results_dir, "*.json")
        ]
        
        json_files = []
        for pattern in patterns:
            json_files.extend(glob.glob(pattern))
        
        # Remove duplicates
        json_files = list(set(json_files))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract timestamp from filename
                    filename = os.path.basename(file_path)
                    # Try different timestamp extraction methods
                    if '_' in filename:
                        timestamp_str = filename.split('_')[-1].replace('.json', '')
                    else:
                        timestamp_str = filename.replace('.json', '')
                    data['timestamp'] = timestamp_str
                    data['filename'] = filename
                    self.results_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        print(f"Loaded {len(self.results_data)} result files")
    
    
    def create_flow_network_visualization(self):
        """Create a simple and clear flow network visualization."""
        if not self.results_data:
            print("No results data available")
            return
        
        # Use the most recent result
        latest_result = max(self.results_data, key=lambda x: x.get('timestamp', ''))
        flows = latest_result.get('flows', [])
        
        if not flows:
            print("No flow data available")
            return
        
        # Create a larger figure for better readability
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Extract unique nodes
        supply_nodes = sorted(list(set([flow['from'] for flow in flows])))
        demand_nodes = sorted(list(set([flow['to'] for flow in flows])))
        
        # Better positioning with more spacing
        supply_positions = {}
        demand_positions = {}
        
        # Position supply nodes on the left with more spacing
        for i, node in enumerate(supply_nodes):
            supply_positions[node] = (2, (i + 1) * 1.2)
        
        # Position demand nodes on the right, centered vertically
        total_height = max(len(supply_nodes), len(demand_nodes)) * 1.2
        demand_start_y = (total_height - len(demand_nodes) * 1.2) / 2 + 1.2
        
        for i, node in enumerate(demand_nodes):
            demand_positions[node] = (8, demand_start_y + i * 1.2)
        
        # Draw supply nodes with text on the left
        for node, pos in supply_positions.items():
            circle = patches.Circle(pos, 0.2, color='lightblue', ec='navy', linewidth=2)
            ax.add_patch(circle)
            # Position text to the left of the circle
            ax.text(pos[0] - 0.4, pos[1], node[:12], ha='right', va='center', 
                   fontsize=9, fontweight='bold', color='navy')
        
        # Draw demand nodes with text on the right
        for node, pos in demand_positions.items():
            circle = patches.Circle(pos, 0.2, color='lightcoral', ec='darkred', linewidth=2)
            ax.add_patch(circle)
            # Position text to the right of the circle
            ax.text(pos[0] + 0.4, pos[1], node[:12], ha='left', va='center', 
                   fontsize=9, fontweight='bold', color='darkred')
        
        # Draw flows with external label positioning
        max_flow = max([flow['flow'] for flow in flows])
        max_cost = max([flow['unit_cost_eur'] for flow in flows])
        
        # Sort flows by cost to prioritize showing most expensive flows
        sorted_flows = sorted(flows, key=lambda x: x['arc_cost_eur'], reverse=True)
        
        # Clean visualization without textboxes
        
        for i, flow in enumerate(sorted_flows):
            from_pos = supply_positions[flow['from']]
            to_pos = demand_positions[flow['to']]
            
            # Calculate arc start and end points on the circle perimeters
            # Vector from center to center
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            distance = (dx**2 + dy**2)**0.5
            
            # Normalize the vector
            if distance > 0:
                dx_norm = dx / distance
                dy_norm = dy / distance
            else:
                dx_norm = 1
                dy_norm = 0
            
            # Calculate start point (on supply node perimeter)
            circle_radius = 0.2
            start_x = from_pos[0] + dx_norm * circle_radius
            start_y = from_pos[1] + dy_norm * circle_radius
            
            # Calculate end point (on demand node perimeter)
            end_x = to_pos[0] - dx_norm * circle_radius
            end_y = to_pos[1] - dy_norm * circle_radius
            
            # Line width based on flow
            line_width = 1 + (flow['flow'] / max_flow) * 5
            
            # Color based on cost
            cost_ratio = flow['unit_cost_eur'] / max_cost
            color = plt.cm.Reds(cost_ratio)
            
            # Draw arrow from perimeter to perimeter
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=line_width, color=color, alpha=0.7))
            
            # No textboxes - clean visualization focusing on the network structure
        
        # Add title
        ax.set_title('EV Charging Flow Network', fontsize=16, fontweight='bold', pad=20)
        ax.text(3, 0.5, f'Algorithm: {latest_result.get("algorithm", "unknown").upper()}', 
               ha='center', fontsize=12, color='darkgreen')
        ax.text(3, 0.3, f'Total Cost: €{latest_result.get("objective_eur", 0):.2f}', 
               ha='center', fontsize=12, color='darkred')
        
        # Add legend in top right corner
        legend_x = 9.5
        legend_y = 0.8
        
        # Legend title
        ax.text(legend_x, legend_y, 'Legend', ha='center', fontsize=12, fontweight='bold', 
               color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
        
        # Arc color explanation
        ax.text(legend_x, legend_y - 0.15, 'Arc Color = Unit Cost', ha='center', fontsize=10, 
               fontweight='bold', color='black')
        
        # Color scale examples
        color_examples = [
            ('High Cost', 'red'),
            ('Medium Cost', 'orange'), 
            ('Low Cost', 'green')
        ]
        
        for i, (label, color) in enumerate(color_examples):
            y_pos = legend_y - 0.4 - i * 0.6  # Very large spacing of 0.6 for excellent readability
            # Draw colored line
            ax.plot([legend_x - 0.3, legend_x - 0.1], [y_pos, y_pos], color=color, linewidth=3)
            # Add label
            ax.text(legend_x - 0.05, y_pos, label, ha='left', va='center', fontsize=14, color='black')
        
        # Arc thickness explanation
        ax.text(legend_x, legend_y - 0.55, 'Arc Thickness = Flow Amount', ha='center', fontsize=10, 
               fontweight='bold', color='black')
        
        # Thickness examples
        thickness_examples = [
            ('High Flow', 4),
            ('Medium Flow', 2.5),
            ('Low Flow', 1)
        ]
        
        for i, (label, thickness) in enumerate(thickness_examples):
            y_pos = legend_y - 1.2 - i * 0.6  # Very large spacing of 0.6 and adjusted start position
            # Draw line with different thickness
            ax.plot([legend_x - 0.3, legend_x - 0.1], [y_pos, y_pos], color='blue', linewidth=thickness)
            # Add label
            ax.text(legend_x - 0.05, y_pos, label, ha='left', va='center', fontsize=14, color='black')
        
        # Set axis properties to accommodate external labels
        ax.set_xlim(0, 12)
        ax.set_ylim(0, max(len(supply_nodes), len(demand_nodes)) * 1.2 + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save the figure
        plt.savefig('visualizations/flow_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Flow network visualization saved as 'visualizations/flow_network.png'")
    
    def create_cost_distribution_analysis(self):
        """Create cost distribution analysis."""
        if not self.results_data:
            print("No results data available")
            return
        
        # Get the most recent result
        latest_result = max(self.results_data, key=lambda x: x.get('timestamp', ''))
        flows = latest_result.get('flows', [])
        
        if not flows:
            print("No flow data available")
            return
        
        # Extract cost data
        unit_costs = [flow['unit_cost_eur'] for flow in flows]
        arc_costs = [flow['arc_cost_eur'] for flow in flows]
        flow_amounts = [flow['flow'] for flow in flows]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cost Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Unit cost distribution
        ax1.hist(unit_costs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Unit Costs (€/unit)')
        ax1.set_xlabel('Unit Cost (EUR)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Arc cost distribution
        ax2.hist(arc_costs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Distribution of Arc Costs (€)')
        ax2.set_xlabel('Arc Cost (EUR)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Flow amount distribution
        ax3.hist(flow_amounts, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Distribution of Flow Amounts')
        ax3.set_xlabel('Flow Amount (units)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cost vs Flow scatter plot
        ax4.scatter(flow_amounts, arc_costs, alpha=0.6, s=60, c=unit_costs, cmap='viridis')
        ax4.set_title('Flow Amount vs Arc Cost')
        ax4.set_xlabel('Flow Amount (units)')
        ax4.set_ylabel('Arc Cost (EUR)')
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Unit Cost (EUR/unit)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/cost_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure instead of showing it
        print("Cost distribution analysis saved as 'visualizations/cost_distribution.png'")
    
    def create_results_dashboard(self):
        """Create a comprehensive results dashboard."""
        if not self.results_data:
            print("No results data available")
            return
        
        # Create a large dashboard
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('EV Charging Optimization - Results Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Algorithm comparison (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        algorithms = {}
        for result in self.results_data:
            algo = result.get('algorithm', 'unknown')
            if algo not in algorithms:
                algorithms[algo] = {'objectives': [], 'times': []}
            algorithms[algo]['objectives'].append(result.get('objective_eur', 0))
            algorithms[algo]['times'].append(result.get('timing_sec', 0))
        
        x_pos = np.arange(len(algorithms))
        objectives = [np.mean(algorithms[algo]['objectives']) for algo in algorithms.keys()]
        times = [np.mean(algorithms[algo]['times']) * 1000 for algo in algorithms.keys()]  # Convert to ms
        
        ax1_twin = ax1.twinx()
        bars1 = ax1.bar(x_pos - 0.2, objectives, 0.4, label='Objective (€)', color='skyblue')
        bars2 = ax1_twin.bar(x_pos + 0.2, times, 0.4, label='Time (ms)', color='lightcoral')
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Objective Cost (€)', color='blue')
        ax1_twin.set_ylabel('Execution Time (ms)', color='red')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([algo.upper() for algo in algorithms.keys()])
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Results timeline (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        timestamps = []
        objectives_timeline = []
        for result in sorted(self.results_data, key=lambda x: x.get('timestamp', '')):
            timestamps.append(result.get('timestamp', ''))
            objectives_timeline.append(result.get('objective_eur', 0))
        
        ax2.plot(range(len(timestamps)), objectives_timeline, marker='o', linewidth=2, markersize=6)
        ax2.set_title('Objective Values Over Time')
        ax2.set_xlabel('Run Number')
        ax2.set_ylabel('Objective Cost (€)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Flow statistics (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        latest_result = max(self.results_data, key=lambda x: x.get('timestamp', ''))
        flows = latest_result.get('flows', [])
        
        if flows:
            flow_amounts = [flow['flow'] for flow in flows]
            ax3.hist(flow_amounts, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('Flow Amount Distribution')
            ax3.set_xlabel('Flow Amount (units)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Cost breakdown (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if flows:
            # Top 10 most expensive flows
            top_flows = sorted(flows, key=lambda x: x['arc_cost_eur'], reverse=True)[:10]
            flow_labels = [f"{f['from'][:10]}...\n→ {f['to'][:10]}..." for f in top_flows]
            costs = [f['arc_cost_eur'] for f in top_flows]
            
            bars = ax4.barh(range(len(flow_labels)), costs, color='lightcoral')
            ax4.set_yticks(range(len(flow_labels)))
            ax4.set_yticklabels(flow_labels, fontsize=8)
            ax4.set_xlabel('Arc Cost (€)')
            ax4.set_title('Top 10 Most Expensive Flows')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + 10, bar.get_y() + bar.get_height()/2, 
                        f'€{width:.0f}', ha='left', va='center', fontsize=8)
        
        # 5. Summary statistics (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Calculate summary statistics
        total_runs = len(self.results_data)
        unique_algos = list(set([r.get('algorithm', 'unknown') for r in self.results_data]))
        avg_objective = np.mean([r.get('objective_eur', 0) for r in self.results_data])
        min_time = min([r.get('timing_sec', 0) for r in self.results_data])
        max_time = max([r.get('timing_sec', 0) for r in self.results_data])
        
        summary_text = f"""
        SUMMARY STATISTICS
        ==================
        Total Runs: {total_runs}
        Algorithms Used: {', '.join([algo.upper() for algo in unique_algos])}
        Average Objective: €{avg_objective:.2f}
        Fastest Execution: {min_time*1000:.1f}ms
        Slowest Execution: {max_time*1000:.1f}ms
        Latest Result: {latest_result.get('algorithm', 'unknown').upper()} - €{latest_result.get('objective_eur', 0):.2f}
        """
        
        ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=14,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.savefig('visualizations/results_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure instead of showing it
        print("Results dashboard saved as 'visualizations/results_dashboard.png'")
    
    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
        print("Generating all visualizations...")
        print("=" * 50)
        
        try:
            self.create_flow_network_visualization()
            print()
        except Exception as e:
            print(f"Error creating flow network visualization: {e}")
        
        try:
            self.create_cost_distribution_analysis()
            print()
        except Exception as e:
            print(f"Error creating cost distribution analysis: {e}")
        
        try:
            self.create_results_dashboard()
            print()
        except Exception as e:
            print(f"Error creating results dashboard: {e}")
        
        print("=" * 50)
        print("All visualizations completed!")
        print(f"Images saved in: {os.path.abspath('visualizations')}")
        print("Files created:")
        for file in os.listdir('visualizations'):
            if file.endswith('.png'):
                print(f"  - {file}")


def main():
    """Main function to run the visualization script."""
    print("EV Charging Optimization - Results Visualizer")
    print("=" * 50)
    
    # Check if results directory exists
    if not os.path.exists("results"):
        print("Error: 'results' directory not found!")
        print("Please run the optimization first to generate results.")
        return
    
    # Create visualizer
    visualizer = ResultsVisualizer()
    
    if not visualizer.results_data:
        print("No result files found in the results directory!")
        print("Please run the optimization first to generate results.")
        return
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
