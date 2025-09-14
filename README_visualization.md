# Results Visualization

This directory contains a standalone visualization script that reads optimization results and creates graphical representations.

## Files

- `visualize_results.py` - Main visualization script
- `requirements_visualization.txt` - Python dependencies for visualization
- `README_visualization.md` - This documentation

## Installation

```bash
# Install visualization dependencies
pip install -r requirements_visualization.txt
```

## Usage

```bash
# Run the visualization script
python visualize_results.py
```

## Generated Visualizations

The script creates three types of visualizations and saves them in the `visualizations/` folder:

### 1. Flow Network Visualization (`visualizations/flow_network.png`)
- **Enhanced clarity**: Much larger figure (24x16) with generous spacing
- **Clear node labels**: Larger circles with bold, readable text
- **Detailed flow information**: Each arc shows flow amount, unit cost, and total cost
- **Visual encoding**: Arrow thickness represents flow amount, color represents unit cost
- **Comprehensive legend**: Clear explanation of all visual elements
- **Better positioning**: Smart label placement to avoid overlaps

### 2. Cost Distribution Analysis (`visualizations/cost_distribution.png`)
- Distribution of unit costs, arc costs, and flow amounts
- Scatter plot of flow amount vs arc cost
- Statistical analysis of cost patterns

### 3. Results Dashboard (`visualizations/results_dashboard.png`)
- Comprehensive overview of all results
- Algorithm comparison charts
- Timeline of objective values
- Top 10 most expensive flows
- Summary statistics

## Features

- **Automatic Data Loading**: Reads all JSON result files from the `results/` directory
- **Multiple Visualizations**: Creates 3 different types of analysis
- **High-Quality Output**: Saves images at 300 DPI for publication quality
- **Non-Interactive Mode**: Saves images directly without opening display windows
- **Organized Output**: Creates a dedicated `visualizations/` folder for all images
- **Error Handling**: Gracefully handles missing or corrupted data files
- **No Dependencies on Main Code**: Completely independent of the optimization scripts

## Requirements

- Python 3.7+
- matplotlib
- seaborn
- numpy
- pandas

## Notes

- The script only reads from the `results/` folder and does not modify any existing project files
- All visualizations are saved as PNG files in the `visualizations/` folder
- The script automatically creates the `visualizations/` directory if it doesn't exist
- No interactive windows are opened - images are saved directly to disk
- The script automatically detects and processes all available result files
- Results are sorted by timestamp to show the most recent data
