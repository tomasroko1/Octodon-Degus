# Octodon Degus Analysis Pipeline

This repository contains the core pipeline for modeling spatially tuned neurons in Octodon degus (e.g., place cells, head direction cells, and viewpoint cells) using Generalized Additive Models (GAMs).

## Repository Structure

- `data/`: Put your HDF5 database files (`*.db`), cluster data (`*.db_clnew`), and master cell lists here. These files are ignored by git.
- `models/`: Saved `pygam` models are cached here.
- `results/`: CSV outputs and serialized analysis results.
- `reference/matlab/`: Legacy MATLAB scripts for reference.
- `scripts/`: Main execution scripts.
  - `step1_viewpoint_analysis.m`: Step 1 of the pipeline (MATLAB). Runs continuous perspective tuning and circular shuffling.
  - `convert_mat_to_csv.py`: Translates the MATLAB output into a CSV for Step 2.
  - `step2_gam_analysis.py`: Step 2 of the pipeline (Python). Trains cross-validated GAMs (Pos, View, HD) for the significant cells found in Step 1.
  - `plot_viewpoints.py`: Visualizes the learned viewpoint representations.
  - `utils/`: Core reusable components.

## Configuration

The pipeline defaults to reading data from the `./data/` folder. For cluster environments or custom paths, set the `DEGUS_DATA_DIR` environment variable:
```bash
export DEGUS_DATA_DIR="/mnt/NAS/Degus/merged_files"
```
*Note: `step2_gam_analysis.py` will automatically fallback to the NAS path if local data is not found.*

## Running the Pipeline

The analysis is split into two explicit steps to make it modular and cluster-friendly:

### Step 1: Viewpoint Analysis (MATLAB)
Since the spike shuffling significance test is heavily matrix-optimized, Step 1 is run natively in MATLAB on the cluster.
Run the script `scripts/step1_viewpoint_analysis.m` inside MATLAB.
It will generate `results/viewpoint_results.mat`.

Convert this output to CSV so Step 2 can read it:
```bash
python scripts/convert_mat_to_csv.py ../results/viewpoint_results.mat
```
This step outputs `results/viewpoint_results.csv`.

### Step 2: GAMs Analysis (Python)
To train the Spatial, Head Direction, and Joint (Pos+View) GAMs, calculating cross-validated Log-Likelihoods and Shapley values on the significant cells:
```bash
python scripts/step2_gam_analysis.py
```
This script automatically reads `viewpoint_results.csv`, filters for `is_significant == True`, and outputs `results/gam_results.csv`.

## Notebook Template
Review the `template_analysis.ipynb` for a notebook-based demonstration of loading and modeling a single cell interactively.
