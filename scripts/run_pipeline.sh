#!/bin/bash
# run_pipeline.sh
# Master script to run the unified Viewpoint + GAMs pipeline
# It checks if we are on a Slurm cluster and handles paths accordingly.

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "      Starting Unified Pipeline: Viewpoint + GAMs       "
echo "=========================================================="

# 1. Environment Setup
if [ -z "$DEGUS_DATA_DIR" ]; then
    # If not set, try to define a default
    if [ -d "/mnt/NAS/Degus/merged_files/" ]; then
        export DEGUS_DATA_DIR="/mnt/NAS/Degus/merged_files/"
        export DEGUS_DB_PATH="/mnt/NAS/Mati/MATLAB/2019-20 _ Degus/Degus-2020-Mati/AllData2.db"
        echo "[INFO] Using cluster default NAS paths."
    else
        # Fallback to local 'data' folder
        BASE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
        export DEGUS_DATA_DIR="${BASE_DIR}/data"
        export DEGUS_DB_PATH="${BASE_DIR}/data/AllData2.db" # If exists locally
        echo "[INFO] Using local path: $DEGUS_DATA_DIR"
    fi
else
    echo "[INFO] Using user-provided DEGUS_DATA_DIR: $DEGUS_DATA_DIR"
fi

# Ensure results directory exists
RESULTS_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")/results"
mkdir -p "$RESULTS_DIR"

echo "----------------------------------------------------------"
echo " Step 1: MATLAB Viewpoint Analysis"
echo "----------------------------------------------------------"
# Move to scripts directory to run MATLAB
cd "$(dirname "$0")"

# Execute MATLAB script without opening the GUI
# -batch flag requires R2019b or later, for older use -nodisplay -r "step1_viewpoint_analysis; exit"
matlab -nodisplay -nosplash -nodesktop -r "try, run('step1_viewpoint_analysis.m'), catch e, disp(getReport(e)), exit(1), end, exit(0)"

# Verify MATLAB output exists
if [ ! -f "../results/viewpoint_results.mat" ] && [ ! -f "../results/viewpoint_results.csv" ]; then
    echo "[ERROR] Step 1 did not produce viewpoint_results.mat/csv. Exiting."
    exit 1
fi

echo "----------------------------------------------------------"
echo " Step 2: Python GAM Analysis"
echo "----------------------------------------------------------"
# Run Python script
# Assuming python3 is available in the environment (e.g. via virtualenv or conda)
python3 step2_gam_analysis.py --data_dir "$DEGUS_DATA_DIR"

echo "=========================================================="
echo "               Pipeline Finished Successfully             "
echo "=========================================================="
