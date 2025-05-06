#!/bin/bash

echo "Starting evaluation of the original model (non-modified) with default parameters"

# --- Configuration ---
CONFIG_FILE="/home/kmbhatt/home/dr_sem4/gits/MEFormer/projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py"
# This CHECKPOINT_FILE points to the original, non-modified model weights.
CHECKPOINT_FILE="/home/kmbhatt/home/dr_sem4/gits/MEFormer/ckpts/meformer_voxel0075_vov_1600x640_cbgs.pth"
GPUS=4 # Number of GPUs to use for testing
# --- End Configuration ---

# We'll use just one set of post-processing parameters for the baseline comparison
# You could run multiple tests like before, but one is often sufficient for a baseline
# Let's pick the default or a common one, like NMS 0.2 and MaxNum 200

NMS_THR=0.20
MAX_NUM=200
OUTPUT_DIR="./test_results/original_model_nms${NMS_THR}_max${MAX_NUM}"
echo "Running evaluation of ORIGINAL model with NMS Threshold: $NMS_THR, Max Detections: $MAX_NUM"
# Create output dir if it doesn't exist
mkdir -p $OUTPUT_DIR
# Run distributed test
bash ./tools/dist_test.sh $CONFIG_FILE $CHECKPOINT_FILE $GPUS --out $OUTPUT_DIR/results.pkl --eval bbox \
  --cfg-options model.test_cfg.pts.nms_thr=$NMS_THR model.test_cfg.pts.max_num=$MAX_NUM \
> $OUTPUT_DIR/eval_log.txt 2>&1 # Redirect output to a log file
echo "Results saved in $OUTPUT_DIR"
echo "---------------------------------"

echo "Finished testing the original model."