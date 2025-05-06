#!/bin/bash
echo "Starting frozen 0.1 with aux loss test script"

# --- Configuration ---
CONFIG_FILE="/home/kmbhatt/home/dr_sem4/gits/MEFormer/projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py"
# CHECKPOINT_FILE="ckpts/moad_voxel0075_vov_1600x640_cbgs.pth" # Or your trained one
# CHECKPOINT_FILE="/home/kmbhatt/home/dr_sem4/gits/MEFormer/ckpts/meformer_voxel0075_vov_1600x640_cbgs.pth" # Or epoch_X.pth
CHECKPOINT_FILE="/home/kmbhatt/home/dr_sem4/gits/MEFormer/work_dirs/meformer_voxel0075_vov_1600x640_cbgs/20250501-170310/latest.pth"
GPUS=4 # Number of GPUs to use for testing
# --- End Configuration ---

# --- Parameters to Test (Example 1) ---
NMS_THR=0.15
MAX_NUM=200
OUTPUT_DIR="./test_results/unfrozen/$(basename $CONFIG_FILE .py)_nms${NMS_THR}_max${MAX_NUM}"
echo "Running evaluation with NMS Threshold: $NMS_THR, Max Detections: $MAX_NUM"
# Create output dir if it doesn't exist
mkdir -p $OUTPUT_DIR
# Run distributed test
bash ./tools/dist_test.sh $CONFIG_FILE $CHECKPOINT_FILE $GPUS --out $OUTPUT_DIR/results.pkl --eval bbox \
  --cfg-options model.test_cfg.pts.nms_thr=$NMS_THR model.test_cfg.pts.max_num=$MAX_NUM \
> $OUTPUT_DIR/eval_log.txt 2>&1 # Redirect output to a log file
echo "Results saved in $OUTPUT_DIR"
echo "---------------------------------"

# --- Parameters to Test (Example 2) ---
NMS_THR=0.25
MAX_NUM=200
OUTPUT_DIR="./test_results/unfrozen/$(basename $CONFIG_FILE .py)_nms${NMS_THR}_max${MAX_NUM}"
echo "Running evaluation with NMS Threshold: $NMS_THR, Max Detections: $MAX_NUM"
mkdir -p $OUTPUT_DIR
bash ./tools/dist_test.sh $CONFIG_FILE $CHECKPOINT_FILE $GPUS --out $OUTPUT_DIR/results.pkl --eval bbox \
  --cfg-options model.test_cfg.pts.nms_thr=$NMS_THR model.test_cfg.pts.max_num=$MAX_NUM \
> $OUTPUT_DIR/eval_log.txt 2>&1
echo "Results saved in $OUTPUT_DIR"
echo "---------------------------------"

# --- Parameters to Test (Example 3) ---
NMS_THR=0.20 # Back to default NMS
MAX_NUM=150 # Try fewer max detections
OUTPUT_DIR="./test_results/unfrozen/$(basename $CONFIG_FILE .py)_nms${NMS_THR}_max${MAX_NUM}"
echo "Running evaluation with NMS Threshold: $NMS_THR, Max Detections: $MAX_NUM"
mkdir -p $OUTPUT_DIR
bash ./tools/dist_test.sh $CONFIG_FILE $CHECKPOINT_FILE $GPUS --out $OUTPUT_DIR/results.pkl --eval bbox \
  --cfg-options model.test_cfg.pts.nms_thr=$NMS_THR model.test_cfg.pts.max_num=$MAX_NUM \
> $OUTPUT_DIR/eval_log.txt 2>&1
echo "Results saved in $OUTPUT_DIR"
echo "---------------------------------"

# --- Add more blocks like above for other parameter combinations ---

echo "Finished testing different post-processing parameters."