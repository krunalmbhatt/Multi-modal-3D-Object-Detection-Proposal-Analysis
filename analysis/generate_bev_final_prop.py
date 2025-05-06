import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
import sys
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

# Monkey patch registry
orig_register_module = mmcv.utils.registry.Registry._register_module
def safe_register(self, module, module_name=None, force=False):
    return orig_register_module(self, module, module_name, force=True)
mmcv.utils.registry.Registry._register_module = safe_register

# Plugin path
PLUGIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects'))
sys.path.insert(0, PLUGIN_PATH)
import mmdet3d_plugin

# --- CONFIGURATION ---
detections_pkl = 'final_detections.pkl'
config_file = 'projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py'
num_frames_to_plot = 3

# --- Load Final Detections ---
detections = mmcv.load(detections_pkl)
if isinstance(detections[0], dict) and 'pts_bbox' in detections[0]:
    detections = [d['pts_bbox'] for d in detections]
num_samples = len(detections)

# --- Load Dataset ---
cfg = Config.fromfile(config_file)
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

assert len(dataset) == num_samples, "Mismatch between detections and dataset size!"

# --- Random Frame Selection ---
random.seed(42)
frame_indices = random.sample(range(num_samples), num_frames_to_plot)
print(f"Selected frames: {frame_indices}")

# --- BEV Plot Loop ---
for frame_idx in frame_indices:
    det_entry = detections[frame_idx]
    ann_info = dataset.get_ann_info(frame_idx)
    gt_bboxes = ann_info['gt_bboxes_3d']

    pred_boxes = det_entry['boxes_3d'].tensor.cpu().numpy() if hasattr(det_entry['boxes_3d'], 'tensor') else det_entry['boxes_3d']
    gt_boxes = gt_bboxes.tensor.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8,8))

    # Plot Predictions (Blue)
    if pred_boxes.shape[1] >= 7:
        x = pred_boxes[:, 0]
        y = pred_boxes[:, 1]
        ax.scatter(x, y, c='blue', s=5, label=f'Final Detections ({len(x)})')

    # Plot GT (Green)
    if gt_boxes.shape[1] >= 7:
        xg = gt_boxes[:, 0]
        yg = gt_boxes[:, 1]
        ax.scatter(xg, yg, c='green', s=20, marker='s', label=f'Ground Truth ({len(xg)})')

    # Draw eval boundary
    pcd_range = np.array([-54, -54, -5, 54, 54, 3], dtype=np.float32)
    bev_boundary = np.array([
        [pcd_range[0], pcd_range[1]],
        [pcd_range[3], pcd_range[1]],
        [pcd_range[3], pcd_range[4]],
        [pcd_range[0], pcd_range[4]],
        [pcd_range[0], pcd_range[1]],
    ])
    ax.plot(bev_boundary[:, 0], bev_boundary[:, 1], 'k--', label='Eval Range')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.title(f"BEV View - Frame {frame_idx} (Final Detections)")

    out_path = f'bev_final_frame{frame_idx}.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved BEV plot to {out_path}")
    plt.close()
