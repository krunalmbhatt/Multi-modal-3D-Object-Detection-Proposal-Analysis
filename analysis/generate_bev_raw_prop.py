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

# Add plugin path
PLUGIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects'))
sys.path.insert(0, PLUGIN_PATH)
import mmdet3d_plugin

# --- CONFIGURATION ---
proposal_pkl = 'proposal_dumps_old.pkl'
config_file = 'projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py'
num_frames_to_plot = 3

# --- Load Proposals ---
proposals = mmcv.load(proposal_pkl)
num_samples = len(proposals)

# --- Load Dataset ---
cfg = Config.fromfile(config_file)
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

# --- Random Frame Selection ---
random.seed(42)
frame_indices = random.sample(range(num_samples), num_frames_to_plot)
print(f"Selected frames: {frame_indices}")

# --- BEV Plot Loop ---
for frame_idx in frame_indices:
    prop_entry = proposals[frame_idx]
    ann_info = dataset.get_ann_info(frame_idx)
    gt_bboxes = ann_info['gt_bboxes_3d']

    prop_boxes = np.array(prop_entry['boxes'])
    gt_boxes = gt_bboxes.tensor.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8,8))

    # Plot Proposals (Red)
    if prop_boxes.shape[1] >= 7:
        x = prop_boxes[:, 0]
        y = prop_boxes[:, 1]
        ax.scatter(x, y, c='red', s=5, label=f'Raw Proposals ({len(x)})')

    # Plot GT (Green)
    if gt_boxes.shape[1] >= 7:
        xg = gt_boxes[:, 0]
        yg = gt_boxes[:, 1]
        ax.scatter(xg, yg, c='green', s=20, marker='s', label=f'Ground Truth ({len(xg)})')

    # Draw eval boundary
    pcd_range = np.array([-54, -54, -5, 54, 54, 3], dtype=np.float32)
    bev_boundary = [
        [pcd_range[0], pcd_range[1]],
        [pcd_range[3], pcd_range[1]],
        [pcd_range[3], pcd_range[4]],
        [pcd_range[0], pcd_range[4]],
        [pcd_range[0], pcd_range[1]],
    ]
    bev_boundary = np.array(bev_boundary)
    ax.plot(bev_boundary[:, 0], bev_boundary[:, 1], 'k--', label='Eval Range')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.title(f"BEV View - Frame {frame_idx} (Raw Proposals)")

    out_path = f'bev_proposals_frame{frame_idx}.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved BEV plot to {out_path}")
    plt.close()
