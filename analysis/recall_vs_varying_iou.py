import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import mmcv
import os
import sys
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core import bbox_overlaps_3d

# Monkey patch registry
orig_register_module = mmcv.utils.registry.Registry._register_module
def safe_register(self, module, module_name=None, force=False):
    return orig_register_module(self, module, module_name, force=True)
mmcv.utils.registry.Registry._register_module = safe_register

# Plugin path
PLUGIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects'))
sys.path.insert(0, PLUGIN_PATH)
import mmdet3d_plugin

# --- Configuration ---
proposal_pkl = 'proposal_dumps_old.pkl'
config_file = 'projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py'
fixed_top_k = 100  # Always top 100 proposals

# IoU thresholds to test
iou_thresholds = np.linspace(0.3, 0.8, 11)  # 0.3, 0.35, 0.4, ..., 0.8

# --- Load Proposals ---
proposals = mmcv.load(proposal_pkl)
num_samples = len(proposals)

# --- Load Dataset ---
cfg = Config.fromfile(config_file)
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

assert len(dataset) == num_samples, "Mismatch between proposals and dataset size!"

# --- Initialize Counters ---
total_gt_count = 0
covered_gt_counts = {iou: 0 for iou in iou_thresholds}

# --- Main Loop ---
for idx in range(num_samples):
    ann_info = dataset.get_ann_info(idx)
    gt_bboxes = ann_info['gt_bboxes_3d']
    gt_num = gt_bboxes.tensor.shape[0] if hasattr(gt_bboxes, 'tensor') else len(gt_bboxes)

    if gt_num == 0:
        continue
    total_gt_count += gt_num

    prop_entry = proposals[idx]
    if prop_entry is None or len(prop_entry['boxes']) == 0:
        continue

    prop_boxes = prop_entry['boxes']
    prop_scores = prop_entry['scores']

    prop_boxes_tensor = torch.tensor(prop_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(prop_scores, dtype=torch.float32)

    sorted_indices = torch.argsort(scores_tensor, descending=True)
    prop_boxes_sorted = prop_boxes_tensor[sorted_indices]

    gt_boxes_tensor = torch.tensor(gt_bboxes.tensor if hasattr(gt_bboxes, 'tensor') 
                                   else np.array(gt_bboxes), dtype=torch.float32)

    # Take top-K proposals
    num_props = prop_boxes_sorted.shape[0]
    topk = min(fixed_top_k, num_props)
    topk_props = prop_boxes_sorted[:topk]

    # Compute IoUs
    ious = bbox_overlaps_3d(topk_props, gt_boxes_tensor, mode='iou', coordinate='lidar')

    # For each IoU threshold
    for thr in iou_thresholds:
        max_ious, _ = ious.max(dim=0)  # for each GT box
        covered = (max_ious >= thr).sum().item()
        covered_gt_counts[thr] += covered

# --- Compute Recall ---
recalls = {thr: (covered_gt_counts[thr] / total_gt_count) for thr in iou_thresholds}

# --- Print and Plot ---
print("\nRecall vs IoU (Top-100 Proposals):")
for thr in iou_thresholds:
    print(f"IoU >= {thr:.2f}: Recall = {recalls[thr]*100:.2f}%")

# Plot
plt.figure(figsize=(6,5))
plt.plot(iou_thresholds, [recalls[thr]*100 for thr in iou_thresholds], marker='o')
plt.title('Recall vs IoU Threshold (Top-100 Proposals)')
plt.xlabel('IoU Threshold')
plt.ylabel('Recall [%]')
plt.grid(True)
plt.savefig('recall_vs_iou_threshold.png')
plt.show()
