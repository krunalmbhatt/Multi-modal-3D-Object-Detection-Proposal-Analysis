import pickle
import numpy as np
import torch
import mmcv
from mmdet3d.core import bbox_overlaps_3d  # IoU calculation for 3D boxes
from mmdet3d.datasets import build_dataset
from mmcv import Config
import matplotlib.pyplot as plt
import sys
import os

# Monkey patch MMCV Registry to force overwrite duplicates (avoids registration errors)
orig_register_module = mmcv.utils.registry.Registry._register_module
def safe_register(self, module, module_name=None, force=False):
    return orig_register_module(self, module, module_name, force=True)
mmcv.utils.registry.Registry._register_module = safe_register

# Add custom plugin path to sys.path and import it to register all custom modules
PLUGIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects'))
sys.path.insert(0, PLUGIN_PATH)
import mmdet3d_plugin

# --- Configuration ---
proposal_pkl = 'proposal_dumps_old.pkl'               # Path to the proposals pickle file
config_file = 'projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py'     # Path to the model config (for dataset)
iou_thr = 0.3                                     # IoU threshold for considering a GT covered
max_k = 100                                       # Evaluate recall up to top-K proposals

# --- Load raw proposals ---
# Using mmcv or pickle to load the proposals list
proposals_list = mmcv.load(proposal_pkl)  # mmcv.load can handle pkl similar to pickle.load

# Ensure proposals_list is a list of dicts with 'boxes' and 'scores'
num_samples = len(proposals_list)
print(f"Loaded proposals for {num_samples} samples.")

# --- Build the dataset to get ground truth annotations ---
cfg = Config.fromfile(config_file)
# If the config has separate validation/test pipeline, ensure test_mode
if hasattr(cfg.data, 'test') and cfg.data.test.get('test_mode', None) is False:
    cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

assert len(dataset) == num_samples, "Dataset size and proposals list size do not match!"

# --- Initialize counters for recall ---
total_gt_count = 0        # total number of GT boxes in dataset
covered_gt_counts = {K: 0 for K in range(1, max_k+1)}  # covered GT count at each K

# --- Iterate over each sample to compute coverage ---
for idx in range(num_samples):
    ann_info = dataset.get_ann_info(idx)
    gt_bboxes = ann_info['gt_bboxes_3d']         # ground truth 3D boxes (LiDARInstance3DBoxes or similar)
    gt_num = gt_bboxes.tensor.shape[0] if hasattr(gt_bboxes, 'tensor') else len(gt_bboxes)
    # If dataset returns LiDARInstance3DBoxes, use .tensor to get a torch tensor of shape (gt_num, dims)
    if gt_num == 0:
        # No ground truth in this frame
        continue
    total_gt_count += gt_num

    # Get proposals for this sample
    prop_entry = proposals_list[idx]
    if prop_entry is None:
        # No proposals were generated for this frame
        continue
    prop_boxes = prop_entry['boxes']
    prop_scores = prop_entry['scores']
    if len(prop_boxes) == 0:
        # No proposals in this frame
        continue

    # Convert proposals and GT to torch tensors (for IoU computation)
    # Ensure they are of shape (N, dim)
    prop_boxes_tensor = torch.tensor(prop_boxes, dtype=torch.float32)
    gt_boxes_tensor = torch.tensor(gt_bboxes.tensor if hasattr(gt_bboxes, 'tensor') 
                                   else np.array(gt_bboxes), dtype=torch.float32)

    # Sort proposals by score in descending order
    scores_tensor = torch.tensor(prop_scores, dtype=torch.float32)
    sorted_indices = torch.argsort(scores_tensor, descending=True)
    prop_boxes_sorted = prop_boxes_tensor[sorted_indices]

    # Compute IoU overlaps between all proposals and all GT for this sample
    # Use the appropriate coordinate system ('lidar' for LiDAR coordinates, 'camera' if in camera coords)
    ious = bbox_overlaps_3d(prop_boxes_sorted, gt_boxes_tensor, mode='iou', coordinate='lidar')
    # ious shape: [GT_count, Proposal_count] (each entry is IoU of (gt, prop))

    # Determine coverage for each top-K
    # We iterate K from 1 to max_k (or up to number of proposals if fewer than max_k available)
    num_props = prop_boxes_sorted.shape[0]
    for K in range(1, max_k+1):
        if K > num_props:
            # If fewer proposals than K for this sample, then from this K onward nothing new covers
            break
        # IoUs of GT vs top-K proposals (take first K columns of ious matrix)
        ious_topK = ious[:, :K]
        # Check which GT boxes have IoU >= threshold with any of the top K proposals
        # (dim 0 corresponds to each GT, so take max IoU over the K proposals for each GT)
        max_iou_per_gt, _ = ious_topK.max(dim=1)
        # Count GTs covered by at least one proposal among top K (IoU >= threshold)
        covered = (max_iou_per_gt >= iou_thr).sum().item()
        covered_gt_counts[K] += covered

# --- Compute recall at each K ---
recall_values = []
for K in range(1, max_k+1):
    # Recall = (number of GTs covered by at least one of top-K proposals) / (total GTs)
    if total_gt_count == 0:
        recall = 0.0
    else:
        recall = covered_gt_counts[K] / total_gt_count
    recall_values.append(recall)
    print(f"Recall@{K}: {recall:.4f}")

# --- Plot Recall vs Top-K ---
plt.figure(figsize=(6,5))
Ks = list(range(1, max_k+1))
# Convert recall to percentage for plotting
recall_percent = [val * 100 for val in recall_values]
plt.plot(Ks, recall_percent, marker='o', color='C0')
plt.title('Recall vs Top-K Proposals')
plt.xlabel('Top-K Proposals')
plt.ylabel(f'Recall (IoU >= {iou_thr}) [%]')
plt.grid(True)

# Annotate a few key points on the curve
key_Ks = [1, 5, 10, 50, 100]  # key points to annotate (modify as needed)
for K in key_Ks:
    if K <= max_k:
        plt.annotate(f"{recall_percent[K-1]:.1f}%", (K, recall_percent[K-1] + 1),
                     ha='center', color='C0')

plt.savefig('recall_vs_topK_raw_proposals.png')
plt.show()
