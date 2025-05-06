import mmcv
import numpy as np
import torch
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import bbox_overlaps_3d
import sys
import os

# --- MONKEY PATCH MMCV REGISTRY ---
orig_register_module = mmcv.utils.registry.Registry._register_module
def safe_register(self, module, module_name=None, force=False):
    return orig_register_module(self, module, module_name, force=True)
mmcv.utils.registry.Registry._register_module = safe_register

# --- IMPORT PLUGIN MODULES ---
PLUGIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects'))
sys.path.insert(0, PLUGIN_PATH)
import mmdet3d_plugin  # ensures all custom datasets, coders, hooks, etc. are registered

# --- CONFIGURATION ---
CONFIG_PATH = 'projects/configs/meformer_voxel0075_vov_1600x640_cbgs.py'
DET_PKL_PATH = 'final_detections.pkl'
IOU_THRESHOLD = 0.5
Ks = [1, 5, 10, 50, 100]

# --- LOAD CONFIG AND DATASET ---
cfg = mmcv.Config.fromfile(CONFIG_PATH)
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)

# --- LOAD DETECTIONS ---
results = mmcv.load(DET_PKL_PATH)
if isinstance(results[0], dict) and 'pts_bbox' in results[0]:
    results = [res['pts_bbox'] for res in results]

# --- RECALL TRACKERS ---
total_gt = 0
covered_gt_at_K = {K: 0 for K in Ks}

# --- MAIN LOOP ---
for idx in range(len(dataset)):
    ann_info = dataset.get_ann_info(idx)
    gt_boxes = ann_info['gt_bboxes_3d']
    gt_labels = ann_info.get('gt_labels_3d', None)

    if gt_labels is not None:
        valid_mask = (gt_labels != -1)
        gt_boxes = gt_boxes[valid_mask]

    num_gt = len(gt_boxes)
    if num_gt == 0:
        continue
    total_gt += num_gt

    det = results[idx]
    pred_boxes = det['boxes_3d']
    pred_scores = det['scores_3d']

    if len(pred_scores) == 0:
        continue  # skip frame with no predictions

    scores_np = pred_scores.cpu().numpy() if hasattr(pred_scores, 'cpu') else np.asarray(pred_scores)
    order = scores_np.argsort()[::-1]

    pred_boxes_tensor = torch.from_numpy(pred_boxes.tensor.cpu().numpy()[order].copy()).to(pred_boxes.tensor.device)

    gt_boxes_tensor = gt_boxes.tensor

    ious = bbox_overlaps_3d(pred_boxes_tensor, gt_boxes_tensor).cpu().numpy()

    for K in Ks:
        iou_subset = ious[:K, :] if K <= ious.shape[0] else ious
        covered = (iou_subset >= IOU_THRESHOLD).any(axis=0)
        covered_gt_at_K[K] += covered.sum()

# --- PRINT RECALL RESULTS ---
print(f"\nRecall results (IoU ≥ {IOU_THRESHOLD}):")
for K in Ks:
    recall = covered_gt_at_K[K] / total_gt if total_gt > 0 else 0
    print(f"  Recall@{K:<3} = {recall * 100:.2f}%  ({covered_gt_at_K[K]}/{total_gt} GT covered)")

# --- OPTIONAL PLOT ---
try:
    import matplotlib.pyplot as plt
    plt.figure()
    recalls = [covered_gt_at_K[K] / total_gt for K in Ks]
    plt.plot(Ks, [r * 100 for r in recalls], marker='o')
    plt.xlabel('Top-K Predictions')
    plt.ylabel(f'Recall (IoU ≥ {IOU_THRESHOLD}) [%]')
    plt.title('Recall vs Top-K Predictions')
    plt.grid(True)
    for i, r in zip(Ks, recalls):
        plt.text(i, r * 100 + 0.5, f'{r * 100:.1f}%', ha='center', fontsize=8)
    plt.savefig('recall_curve.png', dpi=200, bbox_inches='tight')
    print("Saved recall plot to recall_curve.png")
except ImportError:
    print("matplotlib not installed, skipping plot.")
