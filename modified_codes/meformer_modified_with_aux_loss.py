# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

# Modified by: KRUNAL M BHATT
#MOSTLY EXPERIMENTS HAVE BEEN DONE USING THIS FILE. A MODIFIED VERSION OF THIS IMPROVES THE FWD_PTS_TRAIN FUNCTION
# ------------------------------------------------------------------------


import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import SPConvVoxelization
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class MEFormerDetector(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 **kwargs):
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(MEFormerDetector, self).__init__(**kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def init_weights(self):
        """Initialize model weights."""
        super(MEFormerDetector, self).init_weights()

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(
                pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
            )
            losses.update(losses_pts)
        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            img_feats (list[torch.Tensor]): Features of image branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample.
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        # Handle None features
        num_task_heads = len(getattr(self.pts_bbox_head, 'task_heads', []))
        if num_task_heads == 0: num_task_heads = 1 # Basic fallback
        _pts_feats = pts_feats if pts_feats is not None else [None] * num_task_heads
        _img_feats = img_feats if img_feats is not None else [None] * num_task_heads

        outs = self.pts_bbox_head(_pts_feats, _img_feats, img_metas)

        # --- Auxiliary loss on decoded centers ---
        loss_aux_total = 0
        valid_sample_count = 0

        # Basic validation of inputs
        if not gt_labels_3d: raise ValueError("gt_labels_3d cannot be None or empty.")
        if not gt_bboxes_3d: raise ValueError("gt_bboxes_3d cannot be None or empty.")
        if len(gt_labels_3d) != len(gt_bboxes_3d): raise ValueError("gt_labels_3d and gt_bboxes_3d must have the same length.")
        if not isinstance(gt_labels_3d[0], torch.Tensor): raise ValueError("gt_labels_3d[0] must be a tensor.")

        # Determine batch size (use sample 0's label tensor size)
        # Ensure batch_size corresponds to the actual dimension in the output tensor later
        batch_size_from_labels = gt_labels_3d[0].size(0) if gt_labels_3d[0].numel() > 0 else 0
        # Fallback using gt_bboxes if labels are empty (less common)
        if batch_size_from_labels == 0 and hasattr(gt_bboxes_3d[0], 'tensor') and gt_bboxes_3d[0].tensor.numel() > 0:
             batch_size_from_labels = gt_bboxes_3d[0].tensor.size(0)
        if batch_size_from_labels == 0:
             print("Warning: Batch size determined to be 0 from labels/bboxes. Skipping aux loss.")


        # Get the predicted centers tensor [1, bs, num_q, 2] from the last (and only) decoder layer
        pred_centers_all_tasks_last_layer = []
        if outs and isinstance(outs, (list, tuple)): # outs structure is tuple(list(dict))
             num_tasks = len(outs[0]) # Number of task dictionaries in the list
             train_modalities = self.pts_bbox_head.modalities.get('train', ['fused'])
             modality_idx_to_use = -1 # Default invalid index
             # Find the index for the desired modality ('ensemble' based on logs)
             # Or use 'fused'/'bev' if specified differently and present
             target_modality = 'ensemble' # Assuming 'ensemble' is the key output modality name shown in logs
             # You might need to adjust target_modality if 'fused' or 'bev' is intended
             # For now, we trust the log showing 'ensemble' under 'modalities' key

             # NOTE: The 'outs' structure from log is tuple(list(dict))
             # The dict seems to contain ALL modalities/outputs, not nested per modality.
             # The 'modalities' key seems to just list ['ensemble'].
             # Let's assume there's effectively only one task output dict to process.
             if num_tasks != 1:
                 print(f"Warning: Expected 1 task dict in outs[0], found {num_tasks}. Processing only the first.")

             task_dict = outs[0][0] # Access the first (and likely only) task dictionary

             if isinstance(task_dict, dict) and 'center' in task_dict:
                  center_outputs = task_dict['center']
                  if isinstance(center_outputs, (list, tuple)) and len(center_outputs) > 0:
                       last_layer_center_tensor = center_outputs[-1] # Get tensor from last/only layer
                       if isinstance(last_layer_center_tensor, torch.Tensor):
                            # Store the tensor for the (single) task
                            pred_centers_all_tasks_last_layer.append(last_layer_center_tensor)
                       else:
                            pred_centers_all_tasks_last_layer.append(None)
                  elif isinstance(center_outputs, torch.Tensor): # Handle if not a list
                       pred_centers_all_tasks_last_layer.append(center_outputs)
                  else:
                       pred_centers_all_tasks_last_layer.append(None)
             else:
                 print(f"Warning: Task dict is not a dict or missing 'center': {type(task_dict)}. Cannot extract centers.")
                 pred_centers_all_tasks_last_layer.append(None) # Append None for this task

        else:
             print("Warning: 'outs' from bbox head is empty or not structured as expected (tuple(list(dict))).")
             num_tasks = 0 # Set num_tasks based on extraction result


        # Iterate through samples in the batch
        actual_batch_size = 0 # Determine actual batch size from tensor if possible
        if pred_centers_all_tasks_last_layer and pred_centers_all_tasks_last_layer[0] is not None:
             # Expect shape [1, bs, num_q, 2]
             if pred_centers_all_tasks_last_layer[0].ndim == 4:
                  actual_batch_size = pred_centers_all_tasks_last_layer[0].size(1)

        # Use the smaller of label-derived batch size and tensor-derived batch size
        batch_size = min(batch_size_from_labels, actual_batch_size) if actual_batch_size > 0 else batch_size_from_labels

        for b in range(batch_size):
            if b >= len(gt_bboxes_3d): continue
            gt_bboxes_sample_b = gt_bboxes_3d[b]
            if not hasattr(gt_bboxes_sample_b, 'gravity_center') or gt_bboxes_sample_b.gravity_center is None or gt_bboxes_sample_b.gravity_center.numel() == 0: continue
            gt_centers_b = gt_bboxes_sample_b.gravity_center[:, :2]
            num_gt_b = gt_centers_b.size(0)
            if num_gt_b == 0: continue

            loss_aux_sample_tasks = 0
            tasks_contributing = 0
            # Loop through the extracted tensors (likely just one task)
            for task_idx in range(len(pred_centers_all_tasks_last_layer)):
                 pred_center_task = pred_centers_all_tasks_last_layer[task_idx] # Tensor shape [1, bs, num_q, 2]
                 if pred_center_task is None: continue
                 if not isinstance(pred_center_task, torch.Tensor): continue

                 # *** FIX 1: Adjust shape check for 4 dimensions ***
                 # Check shape: [num_layers=1, batch_size, num_queries, features=2]
                 if not (pred_center_task.ndim == 4 and pred_center_task.size(0) == 1 and pred_center_task.size(1) == actual_batch_size and pred_center_task.size(3) == 2):
                     # Print warning only if shape is unexpected
                     # print(f"Warning: Unexpected shape for pred_center_task[{task_idx}]: {pred_center_task.shape}. Expected [1, {actual_batch_size}, num_queries, 2]. Skipping task.")
                     continue
                 # Ensure sample index b is valid for the batch dimension (dim 1)
                 if b >= pred_center_task.size(1): continue

                 # *** FIX 2: Adjust indexing for the specific sample b ***
                 # Select the data for the first layer [0] and the current sample [b]
                 pred_center_b_task = pred_center_task[0, b] # Shape: [num_queries, 2]
                 gt_centers_b = gt_centers_b.to(pred_center_b_task.device)

                 try:
                     dist = torch.cdist(pred_center_b_task, gt_centers_b, p=2)
                 except RuntimeError as e:
                     print(f"\n=== RUNTIME ERROR DEBUG (torch.cdist) ===")
                     print(f"Sample index b: {b}")
                     print(f"Task index: {task_idx}")
                     print(f"pred_center_b_task shape: {pred_center_b_task.shape}, dtype: {pred_center_b_task.dtype}, device: {pred_center_b_task.device}")
                     print(f"gt_centers_b shape: {gt_centers_b.shape}, dtype: {gt_centers_b.dtype}, device: {gt_centers_b.device}")
                     print(f"Error during torch.cdist: {e}")
                     raise e

                 assigned_gt_inds = dist.argmin(dim=-1)
                 if assigned_gt_inds.numel() > 0 and assigned_gt_inds.max().item() >= num_gt_b:
                     # print(f"Warning: Invalid index in assigned_gt_inds. Max index: {assigned_gt_inds.max().item()}, num_gt_b: {num_gt_b}. Skipping task {task_idx} for sample {b}.")
                     continue

                 assigned_gt_centers = gt_centers_b[assigned_gt_inds]

                 loss_aux = torch.nn.functional.smooth_l1_loss(
                     pred_center_b_task, assigned_gt_centers, reduction='mean'
                 )

                 if torch.isnan(loss_aux).any() or torch.isinf(loss_aux).any():
                      # print(f"Warning: NaN/Inf detected in loss_aux for task {task_idx}, sample {b}. Skipping contribution.")
                      continue

                 loss_aux_sample_tasks += loss_aux
                 tasks_contributing += 1

            if tasks_contributing > 0:
                loss_aux_total += (loss_aux_sample_tasks / tasks_contributing)
                valid_sample_count += 1

        # Average loss over valid samples
        if valid_sample_count > 0:
            loss_aux_total = loss_aux_total / valid_sample_count
        else:
            # Create zero tensor logic (keep previous version)
            # print("Warning: No valid samples contributed to the auxiliary loss for this batch.") # Keep this warning
            device = next(self.parameters()).device if next(self.parameters(), None) is not None else 'cpu'
            example_tensor = None
            # Try finding an example tensor more robustly
            def find_first_tensor(data):
                if isinstance(data, torch.Tensor): return data
                if isinstance(data, (list, tuple)):
                    for item in data:
                        found = find_first_tensor(item)
                        if found is not None: return found
                if isinstance(data, dict):
                    for val in data.values():
                        found = find_first_tensor(val)
                        if found is not None: return found
                return None
            example_tensor = find_first_tensor(outs) # Search within the original outs

            if example_tensor is not None:
                 device = example_tensor.device
                 dtype = example_tensor.dtype
                 loss_aux_total = torch.tensor(0.0, device=device, dtype=dtype)
            else: # Fallback
                 loss_aux_total = torch.tensor(0.0, device=device, dtype=torch.float32)


        # Scale auxiliary loss
        loss_aux_total = 0.1 * loss_aux_total

        # Calculate main losses
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        # Add the auxiliary loss
        if torch.isnan(loss_aux_total).any() or torch.isinf(loss_aux_total).any():
             print(f"ERROR: Final loss_aux_total is NaN or Inf! Setting to zero. Value: {loss_aux_total}")
             first_loss_tensor = next((v for v in losses.values() if isinstance(v, torch.Tensor)), None)
             device = first_loss_tensor.device if first_loss_tensor is not None else 'cpu'
             dtype = first_loss_tensor.dtype if first_loss_tensor is not None else torch.float32
             loss_aux_total = torch.tensor(0.0, device=device, dtype=dtype)

        losses['loss_aux_proposal'] = loss_aux_total
        # print(f"DEBUG: Added loss_aux_proposal: {losses['loss_aux_proposal'].item()}") # Optional final check
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)

    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]

        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
