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
        # Handle None features before passing to head
        _pts_feats = pts_feats
        _img_feats = img_feats

        # Step 1: Extract outputs from the bbox head
        outs = self.pts_bbox_head(_pts_feats, _img_feats, img_metas)


        # --- Auxiliary loss on decoded centers ---
        loss_aux_total = 0.0
        valid_sample_count = 0

        # Basic validation of GT inputs
        if not gt_labels_3d or not gt_bboxes_3d or len(gt_labels_3d) == 0 or len(gt_bboxes_3d) == 0:
             #print("Warning: GT labels or bboxes are None or empty. Skipping aux loss calculation for the batch.")
             device = next(self.parameters()).device if next(self.parameters(), None) is not None else 'cpu'
             example_tensor = None
             # Try getting dtype from outs structure if valid, or fallback to model parameters
             if outs and isinstance(outs, (list, tuple)) and len(outs) > 0 and \
                isinstance(outs[-1], (list, tuple)) and len(outs[-1]) > 0 and \
                isinstance(outs[-1][0], dict) and 'center' in outs[-1][0] and \
                isinstance(outs[-1][0]['center'], (list, tuple)) and len(outs[-1][0]['center']) > 0 and \
                isinstance(outs[-1][0]['center'][-1], torch.Tensor): # Check access path to the tensor
                 example_tensor = outs[-1][0]['center'][-1] # Access the tensor


             dtype = example_tensor.dtype if example_tensor is not None else torch.float32
             loss_aux_averaged = torch.tensor(0.0, device=device, dtype=dtype)


        else: # Only proceed with aux loss calculation if GT is valid
            batch_size = len(gt_bboxes_3d)

            # Step 2: Correctly extract the predicted centers tensor from the LAST decoder layer, FIRST task, LAST PME layer
            # The tensor is expected to have shape [1, batch_size, num_queries, 2]
            pred_centers_tensor = None
            try:
                # Access output from the last main decoder layer (e.g., layer 5 if 6 layers total)
                if isinstance(outs, (list, tuple)) and len(outs) > 0:
                    last_layer_outputs = outs[-1] # This should be a list of task output dicts

                    # Access the dictionary for the first task head from the last layer's outputs
                    if isinstance(last_layer_outputs, (list, tuple)) and len(last_layer_outputs) > 0:
                        first_task_outputs = last_layer_outputs[0] # Assuming the first task head corresponds to the main task.

                        # Access the list of center prediction tensors from the PME layers
                        if isinstance(first_task_outputs, dict) and 'center' in first_task_outputs:
                             center_list_from_head = first_task_outputs['center'] # <-- This is the list of tensors from PME layers

                             # Get the LAST tensor from this list (output of the last PME layer, which is likely the only one)
                             if isinstance(center_list_from_head, (list, tuple)) and len(center_list_from_head) > 0:
                                 # <<<--- CORRECTED ACCESS TO GET THE TENSOR FROM THE LIST ---
                                 temp_tensor = center_list_from_head[-1] # <<<--- temp_tensor is the tensor [1, batch_size, num_queries, 2]

                                 # Validate the shape of the extracted tensor - Expecting [1, batch_size, num_queries, 2]
                                 if isinstance(temp_tensor, torch.Tensor) and temp_tensor.ndim == 4 and temp_tensor.size(0) == 1 and temp_tensor.size(3) == 2:
                                      pred_centers_tensor = temp_tensor # Assign the valid tensor
                                 else:
                                      print(f"Error: Unexpected predicted centers tensor shape after extraction: {temp_tensor.shape if isinstance(temp_tensor, torch.Tensor) else 'N/A'}. Expected [1, batch_size, num_queries, 2]. Skipping aux loss calculation for this batch.")
                             else:
                                print(f"Warning: 'center' value is not a list/tuple or is empty for the first task output from last layer. Skipping aux loss calculation.")
                        else:
                             print(f"Warning: First task output from last layer is not a dict or missing 'center' key. Skipping aux loss calculation.")
                    else:
                        print(f"Warning: Last layer outputs from bbox head are empty or not a list/tuple. Skipping aux loss calculation.")
                else:
                     print(f"Warning: 'outs' from bbox head is empty or not structured as expected (not tuple/list). Skipping aux loss calculation.")

            except (IndexError, KeyError, TypeError) as e:
                 print(f"\n=== Aux Loss Extraction ERROR Exception: {e} ===")
                 print("Could not extract predicted centers from outs structure. Skipping aux loss calculation.")
                 pred_centers_tensor = None


            if pred_centers_tensor is not None: # Only proceed if the tensor was extracted successfully
                # Iterate through samples using the batch dimension (index 1) of the [1, bs, N, 2] tensor
                actual_batch_size_preds = pred_centers_tensor.size(1)
                loop_batch_size = min(batch_size, actual_batch_size_preds) if actual_batch_size_preds > 0 else batch_size


                for b in range(loop_batch_size):
                    gt_bboxes_sample_b = gt_bboxes_3d[b]
                    if not hasattr(gt_bboxes_sample_b, 'gravity_center') or gt_bboxes_sample_b.gravity_center is None or gt_bboxes_sample_b.gravity_center.numel() == 0:
                        continue

                    gt_centers_b = gt_bboxes_sample_b.gravity_center[:, :2] # Shape [num_gt, 2]
                    num_gt_b = gt_centers_b.size(0)
                    if num_gt_b == 0:
                        continue

                    # Get predicted centers for sample b. Shape: [num_queries, 2]
                    # Indexing batch dimension (index 1) from the [1, bs, N, 2] tensor, discarding dim 0.
                    pred_q_b = pred_centers_tensor[0, b] # <<<--- CORRECTED BATCH INDEXING

                    num_q_b = pred_q_b.size(0)
                    if num_q_b == 0:
                         continue

                    gt_centers_b = gt_centers_b.to(pred_q_b.device)

                    # --- Assignment logic (closest GT for each query) ---
                    try:
                        dist = torch.cdist(pred_q_b, gt_centers_b, p=2)
                        assigned_gt_inds = dist.argmin(dim=-1)

                        if assigned_gt_inds.numel() > 0 and assigned_gt_inds.max().item() >= num_gt_b:
                            print(f"Warning: Invalid index in assigned_gt_inds for sample {b}. Max index: {assigned_gt_inds.max().item()}, num_gt_b: {num_gt_b}. Skipping contribution.")
                            continue

                        assigned_gt_centers = gt_centers_b[assigned_gt_inds]

                    except RuntimeError as e:
                        print(f"\n=== RUNTIME ERROR DEBUG (torch.cdist or indexing) for sample {b} ===")
                        print(f"pred_q_b shape: {pred_q_b.shape}, dtype: {pred_q_b.dtype}, device: {pred_q_b.device}")
                        print(f"gt_centers_b shape: {gt_centers_b.shape}, dtype: {gt_centers_b.dtype}, device: {gt_centers_b.device}")
                        print(f"Error during distance calculation or indexing: {e}")
                        continue


                    loss_aux_sample = F.smooth_l1_loss(
                        pred_q_b, assigned_gt_centers, reduction='mean'
                    )

                    if torch.isnan(loss_aux_sample).any() or torch.isinf(loss_aux_sample).any():
                         print(f"Warning: NaN/Inf detected in loss_aux_sample for sample {b}. Value: {loss_aux_sample.item()}. Skipping contribution.")
                         continue

                    loss_aux_total += loss_aux_sample
                    valid_sample_count += 1

            if valid_sample_count > 0:
                 loss_aux_averaged = loss_aux_total / valid_sample_count
            else:
                 device = pred_centers_tensor.device if pred_centers_tensor is not None else (next(self.parameters()).device if next(self.parameters(), None) is not None else 'cpu')
                 dtype = pred_centers_tensor.dtype if pred_centers_tensor is not None else torch.float32
                 loss_aux_averaged = torch.tensor(0.0, device=device, dtype=dtype)


        # --- Step 4: Apply warm-up scaling to the averaged auxiliary loss ---
        # HARDCODED AUX LOSS PARAMETERS
        # Adjust these values directly in the code for your experiment
        hardcoded_total_epochs = 12  # <<<--- Set your desired TOTAL epochs for THIS RUN (Matches config log)
        hardcoded_warmup_epochs = 2 # <<<--- Set your desired warm-up duration in epochs (e.g., 2, 3, or 5)
        hardcoded_max_aux_weight = 0.1 # <<<--- Set your desired max aux loss weight (0.1 or 0.05)


        # Get current epoch number (assuming 1-indexed from logger/runner, as seen in logs)
        # img_metas[0].get('epoch', -1) will return the 1-indexed epoch number if present, -1 otherwise
        current_epoch_1_indexed = img_metas[0].get('epoch', -1)

        scale_factor = 0.0 # Default scale is 0

        # Calculate linear warm-up scale factor based on 1-indexed epoch
        # Scale goes from 0 at the start of Epoch 1 to 1.0 at the start of Epoch (hardcoded_warmup_epochs + 1)
        if current_epoch_1_indexed >= 1 and hardcoded_warmup_epochs > 0:
             # Progress through warm-up: (Current_Epoch - 1) / Warmup_Epochs
             # Use max(0.0, ...) in case current_epoch_1_indexed is 1 and warmup_epochs is 0 (though checks prevent this)
             alpha = float(current_epoch_1_indexed - 1) / hardcoded_warmup_epochs
             scale_factor = min(alpha, 1.0) # Cap scale at 1.0 after warm-up is complete
        elif current_epoch_1_indexed > hardcoded_warmup_epochs:
             # After warm-up epochs are complete, scale factor is 1.0
             scale_factor = 1.0
        # Note: If current_epoch is -1 (key missing) or hardcoded_warmup_epochs <= 0, scale_factor remains 0.0

        # Apply the scale factor and max weight
        scaled_loss_aux = loss_aux_averaged * scale_factor * hardcoded_max_aux_weight


        # Compute main losses using the original mechanism of the head
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)


        # Step 6: Safety check for NaNs/Infs in main loss
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any()):
                print(f"Warning: NaN or Inf detected in main loss key '{k}', setting to zero. Value: {v.item()}")
                losses[k] = torch.tensor(0.0, device=v.device, dtype=v.dtype)


        # Merge the scaled auxiliary loss
        if torch.isnan(scaled_loss_aux).any() or torch.isinf(scaled_loss_aux).any():
             print(f"ERROR: Final scaled_loss_aux is NaN or Inf! Value: {scaled_loss_aux.item()}. Setting to zero.")
             first_loss_tensor = next((v for v in losses.values() if isinstance(v, torch.Tensor)), None)
             device = first_loss_tensor.device if first_loss_tensor is not None else 'cpu'
             dtype = first_loss_tensor.dtype if first_loss_tensor is not None else torch.float32
             losses['loss_aux_proposal'] = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            losses['loss_aux_proposal'] = scaled_loss_aux
            # Optional final check - helps debug warm-up scale
            # print(f"DEBUG: Added loss_aux_proposal: {losses['loss_aux_proposal'].item()} (scale: {scale_factor * hardcoded_max_aux_weight})")


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
