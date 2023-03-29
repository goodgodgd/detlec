import torch
import numpy as np

import config as cfg
import pytch.utils.util_function as puf


class SinglePositivePolicy:
    def __init__(self):
        self.anchor_ratio = torch.from_numpy(cfg.ModelOutput.ANCHORS_RATIO).to(puf.device())
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.grtr_compos = cfg.ModelOutput.GRTR_FMAP_COMPOSITION
        self.num_anchor = len(self.anchor_ratio) // len(self.feat_scales)

    def __call__(self, features):
        """
        :param features: {"image": (B,3,H,W), "inst": (B,N,5+K)}
        :return: {"image": (B,3,H,W), "inst": {"whole": (B,N,5+K), "yxhw": (B,N,4), ...},
                  "fmap": {"whole": (B,5+K,A,GH,GW), "yxhw": (B,4,A*GH*GW), ...}}
        """
        imshape = features["image"].shape[2:]
        feat_shapes = [np.array(imshape) // scale for scale in self.feat_scales]
        new_inst = {"whole": features["inst"]}
        new_inst.update(puf.slice_feature(features["inst"], self.grtr_compos, dim=-1))

        bbox_hw = new_inst["yxhw"][:, :, None, 2:4]     # (B, N, 1, 2)
        anchors_hw = self.anchor_ratio                  # (9, 2)
        inter_hw = torch.minimum(bbox_hw, anchors_hw)      # (B, N, 9, 2)
        inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (B, N, 9)
        # (B, 1, N) + (1, 9, 1) = (B, 9, N)
        union_area = bbox_hw[..., 0] * bbox_hw[..., 1] + anchors_hw[..., 0] * anchors_hw[..., 1] - inter_area   # (B, N, 9)
        iou = inter_area / union_area
        best_anchor_indices = torch.argmax(iou, dim=-1)   # (B, N, 9) -> (B, N)
        batch, num_inst, channel = new_inst["whole"].shape
        feat_map = [torch.zeros((batch, fmap_hw[0], fmap_hw[1], self.num_anchor, channel), dtype=torch.float32)
                    for fmap_hw in feat_shapes]

        scale_index = best_anchor_indices // self.num_anchor    # (B, N)
        anchor_index = best_anchor_indices % self.num_anchor    # (B, N)
        valid_inst_mask = features["inst"][..., 2] > 0
        for scale, fmap in enumerate(feat_map):
            scale_mask = (scale_index == scale) * valid_inst_mask   # (B, N)
            anchor_index_in_scale = anchor_index * scale_mask       # (B, N)
            anchor_index_in_scale = puf.convert_to_numpy(anchor_index_in_scale).astype(np.int32)
            fmap_hw = torch.tensor(fmap.shape[1:3])
            grid_yx = features["inst"][..., :2] * torch.unsqueeze(scale_mask, 2) * fmap_hw  # (B, N, 2) * (B, N, 1) * 2
            grid_yx = puf.convert_to_numpy(grid_yx).astype(np.int32)
            frame_ind = np.arange(0, batch)[:, np.newaxis] * np.ones((batch, num_inst)) # (B, 1) * (B, N)
            scale_feature = features["inst"] * torch.unsqueeze(scale_mask, 2)
            fmap[frame_ind, grid_yx[..., 0], grid_yx[..., 1], anchor_index_in_scale] = scale_feature # (B, H, W, A, C)
            # print("objc", torch.sum(fmap[..., 4]), torch.sum(scale_mask))
            feat_map[scale] = torch.permute(fmap, (0, 4, 3, 1, 2))

        feat_slices = puf.slice_features_and_merge_dims(feat_map, self.grtr_compos)
        feat_map = {"whole": feat_map}
        feat_map.update(feat_slices)
        features["inst"] = new_inst
        features["fmap"] = feat_map
        return features
