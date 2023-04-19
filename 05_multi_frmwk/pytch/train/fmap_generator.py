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
        :param features: {"image": (B,3,H,W), "inst": (B,5+K,N)}
        :return: {"image": (B,3,H,W), "inst": {"whole": (B,5+K,N), "yxhw": (B,4,N), ...},
                  "fmap": {"whole": (B,5+K,A,GH,GW), "yxhw": (B,4,A*GH*GW), ...}}
        """
        imshape = features["image"].shape[2:]
        feat_shapes = [np.array(imshape) // scale for scale in self.feat_scales]
        new_inst = {"whole": features["inst"]}
        new_inst.update(puf.slice_feature(features["inst"], self.grtr_compos))

        bbox_hw = new_inst["yxhw"][:, 2:4, :, None]     # (B, 2, N, 1)
        anchors_hw = torch.transpose(self.anchor_ratio, 0, 1)[:, None, :]   # (2, 1, 9)
        inter_hw = torch.minimum(bbox_hw, anchors_hw)   # (B, 2, N, 9)
        inter_area = inter_hw[:, 0] * inter_hw[:, 1]    # (B, N, 9)
        # (B, N, 1) + (1, 9) = (B, N, 9)
        union_area = bbox_hw[:, 0] * bbox_hw[:, 1] + anchors_hw[0] * anchors_hw[1] - inter_area   # (B, N, 9)
        iou = inter_area / union_area
        best_anchor_indices = torch.argmax(iou, dim=-1)   # (B, N, 9) -> (B, N)
        batch, channel, num_inst = new_inst["whole"].shape
        feat_map = [torch.zeros((batch, self.num_anchor, fmap_hw[0], fmap_hw[1], channel), dtype=torch.float32)
                    for fmap_hw in feat_shapes]

        scale_index = best_anchor_indices // self.num_anchor    # (B, N)
        anchor_index = best_anchor_indices % self.num_anchor    # (B, N)
        valid_inst_mask = features["inst"][:, 2] > 0            # (B, N)
        frame_ind = np.arange(0, batch)[:, np.newaxis] * np.ones((batch, num_inst))  # (B, 1) * (B, N)
        for scale, fmap in enumerate(feat_map):
            scale_mask = (scale_index == scale) * valid_inst_mask   # (B, N)
            anchor_index_in_scale = anchor_index * scale_mask       # (B, N)
            anchor_index_in_scale = puf.convert_to_numpy(anchor_index_in_scale).astype(np.int32)
            fmap_hw = torch.tensor(fmap.shape[2:4])
            # (B, 2, N) * (B, 1, N) * (2, 1) = (B, 2, N)
            grid_yx = features["inst"][:, :2] * torch.unsqueeze(scale_mask, 1) * torch.unsqueeze(fmap_hw, 1)
            grid_yx = puf.convert_to_numpy(grid_yx).astype(np.int32)
            scale_feature = features["inst"] * torch.unsqueeze(scale_mask, 1)   # (B, C, N)
            scale_feature = torch.transpose(scale_feature, 1, 2)
            fmap[frame_ind, anchor_index_in_scale, grid_yx[:, 0], grid_yx[:, 1]] = scale_feature # (B, A, H, W, C)
            # print("objc", torch.sum(fmap[..., 4]), torch.sum(scale_mask))
            feat_map[scale] = torch.transpose(fmap, -1, 1)  # (B, C, A, H, W)

        feat_slices = puf.slice_features_and_merge_dims(feat_map, self.grtr_compos)
        feat_map = {"whole": feat_map}
        feat_map.update(feat_slices)
        features["inst"] = new_inst
        features["fmap"] = feat_map
        return features
