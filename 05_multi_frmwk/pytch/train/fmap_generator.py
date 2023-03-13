import torch
import numpy as np

import config as cfg
import pytch.utils.util_function as puf


class SinglePositivePolicy:
    def __init__(self):
        self.anchor_ratio = cfg.ModelOutput.ANCHORS_RATIO
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

        bbox_hw = new_inst["yxhw"][:, np.newaxis, 2:4]  # (B, 1, 2, N)
        anchors_hw = self.anchor_ratio[np.newaxis, np.newaxis]  # (1, 9, 2, 1)
        inter_hw = np.minimum(bbox_hw, anchors_hw)  # (B, 9, 2, N)
        inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]  # (B, 9, N)
        # (B, 1, N) + (1, 9, 1) = (B, 9, N)
        union_area = bbox_hw[:, :, 0] * bbox_hw[:, :, 1] + anchors_hw[:, :, 0, 0] * anchors_hw[:, :, 1, 0] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=-1)   # (B, 9, N) -> (B, 9)
        batch, channel = bbox_hw.shape[0], new_inst["whole"].shape[1]
        feat_map = [np.zeros((batch, channel, self.num_anchor, fmap_hw[0], fmap_hw[1]), dtype=np.float32)
                    for fmap_hw in feat_shapes]

        for batch in range(batch):
            for anchor_index, inst in zip(best_anchor_indices[batch], new_inst["whole"][batch]):
                if np.all(inst == 0):
                    break
                scale_index = anchor_index // self.num_anchor
                anchor_index_in_scale = anchor_index % self.num_anchor
                # inst: [y, x, h, w, object, category]
                grid_yx = (inst[:2] * feat_shapes[scale_index]).type(torch.int32)
                assert torch.all(grid_yx >= 0) and torch.all(grid_yx < feat_shapes[scale_index])
                feat_map[scale_index][batch, grid_yx[0], grid_yx[1], anchor_index_in_scale] = inst

        feat_slices = puf.slice_features_and_merge_dims(feat_map, self.grtr_compos)
        feat_map = {"whole": feat_map}
        feat_map.update(feat_slices)
        features["inst"] = new_inst
        features["fmap"] = feat_map
        return features
