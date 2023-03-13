import tensorflow as tf
import numpy as np

import config as cfg
import utils.util_function as uf


class SinglePositivePolicy:
    def __init__(self):
        self.anchor_ratio = cfg.ModelOutput.ANCHORS_RATIO
        self.feat_scales = cfg.ModelOutput.FEATURE_SCALES
        self.grtr_compos = cfg.ModelOutput.GRTR_FMAP_COMPOSITION
        self.num_anchor = len(self.anchor_ratio) // len(self.feat_scales)

    def __call__(self, features):
        """
        :param features: {"image": (B,H,W,3), "inst": (B,N,5+K)}
        :return: {"image": (B,H,W,3), "inst": {"whole": (B,N,5+K), "yxhw": (B,N,4), ...},
                  "fmap": {"whole": (B,GH,GW,A,5+K), "yxhw": (B,GH*GW*A,4), ...}}
        """
        imshape = features["image"].shape[1:3]
        feat_shapes = [np.array(imshape) // scale for scale in self.feat_scales]
        new_inst = {"whole": features["inst"]}
        new_inst.update(uf.slice_feature(features["inst"], self.grtr_compos))
        # channel = 0
        # for key, depth in self.grtr_compos.items():
        #     new_inst[key] = features["inst"][:, channel:channel+depth]
        #     channel += depth

        bbox_hw = new_inst["yxhw"][..., np.newaxis, 2:4]                # (B, N, 1, 2)
        anchors_hw = self.anchor_ratio[np.newaxis, np.newaxis, :, :]    # (1, 1, 9, 2)
        inter_hw = np.minimum(bbox_hw, anchors_hw)                      # (B, N, 9, 2)
        inter_area = inter_hw[..., 0] * inter_hw[..., 1]                # (B, N, 9)
        union_area = bbox_hw[..., 0] * bbox_hw[..., 1] + anchors_hw[..., 0] * anchors_hw[..., 1] - inter_area
        iou = inter_area / union_area
        best_anchor_indices = np.argmax(iou, axis=-1)
        batch, channel = bbox_hw.shape[0], new_inst["whole"].shape[-1]
        feat_map = [np.zeros((batch, feat_shape[0], feat_shape[1], self.num_anchor, channel), dtype=np.float32)
                    for feat_shape in feat_shapes]
        
        for batch in range(batch):
            for anchor_index, inst in zip(best_anchor_indices[batch], new_inst["whole"][batch]):
                if np.all(inst == 0):
                    break
                scale_index = anchor_index // self.num_anchor
                anchor_index_in_scale = anchor_index % self.num_anchor
                # inst: [y, x, h, w, object, category]
                grid_yx = tf.cast(inst[:2] * feat_shapes[scale_index], dtype=tf.int32)
                assert tf.reduce_all(grid_yx >= 0) and tf.reduce_all(grid_yx < feat_shapes[scale_index])
                feat_map[scale_index][batch, grid_yx[0], grid_yx[1], anchor_index_in_scale] = inst

        feat_slices = uf.slice_features_and_merge_dims(feat_map, self.grtr_compos)
        feat_map = {"whole": feat_map}
        feat_map.update(feat_slices)
        features["inst"] = new_inst
        features["fmap"] = feat_map
        return features
