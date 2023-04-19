import torch
import torchvision as tv

import config as cfg
import pytch.utils.util_function as puf


class DetectorPostProcess:
    def __init__(self, anchors_ratio = cfg.ModelOutput.ANCHORS_RATIO,
                 num_scales = len(cfg.ModelOutput.FEATURE_SCALES)):
        self.num_anchors_per_scale = cfg.ModelOutput.NUM_ANCHORS_PER_SCALE
        # (9,2) -> (3,3,2)
        anchors_per_scale = [anchors_ratio[i * self.num_anchors_per_scale:(i + 1) * self.num_anchors_per_scale] for i in range(num_scales)]
        self.decoder = FeatureDecoder(anchors_per_scale)

    def __call__(self, feat_logit):
        feat_logit = self.insert_anchor_dimension(feat_logit)
        head_logit = [val for key, val in feat_logit.items() if 'head' in key]
        head_slices = puf.slice_features_and_merge_dims(head_logit, cfg.ModelOutput.PRED_FMAP_COMPOSITION)
        decode_slices = self.decoder(head_slices, head_logit)
        # TODO: add "bkbn_logit": bkbn_features
        features = {"head_logit": head_logit}
        features.update(decode_slices)
        output = {"fmap": features}
        # TODO: NMS
        return output

    def insert_anchor_dimension(self, feat_logit):
        new_logit = {}
        for key, logit in feat_logit.items():
            batch, channel, height, width = logit.shape
            out_channel = channel // self.num_anchors_per_scale
            logit = torch.reshape(logit, (batch, out_channel, self.num_anchors_per_scale, height, width))
            new_logit[key] = logit
        return new_logit


class FeatureDecoder:
    def __init__(self, anchors_per_scale):
        """
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.anchors_per_scale = anchors_per_scale
        self.const_3 = torch.tensor(3, dtype=torch.float32)
        self.const_log_2 = torch.log(torch.tensor(2, dtype=torch.float32))

    def __call__(self, slices, features):
        """
        :param slices: sliced head feature maps {"yxhw": [(batch, channel, anchor*grid_h*grid_w) x 3], ...}
        :param features: whole head feature maps [(batch, channel, anchor*grid_h*grid_w) x 3]
        :return: decoded feature in the same shape e.g. {"yxhw": [...], "object": [...], "category": [...]}
        """
        decoded = {key: [] for key in slices.keys()}
        for si, anchors_ratio in enumerate(self.anchors_per_scale):
            box_yx = self.decode_yx(slices["yxhw"][si][:, :2], features[si].shape)
            box_hw = self.decode_hw(slices["yxhw"][si][:, 2:], anchors_ratio, features[si].shape)
            decoded["yxhw"].append(torch.cat([box_yx, box_hw], dim=1))
            decoded["object"].append(torch.sigmoid(slices["object"][si]))
            decoded["category"].append(torch.sigmoid(slices["category"][si]))
        return decoded

    def decode_yx(self, yx_raw, feat_shape):
        """
        :param yx_raw: (batch, 2, anchor*grid_h*grid_w)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, 2, anchor*grid_h*grid_w)
        
        Original yolo v3 implementation: yx_dec = torch.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw 
        """
        batch, channel, num_anc, grid_h, grid_w = feat_shape
        # grid_x: (grid_h, grid_w)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, grid_h), torch.arange(0, grid_w))
        # grid: (2, 1, grid_h, grid_w)
        grid = torch.unsqueeze(torch.stack([grid_y, grid_x], dim=0), dim=1)
        divider = torch.reshape(torch.tensor([grid_h, grid_w], dtype=torch.float32), (2, 1, 1, 1))

        yx_5d = torch.reshape(yx_raw, (batch, 2, num_anc, grid_h, grid_w))
        yx_box = torch.sigmoid(yx_5d) * 1.4 - 0.2
        # [(batch, 2, anchor, grid_h, grid_w) + (2, 1, grid_h, grid_w)] / (2, 1, 1, 1)
        yx_dec = (yx_box + grid) / divider
        yx_dec = torch.reshape(yx_dec, (batch, 2, -1))
        return yx_dec

    def decode_hw(self, hw_raw, anchors_ratio, feat_shape):
        """
        :param hw_raw: (batch, 2, anchor*grid_h*grid_w)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, 2, anchor*grid_h*grid_w)
        """
        batch, channel, num_anc, grid_h, grid_w = feat_shape
        anchors_tensor = torch.tensor(anchors_ratio)            # (anchor, 2)
        anchors_tensor = torch.transpose(anchors_tensor, 0, 1)  # (2, anchor)
        anchors_tensor = torch.reshape(anchors_tensor, (2, num_anc, 1, 1))
        hw_5d = torch.reshape(hw_raw, (batch, 2, num_anc, grid_h, grid_w))
        # NOTE: exp activation may result in infinity
        # hw_dec = torch.exp(hw_raw) * anchors_tensor      (batch, 2, anchor, grid_h, grid_w) * (2, anchor, 1, 1)
        # hw_dec: 0~3 times of anchor, the delayed sigmoid passes through (0, 1)
        hw_dec = self.const_3 * torch.sigmoid(hw_5d - self.const_log_2) * anchors_tensor
        hw_dec = torch.reshape(hw_dec, (batch, 2, -1))
        return hw_dec


class NonMaximumSuppressionBox:
    def __init__(self, max_out=cfg.NmsInfer.MAX_OUT,
                 iou_thresh=cfg.NmsInfer.IOU_THRESH,
                 score_thresh=cfg.NmsInfer.SCORE_THRESH,
                 ):
        self.max_out = max_out
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.num_top = 300

    def __call__(self, pred):
        """
        :param pred: {'yxhw': [(batch, N, 4)]x3, 'object': ..., 'category': ...}
        :return: (batch, 6, max_out), 6: bbox, score, category
        """
        pred = {key: torch.cat(data, dim=2) for key, data in pred.items() if isinstance(data, list) and not key.endswith("_logit")}
        boxes = torch.transpose(puf.convert_box_format_yxhw_to_tlbr(pred["yxhw"]), 1, 2)   # (batch, N, 4)
        categories = torch.argmax(pred["category"], dim=1)      # (batch, N)
        best_probs = torch.max(pred["category"], dim=1)         # (batch, N)
        objectness = pred["object"][:, 0]                       # (batch, N)
        scores = objectness * best_probs                        # (batch, N)
        batch, numctgr, numbox = pred["category"].shape

        batch_indices = [[] for i in range(batch)]
        for ctgr_idx in range(numctgr):
            ctgr_mask = (categories == ctgr_idx).type(torch.float32)    # (batch, N)
            score_mask = (scores > self.score_thresh[ctgr_idx]).type(torch.float32) # (batch, N)
            ctgr_boxes = boxes * ctgr_mask[..., None] * score_mask      # (batch, N, 4)
            ctgr_scores = scores * ctgr_mask * score_mask               # (batch, N)

            for frame_idx in range(batch):
                top_scores, top_indices = torch.topk(ctgr_scores[frame_idx], self.num_top) # (N,) -> (M,)
                boxes_in_frame = ctgr_boxes[frame_idx, top_indices]     # (M, 4)
                selected_indices = tv.ops.nms(                          # (M,) -> (L,)
                    boxes=boxes_in_frame,
                    scores=ctgr_scores[frame_idx],
                    iou_threshold=self.iou_thresh[ctgr_idx],
                )
                selected_indices = selected_indices[:self.max_out[ctgr_idx]]
                padlen = self.max_out[ctgr_idx] - selected_indices.shape[0]
                selected_indices = torch.nn.functional.pad(selected_indices, (0, padlen), mode='constant', value=-1)
                batch_indices[frame_idx].append(selected_indices)       # (B, K, max_out)

        # make batch_indices, valid_mask as fixed shape tensor
        batch_indices = [torch.cat(ctgr_indices, dim=0) for ctgr_indices in batch_indices]  # (B, K*max_out)
        batch_indices = torch.stack(batch_indices, dim=0)       # (batch, K*max_output)
        valid_mask = (batch_indices >= 0).type(torch.float32)   # (batch, K*max_output)
        batch_indices = torch.maximum(batch_indices, 0)

        categories = categories.type(torch.float32)
        # "bbox": 4, "object": 1, "category": 1
        result = torch.stack([scores, categories], dim=-1)      # (B, N) x2 -> (B, N, 2)
        result = torch.cat([pred["yxhw"], result], dim=-1)      # (B, N, 6)
        result = torch.gather(result, dim=1, index=batch_indices)    # (batch, K*max_output, 6)
        result = result * valid_mask[..., None]                 # (batch, K*max_output, 6)
        return result
