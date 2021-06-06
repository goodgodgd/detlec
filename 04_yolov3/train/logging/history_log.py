import numpy as np
import tensorflow as tf
import pandas as pd
from timeit import default_timer as timer

from train.logging.metric import count_true_positives


class HistoryLog:
    def __init__(self):
        self.batch_data_table = pd.DataFrame()
        self.start = timer()
        self.summary = dict()

    def __call__(self, step, grtr, pred, total_loss, loss_by_type):
        """
        :param step: integer step index
        :param grtr: slices of GT data {'image': (B,H,W,3), 'bboxes': (B,N,6), 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param pred: slices of pred. data {'nms': (B,M,8), 'feature_l': {'bbox': (B,HWA,4), ...}, ...}
        :param total_loss:
        :param loss_by_type:
        """
        loss_list = [loss_name for loss_name, loss_tensor in loss_by_type.items() if loss_tensor.ndim == 0]
        batch_data = {loss_name: loss_by_type[loss_name].numpy() for loss_name in loss_list}
        batch_data["total_loss"] = total_loss.numpy()

        objectness = self.analyze_objectness(grtr, grtr)
        batch_data.update(objectness)

        num_ctgr = pred["feature_l"]["category"].shape[-1]
        metric = count_true_positives(grtr["bboxes"], pred["bboxes"], num_ctgr)
        batch_data.update(metric)

        batch_data = self.set_precision(batch_data, 5)
        col_order = list(batch_data.keys())
        self.batch_data_table = self.batch_data_table.append(batch_data, ignore_index=True)
        self.batch_data_table = self.batch_data_table.loc[:, col_order]

        if step % 200 == 10:
            print("\n--- batch_data:", batch_data)
        #     self.check_pred_scales(pred)

    def analyze_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        scales = [key for key in grtr if "feature_" in key]
        for scale_name in scales:
            grtr_obj_mask = grtr[scale_name]["object"]      # (batch, HWA, 1)
            pred_obj_prob = pred[scale_name]["object"]      # (batch, HWA, 1)
            obj_num = tf.maximum(tf.reduce_sum(grtr_obj_mask), 1)
            # average positive objectness probability
            pos_obj += tf.reduce_sum(grtr_obj_mask * pred_obj_prob) / obj_num
            # average top 50 negative objectness probabilities per frame
            neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
            neg_obj_map = tf.squeeze(neg_obj_map)
            neg_obj_map = tf.sort(neg_obj_map, axis=-1, direction="DESCENDING")
            neg_obj_map = neg_obj_map[:, :50]
            neg_obj += tf.reduce_mean(neg_obj_map)
        objectness = {"pos_obj": pos_obj.numpy() / len(scales), "neg_obj": neg_obj.numpy() / len(scales)}
        return objectness

    def check_pred_scales(self, pred):
        raw_features = {key: tensor for key, tensor in pred.items() if key.endswith("raw")}
        pred_scales = dict()
        for key, feat in raw_features.items():
            pred_scales[key] = np.quantile(feat.numpy(), np.array([0.05, 0.5, 0.95]))
        print("--- pred_scales:", pred_scales)

    def set_precision(self, logs, precision):
        new_logs = {key: np.around(val, precision) for key, val in logs.items()}
        return new_logs

    def make_summary(self):
        mean_result = self.batch_data_table.mean(axis=0).to_dict()
        sum_result = self.batch_data_table.sum(axis=0).to_dict()
        sum_result = {"recall": sum_result["trpo"] / (sum_result["grtr"] + 1e-5),
                      "precision": sum_result["trpo"] / (sum_result["pred"] + 1e-5)}
        metric_keys = ["trpo", "grtr", "pred"]
        summary = {key: val for key, val in mean_result.items() if key not in metric_keys}
        summary.update(sum_result)
        summary["time_m"] = round((timer() - self.start ) /60., 5)
        print("epoch summary:", summary)
        self.summary = summary

    def get_summary(self):
        return self.summary
