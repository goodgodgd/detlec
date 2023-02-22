import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import neutr.utils.util_function as uf
import tflow.model.model_util as mu
import neutr.log.metric as mt
import config as cfg


class HistoryLog:
    def __init__(self, epoch, ckpt_path, is_train):
        self.epoch = epoch
        self.history_file = op.join(ckpt_path, "history.csv")
        self.is_train = is_train
        self.log_df = pd.DataFrame()
        self.valid_log = pd.DataFrame()
        self.start = timer()
        self.summary = dict()
        self.nms_box = mu.NonMaximumSuppressionBox()
        self.num_ctgr = len(cfg.Tfrdata.CATEGORY_NAMES)

    def __call__(self, step, grtr, pred, total_loss, loss_by_type):
        batch_data = {loss_name: loss_tensor for loss_name, loss_tensor in loss_by_type.items() if not isinstance(loss_tensor, list)}
        metric_counts = mt.count_true_positives(grtr["inst"], pred["inst"], self.num_ctgr)
        batch_data.update(metric_counts)
        objectness = self.analyze_objectness(grtr, pred)
        batch_data.update(objectness)
        batch_data["total_loss"] = total_loss
        batch_data = {key: np.around(val, 5) for key, val in batch_data.items()}

        self.log_df = pd.concat([self.log_df, pd.DataFrame([batch_data])], axis=0, ignore_index=True)
        if step % 200 == 10:
            print("\n--- batch_data:", batch_data)

    def analyze_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        num_scales = len(pred["fmap"]["object"])
        for grtr_obj_mask, pred_obj_prob in zip(grtr["fmap"]["object"], pred["fmap"]["object"]):
            obj_num = np.maximum(np.sum(grtr_obj_mask), 1)
            # average positive objectness probability
            pos_obj += np.sum(grtr_obj_mask * pred_obj_prob) / obj_num
            # average top 20 negative objectness probabilities per frame
            neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
            neg_obj_map = np.squeeze(neg_obj_map, axis=-1)
            neg_obj_map = np.sort(neg_obj_map, axis=-1)[:, ::-1]
            neg_obj_map = neg_obj_map[:, :20]
            neg_obj += np.mean(neg_obj_map)
        
        objectness = {"pos_obj": pos_obj / num_scales, "neg_obj": neg_obj / num_scales}
        return objectness

    def finalize(self):
        summary = self.make_summary()
        print("epoch summary:", summary)
        self.update_csv_file(summary)

    def make_summary(self):
        result_mean = self.log_df.mean(axis=0).to_dict()
        result_sum = self.log_df.sum(axis=0).to_dict()
        summary = {key: val for key, val in result_mean.items() if key not in ["trpo", "grtr", "pred"]}
        metric = {"recall": result_sum["trpo"] / (result_sum["grtr"] + 1e-5),
                  "precision": result_sum["trpo"] / (result_sum["pred"] + 1e-5)
                  }
        summary.update(metric)
        summary["time_m"] = round((timer() - self.start)/60., 5)
        return summary

    def update_csv_file(self, summary_in):
        if self.is_train:
            epoch_summary = {"epoch": self.epoch}
            epoch_summary.update({"!" + key: val for key, val in summary_in.items()})
            epoch_summary.update({"|": "|"})
            if op.isfile(self.history_file):
                history = pd.read_csv(self.history_file, encoding='utf-8', converters={'epoch': lambda c: int(c)})
                history = pd.concat([history, pd.DataFrame([epoch_summary])], axis=0, ignore_index=True)
            else:
                history = pd.DataFrame([epoch_summary])
        else:
            history = pd.read_csv(self.history_file, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            for key, val in summary_in.items():
                history.loc[self.epoch, "`" + key] = val

        history["epoch"] = history["epoch"].astype(int)
        print("=== history\n", history)
        history.to_csv(self.history_file, encoding='utf-8', index=False, float_format='%.4f')
