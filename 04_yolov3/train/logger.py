import os.path as op
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import utils.util_function as uf
import model.model_util as mu
import train.metric as mt
import config as cfg


class LogFile:
    def save_log(self, epoch, train_log, val_log):
        summary = self.merge_logs(epoch, train_log, val_log)
        filename = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME, "history.csv")
        if op.isfile(filename):
            history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(summary, ignore_index=True)
        else:
            history = pd.DataFrame([summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        history.to_csv(filename, encoding='utf-8', index=False, float_format='%.4f')

    def merge_logs(self, epoch, train_log, val_log):
        summary = dict()
        summary["epoch"] = epoch
        train_summary = train_log.get_summary()
        train_summary = {"!" + key: val for key, val in train_summary.items()}
        summary.update(train_summary)
        summary["|"] = 0
        val_summary = val_log.get_summary()
        val_summary = {"`" + key: val for key, val in val_summary.items()}
        summary.update(val_summary)
        return summary


class LogData:
    def __init__(self):
        self.log = pd.DataFrame()
        self.start = timer()
        self.summary = dict()
        self.nms_box = mu.NonMaximumSuppressionBox()
        self.num_ctgr = len(cfg.Tfrdata.CATEGORY_NAMES)

    def append_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        grtr = uf.convert_to_numpy(grtr)
        pred = uf.convert_to_numpy(pred)
        loss_by_type = uf.convert_to_numpy(loss_by_type)
        total_loss = total_loss.numpy()

        assert self.check_scales("[pred scale]", pred) == 0
        assert self.check_scales("[loss scale]", loss_by_type) == 0

        batch_data = {loss_name: loss_tensor for loss_name, loss_tensor in loss_by_type.items() if not isinstance(loss_tensor, list)}
        metric = self.compute_metric(grtr, pred)
        batch_data.update(metric)
        objectness = self.analyze_objectness(grtr, pred)
        batch_data.update(objectness)
        batch_data["total_loss"] = total_loss
        batch_data = {key: np.around(val, 5) for key, val in batch_data.items()}
        self.log = pd.concat([self.log, pd.DataFrame([batch_data])], axis=0, ignore_index=True)
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

    def compute_metric(self, grtr, pred):
        pred_inst = self.nms_box(pred["fmap"])
        pred_inst = uf.slice_feature(pred_inst, cfg.ModelOutput.GRTR_FMAP_COMPOSITION)
        counts = mt.count_true_positives(grtr["inst"], pred_inst, self.num_ctgr)
        metric = {"recall": counts["trpo"] / counts["grtr"], "precision": counts["trpo"] / counts["pred"]}
        return metric

    def check_scales(self, title, data, key=""):
        div_count = 0
        if isinstance(data, list):
            for i, datum in enumerate(data):
                div_count += self.check_scales(title, datum, f"{key}/{i}")
        elif isinstance(data, dict):
            for subkey, datum in data.items():
                div_count += self.check_scales(title, datum, f"{key}/{subkey}")
        elif type(data) == np.ndarray:
            quant = np.quantile(data, np.array([0.05, 0.5, 0.95]))
            if np.max(np.abs(quant)) > 1e+6:
                print(title, key, data.shape, type(data), quant)
                div_count += 1
        return div_count

    def finalize(self):
        self.summary = self.log.mean(axis=0).to_dict()
        self.summary["time_m"] = round((timer() - self.start)/60., 5)
        print("result summary:", self.summary)
    
    def get_summary(self):
        return self.summary
