import os.path as op
import numpy as np
import tensorflow as tf
import pandas as pd

import utils.util_function as uf
import model.model_util as mu
from train.logging.history_log import HistoryLog
from train.logging.visual_log import VisualLog
from train.logging.anchor_log import AnchorLog
import config as cfg


class LogFile:
    def __init__(self, ckpt_path):
        self.filename = op.join(ckpt_path, "history.csv")

    def save_log(self, epoch, train_log, val_log):
        summary = self.merge_logs(epoch, train_log, val_log)
        if op.isfile(self.filename):
            history = pd.read_csv(self.filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
            history = history.append(summary, ignore_index=True)
        else:
            history = pd.DataFrame([summary])
        print("=== history\n", history)
        history["epoch"] = history["epoch"].astype(int)
        history.to_csv(self.filename, encoding='utf-8', index=False, float_format='%.4f')

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

    def save_val_log(self, val_log):
        pass


class Logger:
    def __init__(self, visual_log, anchor_log, ckpt_path, epoch):
        self.history_logger = HistoryLog()
        self.visual_logger = VisualLog(ckpt_path, epoch) if visual_log else None
        self.anchor_logger = AnchorLog(ckpt_path, epoch) if anchor_log else None
        self.nms = mu.NonMaximumSuppression()

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        self.check_nan(grtr, pred, loss_by_type)
        grtr_slices = uf.merge_and_slice_features(grtr, True)
        pred_slices = uf.merge_and_slice_features(pred, False)
        nms_boxes = self.nms(pred_slices)
        pred_slices["bboxes"] = uf.slice_feature(nms_boxes, cfg.Model.Output.get_bbox_composition(False))

        self.history_logger(step, grtr_slices, pred_slices, total_loss, loss_by_type)
        if self.visual_logger:
            self.visual_logger(step, grtr_slices, pred_slices)
        if self.anchor_logger:
            self.anchor_logger(step, pred_slices, pred_slices, loss_by_type)

    def finalize(self):
        self.history_logger.make_summary()

    def get_summary(self):
        return self.history_logger.get_summary()

    def get_anchor_log(self):
        if self.anchor_logger:
            return self.anchor_logger.get_result()
        else:
            return None

    def check_nan(self, grtr, pred, loss_by_type):
        valid_result = True
        for name, tensor in grtr.items():
            if not np.isfinite(tensor.numpy()).all():
                print(f"nan grtr:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        for name, tensor in pred.items():
            if not np.isfinite(tensor.numpy()).all():
                print(f"nan pred:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        for name, loss in loss_by_type.items():
            if loss.ndim == 0 and (np.isnan(loss) or np.isinf(loss) or loss > 100000):
                print(f"nan loss: {name}, {loss}")
                valid_result = False
        assert valid_result
