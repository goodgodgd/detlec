import numpy as np
import tensorflow as tf
import pandas as pd

import utils.util_function as uf


class Logger:
    def __init__(self):
        pass

    def save_log(self, epoch, train_log, val_log):
        print("save_log() will be implemented")
        pass


class ModelLog:
    def __init__(self):
        self.batch = pd.DataFrame()
        self.epoch = dict()

    def append_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        loss_list = [loss_name for loss_name, loss_tensor in loss_by_type.items() if loss_tensor.ndim == 0]
        batch_data = {loss_name: loss_by_type[loss_name].numpy() for loss_name in loss_list}
        batch_data["total_loss"] = total_loss.numpy()
        objectness = self.analyze_objectness(grtr, pred)
        batch_data.update(objectness)
        self.check_nan(batch_data, grtr, pred)
        # self.check_pred_scales(pred)
        if step % 100 == 10:
            print("--- batch_data:", batch_data)
        self.batch.append(batch_data, ignore_index=True)

    def analyze_objectness(self, grtr, pred):
        pos_obj, neg_obj = 0, 0
        scales = [key for key in grtr if "feature_" in key]
        for scale_name in scales:
            grtr_slices = uf.slice_features(grtr[scale_name])
            pred_slices = uf.slice_features(pred[scale_name])
            grtr_obj_mask = grtr_slices["object"]
            pred_obj_prob = pred_slices["object"]
            obj_num = tf.maximum(tf.reduce_sum(grtr_obj_mask), 1)
            # average positive objectness probability
            pos_obj += tf.reduce_sum(grtr_obj_mask * pred_obj_prob) / obj_num
            # average top 50 negative objectness probabilities per frame
            neg_obj_map = (1. - grtr_obj_mask) * pred_obj_prob
            batch, grid_h, grid_w, anchor, _ = neg_obj_map.shape
            neg_obj_map = tf.reshape(neg_obj_map, (batch, grid_h * grid_w * anchor))
            neg_obj_map = tf.sort(neg_obj_map, axis=-1, direction="DESCENDING")
            neg_obj_map = neg_obj_map[:, :50]
            neg_obj += tf.reduce_mean(neg_obj_map)
        objectness = {"pos_obj": pos_obj.numpy() / len(scales), "neg_obj": neg_obj.numpy() / len(scales)}
        return objectness

    def check_nan(self, losses, grtr, pred):
        valid_result = True
        for name, loss in losses.items():
            if np.isnan(loss) or np.isinf(loss):
                print(f"nan loss: {name}, {loss}")
                valid_result = False
        for name, tensor in pred.items():
            if np.isnan(tensor.numpy()).any():
                print(f"nan pred:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        for name, tensor in grtr.items():
            if np.isnan(tensor.numpy()).any():
                print(f"nan grtr:", name, np.quantile(tensor.numpy(), np.linspace(0, 1, 11)))
                valid_result = False
        assert valid_result

    def check_pred_scales(self, pred):
        raw_features = {key: tensor for key, tensor in pred.items() if key.endswith("raw")}
        pred_scales = dict()
        for key, feat in raw_features.items():
            pred_scales[key] = np.quantile(feat.numpy(), np.array([0.05, 0.5, 0.95]))
        print("--- pred_scales:", pred_scales)

    def append_epoch_result(self, **kwargs):
        self.epoch = kwargs

