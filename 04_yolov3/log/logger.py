import numpy as np
import os
import os.path as op

from log.history_log import HistoryLog
# from log.visual_log import VisualLog
import utils.util_function as uf
import model.model_util as mu
import config as cfg


class Logger:
    def __init__(self, epoch: int, ckpt_path: str, visual_log: bool, is_train: bool):
        self.history_logger = HistoryLog(epoch, ckpt_path, is_train)
        # self.visual_logger = VisualLog(epoch, ckpt_path) if visual_log else None
        self.nms_box = mu.NonMaximumSuppressionBox()
        self.num_ctgr = len(cfg.Tfrdata.CATEGORY_NAMES)
        self.epoch = epoch
        self.ckpt_path = ckpt_path
        assert op.isdir(op.dirname(op.dirname(ckpt_path)))
        os.makedirs(ckpt_path, exist_ok=True)

    def log_batch_result(self, step, grtr, pred, total_loss, loss_by_type):
        grtr = uf.convert_to_numpy(grtr)
        pred = uf.convert_to_numpy(pred)
        loss_by_type = uf.convert_to_numpy(loss_by_type)
        total_loss = total_loss.numpy()
        assert self.check_scales("[pred scale]", pred) == 0
        assert self.check_scales("[loss scale]", loss_by_type) == 0

        pred_inst = self.nms_box(pred["fmap"])
        pred["inst"] = uf.slice_feature_np(pred_inst, cfg.ModelOutput.GRTR_FMAP_COMPOSITION)

        if step == 0 and self.epoch == 0:
            structure = {"grtr": grtr, "pred": pred, "loss": loss_by_type}
            self.save_data_structure(structure)
        
        self.history_logger(step, grtr, pred, total_loss, loss_by_type)
        # if self.visual_logger:
        #    self.visual_logger(step, grtr, pred)

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

    def save_data_structure(self, structures):
        structure_file = op.join(self.ckpt_path, "structure.md")
        f = open(structure_file, "w")
        for key, structure in structures.items():
            f.write(f"- {key}\n")
            space_count = 1
            self.analyze_structure(structure, f, space_count)
        f.close()

    def analyze_structure(self, data, f, space_count, key=""):
        space = "    " * space_count
        if isinstance(data, list):
            for i, datum in enumerate(data):
                if isinstance(datum, dict):
                    # space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    # space_count -= 1
                elif type(datum) == np.ndarray:
                    f.write(f"{space}- {key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {datum}\n")
                    space_count += 1
                    self.analyze_structure(datum, f, space_count)
                    space_count -= 1
        elif isinstance(data, dict):
            for sub_key, datum in data.items():
                if type(datum) == np.ndarray:
                    f.write(f"{space}- {sub_key}: {datum.shape}\n")
                else:
                    f.write(f"{space}- {sub_key}\n")

                space_count += 1
                self.analyze_structure(datum, f, space_count, sub_key)
                space_count -= 1

    def finalize(self):
        self.history_logger.finalize()
