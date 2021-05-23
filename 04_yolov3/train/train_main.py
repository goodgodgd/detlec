import os
import os.path as op
import tensorflow as tf
import numpy as np
import pandas as pd

import settings
from config import Config as cfg
from tfrecord.tfrecord_reader import TfrecordReader
from model.model_factory import ModelFactory
from train.loss_factory import IntegratedLoss
from train.logger import LogFile
import train.train_val as tv
import utils.util_function as uf


def train_main():
    uf.set_gpu_configs()
    end_epoch = 0
    for dataset_name, epochs, learning_rate, loss_weights, model_save in cfg.Train.TRAINING_PLAN:
        end_epoch += epochs
        train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save)


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    tfrd_path, ckpt_path = cfg.Paths.TFRECORD, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    valid_category = cfg.get_valid_category_mask(dataset_name)
    start_epoch = read_previous_epoch(ckpt_path)
    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    dataset_train, train_steps, imshape, anchors_per_scale = \
        get_dataset(tfrd_path, dataset_name, False, batch_size, "train")
    dataset_val, val_steps, _, _ = get_dataset(tfrd_path, dataset_name, False, batch_size, "val")

    model, loss_object, optimizer = create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path,
                                                          learning_rate, loss_weights, valid_category)
    trainer = tv.trainer_factory(train_mode, model, loss_object, optimizer, train_steps)
    validater = tv.validater_factory(train_mode, model, loss_object, val_steps)
    log_file = LogFile()

    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        train_result = trainer.run_epoch(dataset_train)
        val_result = validater.run_epoch(dataset_val)
        save_model_ckpt(ckpt_path, model)
        log_file.save_log(epoch, train_result, val_result)

    if model_save:
        save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def get_dataset(tfrd_path, dataset_name, shuffle, batch_size, split):
    tfrpath = op.join(tfrd_path, f"{dataset_name}_{split}")
    reader = TfrecordReader(tfrpath, shuffle, batch_size, 1)
    dataset = reader.get_dataset()
    frames = reader.get_total_frames()
    tfr_cfg = reader.get_tfr_config()
    image_shape = tfr_cfg["image"]["shape"]
    # anchor sizes per scale in pixel
    anchors_per_scale = {key: np.array(val) / np.array([image_shape[:2]]) for key, val in tfr_cfg.items() if key.startswith("anchor")}
    anchors_per_scale = {key: val.astype(np.float32) for key, val in anchors_per_scale.items()}
    print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, "
          f"frames={frames},\n\tanchors={anchors_per_scale}")
    return dataset, frames // batch_size, image_shape, anchors_per_scale


def create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path, learning_rate,
                          loss_weights, valid_category, weight_suffix='latest'):
    model = ModelFactory(batch_size, imshape, anchors_per_scale).get_model()
    model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = IntegratedLoss(loss_weights, valid_category)
    optimizer = tf.optimizers.Adam(lr=learning_rate)
    return model, loss_object, optimizer


def save_model_ckpt(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if not op.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    print("=== save model:", ckpt_file)
    model.save_weights(ckpt_file)


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model.load_weights(ckpt_file)
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def read_previous_epoch(ckpt_path):
    filename = op.join(ckpt_path, 'history.csv')
    if op.isfile(filename):
        history = pd.read_csv(filename, encoding='utf-8', converters={'epoch': lambda c: int(c)})
        if history.empty:
            print("[read_previous_epoch] EMPTY history:", history)
            return 0

        epochs = history['epoch'].tolist()
        epochs.sort()
        prev_epoch = epochs[-1]
        print(f"[read_previous_epoch] start from epoch {prev_epoch + 1}")
        return prev_epoch + 1
    else:
        print(f"[read_previous_epoch] NO history in {filename}")
        return 0


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    train_main()
