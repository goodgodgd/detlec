import os
import os.path as op
import tensorflow as tf
import numpy as np
import pandas as pd

from config import Config as cfg
from tfrecord.tfrecord_reader import TfrecordReader
from model.model_factory import ModelFactory
from train.loss_factory import LossFactory
from train.logger import Logger
import train.train_val as tv
import utils.util_function as uf
import settings


def train_main():
    uf.set_gpu_configs()
    np.set_printoptions(precision=4, suppress=True, linewidth=100)
    for plan in cfg.Train.TRAINING_PLAN:
        train_by_plan(plan)


def train_by_plan(plan):
    dataset_name, epochs, learning_rate, loss_weights, model_save = plan
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    tfr_path, ckpt_path = cfg.Paths.TFRECORD, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    initial_epoch = read_previous_epoch(ckpt_path)

    dataset_train, train_steps, imshape, anchors_per_scale = \
        get_dataset(tfr_path, dataset_name, False, batch_size, "train")
    dataset_val, val_steps, _, _ = get_dataset(tfr_path, dataset_name, False, batch_size, "val")

    model, loss, optimizer = create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path,
                                                   learning_rate, loss_weights, dataset_name)

    trainer = tv.trainer_factory(train_mode, model, loss, optimizer, train_steps)
    validater = tv.validater_factory(train_mode, model, loss, val_steps)
    logger = Logger()
    end_epoch = initial_epoch + epochs

    for epoch in range(initial_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        train_result = trainer.run_epoch(dataset_train)
        val_result = validater.run_epoch(dataset_val)
        save_model_ckpt(ckpt_path, model)
        logger.save_log(epoch, train_result, val_result)

    if model_save:
        save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def get_dataset(tfr_path, dataset_name, shuffle, batch_size, split):
    tfrpath = op.join(tfr_path, f"{dataset_name}_{split}")
    reader = TfrecordReader(tfrpath, shuffle, batch_size, 1)
    dataset = reader.get_dataset()
    frames = reader.get_total_frames()
    tfr_cfg = reader.get_tfr_config()
    image_shape = tfr_cfg["image"]["shape"]
    # anchor sizes per scale in pixel
    anchors_per_scale = {key: val for key, val in tfr_cfg.items() if key.startswith("anchor")}
    print(f"[get_dataset] dataset={dataset_name}, image shape={image_shape}, frames={frames}")
    return dataset, frames // batch_size, image_shape, anchors_per_scale


def create_training_parts(batch_size, imshape, anchors_per_scale, ckpt_path, learning_rate,
                          loss_weights, dataset_name, weight_suffix='latest'):
    model = ModelFactory(batch_size, imshape, anchors_per_scale).get_model()
    model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = LossFactory(loss_weights, dataset_name).get_loss()
    optimizer = tf.optimizers.Adam(lr=learning_rate)
    return model, loss_object, optimizer


def save_model_ckpt(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"yolo_{weights_suffix}.h5")
    if not op.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    model.save_weights(ckpt_file)


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"{weights_suffix}.h5")
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        model.load_weights(ckpt_file)
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model


def read_previous_epoch(ckpt_path):
    filename = op.join(ckpt_path, 'history.txt')
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
    train_main()
