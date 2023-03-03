import os.path as op
import pandas as pd
import torch
import torch.optim as optim
import numpy as np

import settings
import config as cfg
from pytch.model.model_factory import ModelTemplate
from pytch.data.datasets import Cifar10Dataset
from neutr.loss_factory import IntegratedLoss
import pytch.train.train_val as tv
import pytch.train.loss_pool as loss_pool


def train_main():
    end_epoch = 0
    for dataset_name, epochs, learning_rate, loss_weights, model_save in cfg.Train.TRAINING_PLAN:
        end_epoch += epochs
        train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save)


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save):
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    tfrd_path, ckpt_path = cfg.Paths.TFRECORD, op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    start_epoch = read_previous_epoch(ckpt_path)
    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    dataset_train, train_steps, imshape = Cifar10Dataset("train", batch_size, False).get_dataloader()
    dataset_val, val_steps, _ = Cifar10Dataset("val", batch_size, False).get_dataloader()
    model, loss_object, optimizer = create_training_parts(batch_size, imshape, ckpt_path,
                                                          learning_rate, loss_weights)
    trainer = tv.trainer_factory(train_mode, model, loss_object, train_steps, ckpt_path, optimizer)
    validater = tv.validater_factory(train_mode, model, loss_object, val_steps, ckpt_path)

    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start dataset : {dataset_name} epoch: {epoch + 1}/{end_epoch} ==========")
        trainer.run_epoch(dataset_train, epoch, False)
        validater.run_epoch(dataset_val, epoch, epoch%5==0 or epoch%5==4)
        save_model_ckpt(ckpt_path, model)

    if model_save:
        save_model_ckpt(ckpt_path, model, f"ep{end_epoch:02d}")


def get_dataset(tfrd_path, dataset_name, shuffle, batch_size, split):
    raise NotImplementedError()


def create_training_parts(batch_size, imshape, ckpt_path, learning_rate,
                          loss_weights, weight_suffix='latest'):
    model = ModelTemplate(cfg.Architecture, imshape)
    # model = try_load_weights(ckpt_path, model, weight_suffix)
    loss_object = IntegratedLoss(loss_weights, loss_pool)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_object, optimizer


def save_model_ckpt(ckpt_path, model, weights_suffix='latest'):
    # ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    # if not op.isdir(ckpt_path):
    #     os.makedirs(ckpt_path, exist_ok=True)
    # print("=== save model:", ckpt_file)
    # model.save_weights(ckpt_file)
    raise NotImplementedError()


def try_load_weights(ckpt_path, model, weights_suffix='latest'):
    # ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.h5")
    # if op.isfile(ckpt_file):
    #     print(f"===== Load weights from checkpoint: {ckpt_file}")
    #     model.load_weights(ckpt_file)
    # else:
    #     print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    # return model
    raise NotImplementedError()


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



