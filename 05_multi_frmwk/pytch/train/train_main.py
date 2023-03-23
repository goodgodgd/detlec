import os
import os.path as op
import pandas as pd
import torch
import torch.optim as optim
import numpy as np

import settings
import config as cfg
from pytch.model.model_factory import ModelTemplate
import pytch.data.dataloader as dl
from neutr.loss_factory import IntegratedLoss
import pytch.train.train_val as tv
import pytch.train.loss_pool as loss_pool


def train_main():
    end_epoch = 0
    for dataset_name, epochs, learning_rate, loss_weights, model_save in cfg.Train.TRAINING_PLAN:
        end_epoch += epochs
        train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save)


def train_by_plan(dataset_name, end_epoch, learning_rate, loss_weights, model_save):
    print("==== train by plan")
    batch_size, train_mode = cfg.Train.BATCH_SIZE, cfg.Train.MODE
    ckpt_path = op.join(cfg.Paths.CHECK_POINT, cfg.Train.CKPT_NAME)
    dataset_train, train_steps, imshape = get_dataset(dataset_name, 'train', batch_size, True)
    dataset_val, val_steps, _ = get_dataset(dataset_name, 'val', batch_size, False)
    model, loss_object, optimizer, start_epoch = create_training_parts(imshape, ckpt_path, learning_rate, loss_weights)
    return

    if end_epoch <= start_epoch:
        print(f"!! end_epoch {end_epoch} <= start_epoch {start_epoch}, no need to train")
        return

    trainer = tv.ModelTrainer(model, loss_object, train_steps, ckpt_path, optimizer)
    validater = tv.ModelValidater(model, loss_object, val_steps, ckpt_path)

    for epoch in range(start_epoch, end_epoch):
        print(f"========== Start epoch: {epoch}/{end_epoch}, dataset: {dataset_name} ==========")
        trainer.run_epoch(dataset_train, epoch, False)
        validater.run_epoch(dataset_val, epoch, epoch==0 or epoch%5==4)
        save_model_ckpt(ckpt_path, model, optimizer, epoch)

    if model_save:
        save_model_ckpt(ckpt_path, model, optimizer, end_epoch, f"ep{end_epoch:02d}")


def get_dataset(dataset_name, split, batch_size=4, shuffle=False):
    if dataset_name == 'kitti':
        data_cfg = cfg.Datasets.Kitti
    elif dataset_name == 'cifar10':
        data_cfg = cfg.Datasets.Cifar10
    else:
        raise ValueError(f"No dataset named {dataset_name} is prepared")
    print("data_c2qfg", data_cfg)
    return dl.make_dataloader(data_cfg, split, batch_size, shuffle)


def create_training_parts(imshape, ckpt_path, learning_rate, loss_weights, weight_suffix='latest'):
    print("===== create model template")
    model = ModelTemplate(cfg.Architecture, imshape)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer, epoch = try_load_weights(ckpt_path, model, optimizer, weight_suffix)
    loss_object = IntegratedLoss(loss_weights, loss_pool)
    return model, loss_object, optimizer, epoch


def save_model_ckpt(ckpt_path, model, optimizer, epoch, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.pt")
    if not op.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    print("=== save model:", ckpt_file)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, ckpt_file)


def try_load_weights(ckpt_path, model, optimizer, weights_suffix='latest'):
    ckpt_file = op.join(ckpt_path, f"model_{weights_suffix}.pt")
    latest_epoch = 0
    if op.isfile(ckpt_file):
        print(f"===== Load weights from checkpoint: {ckpt_file}")
        checkpoint = torch.load(ckpt_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        latest_epoch = int(checkpoint['epoch'])
    else:
        print(f"===== Failed to load weights from {ckpt_file}\n\ttrain from scratch ...")
    return model, optimizer, latest_epoch


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



