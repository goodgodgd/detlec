import os.path as op
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from config import Config as cfg
from tfrecord.tfrecord_writer import TfrecordMaker
import settings


def create_tfrecords():
    datasets = cfg.Dataset.DATASETS_FOR_TFRECORD
    for dataset, splits in datasets.items():
        for split in splits:
            dataset_cfg = cfg.Dataset.get_dataset_config(dataset)
            tfrpath = op.join(cfg.Paths.TFRECORD, f"{dataset}_{split}")
            if op.isdir(tfrpath):
                print("[convert_to_tfrecords] tfrecord already created in", op.basename(tfrpath))
                continue

            tfrmaker = TfrecordMaker(dataset_cfg, split, tfrpath, cfg.Dataset.SHARD_SIZE)
            tfrmaker.make()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    create_tfrecords()
