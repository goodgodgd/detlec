import cv2
import tensorflow as tf
import numpy as np
import os
import os.path as op
import json

from tfrecord.readers.kitti_reader import KittiReader
from preprocess import ExamplePreprocess
import tfrecord.tfr_util as tu
import utils.util_function as uf
import config as cfg


def create_tfrecords():
    dataset_cfg = cfg.Datasets.Kitti
    preprocess_example = ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                           dataset_cfg=dataset_cfg,
                                           category_names=cfg.Tfrdata.CATEGORY_NAMES,
                                           max_bbox=20)
    serializer = tu.TfrSerializer()

    for split in ['train', 'val']:
        tfr_path = check_path(split)
        if tfr_path is None:
            continue
        kitti_path = op.join(cfg.Datasets.Kitti.PATH, 'training', 'image_2')
        data_reader = KittiReader(kitti_path, split, cfg.Datasets.Kitti)
        tfr_file = op.join(tfr_path, "data.tfrecord")
        tfr_writer = tf.io.TFRecordWriter(tfr_file)
        frame_names = data_reader.get_frame_names()
        count = 0
        for index, frame in enumerate(frame_names):
            example = dict()
            try:
                example["image"] = data_reader.get_image(index)
                bbox = data_reader.get_bboxes(index)
                category = data_reader.get_categories(index)
                example["inst"] = np.concatenate([bbox, np.ones_like(category), category], axis=-1, dtype=np.float32)
                example = preprocess_example(example)
                count += 1
            except Exception as e:
                print('[Exception]', e)
                continue

            serialized = serializer(example)
            tfr_writer.write(serialized)
            uf.print_progress(f"write frame index={index}, file={frame}")
            if index % 50 == 0:
                show_example(example)
            # if index >= 500:
            #     break

        print('')
        write_tfrecord_config(example, count, tfr_path)
        tfr_writer.close()


def check_path(split):
    tfr_path = op.join(cfg.Paths.TFRECORD, f'kitti_{split}')
    if op.isdir(tfr_path):
        print('Tfrecords are already created!', tfr_path)
        return None
    else:
        print('Create path', tfr_path)
        os.makedirs(tfr_path)
    return tfr_path


def show_example(example):
    category_names = cfg.Tfrdata.CATEGORY_NAMES
    image = tu.draw_boxes(example["image"], example["inst"], category_names)
    cv2.imshow("image with bboxes", image)
    cv2.waitKey(100)


def write_tfrecord_config(example, length, tfr_path):
    assert ('image' in example) and (example['image'] is not None)
    config = tu.inspect_properties(example)
    config["length"] = length
    print("## save config", config)
    with open(op.join(tfr_path, "tfr_config.txt"), "w") as fr:
        json.dump(config, fr)


if __name__ == '__main__':
    create_tfrecords()
