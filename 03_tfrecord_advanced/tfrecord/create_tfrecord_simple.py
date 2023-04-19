import cv2
import tensorflow as tf
import numpy as np

from tfrecord.readers.kitti_reader import KittiReader
from preprocess import ExamplePreprocess
from tfr_util import TfrSerializer, draw_boxes
import config as cfg


def create_tfrcord():
    dataset_cfg = cfg.Datasets.Kitti
    data_reader = KittiReader(cfg.Datasets.Kitti.PATH + '/training/image_2', 'train', cfg.Datasets.Kitti)
    frame_files = data_reader.init_drive(cfg.Datasets.Kitti.PATH + '/training/image_2', 'train')

    preprocess_example = ExamplePreprocess(target_hw=dataset_cfg.INPUT_RESOLUTION,
                                           dataset_cfg=dataset_cfg,
                                           category_names=cfg.Tfrdata.CATEGORY_NAMES,
                                           max_bbox=20)
    serializer = TfrSerializer()
    outfile = cfg.Paths.TFRECORD + "/kitti/train.tfrecord"
    tfr_writer = tf.io.TFRecordWriter(outfile)

    for index, frame in enumerate(frame_files):
        example = dict()
        example["image"] = data_reader.get_image(index)
        bbox = data_reader.get_bboxes(index)
        category = data_reader.get_categories(index)
        example["inst"] = np.concatenate([bbox, np.ones_like(category), category], axis=-1, dtype=np.float32)
        example = preprocess_example(example)
        serialized = serializer(example)
        tfr_writer.write(serialized)
        print("write frame:", index, frame)
        if index % 50 == 0:
            show_example(example)
        if index >= 500:
            break
    tfr_writer.close()


def show_example(example):
    category_names = cfg.Tfrdata.CATEGORY_NAMES
    image = draw_boxes(example["image"], example["inst"], category_names)
    cv2.imshow("image with bboxes", image)
    cv2.waitKey(100)


if __name__ == '__main__':
    create_tfrcord()
