import os.path as op
import json
import tensorflow as tf

import settings
import config as cfg
import tflow.utils.util_function as tuf


class TfrecordReader:
    def __init__(self, tfrpath, shuffle=False, batch_size=cfg.Train.BATCH_SIZE, epochs=1):
        self.tfrpath = tfrpath
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.config = self.read_tfrecord_config(tfrpath)
        self.features_dict = self.get_features(self.config)

    def read_tfrecord_config(self, tfrpath):
        with open(op.join(tfrpath, "tfr_config.txt"), "r") as fr:
            config = json.load(fr)
            for key, properties in config.items():
                if not isinstance(properties, dict):
                    continue
                # convert parse types in string to real type
                properties["parse_type"] = eval(properties["parse_type"])
                properties["decode_type"] = eval(properties["decode_type"])
        return config

    def get_features(self, config):
        features_dict = {}
        for key, properties in config.items():
            if isinstance(properties, dict):
                default = "" if properties["parse_type"] is tf.string else 0
                features_dict[key] = tf.io.FixedLenFeature((), properties["parse_type"], default_value=default)
        return features_dict

    def get_dataset(self):
        """
        :return features: {"image": ..., "inst": ...}
            image: (batch, height, width, 3)
            bbox: [y1, x1, y2, x2, category] (batch, grid_height, grid_width, 5)
        """
        file_pattern = f"{self.tfrpath}/*.tfrecord"
        filenames = tf.io.gfile.glob(file_pattern)
        filenames.sort()
        print("[tfrecord reader] pattern:", file_pattern, "\nfiles:", filenames)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_example)
        return self.dataset_process(dataset)

    def parse_example(self, example):
        parsed = tf.io.parse_single_example(example, self.features_dict)
        decoded = {}
        for key, properties in self.config.items():
            if isinstance(properties, dict):
                decoded[key] = tf.io.decode_raw(parsed[key], properties["decode_type"])
                decoded[key] = tf.reshape(decoded[key], shape=properties["shape"])
        # uint8 image -> image float (-1 ~ 1)
        decoded["image"] = tuf.to_float_image(decoded["image"])
        return decoded

    def dataset_process(self, dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=200)
            print("[dataset] dataset shuffled")
        print(f"[dataset] num epochs={self.epochs}, batch size={self.batch_size}")
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        return dataset

    def get_total_frames(self):
        return self.config["length"]

    def get_tfr_config(self):
        return self.config


# ==================================================
import cv2
import neutr.utils.util_function as nuf


def test_read_dataset():
    print("===== start test_read_dataset")
    dataset = TfrecordReader(op.join(cfg.Paths.TFRECORD, "kitti_train"), batch_size=4).get_dataset()
    for i, x in enumerate(dataset):
        print(f"=== index: {i}, image={x['image'].shape}, bbox={x['inst'].shape}")
        image = tuf.to_uint8_image(x['image'])
        image = image[0].numpy()
        bboxes = x["inst"][0].numpy()
        image = nuf.draw_boxes(image, bboxes, cfg.DataCommon.CATEGORY_NAMES)
        cv2.imshow("image with boxes", image)
        key = cv2.waitKey()
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("!!! test_read_dataset passed")


if __name__ == "__main__":
    test_read_dataset()
