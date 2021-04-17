import os
import tensorflow as tf
import numpy as np
import pickle
from glob import glob


class Config:
    RAW_DATA_PATH = "/home/ian/workspace/detlec/dataset/cifar-10-batches-py"
    TFRECORD_PATH = "/home/ian/workspace/detlec/dataset/tfrecord"
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    CIFAR_IMG_SHAPE = (32, 32, 3)


def write_cifar10_tfrecord():
    test_serializer()
    train_set, test_set = load_cifar10_dataset(Config.RAW_DATA_PATH, Config.CIFAR_IMG_SHAPE)
    make_tfrecord(train_set, "cifar10", "train", Config.CLASS_NAMES, Config.TFRECORD_PATH)
    make_tfrecord(test_set, "cifar10", "test", Config.CLASS_NAMES, Config.TFRECORD_PATH)


def test_serializer():
    example = {"name": "car", "int": 10, "float": 1.1, "np": np.array([1, 2, 3]).astype(np.uint8)}
    features = TfrSerializer().convert_to_feature(example)
    print("=== dict of tf.train.Feature:\n", features)
    features = tf.train.Features(feature=features)
    print("=== tf.train.Features:\n", features)
    tf_example = tf.train.Example(features=features)
    print("=== tf.train.Example\n", tf_example)
    serialized = tf_example.SerializeToString()
    print("=== serialized\n", serialized)
    print("")


def load_cifar10_dataset(data_path, img_shape):
    train_files = glob(os.path.join(data_path, "data_*"))
    train_labels = []
    train_images = []
    for file in train_files:
        labels, images = read_data(file, img_shape)
        train_labels += labels
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)

    test_file = os.path.join(data_path, "test_batch")
    test_labels, test_images = read_data(test_file, img_shape)

    print("[load_cifar10_dataset] train image and label shape:", train_images.shape, len(train_labels))
    print("[load_cifar10_dataset] test image and label shape: ", test_images.shape, len(test_labels))
    return (train_images, train_labels), (test_images, test_labels)


def read_data(file, img_shape):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        labels = data[b"labels"]    # list of category indices, [10000], int
        images = data[b"data"]      # numpy array, [10000, 3072(=32x32x3)], np.uint8
        # CIFAR dataset is encoded in channel-first format
        images = images.reshape((-1, img_shape[2], img_shape[0], img_shape[1]))
        # convert to back channel-last format
        images = np.transpose(images, (0, 2, 3, 1))
    return labels, images


def make_tfrecord(dataset, dataname, split, class_names, tfr_path):
    xs, ys = dataset
    labels = np.array(class_names)
    labels = labels[ys]
    writer = None
    serializer = TfrSerializer()
    examples_per_shard = 10000

    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        if i % examples_per_shard == 0:
            writer = open_tfr_writer(writer, tfr_path, dataname, split, i//examples_per_shard)

        example = {"image": x, "label_index": y, "label_name": label}
        serialized = serializer(example)
        writer.write(serialized)

    writer.close()


def open_tfr_writer(writer, tfr_path, dataname, split, shard_index):
    if writer:
        writer.close()

    tfrdata_path = os.path.join(tfr_path, f"{dataname}_{split}")
    if os.path.isdir(tfr_path) and not os.path.isdir(tfrdata_path):
        os.makedirs(tfrdata_path)
    tfrfile = os.path.join(tfrdata_path, f"shard_{shard_index:03d}.tfrecord")
    writer = tf.io.TFRecordWriter(tfrfile)
    print(f"create tfrecord file: {tfrfile}")
    return writer


class TfrSerializer:
    def __call__(self, raw_example):
        features = self.convert_to_feature(raw_example)
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=features)
        # wrap again as a TensorFlow Example.
        tf_example = tf.train.Example(features=features)
        # serialize the data.
        serialized = tf_example.SerializeToString()
        return serialized

    def convert_to_feature(self, raw_example):
        features = dict()
        for key, value in raw_example.items():
            if value is None:
                continue
            elif isinstance(value, np.ndarray):
                # method 1: encode into raw bytes - fast but losing shape, 2 seconds to make training dataset
                value = value.tobytes()
                # method 2: encode into png format - slow but keeping shape, 10 seconds to make training dataset
                # value = tf.io.encode_png(value)
                # value = value.numpy()  # BytesList won't unpack a tf.string from an EagerTensor.
                features[key] = self._bytes_feature(value)
            elif isinstance(value, str):
                value = bytes(value, 'utf-8')
                features[key] = self._bytes_feature(value)
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._float_feature(value)
            else:
                assert 0, f"[convert_to_feature] Wrong data type: {type(value)}"
        return features

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
    write_cifar10_tfrecord()

