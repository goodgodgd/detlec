import sys
import os.path as op
import tensorflow as tf
import numpy as np
import pickle
from glob import glob


class Config:
    RAW_DATA_PATH = "/home/ian/workspace/detlec/dataset/cifar-10-batches-py"
    TFRECORD_PATH = "/home/ian/workspace/detlec/dataset/cifar-10-tfrecord"
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    test_serializer()
    train_set, test_set = load_cifar10_dataset(Config.RAW_DATA_PATH)
    make_tfrecord(train_set, "train", Config.CLASS_NAMES, Config.TFRECORD_PATH)
    make_tfrecord(test_set, "test", Config.CLASS_NAMES, Config.TFRECORD_PATH)


def test_serializer():
    example = {"name": "car", "int": 10, "float": 1.1, "np": np.array([1, 2, 3]).astype(np.uint8)}
    print(TfrSerializer().convert_to_feature(example))


def load_cifar10_dataset(data_path):
    train_files = glob(op.join(data_path, "data_*"))
    train_labels = []
    train_images = []
    for file in train_files:
        labels, images = read_data(file)
        train_labels += labels
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)

    test_file = op.join(data_path, "test_batch")
    test_labels, test_images = read_data(test_file)

    print("[load_cifar10_dataset] train image and label shape:", train_images.shape, len(train_labels))
    print("[load_cifar10_dataset] test image and label shape: ", test_images.shape, len(test_labels))
    return (train_images, train_labels), (test_images, test_labels)


def read_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        labels = data[b"labels"]    # list of category indices, [10000], int
        images = data[b"data"]      # numpy array, [10000, 3072(=32x32x3)], np.uint8
    return labels, images


def make_tfrecord(dataset, split, class_names, tfr_path):
    xs, ys = dataset
    labels = np.array(class_names)
    labels = labels[ys]
    writer = None
    serializer = TfrSerializer()
    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        if i % 10000 == 0:
            if writer:
                writer.close()
            tfrfile = op.join(tfr_path, f"cifar10-{split}-{i//10000:03d}.tfrecord")
            writer = tf.io.TFRecordWriter(tfrfile)
            print(f"create tfrecord file at {i}: {tfrfile}")

        example = make_example(x, y, label)
        serialized = serializer(example)
        writer.write(serialized)

    writer.close()


def make_example(x, y, label):
    return {"image": x, "class": y, "label": label}


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
                features[key] = self._bytes_feature(value)
            elif isinstance(value, str):
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
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        elif isinstance(value, str):
            value = bytes(value, 'utf-8')
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
    main()
