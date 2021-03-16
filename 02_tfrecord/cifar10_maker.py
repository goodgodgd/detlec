import os.path as op
import tensorflow as tf
import numpy as np

RAW_DATA_PATH = "/home/ian/workspace/dataset/cifar-10-batches-py"
TFRECORD_PATH = "/home/ian/workspace/dataset/cifar-10-tfrecord"
CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    train_set, test_set = load_cifar10_dataset()
    make_tfrecord(train_set, "train")
    make_tfrecord(test_set, "test")


def load_cifar10_dataset():
    return (1, 2), (3, 4)


def make_tfrecord(dataset, split):
    xs, ys = dataset
    labels = np.array(CLASS_NAMES)
    labels = labels[ys]
    writer = tf.io.TFRecordWriter("")
    serializer = TfrSerializer()
    for i, (x, y, label) in enumerate(zip(xs, ys, labels)):
        if i % 10000 == 0:
            tfrfile = op.join(TFRECORD_PATH, f"cifar-10-{split}-{i//10000:03d}.tfrecord")
            writer = tf.io.TFRecordWriter(tfrfile)

        example = make_example(x, y, label)
        serialized = serializer(example)
        writer.write(serialized)


def make_example(x, y, label):
    return {"image": x, "class": y, "label": label}


class TfrSerializer:
    def __call__(self, example_dict):
        features = self.convert_to_feature(example_dict)
        # wrap the data as TensorFlow Features.
        features = tf.train.Features(feature=features)
        # wrap again as a TensorFlow Example.
        example = tf.train.Example(features=features)
        # serialize the data.
        serialized = example.SerializeToString()
        return serialized

    def convert_to_feature(self, example_dict):
        features = dict()
        for key, value in example_dict.items():
            if value is None:
                continue
            elif isinstance(value, np.ndarray):
                features[key] = self._bytes_feature(value.tostring())
            elif isinstance(value, int):
                features[key] = self._int64_feature(value)
            elif isinstance(value, float):
                features[key] = self._int64_feature(value)
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


