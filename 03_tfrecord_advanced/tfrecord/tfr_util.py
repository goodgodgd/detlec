import numpy as np
import cv2
import tensorflow as tf


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


def inspect_properties(example):
    config = dict()
    for key, value in example.items():
        if value is not None:
            config[key] = read_data_config(key, value)
    return config


def read_data_config(key, value):
    parse_type = ""
    decode_type = ""
    shape = ()
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            decode_type = "tf.uint8"
        elif value.dtype == np.int32:
            decode_type = "tf.int32"
        elif value.dtype == np.float32:
            decode_type = "tf.float32"
        else:
            assert 0, f"[read_data_config] Wrong numpy type: {value.dtype}, key={key}"
        parse_type = "tf.string"
        shape = list(value.shape)
    elif isinstance(value, int):
        parse_type = "tf.int64"
        shape = None
    elif isinstance(value, str):
        parse_type = "tf.string"
        shape = None
    else:
        assert 0, f"[read_data_config] Wrong type: {type(value)}, key={key}"

    return {"parse_type": parse_type, "decode_type": decode_type, "shape": shape}


def draw_boxes(image, bboxes, category_names):
    bboxes = bboxes[bboxes[:, 2] > 0, :]
    for bbox in bboxes:
        pt1, pt2 = (bbox[1], bbox[0]), (bbox[3], bbox[2])
        category = category_names[bbox[4]]
        image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), thickness=2)
        image = cv2.putText(image, category, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image
