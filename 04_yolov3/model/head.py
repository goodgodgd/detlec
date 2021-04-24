import tensorflow as tf

from utils.util_class import MyExceptionToCatch
import model.model_util as mu


def head_factory(head, model_cfg):
    if head == "FPN":
        return FPN(model_cfg)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {head}")


class HeadBase:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **model_cfg.Structure.CONV_ARGS)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **model_cfg.Structure.CONV_ARGS)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **model_cfg.Structure.CONV_ARGS)
        self.conv2d_result = mu.CustomConv2D(kernel_size=3, strides=2, bn=False)

    def __call__(self, input_features):
        raise NotImplementedError()

    def conv_5x(self, x, channel):
        x = self.conv2d_k1(x, channel)
        x = self.conv2d(x, channel*2)
        x = self.conv2d_k1(x, channel)
        x = self.conv2d(x, channel*2)
        x = self.conv2d_k1(x, channel)
        return x

    def make_result(self, x, channel):
        x = self.conv2d(x, channel)
        anchors, anc_channels = self.model_cfg.Output.ANCHORS_PER_SCALE, self.model_cfg.Output.OUT_CHANNELS
        x = self.conv2d_result(x, anchors, anc_channels)
        batch, height, width, channel = x.shape
        x_5d = tf.reshape(x, (batch, height, width, anchors, anc_channels))
        return x_5d


class FPN(HeadBase):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

    def __call__(self, input_features):
        large = input_features["feature_l"]
        medium = input_features["feature_m"]
        small = input_features["feature_s"]
        conv = self.conv_5x(large, 512)
        conv_lbbox = self.make_result(conv, 1024)

        conv_medium = self.upsample_concat(large, medium, 256)
        conv = self.conv_5x(conv_medium, 256)
        conv_mbbox = self.make_result(conv, 512)

        conv_small = self.upsample_concat(conv_medium, small, 128)
        conv = self.conv_5x(conv_small, 128)
        conv_sbbox = self.make_result(conv, 1024)
        conv_result = {"feature_l": conv_lbbox, "feature_m": conv_mbbox, "feature_s": conv_sbbox}
        return conv_result

    def upsample_concat(self, upper, lower, channel):
        x = self.conv2d_k1(upper, channel)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.concat([x, lower], axis=-1)
        return x


class FeatureDecoder:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg

    def __call__(self, feature, scale_name: str):
        """
        :param feature: raw feature map predicted by model from one of multiple scales
        :param scale_name: scale name
        :return: decoded feature e.g. (yxhw, objectness, category probabilities)
        """
        slices = mu.slice_features(feature, self.model_cfg.Output.OUT_COMPOSITION)
        anchors = self.get_feature_anchors(feature, scale)
        box_yx, box_hw = self.decode_box(slices["yxhw"])

        anchors_h = anchors[:, 0:1] / cfg.INPUT_SHAPE[0]
        anchors_w = anchors[:, 1:2] / cfg.INPUT_SHAPE[1]
        grid_h, grid_w = tf.shape(pred)[1:3]
        box_yx, box_h, box_w, objectness, class_probs, distances = tf.split(pred, (2, 1, 1, 1, num_class, 1), axis=-1)
        box_yx = tf.sigmoid(box_yx) * 1.4 - 0.2
        box_h = tf.exp(box_h) * anchors_h
        box_w = tf.exp(box_w) * anchors_w
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)

        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.expand_dims(tf.expand_dims(grid, axis=2), axis=0)
        grid = tf.cast(grid, tf.float32, name="grid")
        divider = tf.cast([grid_h, grid_w], tf.float32, name="divider")

        box_yx = (box_yx + grid) / divider

        distance = tf.exp(distances)
        bbox = tf.concat([box_yx, box_h, box_w, objectness, class_probs, distance], axis=-1)
        return tf.cast(bbox, dtype=tf.float32)

    def get_feature_anchors(self, feature, scale):
        pass

    def decode_box(self, yxhw_raw):
        """
        :param yxhw_raw: (batch, grid_h, grid_w, 4)
        :return: (batch, grid_h, grid_w, 2), (batch, grid_h, grid_w, 2)
        """
        grid_h, grid_w = yxhw_raw.shape[1:3]
        yx_raw, hw_raw = yxhw_raw[..., :2]
        """
        Original yolo v3 implementation: yx_dec = tf.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw 
        """
        yx_dec = tf.sigmoid(yx_raw) * 1.4 - 0.2


