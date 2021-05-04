import tensorflow as tf
import numpy as np

from utils.util_class import MyExceptionToCatch
import model.model_util as mu


def head_factory(head, conv_args, num_anchors_per_scale, out_channels):
    if head == "FPN":
        return FPN(conv_args, num_anchors_per_scale, out_channels)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {head}")


class HeadBase:
    def __init__(self, conv_args, num_anchors_per_scale, out_channels):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, **conv_args)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, **conv_args)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, **conv_args)
        self.conv2d_output = mu.CustomConv2D(kernel_size=1, strides=1, scope="output", bn=False)
        self.num_anchors_per_scale = num_anchors_per_scale
        self.out_channels = out_channels

    def __call__(self, input_features):
        raise NotImplementedError()

    def conv_5x(self, x, channel):
        x = self.conv2d_k1(x, channel)
        x = self.conv2d(x, channel*2)
        x = self.conv2d_k1(x, channel)
        x = self.conv2d(x, channel*2)
        x = self.conv2d_k1(x, channel)
        return x

    def make_output(self, x, channel):
        x = self.conv2d(x, channel)
        x = self.conv2d_output(x, self.num_anchors_per_scale * self.out_channels)
        batch, height, width, channel = x.shape
        x_5d = tf.reshape(x, (batch, height, width, self.num_anchors_per_scale, self.out_channels))
        return x_5d


class FPN(HeadBase):
    def __init__(self, model_cfg, num_anchors_per_scale, out_channels):
        super().__init__(model_cfg, num_anchors_per_scale, out_channels)

    def __call__(self, input_features):
        large = input_features["backbone_l"]
        medium = input_features["backbone_m"]
        small = input_features["backbone_s"]
        conv = self.conv_5x(large, 512)
        conv_lbbox = self.make_output(conv, 1024)

        conv_medium = self.upsample_concat(large, medium, 256)
        conv = self.conv_5x(conv_medium, 256)
        conv_mbbox = self.make_output(conv, 512)

        conv_small = self.upsample_concat(conv_medium, small, 128)
        conv = self.conv_5x(conv_small, 128)
        conv_sbbox = self.make_output(conv, 256)
        conv_result = {"feature_l": conv_lbbox, "feature_m": conv_mbbox, "feature_s": conv_sbbox}
        return conv_result

    def upsample_concat(self, upper, lower, channel):
        x = self.conv2d_k1(upper, channel)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.concat([x, lower], axis=-1)
        return x


class FeatureDecoder:
    def __init__(self, output_compos, anchors_per_scale):
        """
        :param output_compos: Config.Model.Output.OUT_COMPOSITION
        :param anchors_per_scale: anchor box sizes in ratio per scale
        """
        self.output_compos = output_compos
        self.anchors_per_scale = anchors_per_scale

    def __call__(self, feature, scale_name: str):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_name: scale name e.g. "feature_l"
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        slices = mu.slice_features(feature, self.output_compos)
        anchors_ratio = self.anchors_per_scale[scale_name.replace("feature", "anchor")]

        box_yx = self.decode_yx(slices["yxhw"])
        box_hw = self.decode_hw(slices["yxhw"], anchors_ratio)
        objectness = tf.sigmoid(slices["object"])
        cat_probs = tf.sigmoid(slices["cat_pr"])

        bbox_pred = tf.concat([box_yx, box_hw, objectness, cat_probs], axis=-1)
        assert bbox_pred.shape == feature.shape
        return tf.cast(bbox_pred, dtype=tf.float32)

    def decode_yx(self, yxhw_raw):
        """
        :param yxhw_raw: (batch, grid_h, grid_w, anchor, 4)
        :return: yx_dec = yx coordinates of box centers in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        grid_h, grid_w = yxhw_raw.shape[1:3]
        yx_raw = yxhw_raw[..., :2]
        """
        Original yolo v3 implementation: yx_dec = tf.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw 
        """
        # grid_x: (grid_h, grid_w)
        grid_x, grid_y = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        # grid: (grid_h, grid_w, 2)
        grid = tf.stack([grid_y, grid_x], axis=-1)
        grid = tf.reshape(grid, (1, grid_h, grid_w, 1, 2))
        grid = tf.cast(grid, tf.float32)
        divider = tf.reshape([grid_h, grid_w], (1, 1, 1, 1, 2))
        divider = tf.cast(divider, tf.float32)

        yx_box = tf.sigmoid(yx_raw) * 1.4 - 0.2
        # [(batch, grid_h, grid_w, anchor, 2) + (1, grid_h, grid_w, 1, 2)] / (1, 1, 1, 1, 2)
        yx_dec = (yx_box + grid) / divider
        return yx_dec

    def decode_hw(self, yxhw_raw, anchors_ratio):
        """
        :param yxhw_raw: (batch, grid_h, grid_w, anchor, 4)
        :param anchors_ratio: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        hw_raw = yxhw_raw[..., 2:]
        num_anc, channel = anchors_ratio.shape     # (3, 2)
        anchors_tf = tf.reshape(anchors_ratio, (1, 1, 1, num_anc, channel))
        hw_dec = tf.exp(hw_raw) * anchors_tf
        return hw_dec


# ==================================================
from config import Config as cfg


def test_feature_decoder():
    print("===== start test_feature_decoder")
    anchors = {"anchor_l": [[10, 20], [30, 40], [50, 60]],
               "anchor_m": [[10, 20], [30, 40], [50, 60]],
               "anchor_s": [[10, 20], [30, 40], [50, 60]],
               }
    imshape = (128, 256, 3)
    decode_feature = FeatureDecoder(cfg.Model, anchors, imshape)
    # feature: (batch, grid_h, grid_w, anchor, channel(yxhw+obj+categories))
    feature = tf.zeros((4, 8, 16, 3, 9))
    decoded = decode_feature(feature, "feature_l")
    # batch=0, grid_y=2, grid_x=2, anchor=0 ([10,20])
    single_pred = decoded[0, 2, 2, 0]
    print("decoded feature in single box prediction", single_pred)
    # objectness, category probabilities: 0 -> 0.5
    assert np.isclose(single_pred[4:].numpy(), 0.5).all()
    # check y, x, h, w
    assert single_pred[0] == (2 + 0.5) / 8.
    assert single_pred[1] == (2 + 0.5) / 16.
    assert single_pred[2] == 10. / 128
    assert single_pred[3] == 20. / 256
    print("!!! test_feature_decoder passed !!!")


if __name__ == "__main__":
    test_feature_decoder()
