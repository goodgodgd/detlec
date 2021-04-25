import tensorflow as tf
import numpy as np

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
        anchors, anc_channels = self.model_cfg.Output.NUM_ANCHORS_PER_SCALE, self.model_cfg.Output.OUT_CHANNELS
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
    def __init__(self, model_cfg, anchors_per_scale, imshape):
        """
        :param model_cfg: Config.Model
        :param anchors_per_scale: anchor box sizes in pixel per scale,
                                  e.g. {"anchor_s": [[8.0, 6.2], [18.5, 9.8], [14.2, 20.3]], ...}
        :param imshape: input image shape
        """
        self.model_cfg = model_cfg
        # rescale anchors in pixels to ratio to image
        anchors_ratio = {key: np.array(anchors) / np.array([imshape[:2]]) for key, anchors in anchors_per_scale.items()}
        self.anchors_per_scale = anchors_ratio

    def __call__(self, feature, scale_name: str):
        """
        :param feature: raw feature map predicted by model (batch, grid_h, grid_w, anchor, channel)
        :param scale_name: scale name e.g. "feature_l"
        :return: decoded feature in the same shape e.g. (yxhw, objectness, category probabilities)
        """
        slices = mu.slice_features(feature, self.model_cfg.Output.OUT_COMPOSITION)
        anchors = self.anchors_per_scale[scale_name.replace("feature", "anchor")]

        box_yx = self.decode_yx(slices["yxhw"])
        box_hw = self.decode_hw(slices["yxhw"], anchors)
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

    def decode_hw(self, yxhw_raw, anchors_np):
        """
        :param yxhw_raw: (batch, grid_h, grid_w, anchor, 4)
        :param anchors_np: [height, width]s of anchors in ratio to image (0~1), (anchor, 2)
        :return: hw_dec = heights and widths of boxes in ratio to image (batch, grid_h, grid_w, anchor, 2)
        """
        hw_raw = yxhw_raw[..., 2:]
        """
        Original yolo v3 implementation: yx_dec = tf.sigmoid(yx_raw)
        For yx_dec to be close to 0 or 1, yx_raw should be -+ infinity
        By expanding activation range -0.2 ~ 1.4, yx_dec can be close to 0 or 1 from moderate values of yx_raw 
        """
        num_anc, channel = anchors_np.shape
        anchors_tf = tf.reshape(anchors_np, (1, 1, 1, num_anc, channel))
        anchors_tf = tf.cast(anchors_tf, tf.float32)
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
    feature = tf.zeros((4, 8, 16, 3, 9))
    decoded = decode_feature(feature, "feature_l")
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
