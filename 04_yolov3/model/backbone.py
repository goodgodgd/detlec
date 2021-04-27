from utils.util_class import MyExceptionToCatch
import model.model_util as mu


def backbone_factory(backbone, conv_kwargs):
    if backbone == "Darknet53":
        return Darknet53(conv_kwargs)
    else:
        raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {backbone}")


class BackboneBase:
    def __init__(self, conv_kwargs):
        self.conv2d = mu.CustomConv2D(kernel_size=3, strides=1, scope="back", **conv_kwargs)
        self.conv2d_k1 = mu.CustomConv2D(kernel_size=1, strides=1, scope="back", **conv_kwargs)
        self.conv2d_s2 = mu.CustomConv2D(kernel_size=3, strides=2, scope="back", **conv_kwargs)

    def residual(self, x, filters):
        short_cut = x
        conv = self.conv2d_k1(x, filters // 2)
        conv = self.conv2d(conv, filters)
        return short_cut + conv


class Darknet53(BackboneBase):
    def __init__(self, conv_kwargs):
        super().__init__(conv_kwargs)

    def __call__(self, input_tensor):
        """
        conv'n' represents a feature map of which resolution is (input resolution / 2^n)
        e.g. input_tensor.shape[:2] == conv0.shape[:2], conv0.shape[:2]/8 == conv3.shape[:2]
        """
        features = dict()
        conv0 = self.conv2d(input_tensor, 32)
        conv1 = self.conv2d_s2(conv0, 64)
        conv1 = self.residual(conv1, 64)

        conv2 = self.conv2d_s2(conv1, 128)
        for i in range(2):
            conv2 = self.residual(conv2, 128)

        conv3 = self.conv2d_s2(conv2, 256)
        for i in range(8):
            conv3 = self.residual(conv3, 256)
        features["feature_s"] = conv3

        conv4 = self.conv2d_s2(conv3, 512)
        for i in range(8):
            conv4 = self.residual(conv4, 512)
        features["feature_m"] = conv4

        conv5 = self.conv2d_s2(conv4, 1024)
        for i in range(4):
            conv5 = self.residual(conv5, 1024)
        features["feature_l"] = conv5

        return features



