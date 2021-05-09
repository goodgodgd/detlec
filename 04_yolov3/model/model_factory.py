import tensorflow as tf
import model.backbone as back
import model.head as head
import utils.util_function as uf

from config import Config as cfg
import model.model_util as mu


class ModelFactory:
    def __init__(self, batch_size, input_shape, anchors_per_scale,
                 backbone_name=cfg.Model.Structure.BACKBONE,
                 head_name=cfg.Model.Structure.HEAD,
                 backbone_conv_args=cfg.Model.Structure.BACKBONE_CONV_ARGS,
                 head_conv_args=cfg.Model.Structure.HEAD_CONV_ARGS,
                 num_anchors_per_scale=cfg.Model.Output.NUM_ANCHORS_PER_SCALE,
                 out_channels=cfg.Model.Output.OUT_CHANNELS,
                 ):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors_per_scale = anchors_per_scale
        self.backbone_name = backbone_name
        self.head_name = head_name
        self.backbone_conv_args = backbone_conv_args
        self.head_conv_args = head_conv_args
        self.num_anchors_per_scale = num_anchors_per_scale
        self.out_channels = out_channels
        mu.CustomConv2D.CALL_COUNT = -1
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={self.backbone_name}, HEAD={self.head_name}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.backbone_name, self.backbone_conv_args)
        head_model = head.head_factory(self.head_name, self.head_conv_args, self.num_anchors_per_scale, self.out_channels)
        input_tensor = tf.keras.layers.Input(shape=self.input_shape, batch_size=self.batch_size)
        backbone_features = backbone_model(input_tensor)
        head_features = head_model(backbone_features)
        output_features = dict()
        decode_features = head.FeatureDecoder(self.anchors_per_scale)
        for i, (scale, feature) in enumerate(head_features.items()):
            output_features[scale] = decode_features(feature, scale)
        yolo_model = tf.keras.Model(inputs=input_tensor, outputs=output_features, name="yolo_model")
        return yolo_model


# ==================================================


def test_model_factory():
    print("===== start test_model_factory")
    anchors = {"anchor_l": [[10, 20], [30, 40], [50, 60]],
               "anchor_m": [[10, 20], [30, 40], [50, 60]],
               "anchor_s": [[10, 20], [30, 40], [50, 60]],
               }
    batch_size = 1
    imshape = (128, 256, 3)
    model = ModelFactory(batch_size, imshape, anchors).get_model()
    input_tensor = tf.zeros((batch_size, 128, 256, 3))
    output = model(input_tensor)
    print("print output key and tensor shape")
    for key, val in output.items():
        print(key, val.shape)
    print("!!! test_model_factory passed !!!")


if __name__ == "__main__":
    uf.set_gpu_configs()
    test_model_factory()
