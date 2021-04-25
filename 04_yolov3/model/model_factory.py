import tensorflow as tf
import model.backbone as back
import model.head as head


class ModelFactory:
    def __init__(self, batch_size, input_shape, anchors_per_scale, model_cfg):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors_per_scale = anchors_per_scale
        self.model_cfg = model_cfg
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={model_cfg.Structure.BACKBONE}, HEAD={model_cfg.Structure.HEAD}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.model_cfg.Structure.BACKBONE, self.model_cfg.Structure.CONV_ARGS)
        head_model = head.head_factory(self.model_cfg.Structure.HEAD, self.model_cfg)
        input_tensor = tf.keras.layers.Input(shape=self.input_shape, batch_size=self.batch_size)
        backbone_features = backbone_model(input_tensor)
        head_features = head_model(backbone_features)
        output_features = dict()
        decode_features = head.FeatureDecoder(self.model_cfg, self.anchors_per_scale, self.input_shape)
        for i, (scale, feature) in enumerate(head_features.items()):
            output_features[scale] = decode_features(feature, scale)
        yolo_model = tf.keras.Model(inputs=input_tensor, outputs=output_features, name="yolo_model")
        return yolo_model


# ==================================================
from config import Config as cfg


def test_model_factory():
    print("===== start test_model_factory")
    anchors = {"anchor_l": [[10, 20], [30, 40], [50, 60]],
               "anchor_m": [[10, 20], [30, 40], [50, 60]],
               "anchor_s": [[10, 20], [30, 40], [50, 60]],
               }
    batch_size = 1
    imshape = (128, 256, 3)
    model = ModelFactory(batch_size, imshape, anchors, cfg.Model).get_model()
    input_tensor = tf.zeros((batch_size, 128, 256, 3))
    output = model(input_tensor)
    print("print output key and tensor shape")
    for key, val in output.items():
        print(key, val.shape)
    print("!!! test_model_factory passed !!!")


def set_gpu_configs():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":
    set_gpu_configs()
    test_model_factory()
