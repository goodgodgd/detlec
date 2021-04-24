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
        decode_features = head.FeatureDecoder(self.model_cfg)
        for i, (scale, feature) in enumerate(head_features.items()):
            output_features[scale] = decode_features(feature, scale)
        yolo_model = tf.keras.Model(inputs=input_tensor, outputs=output_features, name="yolo_model")
        return yolo_model
