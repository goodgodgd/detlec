import tensorflow as tf
import model.backbone as back
import model.head as head
import utils.util_function as uf

import config as cfg
import model.model_util as mu


class ModelFactory:
    def __init__(self, batch_size, input_shape,
                 anchors_ratio=cfg.ModelOutput.ANCHORS_RATIO,
                 backbone_name=cfg.Architecture.BACKBONE,
                 head_name=cfg.Architecture.HEAD,
                 backbone_conv_args=cfg.Architecture.BACKBONE_CONV_ARGS,
                 head_conv_args=cfg.Architecture.HEAD_CONV_ARGS,
                 num_anchors_per_scale=cfg.ModelOutput.NUM_ANCHORS_PER_SCALE,
                 ):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.backbone_name = backbone_name
        self.head_name = head_name
        self.backbone_conv_args = backbone_conv_args
        self.head_conv_args = head_conv_args
        self.num_anchors_per_scale = num_anchors_per_scale
        self.out_channels = sum(list(cfg.ModelOutput.PRED_FMAP_COMPOSITION.values()))
        # slice anchor ratio over scales
        num_scales = len(cfg.ModelOutput.FEATURE_SCALES)
        self.anchors_per_scale = [anchors_ratio[i*num_anchors_per_scale:(i+1)*num_anchors_per_scale] for i in range(num_scales)]
        mu.CustomConv2D.CALL_COUNT = -1
        print(f"[ModelFactory] batch size={batch_size}, input shape={input_shape}")
        print(f"[ModelFactory] backbone={self.backbone_name}, HEAD={self.head_name}")

    def get_model(self):
        backbone_model = back.backbone_factory(self.backbone_name, self.backbone_conv_args)
        head_model = head.head_factory(self.head_name, self.head_conv_args, self.num_anchors_per_scale, self.out_channels)
        feature_decoder = head.FeatureDecoder(self.anchors_per_scale)

        input_tensor = tf.keras.layers.Input(shape=self.input_shape, batch_size=self.batch_size)
        bkbn_features = backbone_model(input_tensor)
        head_features = head_model(bkbn_features)
        head_slices = uf.slice_features_and_merge_dims(head_features, cfg.ModelOutput.PRED_FMAP_COMPOSITION)
        decode_slices = feature_decoder(head_slices, head_features)

        outputs = {"bkbn_logit": bkbn_features, "head_logit": head_features}
        outputs.update(decode_slices)
        outputs = {"fmap": outputs}
        yolo_model = tf.keras.Model(inputs=input_tensor, outputs=outputs, name="yolo_model")
        """
        outputs = {"fmap": 
                    {"bkbn_logit": [(B,GH,GW,C) x 3], 
                     "head_logit": [(B,GH,GW,C) x 3]
                     "yxhw": [(B,GH*GW*A,4) x 3],
                     "object": [(B,GH*GW*A,1) x 3],
                     "category": [(B,GH*GW*A,K) x 3]
                    }
                  }
        (GH,GW = H//S,W//S where S in [8,16,32])
        """
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
