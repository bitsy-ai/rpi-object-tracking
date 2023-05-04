import tensorflow as tf

from rpi_deep_pantilt import __path__ as rpi_deep_pantilt_path
from rpi_deep_pantilt.detect.custom.base_predictors import (
    TFLiteDetectionPostProcessPredictor,
)


class FaceSSDMobileNetV2EdgeTPU(TFLiteDetectionPostProcessPredictor):
    """
    Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models

    Non-max supression op (TFLite_Detection_Postprocess) added to graph via tools/tflite-postprocess-ops-128-uint8-quant.sh
    """

    LABELS = ["face"]

    def __init__(
        self,
        model_uri="https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.1.1/facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2.tar.gz",
        model_name="facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2",
        input_shape=(320, 320),
        min_score_thresh=0.50,
        input_type=tf.uint8,
        tflite_file="model_postprocessed_quantized_128_uint8_edgetpu.tflite",
        label_file=rpi_deep_pantilt_path[0] + "/data/facessd_label_map.pbtxt",
    ):

        super().__init__(
            model_name=model_name,
            tflite_file=tflite_file,
            label_file=label_file,
            model_uri=model_uri,
            input_shape=input_shape,
            min_score_thresh=min_score_thresh,
            input_type=input_type,
            edge_tpu=True,
        )


class FaceSSDMobileNetV2Int8(TFLiteDetectionPostProcessPredictor):
    """
    Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models

    Non-max supression op (TFLite_Detection_Postprocess) added to graph via tools/tflite-postprocess-ops-128-uint8-quant.sh
    """

    LABELS = ["face"]

    def __init__(
        self,
        model_uri="https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.1.1/facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2.tar.gz",
        model_name="facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2",
        input_shape=(320, 320),
        min_score_thresh=0.50,
        input_type=tf.uint8,
        tflite_file="model_postprocessed_quantized_128_uint8.tflite",
        label_file=rpi_deep_pantilt_path[0] + "/data/facessd_label_map.pbtxt",
    ):

        super().__init__(
            model_name=model_name,
            tflite_file=tflite_file,
            label_file=label_file,
            model_uri=model_uri,
            input_shape=input_shape,
            min_score_thresh=min_score_thresh,
            input_type=input_type,
        )


class FaceSSDMobileNetV2Float32(TFLiteDetectionPostProcessPredictor):
    LABELS = ["face"]

    def __init__(
        self,
        model_uri="https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.1.1/facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2.tar.gz",
        model_name="facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2",
        input_shape=(320, 320),
        min_score_thresh=0.50,
        input_type=tf.float32,
        tflite_file="model_postprocessed.tflite",
        label_file=rpi_deep_pantilt_path[0] + "/data/facessd_label_map.pbtxt",
    ):

        super().__init__(
            model_name=model_name,
            tflite_file=tflite_file,
            label_file=label_file,
            model_uri=model_uri,
            input_shape=input_shape,
            min_score_thresh=min_score_thresh,
            input_type=input_type,
        )
