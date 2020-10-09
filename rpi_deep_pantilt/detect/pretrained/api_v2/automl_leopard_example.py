import tensorflow as tf

from rpi_deep_pantilt import __path__ as rpi_deep_pantilt_path
from rpi_deep_pantilt.detect.custom.base_predictors import (
    TFLiteDetectionPostProcessPredictor,
)


class LeopardAutoMLInt8(TFLiteDetectionPostProcessPredictor):

    LABELS = ["leopard"]

    def __init__(
        self,
        model_name="leopard_detector",
        tflite_file=rpi_deep_pantilt_path[0]
        + "/data/leopard.tflite",  # substitute the FULL path to your .tflite file
        label_file=rpi_deep_pantilt_path[0]
        + "/data/leopard_label_map.pbtxt",  # you only need to change this if adding additional labels (background class is implicitly id:0)
        input_shape=(
            320,
            320,
        ),  # make sure height, width match metadata.json inputShape
        min_score_thresh=0.95,  # increase threshold to reduce false positive results
        input_type=tf.uint8,  # do not change
    ):

        super().__init__(
            model_name=model_name,
            tflite_file=tflite_file,
            label_file=label_file,
            model_uri=None,  # model_uri is used to download remote files; None implies tflite_File and label_file are fully qualified local paths
            input_shape=input_shape,
            min_score_thresh=min_score_thresh,
            input_type=input_type,
        )
