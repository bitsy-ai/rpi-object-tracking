
import tensorflow as tf

from rpi_deep_pantilt import __path__ as rpi_deep_pantilt_path
from rpi_deep_pantilt.detect.custom.base_predictors import (
    TFLiteDetectionPostProcessPredictor
)

class SSDMobileNetV3EdgeTPU(TFLiteDetectionPostProcessPredictor):
    '''
        Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models

        Non-max supression op (TFLite_Detection_Postprocess) added to graph via tools/tflite-postprocess-ops-128-uint8-quant.sh
    '''
    LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(
        self,
        model_uri='https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.1.1/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz',
        model_name='ssdlite_mobilenet_edgetpu_coco_quant',
        input_shape=(320, 320),
        min_score_thresh=0.50,
        input_type=tf.uint8,
        tflite_file='model_postprocessed_quantized_128_uint8_edgetpu.tflite',
        label_file=rpi_deep_pantilt_path[0] + '/data/mscoco_label_map.pbtxt'
    ):

        super().__init__(
            model_name=model_name,
            tflite_file=tflite_file,
            label_file=label_file,
            model_uri=model_uri,
            input_shape=input_shape,
            min_score_thresh=min_score_thresh,
            input_type=input_type,
            edge_tpu=True
        )

class SSDMobileNetV3Int8(TFLiteDetectionPostProcessPredictor):
    '''
        Model source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models

        Non-max supression op (TFLite_Detection_Postprocess) added to graph via tools/tflite-postprocess-ops-128-uint8-quant.sh
    '''
    LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(
        self,
        model_uri='https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.1.1/ssdlite_mobilenet_edgetpu_coco_quant.tar.gz',
        model_name='ssd_mobilenet_v3_small_coco_2019_08_14',
        input_shape=(320, 320),
        min_score_thresh=0.50,
        input_type=tf.uint8,
        tflite_file='model_postprocessed_quantized_128_uint8.tflite',
        label_file=rpi_deep_pantilt_path[0] + '/data/mscoco_label_map.pbtxt'
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

class SSDMobileNetV3Float32(TFLiteDetectionPostProcessPredictor):
    LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(
        self,
        model_uri='https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.1.1/ssd_mobilenet_v3_small_coco_2019_08_14.tar.gz',
        model_name='ssd_mobilenet_v3_small_coco_2019_08_14',
        input_shape=(320, 320),
        min_score_thresh=0.50,
        input_type=tf.float32,
        tflite_file='model_postprocessed_quantized.tflite',
        label_file=rpi_deep_pantilt_path[0] + '/data/mscoco_label_map.pbtxt'
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