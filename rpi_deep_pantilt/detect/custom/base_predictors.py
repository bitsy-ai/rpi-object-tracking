
# Python
from abc import ABCMeta, abstractmethod
import logging
import os
import sys


# lib
import numpy as np
from PIL import Image

import tensorflow as tf

from rpi_deep_pantilt.detect.util.label import create_category_index_from_labelmap
from rpi_deep_pantilt.detect.util.visualization import visualize_boxes_and_labels_on_image_array

class BasePredictor(metaclass=ABCMeta):

    EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
    LABELS = []

    def __init__(
        self,
        model_uri,
        model_name,
        tflite_file,
        label_file,
        input_shape=(320,320),
        input_type=tf.uint8,
        edge_tpu=False,
        min_score_thresh=0.50
    ):

        self.model_uri = model_uri
        self.model_name = model_name
        self.tflite_file = tflite_file
        self.input_type = input_type
        self.label_file = label_file


        self.model_dir = tf.keras.utils.get_file(
            fname=self.model_name,
            origin=self.model_uri,
            untar=True,
            cache_subdir='models'
        )

        logging.info(f'Downloaded {model_name} to {self.model_dir}')

        self.min_score_thresh = min_score_thresh

        self.model_path = os.path.splitext(
            os.path.splitext(self.model_dir)[0]
        )[0] + f'/{self.tflite_file}'

        if edge_tpu:
            try:
                logging.warning('Loading Coral tflite_runtime for Edge TPU')
                from tflite_runtime import interpreter as coral_tflite_interpreter
                self.tflite_interpreter = coral_tflite_interpreter.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[
                        tf.lite.experimental.load_delegate(self.EDGETPU_SHARED_LIB)
                    ]
                )
            except ImportError as e:
                logging.warning('Failed to import Coral Edge TPU tflite_runtime. Falling back to TensorFlow tflite runtime. If you are using an Edge TPU, please run: \n')
                logging.warning(
                    '$ pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl')
                self.tflite_interpreter = tf.lite.Interpreter(
                    model_path=self.model_path,
                )
        else:
            self.tflite_interpreter = tf.lite.Interpreter(
                model_path=self.model_path,
            )            

        self.tflite_interpreter.allocate_tensors()
        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        self.category_index = create_category_index_from_labelmap(
            label_file, use_display_name=True)

        logging.info(
            f'loaded labels from {self.label_file} \n {self.category_index}')

        logging.info(f'initialized model {model_name} \n')
        logging.info(
            f'model inputs: {self.input_details} \n {self.input_details}')
        logging.info(
            f'model outputs: {self.output_details} \n {self.output_details}')

    def label_to_category_index(self, labels):
        return tuple(map(
            lambda x: x['id'],
            filter(
                lambda x: x['name'] in labels, self.category_index.values()
            )
        ))

    def label_display_name_by_idx(self, idx):
        return self.category_index[idx]['display_name']

    @abstractmethod
    def create_overlay(self, image_np, output_dict):
        pass
    
    @abstractmethod
    def predict(self, image):
        pass

    @classmethod
    def validate_labels(cls, labels):
        return all([x in cls.LABELS for x in labels])

# TFLite_Detection_PostProcess custom op is a non-max supression op (NMS)
# utilized in TensorFlow's Object Detection API / Model zoo
# https://github.com/tensorflow/models/tree/master/research/object_detection
# TFLite_Detection_PostProcess custom op node has four outputs:
# detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
# locations
# detection_classes: a float32 tensor of shape [1, num_boxes]
# with class indices
# detection_scores: a float32 tensor of shape [1, num_boxes]
# with class scores
# num_boxes: a float32 tensor of size 1 containing the number of detected
# boxes

# Without the PostProcessing ops, the graph has two outputs:
#    'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
#     containing the encoded box predictions.
#    'raw_outputs/class_predictions': a float32 tensor of shape
#     [1, num_anchors, num_classes] containing the class scores for each anchor
#     after applying score conversion.

class TFLiteDetectionPostProcessOverlay(BasePredictor):

    def __init__(self, *args, max_boxes_to_draw=3, **kwargs):

        self.max_boxes_to_draw = max_boxes_to_draw
        super().__init__(*args, **kwargs)

    def create_overlay(self, image_np, output_dict):

        image_np = image_np.copy()

        # draw bounding boxes
        visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=self.min_score_thresh,
            max_boxes_to_draw=self.max_boxes_to_draw
        )

        img = Image.fromarray(image_np)

        return img.tobytes() 

class TFLiteDetectionPostProcessPredictor(TFLiteDetectionPostProcessOverlay):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # float16 is not a supported data type (yet)
        assert self.input_type is tf.float32 or self.input_type is tf.uint8
    
    def predict(self, image):

        image = np.asarray(image)
        # normalize 0 - 255 RGB to values between (-1, 1) if float32 data type
        if self.input_type is tf.float32:
            image = (image / 128.0) - 1
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image, dtype=self.input_type)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        self.tflite_interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)

        self.tflite_interpreter.invoke()

        box_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[0]['index']))
        class_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[1]['index']))
        score_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[2]['index']))
        num_detections = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[3]['index']))

        # hilarious, but it seems like all classes predictions are off by 1 idx
        class_data = tf.squeeze(
            class_data, axis=[0]).numpy().astype(np.int64) + 1
        box_data = tf.squeeze(box_data, axis=[0]).numpy()
        score_data = tf.squeeze(score_data, axis=[0]).numpy()


        return {
            'detection_boxes': box_data,
            'detection_classes':  class_data,
            'detection_scores': score_data,
            'num_detections': len(num_detections)
        }