# Python
import logging
import pathlib
import os
import sys

# lib
import numpy as np
from PIL import Image
import tensorflow as tf

from rpi_deep_pantilt import __path__ as rpi_deep_pantilt_path
from rpi_deep_pantilt.detect.util.label import create_category_index_from_labelmap
from rpi_deep_pantilt.detect.util.visualization import visualize_boxes_and_labels_on_image_array

LABELS = ['face']


class FaceSSD_MobileNet_V2_EdgeTPU(object):

    EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
    PATH_TO_LABELS = rpi_deep_pantilt_path[0] + '/data/facessd_label_map.pbtxt'

    def __init__(
        self,
        base_url='https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.0.1/',
        model_name='facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2',
        input_shape=(320, 320),
        min_score_thresh=0.50,
        tflite_model_file='model_postprocessed_quantized_128_uint8_edgetpu.tflite'
    ):

        self.base_url = base_url
        self.model_name = model_name
        self.model_file = model_name + '.tar.gz'
        self.model_url = base_url + self.model_file
        self.tflite_model_file = tflite_model_file

        self.model_dir = tf.keras.utils.get_file(
            fname=self.model_file,
            origin=self.model_url,
            untar=True,
            cache_subdir='models'
        )

        self.min_score_thresh = min_score_thresh

        self.model_path = os.path.splitext(
            os.path.splitext(self.model_dir)[0]
        )[0] + f'/{self.tflite_model_file}'

        try:
            from tflite_runtime import interpreter as coral_tflite_interpreter
        except ImportError as e:
            logging.error(e)
            logging.error('Please install Edge TPU tflite_runtime:')
            logging.error(
                '$ pip install 	https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl')
            sys.exit(1)

        self.tflite_interpreter = coral_tflite_interpreter.Interpreter(
            model_path=self.model_path,
            experimental_delegates=[
                tf.lite.experimental.load_delegate(self.EDGETPU_SHARED_LIB)
            ]
        )

        self.tflite_interpreter.allocate_tensors()

        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        self.category_index = create_category_index_from_labelmap(
            self.PATH_TO_LABELS, use_display_name=True)

        logging.info(
            f'loaded labels from {self.PATH_TO_LABELS} \n {self.category_index}')

        logging.info(f'initialized model {model_name} \n')
        logging.info(
            f'model inputs: {self.input_details} \n {self.input_details}')
        logging.info(
            f'model outputs: {self.output_details} \n {self.output_details}')

    def label_to_category_index(self, labels):
        # @todo :trashfire:
        return tuple(map(
            lambda x: x['id'],
            filter(
                lambda x: x['name'] in labels, self.category_index.values()
            )
        ))

    def label_display_name_by_idx(self, idx):
        return self.category_index[idx]['display_name']

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
            max_boxes_to_draw=3
        )

        img = Image.fromarray(image_np)

        return img.tobytes()

    def predict(self, image):
        '''
            image - np.array (3 RGB channels)

            returns <dict>
                {
                    'detection_classes': int64,
                    'num_detections': int64
                    'detection_masks': ...
                }
        '''

        image = np.asarray(image)
        # normalize 0 - 255 RGB to values between (-1, 1)
        #image = (image / 128.0) - 1

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.

        input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        self.tflite_interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)

        self.tflite_interpreter.invoke()

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

        box_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[0]['index']))
        class_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[1]['index']))
        score_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[2]['index']))
        num_detections = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[3]['index']))

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


class FaceSSD_MobileNet_V2(object):

    PATH_TO_LABELS = rpi_deep_pantilt_path[0] + '/data/facessd_label_map.pbtxt'

    def __init__(
        self,
        base_url='https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.0.1/',
        model_name='facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2',
        input_shape=(320, 320),
        min_score_thresh=0.6

    ):

        self.base_url = base_url
        self.model_name = model_name
        self.model_file = model_name + '.tar.gz'
        self.model_url = base_url + self.model_file
        self.min_score_thresh = min_score_thresh

        self.model_dir = tf.keras.utils.get_file(
            fname=self.model_name,
            origin=self.model_url,
            untar=True,
            cache_subdir='models'
        )

        self.model_path = os.path.splitext(
            os.path.splitext(self.model_dir)[0]
        )[0] + '/model_postprocessed.tflite'

        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=self.model_path,
        )
        self.tflite_interpreter.allocate_tensors()

        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        self.category_index = create_category_index_from_labelmap(
            self.PATH_TO_LABELS, use_display_name=True)

        logging.info(
            f'loaded labels from {self.PATH_TO_LABELS} \n {self.category_index}')

        logging.info(f'initialized model {model_name} \n')
        logging.info(
            f'model inputs: {self.input_details} \n {self.input_details}')
        logging.info(
            f'model outputs: {self.output_details} \n {self.output_details}')

    def label_to_category_index(self, labels):
        # @todo :trashfire:
        return tuple(map(
            lambda x: x['id'],
            filter(
                lambda x: x['name'] in labels, self.category_index.values()
            )
        ))

    def label_display_name_by_idx(self, idx):
        return self.category_index[idx]['display_name']

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
            max_boxes_to_draw=3
        )

        img = Image.fromarray(image_np)

        return img.tobytes()

    def predict(self, image):
        '''
            image - np.array (3 RGB channels)

            returns <dict>
                {
                    'detection_classes': int64,
                    'num_detections': int64
                    'detection_masks': ...
                }
        '''

        image = np.asarray(image)
        # normalize 0 - 255 RGB to values between (-1, 1)
        image = (image / 128.0) - 1

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.

        input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        self.tflite_interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)

        self.tflite_interpreter.invoke()

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
            'detection_boxes': box_data,  # 10, 4
            'detection_classes':  class_data,  # 10
            'detection_scores': score_data,  # 10,
            'num_detections': len(num_detections)
        }
