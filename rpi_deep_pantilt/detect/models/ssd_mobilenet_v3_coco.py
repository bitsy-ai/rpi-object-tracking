# Python
import argparse
import logging
import pathlib

# lib
from google.protobuf import text_format
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import tensorflow as tf

# from object_detection import export_tflite_ssd_graph

from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
import object_detection.utils.visualization_utils as vis_util
import object_detection.utils.ops as utils_ops
from object_detection.models.keras_models.mobilenet_v2 import mobilenet_v2
from object_detection.core import post_processing

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

class SSDMobileNet_V3_Small_Coco_PostProcessed(object):

    PATH_TO_LABELS = 'lib/tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt'

    # http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2019_08_14.tar.gz

    def __init__(
        self,
        base_url='http://download.tensorflow.org/models/object_detection/',
        model_name= 'ssd_mobilenet_v3_coco',# 'ssd_mobilenet_v3_small_coco_2019_08_14',
        input_shape=(320, 320)
    ):

        self.base_url = base_url
        self.model_name = model_name
        self.model_url = base_url + model_name
        self.model_file = model_name + '.tar.gz'

        # self.model_dir = tf.keras.utils.get_file(
        #     fname=model_name,
        #     origin='https://github.com/tensorflow/models/files/3819458/ssd_mobilenet_v3_coco.zip',# base_url + self.model_file,
        #     #extract=True
        # )

        # self.model_dir = pathlib.Path(self.model_dir)
        self.model_path = '/home/pi/.keras/datasets/ssd_mobilenet_v3_small_coco_2019_08_14_postprocessed/model_postprocessed.tflite' #'/model.tflite'
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=self.model_path
        )
        self.tflite_interpreter.allocate_tensors()

        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        self.category_index = label_map_util.create_category_index_from_labelmap(
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
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.6,
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
        image =  (image / 128.0) - 1

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
        class_data = tf.squeeze(class_data, axis=[0]).numpy().astype(np.int64) + 1
        #background_idxs = np.where(class_data == 0)[0]
        #class_data = np.delete(class_data, background_idxs).astype(np.int64)


        box_data = tf.squeeze(box_data, axis=[0]).numpy()
        #box_data = np.delete(box_data, background_idxs, axis=0)

        score_data =  tf.squeeze(score_data, axis=[0]).numpy()
        #score_data = np.delete(score_data, background_idxs)

        return {
            'detection_boxes': box_data, #tf.squeeze(box_data, axis=[0]).numpy(),
            'detection_classes':  class_data, #tf.squeeze(class_data, axis=[0]).numpy().astype(np.int64),
            'detection_scores': score_data, # tf.squeeze(score_data, axis=[0]).numpy(),
            'num_detections': len(num_detections)
        }


class SSDMobileNet_V3_Small_Coco(object):

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'

    # http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_small_coco_2019_08_14.tar.gz

    def __init__(
        self,
        base_url='http://download.tensorflow.org/models/object_detection/',
        model_name= 'ssd_mobilenet_v3_small_coco_2019_08_14',
        input_shape=(320, 320)
    ):

        self.base_url = base_url
        self.model_name = model_name
        self.model_url = base_url + model_name
        self.model_file = model_name + '.tar.gz'

        self.model_dir = tf.keras.utils.get_file(
            fname=model_name,
            origin=base_url + self.model_file,
            #extract=True
        )

        self.model_dir = pathlib.Path(self.model_dir)
        self.model_path = str(self.model_dir) + '/model.tflite'
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=self.model_path
        )
        self.tflite_interpreter.allocate_tensors()

        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        self.category_index = label_map_util.create_category_index_from_labelmap(
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

    def create_overlay(self, image_np, output_dict):

        image_np = image_np.copy()

        # draw bounding boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
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

        #zero_point = self.input_details[0]['quantization'][0]
        #scale = self.input_details[0]['quantization'][1]
        #image = (image - zero_point) * scale

        #image =  image / 2.0 + 128.0

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        #import pdb; pdb.set_trace()

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        self.tflite_interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)

        self.tflite_interpreter.invoke()

        # class_data = tf.squeeze(tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
        #     self.output_details[0]['index'])))

        box_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[0]['index']))

        # box_data = tf.squeeze(tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
        #     self.output_details[1]['index'])),  axis=[0])

        class_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(self.output_details[1]['index']))

        score_data = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[2]['index']))
        
        num_detections = tf.convert_to_tensor(self.tflite_interpreter.get_tensor(
            self.output_details[3]['index']))

        # TFLite_Detection_PostProcess custom op node has four outputs:
        # detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
        # locations
        # detection_classes: a float32 tensor of shape [1, num_boxes]
        # with class indices
        # detection_scores: a float32 tensor of shape [1, num_boxes]
        # with class scores
        # num_boxes: a float32 tensor of size 1 containing the number of detected
        # boxes

        #   the graph has two outputs:
        #    'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
        #     containing the encoded box predictions.
        #    'raw_outputs/class_predictions': a float32 tensor of shape
        #     [1, num_anchors, num_classes] containing the class scores for each anchor
        #     after applying score conversion.

        # output_array = coco_utils.convert_predictions_to_coco_annotations({
        #     'source_id': [np.arange(0, num_detections)],
        #     'num_detections': num_detections,
        #     'detection_boxes': box_data.numpy(),
        #     'detection_classes': class_data.numpy(),
        #     'detection_scores': score_data.numpy()
        # })
        # odict = post_processing.multiclass_non_max_suppression(box_data,class_data,  0.6, 0.6, 2)   
        # odict = post_processing.batch_multiclass_non_max_suppression(box_data, class_data, 0.2, 0.2, 4)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        # output_dict = {key: value[0, :num_detections].numpy()
        #                for key, value in output_dict.items()
        # }
        # output_dict['num_detections'] = num_detections

        # # detection_classes should be ints.
        # output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        #     np.int64)

        # num_detections = int(num_detections.numpy()[0])
        
        # import pdb; pdb.set_trace()

        # filter out indexes where n=0 
        class_data = tf.squeeze(class_data, axis=[0]).numpy()
        background_idxs = np.where(class_data == 0)[0]
        class_data = np.delete(class_data, background_idxs).astype(np.int64)


        box_data = tf.squeeze(box_data, axis=[0]).numpy()
        box_data = np.delete(box_data, background_idxs, axis=0)

        score_data =  tf.squeeze(score_data, axis=[0]).numpy()
        score_data = np.delete(score_data, background_idxs)



        return {
            'detection_boxes': box_data, #tf.squeeze(box_data, axis=[0]).numpy(),
            'detection_classes':  class_data, #tf.squeeze(class_data, axis=[0]).numpy().astype(np.int64),
            'detection_scores': score_data, # tf.squeeze(score_data, axis=[0]).numpy(),
            'num_detections': len(class_data)
        }

    def label_to_category_index(self, labels):
        # @todo :trashfire:
        return tuple(map(
            lambda x: x['id'],
            filter(
                lambda x: x['name'] in labels, self.category_index.values()
            )
        ))

