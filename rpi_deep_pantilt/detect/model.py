# Python
import argparse
import logging
import pathlib

# lib
import tensorflow as tf
import numpy as np
from PIL import Image

from object_detection.utils import label_map_util
import object_detection.utils.visualization_utils as vis_util
from object_detection.models.keras_models.mobilenet_v2 import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


# App
from .camera import PiCameraStream

logging.basicConfig()

# ssdlite_mobilenet_v2_coco_2018_05_09

class SSDLite_MobileNet_V2_Coco(object):

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'

    def __init__(
        self,
        base_url='http://download.tensorflow.org/models/object_detection/',
        model_name='ssdlite_mobilenet_v2_coco_2018_05_09',
        input_shape=(256,256),
        resolution=(320, 240),
        framerate=24,
        vflip=False,
        hflip=False,
        rotation=0

    ):

        self.base_url = base_url
        self.model_name = model_name
        self.model_url = base_url + model_name
        self.model_file = model_name + '.tar.gz'

        self.model_dir = tf.keras.utils.get_file(
            fname=model_name, 
            origin=base_url + self.model_file,
            untar=True
        )

        self.model_dir = pathlib.Path(self.model_dir)/"saved_model"


        model = tf.saved_model.load(str(self.model_dir))
        
        self.model = model.signatures['serving_default']

        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)
        logging.info(f'loaded labels from {self.PATH_TO_LABELS} \n {self.category_index}')

        logging.info(f'initialized model {model_name} \n')
        logging.info(f'model inputs: {self.model.inputs}')
        logging.info(f'model outputs: {self.model.output_dtypes} \n {self.model.output_shapes}')
        #logging.info(f'model summary: {self.model.summary}')

    def create_overlay(self, image_np, output_dict):

        image_np = image_np.copy()

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        
        return Image.fromarray(image_np).tobytes()

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
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = self.model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
        return output_dict
    
    def tflite_convert():
        pass
    
    def tflite_interpreter():
        pass

