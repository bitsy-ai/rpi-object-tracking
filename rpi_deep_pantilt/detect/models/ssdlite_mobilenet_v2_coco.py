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

from official.vision.detection.evaluation import coco_utils

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

logging.getLogger().setLevel(logging.INFO)
class SSDLite_MobileNet_V2_Coco(object):

    PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'

    def __init__(
        self,
        base_url='http://download.tensorflow.org/models/object_detection/',
        model_name='ssdlite_mobilenet_v2_coco_2018_05_09',
        input_shape=(320, 320)
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
        self.input_shape = input_shape
        self.model = model.signatures['serving_default']

        self.category_index = label_map_util.create_category_index_from_labelmap(
            self.PATH_TO_LABELS, use_display_name=True)
        logging.info(
            f'loaded labels from {self.PATH_TO_LABELS} \n {self.category_index}')

        logging.info(f'initialized model {model_name} \n')
        logging.info(f'model inputs: {self.model.inputs}')
        logging.info(
            f'model outputs: {self.model.output_dtypes} \n {self.model.output_shapes}')
        # logging.info(f'model summary: {self.model.summary}')

    def label_to_category_index(self, labels):
        # @todo :trashfire:
        return tuple(map(
            lambda x: x['id'],
            filter(
                lambda x: x['name'] in labels, self.category_index.values()
            )
        ))

    def create_overlay(self, image_np, output_dict, draw_target=None):

        image_np = image_np.copy()

        # draw bounding boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8
        )

        # draw centroid
        img = Image.fromarray(image_np)
        if draw_target is not None:
            # (uleft, uright, bright, bleft)

            xx = draw_target[:2].mean()
            yy = draw_target[2:].mean()

            draw = ImageDraw.Draw(img)
            draw.point([(xx, yy)], fill='black')

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

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        output_dict = self.model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy()
                       for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(
            np.int64)

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

    # saved_model_cli show --dir ~/.keras/datasets/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model --all

    # MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

    # signature_def['serving_default']:
    #   The given SavedModel SignatureDef contains the following input(s):
    #     inputs['inputs'] tensor_info:
    #         dtype: DT_UINT8
    #         shape: (-1, -1, -1, 3)
    #         name: image_tensor:0
    #   The given SavedModel SignatureDef contains the following output(s):
    #     outputs['detection_boxes'] tensor_info:
    #         dtype: DT_FLOAT
    #         shape: (-1, 100, 4)
    #         name: detection_boxes:0
    #     outputs['detection_classes'] tensor_info:
    #         dtype: DT_FLOAT
    #         shape: (-1, 100)
    #         name: detection_classes:0
    #     outputs['detection_scores'] tensor_info:
    #         dtype: DT_FLOAT
    #         shape: (-1, 100)
    #         name: detection_scores:0
    #     outputs['num_detections'] tensor_info:
    #         dtype: DT_FLOAT
    #         shape: (-1)
    #         name: num_detections:0
    #   Method name is: tensorflow/serving/predict

    def tflite_convert(self, output_dir='includes/'):
        # build_tensor_info() not supported in Eager mode
        # tf.compat.v1.disable_eager_execution()

        # ssdlite_mobilenet_v2_coco_2018_05_09 SavedModel SignatureDef defines input / output shapes using `None`
        # e.g. the input shape is (None, None, None, 3)
        # This raises a ValueError in the 2.0 TFlite converter, which only supports None in the 1st dimension
        # import pdb; pdb.set_trace()

        # import pdb
        # pdb.set_trace()

        t_input = self.model.inputs[0]
        t_input.set_shape(
            (1, self.input_shape[0], self.input_shape[1], 3)
        )
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(self.model_dir),
            signature_keys=[
                {'serving_default': self.model.inputs[0].experimental_ref()}
            ]
        )

        tflite_model = converter.convert()

        if output_dir:
            output_filename = f'{output_dir}{self.model_name}_{self.input_shape[0]}_{self.input_shape[1]}.tflite'

            with open(output_filename, 'wb') as f:
                f.write(tflite_model)
                logging.info('Wrote {}'.format(output_filename))
        return tflite_model

    def tflite_interpreter():
        pass
