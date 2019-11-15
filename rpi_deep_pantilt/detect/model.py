# Python
import logging
import argparse

# lib
import tensorflow as tf
import numpy as np

# App
from rpi_deep_pantilt.detect camera import PiCameraStream

logging.basicConfig()

# ssdlite_mobilenet_v2_coco_2018_05_09

class SSDLite_MobileNet_V2_Coco(object):

    def __init__(
        self,
        base_url='http://download.tensorflow.org/models/object_detection/',
        model_name='ssdlite_mobilenet_v2_coco_2018_05_09',
        input_shape=(256,256)

    )

    self.base_url = base_url
    self.model_name = model_name
    self.model_url = base_url + model_name
    self.model_file = model_name + '.tar.gz'

    self.model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + self.model_file,
        untar=True
    )

    self.model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(self.model_dir))
    
    self.model = model.signatures['serving_default']

    logger.info(f'initialized model {model_name} \n')
    logger.info(f'model inputs: {self.model.inputs}')
    logger.info(f'model outputs: {self.model.output_dtypes} \n {self.model.output_shapes}')
    logger.info(f'model summary: {self.model.summary}')



class MobileNetV2Base():
    def __init__(self,
                 input_shape=None,
                 alpha=1.0,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 ):
        self.tflite_interpreter = None

        self.input_shape = input_shape
        self.include_top = include_top
        self.model_base = tf.keras.applications.mobilenet_v2.MobileNetV2(
            alpha=alpha,
            classes=classes,
            include_top=include_top,
            input_shape=input_shape,
            input_tensor=input_tensor,
            pooling=pooling,
            weights=weights,
        )
        logger.info(self.model_base.summary())

    def predict(self, frame):
        # expand 3D RGB frame into 4D batch
        sample = np.expand_dims(frame, axis=0)
        processed_sample = preprocess_input(sample.astype(np.float32))
        features = self.model_base.predict(processed_sample)
        decoded_features = decode_predictions(features)
        return decoded_features

    def tflite_convert_from_keras_model_file(self, output_dir='includes/', output_filename='mobilenet_v2_imagenet.tflite', keras_model_file='includes/mobilenet_v2_imagenet.h5'):
        # @todo TFLiteConverter.from_keras_model() is only available in the tf-nightly-2.0-preview build right now
        # https://groups.google.com/a/tensorflow.org/forum/#!searchin/developers/from_keras_model%7Csort:date/developers/Mx_EaHM1X2c/rx8Tm-24DQAJ
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        converter = tf.lite.TFLiteConverter.from_keras_model_file(
            keras_model_file)
        tflite_model = converter.convert()
        if output_dir and output_filename:
            with open(output_dir + output_filename, 'wb') as f:
                f.write(tflite_model)
                logger.info('Wrote {}'.format(output_dir + output_filename))
        return tflite_model

    def tflite_convert_from_keras_model(self, output_dir='includes/', output_filename='mobilenet_v2_imagenet.tflite'):
        # @todo TFLiteConverter.from_keras_model() is only available in the tf-nightly-2.0-preview build right now
        # https://groups.google.com/a/tensorflow.org/forum/#!searchin/developers/from_keras_model%7Csort:date/developers/Mx_EaHM1X2c/rx8Tm-24DQAJ
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model_base)
        tflite_model = converter.convert()
        if output_dir and output_filename:
            with open(output_dir + output_filename, 'wb') as f:
                f.write(tflite_model)
                logger.info('Wrote {}'.format(output_dir + output_filename))
        return tflite_model

    def init_tflite_interpreter(self, model_path='includes/mobilenet_v2_imagenet.tflite'):
        '''
            https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/lite/Interpreter
            This makes the TensorFlow Lite interpreter accessible in Python. 
            It is possible to use this interpreter in a multithreaded Python environment, but you must be sure to call functions of a particular instance from only one thread at a time. 
            So if you want to have 4 threads running different inferences simultaneously, create an interpreter for each one as thread-local data. 
            Similarly, if you are calling invoke() in one thread on a single interpreter but you want to use tensor() on another thread once it is done
            you must use a synchronization primitive between the threads to ensure invoke has returned before calling tensor().

        '''
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=model_path)
        self.tflite_interpreter.allocate_tensors()
        logger.info('Initialized tflite Python interpreter \n',
                    self.tflite_interpreter)

        self.tflite_input_details = self.tflite_interpreter.get_input_details()
        logger.info('tflite input details \n', self.tflite_input_details)

        self.tflite_output_details = self.tflite_interpreter.get_output_details()
        logger.info('tflite output details \n',
                    self.tflite_output_details)

        return self.tflite_interpreter

    def tflite_predict(self, frame, input_shape=None):
        if not self.tflite_interpreter:
            self.init_tflite_interpreter()

        dtype = self.tflite_input_details[0].get('dtype')

        # expand 3D RGB frame into 4D batch (of 1 item)
        sample = np.expand_dims(frame, axis=0)
        processed_sample = preprocess_input(sample.astype(dtype))

        self.tflite_interpreter.set_tensor(
            self.tflite_input_details[0]['index'], processed_sample)
        self.tflite_interpreter.invoke()

        features = self.tflite_interpreter.get_tensor(
            self.tflite_output_details[0]['index'])
        decoded_features = decode_predictions(features)

        return decoded_features

    def init_training_model(self, train_dir='data/'):

        if self.include_top is True:
            raise ValueError(
                'FATAL: Cannot re-train a model_base initialized with include_top=True. Init with include_top=False if you want to train additional Dense layers')

        conv_base = self.model_base

        # Freeze the convolutional base to prevent its weights from being re-trained
        # If the base is not frozen, weight updates will be propagated through the network -  this would destroy the learned representation benchmarked in mobilenetv2
        conv_base.trainable = False

        model = tf.keras.models.Sequential()
        model.add(conv_base)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='softmax'))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')
    args = parser.parse_args()
    return args



def main(args):
    model = MobileNetV2Base(include_top=args.include_top)
    capture_manager = PiCameraStream()
    capture_manager.start()

    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            if args.tflite:
                prediction = model.tflite_predict(frame)
            else:
                prediction = model.predict(frame)
            logging.info(prediction)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
