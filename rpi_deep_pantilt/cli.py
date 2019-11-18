# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import logging
import sys
import click

from detect.camera import PiCameraStream
from detect.model import  SSDLite_MobileNet_V2_Coco


# @click.command()
# def main(args=None):
#     """Console script for rpi_deep_pantilt."""

import argparse

logging.basicConfig()


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
    model = SSDLite_MobileNet_V2_Coco()
    capture_manager = PiCameraStream()
    capture_manager.start()

    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            if args.tflite:
                prediction = model.tflite_predict(frame)
            else:
                prediction = model.predict(frame)
                overlay = model.create_overlay(frame, prediction)
                capture_manager.render_overlay(overlay)

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
