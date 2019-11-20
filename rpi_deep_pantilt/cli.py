# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import logging
import sys
import click

import numpy as np

from detect.camera import PiCameraStream
from detect.model import  SSDLite_MobileNet_V2_Coco



import argparse

logging.basicConfig()

@click.group()
def cli():
    pass

def run_detect(capture_manager, model, label):
    label_idx = model.label_to_category_index(label)
    label_idx = label_idx[0] if len(label_idx) else None
    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            prediction = model.predict(frame)
            
            track_target = None
            if label_idx in prediction.get('detection_classes'):
                idx = np.where(prediction.get('detection_classes')==label_idx)
                track_target = prediction.get('detection_boxes')[idx]

            overlay = model.create_overlay(frame, prediction, draw_target=track_target)
            capture_manager.render_overlay(overlay)

@cli.command()
@click.option('--label', required=True, type=str, default='orange')
def detect(label):
    model = SSDLite_MobileNet_V2_Coco()
    capture_manager = PiCameraStream()
    capture_manager.start()
    try:
        run_detect(capture_manager, model, label)
    except KeyboardInterrupt:
        capture_manager.stop()

@cli.command()
def track(args=None):
    pass

if __name__ == "__main__":
    cli()

