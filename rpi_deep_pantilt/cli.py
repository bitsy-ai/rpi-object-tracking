# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import logging
import sys
import time
import click

import numpy as np

from detect.camera import PiCameraStream
from detect.models.ssd_mobilenet_v3_coco import SSDMobileNet_V3_Small_Coco_PostProcessed

import argparse

logging.basicConfig()
LOGLEVEL = logging.getLogger().getEffectiveLevel()


@click.group()
def cli():
    pass


def run_detect(capture_manager, model, label):
    label_idx = model.label_to_category_index(label)
    label_idx = label_idx[0] if len(label_idx) else None

    start_time = time.time()
    fps_counter = 0
    while not capture_manager.stopped:
        if capture_manager.frame is not None:

            frame = capture_manager.read()
            prediction = model.predict(frame)

            track_target = None
            overlay = model.create_overlay(
                frame, prediction)
            capture_manager.overlay_buff = overlay
            if LOGLEVEL is logging.DEBUG and (time.time() - start_time) > 1:
                fps_counter += 1
                fps = fps_counter / (time.time() - start_time)
                logging.debug(f'FPS: {fps}')
                fps_counter = 0
                start_time = time.time()


@cli.command()
@click.option('--label', required=True, type=str, default='orange')
def detect(label):
    #model = SSDLite_MobileNet_V2_Coco()
    model = SSDMobileNet_V3_Small_Coco_PostProcessed()
    capture_manager = PiCameraStream(resolution=(320, 320))
    capture_manager.start()
    capture_manager.start_overlay()
    try:
        run_detect(capture_manager, model, label)
    except KeyboardInterrupt:
        capture_manager.stop()


if __name__ == "__main__":
    cli()
