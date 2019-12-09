# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import logging
import sys
import time
import click

import numpy as np

from rpi_deep_pantilt.detect.camera import PiCameraStream
from rpi_deep_pantilt.detect.ssd_mobilenet_v3_coco import SSDMobileNet_V3_Small_Coco_PostProcessed, SSDMobileNet_V3_Coco_EdgeTPU_Quant
from rpi_deep_pantilt.control.manager import pantilt_process_manager
from rpi_deep_pantilt.control.hardware_test import pantilt_test, camera_test


@click.group()
def cli():
    pass


def run_detect(capture_manager, model):
    LOGLEVEL = logging.getLogger().getEffectiveLevel()

    start_time = time.time()
    fps_counter = 0
    while not capture_manager.stopped:
        if capture_manager.frame is not None:

            frame = capture_manager.read()
            prediction = model.predict(frame)
            overlay = model.create_overlay(
                frame, prediction)
            capture_manager.overlay_buff = overlay
            if LOGLEVEL <= logging.INFO:
                fps_counter += 1
                if (time.time() - start_time) > 1:
                    fps = fps_counter / (time.time() - start_time)
                    logging.info(f'FPS: {fps}')
                    fps_counter = 0
                    start_time = time.time()


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='Run object detection without pan-tilt controls. Pass --loglevel=DEBUG to inspect FPS.')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
def detect(loglevel, edge_tpu):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    if edge_tpu:
        model = SSDMobileNet_V3_Coco_EdgeTPU_Quant()
    else:
        model = SSDMobileNet_V3_Small_Coco_PostProcessed()

    capture_manager = PiCameraStream(resolution=(320, 320))
    capture_manager.start()
    capture_manager.start_overlay()
    try:
        run_detect(capture_manager, model)
    except KeyboardInterrupt:
        capture_manager.stop()


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='List all valid classification labels')
def list_labels(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    model = SSDMobileNet_V3_Small_Coco_PostProcessed()
    print('You can detect / track the following objects:')
    print([x['name'] for x in model.category_index.values()])


@cli.command()
@click.option('--label', required=True, type=str, default='person', help='The class label to track, e.g `orange`. Run `rpi-deep-pantilt list-labels` to inspect all valid values')
@click.option('--loglevel', required=False, type=str, default='WARNING')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
def track(label, loglevel, edge_tpu):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return pantilt_process_manager(edge_tpu=edge_tpu, labels=(label,))


@cli.group()
def test():
    pass


@test.command()
@click.option('--loglevel', required=False, type=str, default='INFO')
def pantilt(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return pantilt_test()


@test.command()
@click.option('--loglevel', required=False, type=str, default='INFO')
def camera(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return camera_test()


def main():
    cli()


if __name__ == "__main__":
    main()
