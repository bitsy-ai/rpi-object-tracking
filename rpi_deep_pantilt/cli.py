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
def track():
	with Manager() as manager:
		# enable the servos
		pth.servo_enable(1, True)
		pth.servo_enable(2, True)

		# set integer values for the object center (x, y)-coordinates
		centerX = manager.Value("i", 0)
		centerY = manager.Value("i", 0)

		# set integer values for the object's (x, y)-coordinates
		objX = manager.Value("i", 0)
		objY = manager.Value("i", 0)

		# pan and tilt values will be managed by independed PIDs
		pan = manager.Value("i", 0)
		tlt = manager.Value("i", 0)

		# set PID values for panning
		panP = manager.Value("f", 0.09)
		panI = manager.Value("f", 0.08)
		panD = manager.Value("f", 0.002)

		# set PID values for tilting
		tiltP = manager.Value("f", 0.11)
		tiltI = manager.Value("f", 0.10)
		tiltD = manager.Value("f", 0.002)

		# we have 4 independent processes
		# 1. objectCenter  - finds/localizes the object
		# 2. panning       - PID control loop determines panning angle
		# 3. tilting       - PID control loop determines tilting angle
		# 4. setServos     - drives the servos to proper angles based
		#                    on PID feedback to keep object in center
		processObjectCenter = Process(target=obj_center,
			args=(args, objX, objY, centerX, centerY))

		processPanning = Process(target=pid_process,
			args=(pan, panP, panI, panD, objX, centerX))

		processTilting = Process(target=pid_process,
			args=(tlt, tiltP, tiltI, tiltD, objY, centerY))
            
		processSetServos = Process(target=set_servos, args=(pan, tlt))

		# start all 4 processes
		processObjectCenter.start()
		processPanning.start()
		processTilting.start()
		processSetServos.start()

		# join all 4 processes
		processObjectCenter.join()
		processPanning.join()
		processTilting.join()
		processSetServos.join()

		# disable the servos
		pth.servo_enable(1, False)
		pth.servo_enable(2, False)

if __name__ == "__main__":
    cli()

