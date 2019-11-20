import logging
from multiprocessing import Manager, Process
import pantilthat as pth
import signal
import sys
import numpy as np

from detect.camera import PiCameraStream
from detect.model import  SSDLite_MobileNet_V2_Coco

from control.pid import PIDController

logging.basicConfig()

RESOLUTION = (320, 240)

# define the range for the motors
servoRange = (-90, 90)

# function to handle keyboard interrupt
def signal_handler(sig, frame):
	# print a status message
	print("[INFO] You pressed `ctrl + c`! Exiting...")

	# disable the servos
	pth.servo_enable(1, False)
	pth.servo_enable(2, False)

	# exit
	sys.exit()

def run_detect(objX, objY, labels):
    model = SSDLite_MobileNet_V2_Coco()
    capture_manager = PiCameraStream()
    capture_manager.start()
    label_idxs = model.label_to_category_index(labels)
    while not capture_manager.stopped:
        if capture_manager.frame is not None:
            frame = capture_manager.read()
            prediction = model.predict(frame)

            if not len(prediction.get('detection_boxes')):
                continue

            if any(item in label_idxs for item in prediction.get('detection_classes')):

                track_target = prediction.get('detection_boxes')[0]
                # [ymin, xmin, ymax, xmax]
                y = RESOLUTION[1] - ((np.take(track_target, [0, 2])).mean() * RESOLUTION[1])
                objY.value = y
                x = RESOLUTION[0] - ((np.take(track_target, [1, 3])).mean() * RESOLUTION[0])
                objX.value = x
                logging.info(f'objX {x} objY {y}')

            overlay = model.create_overlay(frame, prediction)
            capture_manager.render_overlay(overlay)

def in_range(val, start, end):
    # determine the input vale is in the supplied range
    return (val >= start and val <= end)

def set_servos(pan, tlt):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # loop indefinitely
    while True:
        # the pan and tilt angles are reversed
        # panAngle = -1 * pan.value
        # tltAngle = -1 * tlt.value
        panAngle = -1 * pan.value
        tltAngle = tlt.value


        # if the pan angle is within the range, pan

        if in_range(panAngle, servoRange[0], servoRange[1]):
            pth.pan(panAngle)
        else:
            logging.info(f'panAngle not in range {panAngle}')
            # pan.value = 0 
            # pth.pan(0)

        if in_range(tltAngle, servoRange[0], servoRange[1]):
            pth.tilt(tltAngle)
        else:
            logging.info(f'tiltAngle not in range {tltAngle}')
            # tilt.value = 0
            # pth.tilt(0)

def pid_process(output, p, i, d, objCoord, centerCoord, action):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # create a PID and initialize it
    p = PIDController(p.value, i.value, d.value)
    p.initialize()


    # loop indefinitely
    while True:
        # calculate the error
        error = centerCoord.value - objCoord.value

        # if error == 0:
        #     continue
        # else:
        output.value = p.update(error)
        logging.info(f'{action} error {error} angle: {output.value}')


def pantilt_process_manager(labels=('orange', 'apple', 'sports ball', 'cup', 'wine glass', 'book', 'bottle', 'scissors', 'mouse')):
    # start a manager for managing process-safe variables
    with Manager() as manager:
        # enable the servos
        pth.servo_enable(1, True)
        pth.servo_enable(2, True)

        # set integer values for the object center (x, y)-coordinates
        centerX = manager.Value("i", 0)
        centerY = manager.Value("i", 0)
        centerX.value = RESOLUTION[0] // 2
        centerY.value = RESOLUTION[1] // 2

        # set integer values for the object's (x, y)-coordinates
        objX = manager.Value("i", 0)
        objY = manager.Value("i", 0)
        objX.value = RESOLUTION[0] // 2
        objY.value = RESOLUTION[1] // 2

        # pan and tilt values will be managed by independed PIDs
        pan = manager.Value("i", 0)
        tlt = manager.Value("i", 0)

        # set PID values for panning
        panP = manager.Value("f", 0.1)
        # panI = manager.Value("f", 0.08)
        panI = manager.Value("f", 0.00)
        panD = manager.Value("f", 0.01)
        # panD = manager.Value("f", 0.1)

        # set PID values for tilting
        tiltP = manager.Value("f", 0.1)
        #tiltI = manager.Value("f", 0.10)
        #tiltD = manager.Value("f", 0.002)
        tiltI = manager.Value("f", 0.00)
        tiltD = manager.Value("f", 0.1)

        # we have 4 independent processes
        # 1. objectCenter  - finds/localizes the object
        # 2. panning       - PID control loop determines panning angle
        # 3. tilting       - PID control loop determines tilting angle
        # 4. setServos     - drives the servos to proper angles based
        #                    on PID feedback to keep object in center
        processObjectCenter = Process(target=run_detect,
            args=(objX, objY, labels))
            
        processPanning = Process(target=pid_process,
            args=(pan, panP, panI, panD, objX, centerX, 'pan'))

        processTilting = Process(target=pid_process,
            args=(tlt, tiltP, tiltI, tiltD, objY, centerY, 'tilt'))

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
        logging.info('disabling servos')
        pth.servo_enable(1, False)
        pth.servo_enable(2, False)

if __name__ == '__main__':
    pantilt_process_manager()