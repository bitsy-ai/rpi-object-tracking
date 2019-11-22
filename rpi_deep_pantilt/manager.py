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

SERVO_MIN = -90
SERVO_MAX = 90

CENTER = (
    RESOLUTION[0] // 2,
    RESOLUTION[1] // 2
)

# function to handle keyboard interrupt
def signal_handler(sig, frame):
	# print a status message
	print("[INFO] You pressed `ctrl + c`! Exiting...")

	# disable the servos
	pth.servo_enable(1, False)
	pth.servo_enable(2, False)

	# exit
	sys.exit()

def run_detect(center_x, center_y, labels):
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
                center_y.value = y
                x = RESOLUTION[0] - ((np.take(track_target, [1, 3])).mean() * RESOLUTION[0])
                center_x.value = x
                logging.info(f'center_x {x} center_y {y}')

            overlay = model.create_overlay(frame, prediction)
            capture_manager.render_overlay(overlay)

def in_range(val, start, end):
    # determine the input vale is in the supplied range
    return (val >= start and val <= end)

def set_servos(pan, tlt):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        pan_angle = -1 * pan.value
        tilt_angle = tlt.values

        # if the pan angle is within the range, pan
        if in_range(pan_angle, SERVO_MIN, SERVO_MAX):
            pth.pan(pan_angle)
        else:
            logging.info(f'pan_angle not in range {pan_angle}')

        if in_range(tilt_angle, SERVO_MIN, SERVO_MAX):
            pth.tilt(tilt_angle)
        else:
            logging.info(f'tilt_angle not in range {tilt_angle}')


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

        # set initial bounding box (x, y)-coordinates to center
        center_x = manager.Value("i", 0)
        center_y = manager.Value("i", 0)
        center_x.value = RESOLUTION[0] // 2
        center_y.value = RESOLUTION[1] // 2

        # pan and tilt angles updated by independent PID processes
        pan = manager.Value("i", 0)
        tlt = manager.Value("i", 0)

        # PID gains for panning
        pan_p = manager.Value("f", 0.1)
        pan_i = manager.Value("f", 0.00)
        pan_d = manager.Value("f", 0.01)
        # pan_d = manager.Value("f", .1)

        # PID gains for tilt_ing
        tilt_p = manager.Value("f", 0.1)
        tilt_i = manager.Value("f", 0.00)
        tilt_d = manager.Value("f", 0.1)

        # we have 4 independent processes
        # 1. objectCenter  - finds/localizes the object
        # 2. panning       - PID control loop determines panning angle
        # 3. tilt_ing       - PID control loop determines tilt_ing angle
        # 4. setServos     - drives the servos to proper angles based
        #                    on PID feedback to keep object in center
        
        detect_processr = Process(target=run_detect,
            args=(center_x, center_y, labels))
            
        pan_process = Process(target=pid_process,
            args=(pan, pan_p, pan_i, pan_d, center_x, CENTER[0], 'pan'))

        tilt_process = Process(target=pid_process,
            args=(tlt, tilt_p, tilt_i, tilt_d, center_y, CENTER[1], 'tilt'))

        servo_process = Process(target=set_servos, args=(pan, tlt))

        # start all 4 processes
        detect_processr.start()
        pan_process.start()
        tilt_process.start()
        servo_process.start()

        # join all 4 processes
        detect_processr.join()
        pan_process.join()
        tilt_process.join()
        servo_process.join()

        # disable the servos
        logging.info('disabling servos')
        pth.servo_enable(1, False)
        pth.servo_enable(2, False)

if __name__ == '__main__':
    pantilt_process_manager()