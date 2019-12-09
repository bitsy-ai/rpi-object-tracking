import logging
import math
import time

import pantilthat
from picamera import PiCamera


# https://github.com/pimoroni/pantilt-hat/blob/master/examples/smooth.py


def camera_test():
    pantilthat.pan(0)
    pantilthat.tilt(0)
    camera = PiCamera()
    logging.info('Starting Raspberry Pi Camera')
    camera.start_preview()

    try:
        while True:
            continue
    except KeyboardInterrupt:
        logging.info('Stopping Raspberry Pi Camera')
        camera.stop_preview()


def pantilt_test():
    logging.info('Starting Pan-Tilt HAT test!')
    logging.info('Pan-Tilt HAT should follow a smooth sine wave')
    while True:
        # Get the time in seconds
        t = time.time()

        # G enerate an angle using a sine wave (-1 to 1) multiplied by 90 (-90 to 90)
        a = math.sin(t * 2) * 90
        # Cast a to int for v0.0.2
        a = int(a)
        pantilthat.pan(a)
        pantilthat.tilt(a)

        # Sleep for a bit so we're not hammering the HAT with updates
        time.sleep(0.005)


if __name__ == '__main__':
    pantilt_test()
