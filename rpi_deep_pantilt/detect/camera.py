# Python
import logging

import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np

#from concurrent.futures import ThreadPoolExecutor
from threading import Thread

# from multiprocessing.pool import ThreadPool


logging.basicConfig()

# https://github.com/dtreskunov/rpi-sensorium/commit/40c6f3646931bf0735c5fe4579fa89947e96aed7


def _monkey_patch_picamera(overlay):
    original_send_buffer = picamera.mmalobj.MMALPortPool.send_buffer

    def silent_send_buffer(zelf, *args, **kwargs):
        try:
            original_send_buffer(zelf, *args, **kwargs)
        except picamera.exc.PiCameraMMALError as error:
            # Only silence MMAL_EAGAIN for our target instance.
            our_target = overlay.renderer.inputs[0].pool == zelf
            if not our_target or error.status != 14:
                raise error

    picamera.mmalobj.MMALPortPool.send_buffer = silent_send_buffer

class PiCameraStream(object):
    """
      Continuously capture video frames, and optionally render with an overlay

      Arguments
      resolution - tuple (x, y) size 
      framerate - int 
      vflip - reflect capture on x-axis
      hflip - reflect capture on y-axis

    """

    def __init__(self, 
        resolution=(320, 240), 
        framerate=24, 
        vflip=False,
        hflip=False,
        rotation=0,
        max_workers=2
    ):

        #self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.camera.rotation = rotation
        self.overlay = None

        self.data_container = PiRGBArray(self.camera, size=resolution)

        self.stream = self.camera.capture_continuous(
            self.data_container, format="rgb", use_video_port=True
        )

        self.frame = None
        self.stopped = False
        logging.info('starting camera preview')
        self.camera.start_preview()

    def render_overlay(self, image_buff):
        if self.overlay:
            self.overlay.update(image_buff)
        else:
            self.overlay = self.camera.add_overlay(image_buff, layer=3, size=(320, 240))
            _monkey_patch_picamera(self.overlay)

    def start(self):
        '''Begin handling frame stream in a separate thread'''
        Thread(target=self.flush, args=()).start()
        return self

    def flush(self):
        # looping until self.stopped flag is flipped
        # for now, grab the first frame in buffer, then empty buffer
        for f in self.stream:
            self.frame = f.array
            self.data_container.truncate(0)

            if self.stopped:
                self.stream.close()
                self.data_container.close()
                self.camera.close()
                return

    def read(self):
        #return self.frame[0:224, 48:272, :]  # crop the frame
        return self.frame

    def stop(self):
        self.stopped = True
