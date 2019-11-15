from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
from threading import Thread


class PiCameraStream(object):
    """
      Continuously capture video frames, and optionally render with an overlay

      Arguments
      resolution - tuple (x, y) size 
      framerate - int 
      vflip - reflect capture on x-axis
      hflip - reflect capture on y-axis

    """

    def __init__(self, resolution=(320, 240), framerate=24, vflip=True, hflip=True):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.camera.rotation = 270

        self.data_container = PiRGBArray(self.camera, size=resolution)

        self.stream = self.camera.capture_continuous(
            self.data_container, format="bgr", use_video_port=True
        )

        self.frame = None
        self.stopped = False
        print('starting camera preview')
        self.camera.start_preview()

    def render_overlay(self):
        pass

    def start(self):
        """Begin handling frame stream in a separate thread"""
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
        return self.frame[0:224, 48:272, :]  # crop the frame

    def stop(self):
        self.stopped = True
