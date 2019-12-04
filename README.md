# Raspberry Pi Deep PanTilt

[![image](https://img.shields.io/pypi/v/rpi_deep_pantilt.svg)](https://pypi.python.org/pypi/rpi-deep-pantilt)

<!-- [![image](https://img.shields.io/travis/leigh-johnson/rpi_deep_pantilt.svg)](https://travis-ci.org/leigh-johnson/rpi_deep_pantilt) -->

[![Documentation
Status](https://readthedocs.org/projects/rpi-deep-pantilt/badge/?version=latest)](https://rpi-deep-pantilt.readthedocs.io/en/latest/?badge=latest)

# Build List

  - [Raspberry Pi 4 (4GB recommended)](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
  - [Raspberry Pi Camera V2](https://www.raspberrypi.org/products/camera-module-v2/)
  - [Pimoroni Pan-tilt Kit](https://shop.pimoroni.com/products/pan-tilt-hat?variant=22408353287)
  - Micro SD card 16+ GB
  - Micro HDMI Cable
  - [12" CSI/DSI ribbon for Raspberry Pi Camera](https://www.adafruit.com/product/1648) (optional, but highly recommended)
  - [Coral Edge TPU USB Accelerator](https://coral.withgoogle.com/products/accelerator) (optional)
  - [RGB NeoPixel Stick](https://www.adafruit.com/product/1426) (optional, makes lighting conditions more consistent)

An example of deep object detection and tracking with a Raspberry Pi

  - Free software: MIT license
  - Documentation: <https://rpi-deep-pantilt.readthedocs.io>.

# Basic Setup

Before you get started, you should have an up-to-date installation of Raspbian 10 (Buster) running on your Raspberry Pi. You'll also need to configure SSH access into your Pi. 

* [Install Raspbian](https://www.raspberrypi.org/documentation/installation/installing-images/README.md)
* [Configure WiFi](https://www.raspberrypi.org/forums/viewtopic.php?t=191252)
* [Configure SSH Access](https://www.raspberrypi.org/documentation/remote-access/ssh/)

# Installation

1. Install system dependencies

```
sudo apt-get update && sudo apt-get install -y \
    cmake python3-dev libjpeg-dev libatlas-base-dev raspi-gpio libhdf5-dev python3-smbus
```

2. Install the `rpi-deep-pantilt` package.
```
pip install rpi-deep-pantilt
```

# Example Usage

## Real-time object detection

The following will start a PiCamera preview and render detected objects as an overlay. Ensure you're able to detect an object before trying to track it. 

`rpi-deep-pantilt detect`

```
rpi-deep-pantilt detect --help

Usage: rpi-deep-pantilt detect [OPTIONS]

Options:
  --loglevel TEXT  Run object detection without pan-tilt controls. Pass
                   --loglevel=DEBUG to inspect FPS.
  --help           Show this message and exit.
```

## Real-time object tracking

The following will start a PiCamera preview, render detected objects as an overlay, and track an object's movement with the pan-tilt HAT. 

By default, this will track any `person` in the frame. You can track other objects by passing `--label <label>`. For a list of valid labels, run `rpi-deep-pantilt list-labels`. 

`rpi-deep-pantilt track`

```
rpi-deep-pantilt track --help 
Usage: rpi-deep-pantilt track [OPTIONS]

Options:
  --label TEXT     The class label to track, e.g `orange`. Run `rpi-deep-
                   pantilt list-labels` to inspect all valid values
                   [required]
  --loglevel TEXT
  --help           Show this message and exit.
```

## Valid labels

`rpi-deep-pantilt list-labels`

The following labels are valid tracking targets.

```
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

# Credits

The MobileNetV3-SSD model in this package was derived from [TensorFlow's model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), with [post-processing ops added](https://gist.github.com/leigh-johnson/155264e343402c761c03bc0640074d8c).

The PID control scheme in this package was inspired by [Adrian Rosebrock](https://github.com/jrosebr1) tutorial [Pan/tilt face tracking with a Raspberry Pi and OpenCV](https://www.pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/)

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
