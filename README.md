# Raspberry Pi Deep PanTilt

[![image](https://img.shields.io/pypi/v/rpi_deep_pantilt.svg)](https://pypi.python.org/pypi/rpi-deep-pantilt)

<!-- [![image](https://img.shields.io/travis/leigh-johnson/rpi_deep_pantilt.svg)](https://travis-ci.org/leigh-johnson/rpi_deep_pantilt) -->

[![Documentation
Status](https://readthedocs.org/projects/rpi-deep-pantilt/badge/?version=latest)](https://rpi-deep-pantilt.readthedocs.io/en/latest/?badge=latest)

# READ THIS FIRST!

A detailed walk-through is available in [Real-time Object Tracking with TensorFlow, Raspberry Pi, and Pan-tilt HAT](https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134).

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

```bash
$ sudo apt-get update && sudo apt-get install -y \
    cmake python3-dev libjpeg-dev libatlas-base-dev raspi-gpio libhdf5-dev python3-smbus
```

1. Create new virtual environment

```bash
$ python3 -m venv .venv
```

3. Activate virtual environment

```bash
$ source .venv/bin/activate
```

4. Upgrade setuptools

```bash
$ pip install --upgrade setuptools
```

5. Install TensorFlow 2.2 (community-built wheel)

```bash
$ pip install https://github.com/leigh-johnson/Tensorflow-bin/releases/download/v2.2.0/tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl

```

6. Install the `rpi-deep-pantilt` package.

```bash
pip install rpi-deep-pantilt
```

7. Install Coral Edge TPU `tflite_runtime` (optional)

NOTE: This step is only required if you are using [Coral's Edge TPU USB Accelerator](https://coral.withgoogle.com/products/accelerator). If you would like to run TFLite inferences using CPU only, skip this step. 

```bash
$ pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
```

=======
# Configuration

WARNING: Do not skip this section! You will not be able to use `rpi-deep-pantilt` without properly configuring your Pi.

### Enable Pi Camera

1. Run `sudo raspi-config` and select `Interfacing Options` from the Raspberry Pi Software Configuration Tool’s main menu. Press ENTER.

![raspi-config main menu](/images/camera1.png)

2. Select the Enable Camera menu option and press ENTER.

![raspi-config interfacing options menu](/images/camera2.png)

3. In the next menu, use the right arrow key to highlight ENABLE and press ENTER.

![raspi-config enable camera yes/no menu](/images/camera3.png)

### Enable i2c in Device Tree

1. Open `/boot/config.txt` and verify the following `dtparams` lines are uncommented:

```bash
dtparam=i2c1=on
dtparam=i2c_arm=on
```
# Example Usage

## Object Detection

The `detect` command will start a PiCamera preview and render detected objects as an overlay. Verify you're able to detect an object before trying to track it. 

Supports Edge TPU acceleration by passing the `--edge-tpu` option.

`rpi-deep-pantilt detect [OPTIONS] [LABELS]...`

```
rpi-deep-pantilt detect --help
Usage: rpi-deep-pantilt detect [OPTIONS] [LABELS]...

  rpi-deep-pantilt detect [OPTIONS] [LABELS]

    LABELS (optional)     One or more labels to detect, for example:     
    $ rpi-deep-pantilt detect person book "wine glass"

    If no labels are specified, model will detect all labels in this list:
    $ rpi-deep-pantilt list-labels

    Detect command will automatically load the appropriate model

    For example, providing "face" as the only label will initalize
    FaceSSD_MobileNet_V2 model $ rpi-deep-pantilt detect face

    Other labels use SSDMobileNetV3 with COCO labels $ rpi-deep-pantilt detect
    person "wine class" orange

Options:
  --loglevel TEXT  Run object detection without pan-tilt controls. Pass
                   --loglevel=DEBUG to inspect FPS.
  --edge-tpu       Accelerate inferences using Coral USB Edge TPU
  --rotation INTEGER  PiCamera rotation. If you followed this guide, a
                      rotation value of 0 is correct.
                      https://medium.com/@grepLeigh/real-time-object-tracking-
                      with-tensorflow-raspberry-pi-and-pan-tilt-
                      hat-2aeaef47e134
  --help           Show this message and exit.

```

## Object Tracking

The following will start a PiCamera preview, render detected objects as an overlay, and track an object's movement with Pimoroni pan-tilt HAT. 

By default, this will track any `person` in the frame. You can track other objects by passing `--label <label>`. For a list of valid labels, run `rpi-deep-pantilt list-labels`. 

`rpi-deep-pantilt track`

Supports Edge TPU acceleration by passing the `--edge-tpu` option.

```
Usage: rpi-deep-pantilt track [OPTIONS] [LABEL]

  rpi-deep-pantilt track [OPTIONS] [LABEL]

  LABEL (required, default: person) Exactly one label to detect, for example:     
  $ rpi-deep-pantilt track person

  Track command will automatically load the appropriate model

  For example, providing "face" will initalize FaceSSD_MobileNet_V2 model
  $ rpi-deep-pantilt track face

  Other labels use SSDMobileNetV3 model with COCO labels 
  $ rpi-deep-pantilt detect orange

Options:
  --loglevel TEXT  Pass --loglevel=DEBUG to inspect FPS and tracking centroid
                   X/Y coordinates
  --edge-tpu       Accelerate inferences using Coral USB Edge TPU
  --rotation INTEGER  PiCamera rotation. If you followed this guide, a
                      rotation value of 0 is correct.
                      https://medium.com/@grepLeigh/real-time-object-tracking-
                      with-tensorflow-raspberry-pi-and-pan-tilt-
                      hat-2aeaef47e134
  --help           Show this message and exit.
```

## Valid labels for Object Detection/Tracking

`rpi-deep-pantilt list-labels`

The following labels are valid tracking targets.

```
['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

## Face Detection (NEW in v1.1.x)

The following command will detect human faces. 

NOTE: Face detection uses a specialized model (FaceSSD_MobileNet_V2), while other labels are detecting using SSDMobileNetV3_COCO. You cannot detect both face and COCO labels at this time. 

Watch this repo for updates that allow you to re-train these models to support a custom mix of object labels!

```
rpi-deep-pantilt detect face
Usage: cli.py face-detect [OPTIONS]

Options:
  --loglevel TEXT  Run object detection without pan-tilt controls. Pass
                   --loglevel=DEBUG to inspect FPS.
  --edge-tpu       Accelerate inferences using Coral USB Edge TPU
  --help           Show this message and exit.
```

## Face Tracking (NEW in v1.1.x)

The following command will track a human face. 

```
rpi-deep-pantilt track face
Usage: cli.py face-detect [OPTIONS]

Options:
  --loglevel TEXT  Run object detection without pan-tilt controls. Pass
                   --loglevel=DEBUG to inspect FPS.
  --edge-tpu       Accelerate inferences using Coral USB Edge TPU
  --help           Show this message and exit.
```

# Model Summary

The following section describes the models used in this project. 

## Object Detection & Tracking

### `FLOAT32` model (`ssd_mobilenet_v3_small_coco_2019_08_14`)

`rpi-deep-pantilt detect` and `rpi-deep-pantilt track` perform inferences using this model. Bounding box and class predictions render at roughly *6 FPS* on a *Raspberry Pi 4*.  

The model is derived from  `ssd_mobilenet_v3_small_coco_2019_08_14` in [tensorflow/models](https://github.com/tensorflow/models). I extended the model with an NMS post-processing layer, then converted to a format compatible with TensorFlow 2.x (FlatBuffer). 

I scripted the conversion steps in `tools/tflite-postprocess-ops-float.sh`. 


### Quantized `UINT8` model (`ssdlite_mobilenet_edgetpu_coco_quant`)

If you specify `--edge-tpu` option, `rpi-deep-pantilt detect` and `rpi-deep-pantilt track` perform inferences using this model. Rounding box and class predictions render at roughly *24+ FPS (real-time) on Raspberry Pi 4*.

This model *REQUIRES* a [Coral Edge TPU USB Accelerator](https://coral.withgoogle.com/products/accelerator) to run.

This model is derived from  `ssdlite_mobilenet_edgetpu_coco_quant` in [tensorflow/models](https://github.com/tensorflow/models). I reversed the frozen `.tflite` model into a protobuf graph to add an NMS post-processing layer, quantized the model in a `.tflite` FlatBuffer format, then converted using Coral's `edgetpu_compiler` tool. 

I scripted the conversion steps in `tools/tflite-postprocess-ops-128-uint8-quant.sh` and `tools/tflite-edgetpu.sh`. 

## Face Detection & Tracking

I was able to use the same model architechture for FLOAT32 and UINT8 input, `facessd_mobilenet_v2_quantized_320x320_open_image_v4_tflite2`. 

This model is derived from `facessd_mobilenet_v2_quantized_320x320_open_image_v4` in [tensorflow/models](https://github.com/tensorflow/models). 

# Common Issues

### i2c is not enabled

If you run `$ rpi-deep-pantilt test pantilt` and see a similar error, check your Device Tree configuration.

```python
File "/home/pi/projects/rpi-deep-pantilt/.venv/lib/python3.7/site-packages/pantilthat/pantilt.py", line 72, in setup
self._i2c = SMBus(1)
FileNotFoundError: [Errno 2] No such file or directory
```

Open `/boot/config.txt` and ensure the following lines are uncommented:

```bash
dtparam=i2c1=on
dtparam=i2c_arm=on
```

# Credits

The MobileNetV3-SSD model in this package was derived from [TensorFlow's model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), with [post-processing ops added](https://gist.github.com/leigh-johnson/155264e343402c761c03bc0640074d8c).

The PID control scheme in this package was inspired by [Adrian Rosebrock](https://github.com/jrosebr1) tutorial [Pan/tilt face tracking with a Raspberry Pi and OpenCV](https://www.pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/)

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.

