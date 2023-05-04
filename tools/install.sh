#!/bin/bash

set +x

echo "Installing platform dependencies"
sudo apt-get update && sudo apt-get install -y \
    cmake python3-dev libjpeg-dev libatlas-base-dev raspi-gpio libhdf5-dev python3-smbus libopenjp2-7-dev

echo "Installing TensorFlow"
pip install https://github.com/leigh-johnson/Tensorflow-bin/releases/download/v2.2.0/tensorflow-2.2.0-cp37-cp37m-linux_armv7l.whl
echo "Installing Coral Edge TPU runtime"
pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

echo "Enabling Camera via raspi-config nonint"
sudo raspi-config nonint do_camera 1

echo "Enabling i2c dtparam via raspi-config nonint"
sudo raspi-config nonint do_i2c 1