#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

import subprocess
import sys


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()


common_requirements = [
    'Click>=7.0',
    'pillow',
    'h5py',
]

trainer_requirements = [
    'tensorflow==2.0.0',
    'numpy'
]

trainer_requirements = list(map(
    lambda x: x + ';platform_machine=="x86_64"', trainer_requirements
))

rpi_requirements = [
    'smbus',
    'picamera',
    'pantilthat>=0.0.7',
]

rpi_requirements = list(map(
    lambda x: x + ';platform_machine=="armv7l"', rpi_requirements))

requirements = common_requirements + trainer_requirements + rpi_requirements

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

RPI_LIBS = ['python3-dev', 'cmake', 'libjpeg8-dev', 'zlib1g-dev']
RPI_CUSTOM_COMMANDS = [['sudo', 'apt-get', 'update'],
                       ['sudo', 'apt-get', 'install', '-y'] + RPI_LIBS
                       ]

TRAINER_DEBIAN_LIBS = ['python3-dev', 'cmake', 'zlib1g-dev', 'libjpeg-dev']

TRAINER_DEBIAN_CUSTOM_COMMANDS = [['apt-get', 'update'],
                                  ['apt-get', 'install', '-y'] + TRAINER_DEBIAN_LIBS]

TRAINER_DARWIN_LIBS = ['cmake']
TRAINER_DARWIN_CUSTOM_COMMANDS = [['brew', 'update'],
                                  ['brew', 'install'] + TRAINER_DARWIN_LIBS
                                  ]


# $pip install rpi-deep-pantilt==1.0.0rc3
# ERROR: Packages installed from PyPI cannot depend on packages which are not also hosted on PyPI.
# rpi-deep-pantilt depends on tensorflow@ https://github.com/leigh-johnson/Tensorflow-bin/blob/master/tensorflow-2.0.0-cp37-cp37m-linux_armv7l.whl?raw=true;platform_machine=="armv7l"
class PostInstall(install):

    def run(self):
        deps = 'https://github.com/leigh-johnson/Tensorflow-bin/blob/master/tensorflow-2.0.0-cp37-cp37m-linux_armv7l.whl?raw=true'

        install.run(self)
        # https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program
        subprocess.call([sys.executable, '-m', 'pip',
                         'install', deps])


setup(
    author="Leigh Johnson",
    author_email='hi@leighjohnson.me',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="An example of deep object detection and tracking with a Raspberry Pi, PiCamera, and Pimoroni Pantilt Hat",
    entry_points={
        'console_scripts': [
            'rpi-deep-pantilt=rpi_deep_pantilt.cli:main',
        ],
    },
    cmdclass={'install': PostInstall},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='computer vision cv tensorflow raspberrypi detection tracking ',
    name='rpi_deep_pantilt',
    packages=find_packages(include=[
                           'rpi_deep_pantilt', 'rpi_deep_pantilt.*']),
    package_data={'rpi_deep_pantilt': ['data/*.pbtxt']},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/leigh-johnson/rpi_deep_pantilt',
    version='1.0.1',
    zip_safe=False,

)
