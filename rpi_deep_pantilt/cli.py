# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import logging
import sys
import click

from rpi_deep_pantilt.detect.camera import run_stationary_detect

from rpi_deep_pantilt.detect.ssd_mobilenet_v3_coco import (
    SSDMobileNet_V3_Small_Coco_PostProcessed,
    SSDMobileNet_V3_Coco_EdgeTPU_Quant,
    LABELS as SSDMobileNetLabels
)
from rpi_deep_pantilt.detect.facessd_mobilenet_v2 import (
    FaceSSD_MobileNet_V2,
    FaceSSD_MobileNet_V2_EdgeTPU,
    LABELS as FaceSSDLabels
)
from rpi_deep_pantilt.control.manager import pantilt_process_manager
from rpi_deep_pantilt.control.hardware_test import pantilt_test, camera_test


def validate_labels(labels):
    for label in labels:
        if label not in (SSDMobileNetLabels + FaceSSDLabels):
            logging.error(f'''
            Invalid label: {label} \n
            Please choose any of the following labels: \n
            {SSDMobileNetLabels + FaceSSDLabels}
            ''')
            sys.exit(1)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('labels', nargs=-1)
@click.option('--loglevel', required=False, type=str, default='WARNING', help='Run object detection without pan-tilt controls. Pass --loglevel=DEBUG to inspect FPS.')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
@click.option('--rotation', default=0, type=int, help='PiCamera rotation. If you followed this guide, a rotation value of 0 is correct. https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134')
def detect(labels, loglevel, edge_tpu, rotation):
    '''
        rpi-deep-pantilt detect [OPTIONS] [LABELS]...

        LABELS (optional)
            One or more labels to detect, for example:
            $ rpi-deep-pantilt detect person book "wine glass"

            If no labels are specified, model will detect all labels in this list:
            $ rpi-deep-pantilt list-labels

        Detect command will automatically load the appropriate model

        For example, providing "face" as the only label will initalize FaceSSD_MobileNet_V2 model
        $ rpi-deep-pantilt detect face

        Other labels use SSDMobileNetV3 model with COCO labels
        $ rpi-deep-pantilt detect person "wine class" orange
    '''
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    # TypeError: nargs=-1 in combination with a default value is not supported.
    if not labels:
        labels = SSDMobileNetLabels
    # Sanity-check provided labels are supported by model
    else:
        validate_labels(labels)

    if 'face' in labels and len(labels) > 1:
        logging.error(
            f'''Face detector does not support detection for non-face labels \n
               Please re-run with face as the only label argument: \n
               $ rpi-deep-pantilt detect face
            '''
        )

    # FaceSSD model
    if 'face' in labels:
        if edge_tpu:
            model_cls = FaceSSD_MobileNet_V2_EdgeTPU
        else:
            model_cls = FaceSSD_MobileNet_V2
    # All other labels are detected by SSDMobileNetV3 model
    else:
        if edge_tpu:
            model_cls = SSDMobileNet_V3_Coco_EdgeTPU_Quant
        else:
            model_cls = SSDMobileNet_V3_Small_Coco_PostProcessed

    logging.warning(f'Detecting labels: {labels}')
    run_stationary_detect(labels, model_cls, rotation)


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='List all valid classification labels')
def list_labels(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    model = SSDMobileNet_V3_Small_Coco_PostProcessed()
    print('You can detect / track the following objects:')
    print([x['name'] for x in model.category_index.values()])


@cli.command()
@click.argument('label', type=str, default='person')
@click.option('--loglevel', required=False, type=str, default='WARNING', help='Pass --loglevel=DEBUG to inspect FPS and tracking centroid X/Y coordinates')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
@click.option('--rotation', default=0, type=int, help='PiCamera rotation. If you followed this guide, a rotation value of 0 is correct. https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134')
def track(label, loglevel, edge_tpu, rotation):
    '''
        rpi-deep-pantilt track [OPTIONS] [LABEL]

        LABEL (required)
            Exactly one label to detect, for example:
            $ rpi-deep-pantilt track person

        Track command will automatically load the appropriate model

        For example, providing "face" will initalize FaceSSD_MobileNet_V2 model
        $ rpi-deep-pantilt track face

        Other labels use SSDMobileNetV3 model with COCO labels
        $ rpi-deep-pantilt detect orange
    '''
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    validate_labels((label,))

    if label == 'face':
        if edge_tpu:
            model_cls = FaceSSD_MobileNet_V2_EdgeTPU
        else:
            model_cls = FaceSSD_MobileNet_V2
    else:
        if edge_tpu:
            model_cls = SSDMobileNet_V3_Coco_EdgeTPU_Quant
        else:
            model_cls = SSDMobileNet_V3_Small_Coco_PostProcessed

    return pantilt_process_manager(model_cls, labels=(label,),  rotation=rotation)


@cli.group()
def test():
    pass


@test.command()
@click.option('--loglevel', required=False, type=str, default='INFO')
def pantilt(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return pantilt_test()


@test.command()
@click.option('--loglevel', required=False, type=str, default='INFO')
@click.option('--rotation', default=0, type=int, help='PiCamera rotation. If you followed this guide, a rotation value of 0 is correct. https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134')
def camera(loglevel, rotation):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return camera_test(rotation)


def main():
    cli()


if __name__ == "__main__":
    main()
