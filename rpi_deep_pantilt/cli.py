# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import importlib
import logging
import pprint
import sys

import click


from rpi_deep_pantilt.detect.camera import run_stationary_detect
from rpi_deep_pantilt.detect.registry import ModelRegistry

# from rpi_deep_pantilt.detect.v1.ssd_mobilenet_v3_coco import (
#     SSDMobileNet_V3_Small_Coco_PostProcessed,
#     SSDMobileNet_V3_Coco_EdgeTPU_Quant,
#     LABELS as SSDMobileNetLabels
# )
# from rpi_deep_pantilt.detect.v1.facessd_mobilenet_v2 import (
#     FaceSSD_MobileNet_V2,
#     FaceSSD_MobileNet_V2_EdgeTPU,
#     LABELS as FaceSSDLabels
# )

from rpi_deep_pantilt.control.manager import pantilt_process_manager
from rpi_deep_pantilt.control.hardware_test import pantilt_test, camera_test


@click.group()
def cli():
    pass


@cli.command()
@click.argument("labels", nargs=-1)
@click.option(
    "--api-version",
    required=False,
    type=click.Choice(["v1", "v2"]),
    default="v2",
    help="API Version to use (default: 2). API v1 is supported for legacy use cases.",
)
@click.option(
    "--predictor",
    required=False,
    type=str,
    default=None,
    help="Path and module name of a custom predictor class inheriting from rpi_deep_pantilt.detect.custom.base_predictors.BasePredictor",
)
@click.option(
    "--loglevel",
    required=False,
    type=str,
    default="WARNING",
    help="Run object detection without pan-tilt controls. Pass --loglevel=DEBUG to inspect FPS.",
)
@click.option(
    "--edge-tpu",
    is_flag=True,
    required=False,
    type=bool,
    default=False,
    help="Accelerate inferences using Coral USB Edge TPU",
)
@click.option(
    "--rotation",
    default=0,
    type=int,
    help="PiCamera rotation. If you followed this guide, a rotation value of 0 is correct. https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134",
)
@click.option(
    "--dtype",
    type=click.Choice(["uint8", "float32"], case_sensitive=True),
    default="uint8",
)
def detect(api_version, labels, predictor, loglevel, edge_tpu, rotation, dtype):
    """
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
    """
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    if api_version == "v1" and dtype == "uint8":
        logging.warning(
            "WARNING! --dtype=uint8 is not supported by --api-version=v1. Falling back to --dtype=float32."
        )
        dtype = "float32"

    if predictor is not None:
        predictor_cls = importlib.import_module(predictor)
    else:
        # TypeError: nargs=-1 in combination with a default value is not supported.
        model_registry = ModelRegistry(
            edge_tpu=edge_tpu, api_version=api_version, dtype=dtype
        )
        predictor_cls = model_registry.select_model(labels)

    if len(labels) is 0:
        labels = predictor_cls.LABELS

    run_stationary_detect(labels, predictor_cls, rotation)


@cli.command()
@click.option(
    "--api-version",
    required=False,
    type=int,
    default=2,
    help="API Version to use (default: 2). API v1 is supported for legacy use cases.",
)
@click.option("--edge-tpu", is_flag=True, required=False, type=bool, default=False)
@click.option(
    "--dtype",
    type=click.Choice(["uint8", "float32"], case_sensitive=True),
    default="uint8",
)
def list_labels(api_version, edge_tpu, dtype):
    model_registry = ModelRegistry(
        edge_tpu=edge_tpu, api_version=api_version, dtype=dtype
    )
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint("The following labels are supported by pretrained models:")
    pp.pprint(model_registry.label_map())


@cli.command()
@click.argument("label", type=str, default="person")
@click.option(
    "--api-version",
    required=False,
    type=click.Choice(["v1", "v2"]),
    default="v2",
    help="API Version to use (default: 2). API v1 is supported for legacy use cases.",
)
@click.option(
    "--predictor",
    required=False,
    type=str,
    default=None,
    help="Path and module name of a custom predictor class inheriting from rpi_deep_pantilt.detect.custom.base_predictors.BasePredictor",
)
@click.option(
    "--loglevel",
    required=False,
    type=str,
    default="WARNING",
    help="Pass --loglevel=DEBUG to inspect FPS and tracking centroid X/Y coordinates",
)
@click.option(
    "--edge-tpu",
    is_flag=True,
    required=False,
    type=bool,
    default=False,
    help="Accelerate inferences using Coral USB Edge TPU",
)
@click.option(
    "--rotation",
    default=0,
    type=int,
    help="PiCamera rotation. If you followed this guide, a rotation value of 0 is correct. https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134",
)
@click.option(
    "--dtype",
    type=click.Choice(["uint8", "float32"], case_sensitive=True),
    default="uint8",
)
def track(label, api_version, predictor, loglevel, edge_tpu, rotation, dtype):
    """
    rpi-deep-pantilt track [OPTIONS] [LABEL]

    LABEL (required)
        Exactly one label to detect, for example:
        $ rpi-deep-pantilt track person

    Track command will automatically load the appropriate model

    For example, providing "face" will initalize FaceSSD_MobileNet_V2 model
    $ rpi-deep-pantilt track face

    Other labels use SSDMobileNetV3 model with COCO labels
    $ rpi-deep-pantilt detect orange
    """
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    if api_version == "v1" and dtype == "uint8":
        logging.warning(
            "WARNING! --dtype=uint8 is not supported by --api-version=v1. Falling back to --dtype=float32."
        )
        dtype = "float32"

    if predictor is not None:
        predictor_cls = importlib.import_module(predictor)
    else:
        # TypeError: nargs=-1 in combination with a default value is not supported.
        model_registry = ModelRegistry(
            edge_tpu=edge_tpu, api_version=api_version, dtype=dtype
        )
        predictor_cls = model_registry.select_model((label,))

    return pantilt_process_manager(predictor_cls, labels=(label,), rotation=rotation)


@cli.group()
def test():
    pass


@test.command()
@click.option("--loglevel", required=False, type=str, default="INFO")
def pantilt(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return pantilt_test()


@test.command()
@click.option("--loglevel", required=False, type=str, default="INFO")
@click.option(
    "--rotation",
    default=0,
    type=int,
    help="PiCamera rotation. If you followed this guide, a rotation value of 0 is correct. https://medium.com/@grepLeigh/real-time-object-tracking-with-tensorflow-raspberry-pi-and-pan-tilt-hat-2aeaef47e134",
)
def camera(loglevel, rotation):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return camera_test(rotation)


def main():
    cli()


if __name__ == "__main__":
    main()
