from rpi_deep_pantilt.detect.pretrained.api_v1.facessd_mobilenet_v2 import (
    FaceSSDMobileNetV2EdgeTPU,
    FaceSSDMobileNetV2Float32,
)

from rpi_deep_pantilt.detect.pretrained.api_v1.ssd_mobilenet_v3_coco import (
    SSDMobileNetV3EdgeTPU,
    SSDMobileNetV3Float32,
)


class FaceSSDMobileNetV2Int8(object):
    LABELS = []

    def validate_labels(self):
        return False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Please specify --api-version=2 to load FaceSSDMobileNetV2Int8 predictor"
        )


class SSDMobileNetV3Int8(object):
    LABELS = []

    def validate_labels(self):
        return False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Please specify --api-version=2 to load SSDMobileNetV3Int8 predictor"
        )


class LeopardAutoMLInt8(object):
    LABELS = []

    def validate_labels(self):
        return False

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Please specify--api-version=2 to load LeopardAutoMLInt8 predictor"
        )
