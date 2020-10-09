from rpi_deep_pantilt.detect.pretrained.api_v2.facessd_mobilenet_v2 import (
    FaceSSDMobileNetV2EdgeTPU,
    FaceSSDMobileNetV2Float32,
    FaceSSDMobileNetV2Int8
)

from rpi_deep_pantilt.detect.pretrained.api_v2.ssd_mobilenet_v3_coco import (
    SSDMobileNetV3EdgeTPU,
    SSDMobileNetV3Float32,
    SSDMobileNetV3Int8
)

class EfficientDetLeopardFloat32(object):
    LABELS = []
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Please use --api-version=2 to load EfficientDetLeopardFloat32 predictor')