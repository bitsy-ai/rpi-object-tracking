from rpi_deep_pantilt.detect.pretrained.api_v1.facessd_mobilenet_v2 import (
    FaceSSDMobileNetV2EdgeTPU,
    FaceSSDMobileNetV2Float32
)

from rpi_deep_pantilt.detect.pretrained.api_v1.ssd_mobilenet_v3_coco import (
    SSDMobileNetV3CocoEdgeTPU,
    SSDMobileNetV3Float32
)

class EfficientDetLeopardFloat32(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Please use --api-version=2 to load EfficientDetLeopardFloat32 predictor')