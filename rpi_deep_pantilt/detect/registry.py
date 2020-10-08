import importlib
import logging

from rpi_deep_pantilt.detect.util.exceptions import InvalidLabelException


class ModelRegistry(object):

    FLOAT32_CLASSES = (
        'FaceSSDMobileNetV2Float32',
        'SSDMobileNetV3Float32',
        'EfficientDetLeopardFloat32'
    )

    QUANTIZED_CLASSES = (
        'SSDMobileNetV3CocoEdgeTPU',
        'FaceSSDMobileNetV2EdgeTPU',
    )

    def __init__(self, edge_tpu, api_version):

        if api_version not in (1, 2):
            raise Exception('Invalid API version, Please specify 1 or 2')
        
        self.edge_tpu = edge_tpu
        self.api_version = api_version
        self.version_str = f'api_v{api_version}'

        self.import_path = f'rpi_deep_pantilt.detect.pretrained.{self.version_str}'

        self.module = importlib.import_module(self.import_path)

    def select_model(self, labels):
        '''Select best model for provided labels
           Raises InvalidLabelException if any labels are unsupported

        Args:
            labels: List of labels or None. If labels are not provided, default to SSDMobileNet with COCO labels
        '''

        def _select(cls_list):
            for cls_str in cls_list:
                try:
                    predictor_cls = getattr(self.module, cls_str)
                    predictor_cls.validate_labels(labels)
                    return predictor_cls
                except InvalidLabelException:
                    logging.warning(f'Predictor {predictor_cls} does not support all labels {labels}, skipping')
                    continue
            raise InvalidLabelException
        if self.edge_tpu:
            if len(labels) is 0:
                return self.module.SSDMobileNetV3CocoEdgeTPU
            return _select(self.QUANTIZED_CLASSES)
        else:
            if len(labels) is 0:
                return self.module.SSDMobileNetV3Float32
            return _select(self.FLOAT32_CLASSES)