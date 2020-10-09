import importlib
import logging

from rpi_deep_pantilt.detect.util.exceptions import InvalidLabelException


class ModelRegistry(object):

    FLOAT32_CLASSES = (
        "FaceSSDMobileNetV2Float32",
        "SSDMobileNetV3Float32",
    )

    UINT8_CLASSES = (
        "FaceSSDMobileNetV2Int8",
        "SSDMobileNetV3Int8",
        "LeopardAutoMLInt8",
    )

    EDGETPU_CLASSES = (
        "SSDMobileNetV3EdgeTPU",
        "FaceSSDMobileNetV2EdgeTPU",
    )

    def __init__(self, edge_tpu, api_version, dtype):
        self.edge_tpu = edge_tpu
        self.api_version = api_version
        self.version_str = f"api_{api_version}"

        self.import_path = f"rpi_deep_pantilt.detect.pretrained.{self.version_str}"

        self.module = importlib.import_module(self.import_path)

        if edge_tpu:
            self.model_list = self.EDGETPU_CLASSES
            self.default_model = self.module.SSDMobileNetV3EdgeTPU
        elif dtype == "uint8":
            self.model_list = self.UINT8_CLASSES
            self.default_model = self.module.SSDMobileNetV3Int8
        else:
            self.model_list = self.FLOAT32_CLASSES
            self.default_model = self.module.SSDMobileNetV3Float32

    def select_model(self, labels):
        """Select best model for provided labels
           Raises InvalidLabelException if any labels are unsupported

        Args:
            labels: List of labels or None. If labels are not provided, default to SSDMobileNet with COCO labels
        """

        def _select(cls_list):
            for cls_str in cls_list:
                predictor_cls = getattr(self.module, cls_str)
                if predictor_cls.validate_labels(labels):
                    return predictor_cls
                else:
                    continue
            raise InvalidLabelException

        if len(labels) is 0:
            return self.default_model
        else:
            return _select(self.model_list)

    def label_map(self):
        return {x: getattr(self.module, x).LABELS for x in self.model_list}
