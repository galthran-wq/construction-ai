from .const import (
    images_base_path, 
    converted_instance_labels_base_path, 
    converted_class_labels_base_path
)

from ..dataset import FacadeDataset, FacadeDatasetPixelByPixel


class CMPDataset(FacadeDataset):
    IMAGES_BASE_PATH = images_base_path
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_path
    CONVERTED_INSTANCE_LABELS_BASE_PATH = converted_instance_labels_base_path


class CMPSemanticDataset(FacadeDatasetPixelByPixel):
    IMAGES_BASE_PATH = images_base_path
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_path


    