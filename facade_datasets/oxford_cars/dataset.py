import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .const import (
    images_base_paths, 
    converted_instance_labels_base_paths, 
    converted_class_labels_base_paths
)

from ..dataset import FacadeDataset, FacadeDatasetPixelByPixel


class CarsTrainDataset(FacadeDataset):
    """Etrims panoptic dataset."""
    IMAGES_BASE_PATH = images_base_paths[0]
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_paths[0]
    CONVERTED_INSTANCE_LABELS_BASE_PATH = converted_instance_labels_base_paths[0]

    def _filename_to_file(self, filename, is_label=True):
        return filename + ".png"


class CarsTestDataset(FacadeDataset):
    """Etrims panoptic dataset."""
    IMAGES_BASE_PATH = images_base_paths[1]
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_paths[1]
    CONVERTED_INSTANCE_LABELS_BASE_PATH = converted_instance_labels_base_paths[1]

    def _filename_to_file(self, filename, is_label=True):
        return filename + ".png"


class CarsTrainSemanticDataset(FacadeDatasetPixelByPixel):
    """Etrims panoptic dataset."""
    IMAGES_BASE_PATH = images_base_paths[0]
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_paths[0]

    def _filename_to_file(self, filename, is_label=True):
        return filename + ".png"


class CarsTestSemanticDataset(FacadeDatasetPixelByPixel):
    """Etrims panoptic dataset."""
    IMAGES_BASE_PATH = images_base_paths[1]
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_paths[1]

    def _filename_to_file(self, filename, is_label=True):
        return filename + ".png"
    