import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .const import (
    images_base_path, 
    converted_instance_labels_base_path, 
    converted_class_labels_base_path
)

from ..dataset import FacadeDataset


class EtrimsDataset(FacadeDataset):
    """Etrims panoptic dataset."""
    IMAGES_BASE_PATH = images_base_path
    CONVERTED_CLASS_LABELS_BASE_PATH = converted_class_labels_base_path
    CONVERTED_INSTANCE_LABELS_BASE_PATH = converted_instance_labels_base_path

    