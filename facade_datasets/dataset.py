import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FacadeDataset(Dataset):
    """Base class"""
    IMAGES_BASE_PATH = None
    CONVERTED_INSTANCE_LABELS_BASE_PATH = None
    CONVERTED_CLASS_LABELS_BASE_PATH = None

    def __init__(self, transform, processor):
        self.processor = processor
        self.transform = transform

        self.idx2filename = list(set(
            map(lambda x: x.split(".")[0], sorted(os.listdir(self.IMAGES_BASE_PATH)))
        ))
    
    def _filename_to_file(self, filename, is_label=True):
        return f"{filename}{'.png' if is_label else '.jpg'}"
        
    def __len__(self):
        return len(self.idx2filename)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(
            self.IMAGES_BASE_PATH / self._filename_to_file(self.idx2filename[idx], is_label=False)
        ).convert("RGB"))

        instance_seg = np.load(
            self.CONVERTED_INSTANCE_LABELS_BASE_PATH/ (self.idx2filename[idx] + ".npy")
        )
        class_id_map = np.load(
            self.CONVERTED_CLASS_LABELS_BASE_PATH / (self.idx2filename[idx] + ".npy")
        )
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
          inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        return inputs


class FacadeDatasetPixelByPixel(Dataset):
    """Base class"""
    IMAGES_BASE_PATH = None
    CONVERTED_CLASS_LABELS_BASE_PATH = None

    def __init__(self, transform, processor):
        self.processor = processor
        self.transform = transform

        self.idx2filename = list(set(
            map(lambda x: x.split(".")[0], sorted(os.listdir(self.IMAGES_BASE_PATH)))
        ))
    
    def _filename_to_file(self, filename, is_label=True):
        return f"{filename}{'.png' if is_label else '.jpg'}"
        
    def __len__(self):
        return len(self.idx2filename)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(
            self.IMAGES_BASE_PATH / self._filename_to_file(self.idx2filename[idx], is_label=False)
        ).convert("RGB"))

        class_id_map = np.load(
            self.CONVERTED_CLASS_LABELS_BASE_PATH / (self.idx2filename[idx] + ".npy")
        )

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=class_id_map)
            image, class_id_map = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        inputs = self.processor(image, class_id_map, return_tensors="pt")

        for k,v in inputs.items():
          inputs[k].squeeze_() # remove batch dimension

        return inputs