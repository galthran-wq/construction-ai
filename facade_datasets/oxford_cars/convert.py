"""
The challenge with the conversion of this dataset is that the labels have to be
extracted from rgb pixels.

Another problem with this dataset is that the buildings are not instance-segmented.
"""
import os
import shutil
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from .const import (
    cars2general_id,
    instance_labels_base_paths, 
    class_labels_base_paths,

    converted_class_labels_base_paths,
    converted_instance_labels_base_paths,
    CARS_BUILDING_ID
)

from ..const import BACKGROUND_ID, WINDOW_ID, BUILDING_ID

def convert_instance_labels(from_path, to_path, class_path):
    """
    We're **abusing** the fact that there is no panoptic segmentation on buildings (*)
    by marking ```balconies```, etc. to simply (255,255,0) instead of instance-specific id.
    """
    for label_filename in tqdm(os.listdir(from_path)):
        label_name = label_filename.split(".")[0]

        instance_labels_arr = np.array(Image.open(from_path / label_filename))
        class_labels_arr = np.array(Image.open(class_path / label_filename))

        instances = set()
        for i in range(instance_labels_arr.shape[0]):
            for j in range(instance_labels_arr.shape[1]):
                if cars2general_id[
                    tuple(class_labels_arr[i, j, :])
                ] == BUILDING_ID:
                    # (*) here
                    instance_labels_arr[i, j :] = np.array(CARS_BUILDING_ID)

                instances.add(tuple(instance_labels_arr[i, j, :]))
        
        instances_encoding = {
            instance_id: i
            for i, instance_id in enumerate(list(instances))
        }
        instance_labels_arr2 = np.zeros((instance_labels_arr.shape[0], instance_labels_arr.shape[1]))
        for i in range(instance_labels_arr.shape[0]):
            for j in range(instance_labels_arr.shape[1]):
                instance_labels_arr2[i,j] = instances_encoding[
                    tuple(instance_labels_arr[i, j, :])
                ]


        np.save(to_path / label_name, instance_labels_arr2)


def convert_instance_labels(from_path, to_path, class_path):
    """
    We're **abusing** the fact that there is no panoptic segmentation on buildings (*)
    by marking ```balconies```, etc. to simply (255,255,0) instead of instance-specific id.
    """
    for label_filename in tqdm(os.listdir(from_path)):
        label_name = label_filename.split(".")[0]

        instance_labels_arr = np.array(Image.open(from_path / label_filename))
        class_labels_arr = np.array(Image.open(class_path / label_filename))

        instances = set()
        for r,g,b, r1,g1,b1 in zip(
            np.nditer(instance_labels_arr[...,0]), 
            np.nditer(instance_labels_arr[...,1]), 
            np.nditer(instance_labels_arr[...,2]),
            np.nditer(class_labels_arr[...,0]),
            np.nditer(class_labels_arr[...,1]),
            np.nditer(class_labels_arr[...,2]),
        ):
            rgb = (int(r), int(g), int(b))
            rgb1 = (int(r1), int(g1), int(b1))
            if cars2general_id[
                rgb1
            ] == BUILDING_ID:
                # (*) here
                rgb = tuple(CARS_BUILDING_ID)
            instances.add(rgb)
        
        instances_encoding = {
            instance_id: i
            for i, instance_id in enumerate(list(instances))
        }
        instance_labels_arr2 = np.zeros((instance_labels_arr.shape[0], instance_labels_arr.shape[1]))
        for i in range(instance_labels_arr.shape[0]):
            for j in range(instance_labels_arr.shape[1]):
                rgb = tuple(instance_labels_arr[i, j, :])
                if rgb in instances_encoding:
                    instance_labels_arr2[i,j] = instances_encoding[
                        tuple(instance_labels_arr[i, j, :])
                    ]
                else:
                    instance_labels_arr2[i,j] = instances_encoding[
                        tuple(CARS_BUILDING_ID)
                    ]



        np.save(to_path / label_name, instance_labels_arr2)


# result = [
#         rgb2class[(
#             int(r),
#             int(g),
#             int(b)
#         )] for r,g,b in zip(
#             np.nditer(label_arr[...,0]), 
#             np.nditer(label_arr[...,1]), 
#             np.nditer(label_arr[...,2])
#         )
#     ]

def convert_class_labels(from_path, to_path):
    """
    Remap semantic classes to general format
    """
    for label_filename in tqdm(os.listdir(from_path)):
        label_name = label_filename.split(".")[0]
        label_arr = np.array(Image.open(from_path / label_filename))

        converted_label_arr = np.empty((label_arr.shape[0], label_arr.shape[1]))

        # TODO: slow
        for i in range(label_arr.shape[0]):
            for j in range(label_arr.shape[1]):
                converted_label_arr[i, j] = cars2general_id[
                    tuple(label_arr[i, j, :])
                ]

        np.save(to_path / label_name, converted_label_arr)


def _initialize_converted_directories():
    for paths in [
        converted_instance_labels_base_paths, 
        converted_class_labels_base_paths 
    ]:
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)


def main():
    _initialize_converted_directories()

    for from_path, to_path, class_path in zip(
        instance_labels_base_paths,
        converted_instance_labels_base_paths,
        class_labels_base_paths,
    ):
        convert_instance_labels(from_path, to_path, class_path)
    
    for from_path, to_path in zip(
        class_labels_base_paths,
        converted_class_labels_base_paths,
    ):
        convert_class_labels(from_path, to_path)


if __name__ == "__main__":
    main()