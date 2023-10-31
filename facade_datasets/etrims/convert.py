import os
import shutil
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from .const import (
    ETRIMS_BACKGROUND_ID,
    etrims2general_id,

    instance_labels_base_path, 
    class_labels_base_path,

    converted_class_labels_base_path,
    converted_instance_labels_base_path,
)

from ..const import BACKGROUND_ID

def convert_instance_labels():
    """
    (?Several semantic categories are merged.)
    The categories which are merge into background should not be considered as separate instances 


    Therefore, we don't need separate instances for those. We have to merge them
    Read an image and save it as a np array.
    No additional preprocessing.
    """
    for label_filename in tqdm(os.listdir(instance_labels_base_path)):
        label_name = label_filename.split(".")[0]

        instance_labels_arr = np.array(Image.open(instance_labels_base_path / label_filename))
        class_labels_arr = np.array(Image.open(class_labels_base_path / label_filename))

        # find **the** background instance
        # background_instance_id = None
        # for i in range(instance_labels_arr.shape[0]):
        #     if background_instance_id is not None:
        #         break
        #     for j in range(instance_labels_arr.shape[1]):
        #         if background_instance_id is not None:
        #             break
        #         if class_labels_arr[i][j] == ETRIMS_BACKGROUND_ID:
        #             background_instance_id = instance_labels_arr[i, j]

        # Find all the instances which are mapped to background
        # TODO: inefficient, ugly
        background_elem_instance_id = None
        mapped_to_background_instances = set()
        for i in range(instance_labels_arr.shape[0]):
            for j in range(instance_labels_arr.shape[1]):
                if etrims2general_id[
                    class_labels_arr[i][j]
                ] == BACKGROUND_ID:
                    mapped_to_background_instances.add(instance_labels_arr[i, j])
                    if background_elem_instance_id is None:
                        background_elem_instance_id = instance_labels_arr[i, j]
        
        assert len(mapped_to_background_instances) > 0

        @np.vectorize
        def replace(elem):
            return (
                background_elem_instance_id 
                if elem in mapped_to_background_instances 
                else elem
            )
        instance_labels_arr = replace(instance_labels_arr)

        # to each instance which is **merged with** background assign the same instance id
        # instance_labels_arr[
        #     instance_labels_arr.isin(mapped_to_background_instances)
        # ] = mapped_to_background_instances[0]

        np.save(converted_instance_labels_base_path / label_name, instance_labels_arr)


def convert_class_labels():
    """
    Remap semantic classes to general format
    """
    for label_filename in tqdm(os.listdir(class_labels_base_path)):
        label_name = label_filename.split(".")[0]
        label_arr = np.array(Image.open(class_labels_base_path / label_filename))

        for etrims_label, general_label in etrims2general_id.items():
            label_arr[label_arr == etrims_label] = general_label

        np.save(converted_class_labels_base_path / label_name, label_arr)


def _initialize_converted_directories():
    for path in [
        converted_instance_labels_base_path, 
        converted_class_labels_base_path 
    ]:
        shutil.rmtree(path)
        os.makedirs(path)


def main():
    _initialize_converted_directories()
    convert_instance_labels()
    convert_class_labels()


if __name__ == "__main__":
    main()