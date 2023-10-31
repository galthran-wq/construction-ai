import os
import shutil
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import xml.etree.ElementTree as etree
from pathlib import Path

from .const import (
    TO_BUILDING_IDs,

    class_labels_base_path,

    converted_class_labels_base_path,
    converted_instance_labels_base_path,
)

from ..const import BACKGROUND_ID, BUILDING_ID, WINDOW_ID, DOOR_ID


def parse_object(path):
    """
    We use .xml to get info about windows and doors instances
    """
    file = Path(path)
    data = b'<rules>' + file.read_bytes() + b'</rules>'
    arr = etree.fromstring(data)
    arr = filter(
        lambda x: x.find("labelname").text.strip() in ["window", "door"], 
        arr
    )
    return arr


def populate(arr, box, with_):
    h, w = arr.shape
    x_coord = np.array([float(coord.text) for coord in box.find("points").findall("x")])
    y_coord = np.array([float(coord.text) for coord in box.find("points").findall("y")])

    for i in range(w):
        for j in range(h):
            if (
                (i >= (x_coord*w)[0]) and
                (i <= (x_coord*w)[1]) and
                (j <= (y_coord*h)[1]) and
                (j >= (y_coord*h)[0])
            ):
                try:
                    arr[i, j] = with_
                except IndexError:
                    pass


def populate(arr, box, with_):
    h, w = arr.shape
    x_coord = np.array([float(coord.text) for coord in box.find("points").findall("x")])*h
    y_coord = np.array([float(coord.text) for coord in box.find("points").findall("y")])*w
    x_idx = (np.arange(h).reshape(-1, 1))
    y_idx = (np.arange(w))

    arr[
        (x_idx >= x_coord[0]) * (x_idx <= x_coord[1]) *
        (y_idx >= y_coord[0]) * (y_idx <= y_coord[1])
    ] = with_


def convert():
    """
    We use the fact that there is only one building on every picture.
    """
    filenames = set([ 
        label_filename.split(".")[0] 
        for label_filename 
        in os.listdir(class_labels_base_path) 
    ])

    for label_filename in tqdm(filenames):
        class_labels_arr = np.array(Image.open(
            class_labels_base_path / (label_filename + ".png")
        ))
        instances = parse_object(class_labels_base_path / (label_filename + ".xml"))

        
        class_labels_arr_converted = np.zeros(
            (class_labels_arr.shape[0], class_labels_arr.shape[1])
        )
        instance_labels_arr_converted = np.zeros(
            (class_labels_arr.shape[0], class_labels_arr.shape[1])
        )

        # 1. let building be the first instance
        current_instance_id = 1

        @np.vectorize
        def to_building_mask(el):
            return el in TO_BUILDING_IDs

        instance_labels_arr_converted[
            to_building_mask(class_labels_arr )
        ] = current_instance_id

        class_labels_arr_converted[
            to_building_mask(class_labels_arr )
        ] = BUILDING_ID

        current_instance_id += 1

        for instance in instances:
            if instance.find("labelname").text.strip() == "window":
                with_ = WINDOW_ID
            if instance.find("labelname").text.strip() in ["door", "shop"]:
                with_ = DOOR_ID
            populate(class_labels_arr_converted, with_=with_, box=instance)
            populate(instance_labels_arr_converted, with_=current_instance_id, box=instance)
            current_instance_id += 1

        np.save(converted_instance_labels_base_path / label_filename, instance_labels_arr_converted)
        np.save(converted_class_labels_base_path / label_filename, class_labels_arr_converted)


def _initialize_converted_directories():
    for path in [
        converted_instance_labels_base_path, 
        converted_class_labels_base_path 
    ]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def main():
    _initialize_converted_directories()
    convert()


if __name__ == "__main__":
    main()