
from pathlib import Path
from ..const import class2id_general, data_path

CARS_BUILDING_ID = (255, 255, 0)
CARS_WINDOW_ID = (255, 0, 0) 
id2class_cars = {
    CARS_WINDOW_ID: "window",
    (255, 128, 0): "door",
    (128, 0, 255): "balcony",
    (0, 255, 0): "shop",
    CARS_BUILDING_ID: "facade",
    (0, 0, 0): "background"
}

cars2general_class = {
    "facade": "building",
    "balcony": "building",
    "shop": "building",
    # 
    "door": "door",
    "window": "window",
    "background": "background",
}

cars2general_id = {
    cars_id: class2id_general[
        cars2general_class[
            cars_class
        ]
    ] for cars_id, cars_class in id2class_cars.items()
}

cars_base_path = data_path / "oxford-cars"
images_base_paths = [
    (
        cars_base_path / partition / 
        f"images_{100 if partition == 'test' else 400}"
    )
    for partition in ["train", "test"]
]
instance_labels_base_paths = [
    (
        cars_base_path / partition / 
        f"labels_{100 if partition == 'test' else 400}_panoptic"
    )
    for partition in ["train", "test"]
]
class_labels_base_paths = [
    (
        cars_base_path / partition / 
        f"labels_{100 if partition == 'test' else 400}_semantic"
    )
    for partition in ["train", "test"]
]

converted_instance_labels_base_paths = [
    (
        cars_base_path / partition / 
        f"labels_{100 if partition == 'test' else 400}_panoptic_converted"
    )
    for partition in ["train", "test"]
]

converted_class_labels_base_paths = [
    (
        cars_base_path / partition / 
        f"labels_{100 if partition == 'test' else 400}_semantic_converted"
    )
    for partition in ["train", "test"]
]