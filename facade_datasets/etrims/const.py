from pathlib import Path
from ..const import class2id_general, data_path

ETRIMS_BACKGROUND_ID = 0
id2class_etrims = {
    1: 'building',
    2: 'car',
    3: 'door',
    4: 'pavement',
    5: 'road',
    6: 'sky',
    7: 'vegetation',
    8: 'window',
    ETRIMS_BACKGROUND_ID: 'background'
}

etrims2general_class = {
    "car": "background",
    "pavement": "background",
    "road": "background",
    "sky": "background",
    "vegetation": "background",
    # 
    "building": "building",
    "door": "door",
    "window": "window",
    "background": "background",
}

etrims2general_id = {
    etrims_id: class2id_general[
        etrims2general_class[
            etrims_class
        ]
    ] for etrims_id, etrims_class in id2class_etrims.items()
}

etrims_base_path = data_path / "etrims-db_v1"
images_base_path = etrims_base_path / "images" / "08_etrims-ds"
instance_labels_base_path = etrims_base_path / "annotations-object" / "08_etrims-ds"
class_labels_base_path = etrims_base_path / "annotations" / "08_etrims-ds"

# no need to convert
converted_instance_labels_base_path = etrims_base_path / "annotations-objecte-converted" / "08_etrims-ds"

converted_class_labels_base_path = etrims_base_path / "annotations-converted" / "08_etrims-ds"