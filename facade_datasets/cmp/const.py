from pathlib import Path
from ..const import class2id_general, data_path

id2class_cmp = {
    1: 'building',
    2: 'facade',
    3: 'window',
    4: 'door',
    5: 'cornice',
    6: 'sill',
    7: 'balcony',
    8: 'blind',
    9: 'deco',
    10: 'molding',
    11: 'pillar',
    12: 'shop',
    0: 'background'
}

etrims2general_class = {
    "facade": "building",
    "cornice": "building",
    "sill": "building",
    "balcony": "building",
    "blind": "window",
    "building": "background",
    # 
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

base_path = data_path / "cmp"
images_base_path = class_labels_base_path = base_path / "merged"

converted_instance_labels_base_path = base_path / "instance_converted" 
converted_class_labels_base_path = base_path / "class_converted" 