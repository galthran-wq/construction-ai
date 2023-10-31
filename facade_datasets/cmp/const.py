from pathlib import Path
from ..const import class2id_general, data_path

CMP_BUILDING_ID = 2
id2class_cmp = {
    1: 'background',
    CMP_BUILDING_ID: 'facade',
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
}

TO_BUILDING_IDs = [CMP_BUILDING_ID, 5, 6, 7, 8, 9, 10, 11 ]

etrims2general_class = {
    "facade": "building",
    "cornice": "building",
    "sill": "building",
    "balcony": "building",
    "blind": "window",
    # 
    "door": "door",
    "window": "window",
    "background": "background",
}

# etrims2general_id = {
#     etrims_id: class2id_general[
#         etrims2general_class[
#             etrims_class
#         ]
#     ] for etrims_id, etrims_class in id2class_etrims.items()
# }

base_path = data_path / "cmp"
images_base_path = class_labels_base_path = base_path / "merged"

converted_instance_labels_base_path = base_path / "instance_converted" 
converted_class_labels_base_path = base_path / "class_converted" 