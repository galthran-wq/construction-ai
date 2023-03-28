"""
Define
"""
from pathlib import Path

data_path = Path(__file__).parent / "data"

WINDOW_ID = 3
BUILDING_ID = 1
BACKGROUND_ID = 0
DOOR_ID = 2

id2class_general = { 
    BACKGROUND_ID: "background",
    BUILDING_ID: "building",
    DOOR_ID: "door",
    WINDOW_ID: "window",
}
class2id_general = { 
    value: key 
    for key, value in id2class_general.items() 
}
