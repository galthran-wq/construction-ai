"""
Define
"""
from pathlib import Path

data_path = Path(__file__).parent / "data"

WINDOW_ID = 3
BUILDING_ID = 1
BACKGROUND_ID = 0

id2class_general = { 
    BACKGROUND_ID: "background",
    BUILDING_ID: "building",
    2: "door",
    WINDOW_ID: "window",
}
class2id_general = { 
    value: key 
    for key, value in id2class_general.items() 
}
