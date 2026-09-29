from json import load as loadJson
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
EPOCHS: int= 20
ON_TRAINING_BATCH: int= 32
QUANTITY_FRAME: int= 8
LANDMARK_SHAPE: tuple= (36 +8 +21*2, 2) # ie. (86, 2)
DS_LANDMARK_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset" /"ds_landmark"
ds_landmark: list= []
with open(f"{PROJ_ROOT /"dataset" /"clean_dataset" /"ds_landmark.json"}", "r") as f:
    ds_landmark= loadJson(f)
    '''
    face full is (468, 2) --> face worthy is (36, 2)
    pose full is (33, 2) --> pose worthy is (8, 2)
    left_hand is (21, 2)
    right_hand is (21, 2)
    '''
LEN_GLOSS: int= len(ds_landmark)
KEY_G: str= 'gloss'
KEY_GID: str= 'gloss_id'
KEY_LANDMARK: str= 'landmark'

KEY_PFOLDER: str= 'parent_folder'
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'

KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
