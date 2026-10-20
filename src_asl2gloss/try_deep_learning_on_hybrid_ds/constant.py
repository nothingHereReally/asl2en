from json import load as loadJson
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
EPOCHS: int= 20
BATCH: int= 16
QUANTITY_FRAME: int= 8
MOD_PART_3: tuple= (2, 3, 4, 5, 6)
LANDMARK_SHAPE: tuple= (36 +8 +21*2, 2) # ie. (86, 2)
'''
face full is (468, 2) --> face worthy is (36, 2)
pose full is (33, 2) --> pose worthy is (8, 2)
left_hand is (21, 2)
right_hand is (21, 2)
'''
LANDMARK_DIR_WLASL: Path= PROJ_ROOT /"dataset" /"clean_dataset" /"fix_recent_n_classic_landmark"
LANDMARK_DIR_GLASL: Path= PROJ_ROOT /"dataset" /"glasl" /"glasl_v4_landmark"
wlasl_ds: list= []
with open(f"{PROJ_ROOT /"dataset" /"clean_dataset" /"fix_recent_n_classic_landmark.json"}", "r") as f:
    wlasl_ds= loadJson(f)
glasl_ds: list= []
with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl_v4_landmark.json"}", 'r') as f:
    glasl_ds= loadJson(f)
LEN_GLOSS: int= len(wlasl_ds)
KEY_G: str= 'gloss'
KEY_GID: str= 'gloss_id'
KEY_VIDS: str= 'videos'
KEY_LANDMARK: str= 'landmark'

KEY_PFOLDER_FROM: str= 'parent_folder_from'
KEY_PFOLDER: str= 'parent_folder'
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'

KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
LM_IDX_TOP_LIP: int        =  0
LM_IDX_TIP_OF_FOREHEAD: int=  3
LM_IDX_RIGHT_MOST_EYE: int =  7
LM_IDX_RIGHT_MOST_LIP: int =  8
LM_IDX_RIGHT_EAR: int      = 19
LM_IDX_UP_TIP_NOSE: int    = 20
LM_IDX_LEFT_MOST_EYE: int  = 23
LM_IDX_LEFT_MOST_LIP: int  = 24
LM_IDX_LEFT_EAR: int       = 34
LM_IDX_LEFT_ELBOW: int     = 38
LM_IDX_RIGHT_ELBOW: int    = 39
LM_IDX_LEFT_ARM_TIP: int   = 40
LM_IDX_RIGHT_ARM_TIP: int  = 41
# -----------------------------
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
LM_IDX_LEFT_HAND_F_THUMB: int= 48
LM_IDX_LEFT_HAND_F_INDEX: int= 52
LM_IDX_LEFT_HAND_F_MIDLE: int= 56
LM_IDX_LEFT_HAND_F_RING: int = 60
LM_IDX_LEFT_HAND_F_PINKY: int= 64
# -----------------------------
LM_IDX_RIGHT_HAND_F_THUMB: int= 48
LM_IDX_RIGHT_HAND_F_INDEX: int= 52
LM_IDX_RIGHT_HAND_F_MIDLE: int= 56
LM_IDX_RIGHT_HAND_F_RING: int = 60
LM_IDX_RIGHT_HAND_F_PINKY: int= 64
# -----------------------------
LM_IMPORTANT_FACE_IDX: tuple= (
    LM_IDX_TIP_OF_FOREHEAD, # ok

    LM_IDX_LEFT_MOST_EYE,   # ok
    LM_IDX_RIGHT_MOST_EYE,  # ok

    LM_IDX_LEFT_EAR,        # ok
    LM_IDX_RIGHT_EAR,       # ok

    LM_IDX_UP_TIP_NOSE,     # ok

    LM_IDX_LEFT_MOST_LIP,   # ok
    LM_IDX_RIGHT_MOST_LIP,  # ok
    LM_IDX_TOP_LIP,         # ok
)
# LM_IMPORTANT_FACE_IDX: tuple= (
#     LM_IDX_TOP_LIP,
#     LM_IDX_TIP_OF_FOREHEAD,
#     LM_IDX_UP_TIP_NOSE,
#     # -- on later process here to insert up and left body dots
#     LM_IDX_LEFT_ARM_TIP,
#     LM_IDX_RIGHT_ARM_TIP,
# )
LM_LEFT_HAND_TIP: tuple= (
    LM_IDX_LEFT_HAND_F_THUMB,
    LM_IDX_LEFT_HAND_F_INDEX,
    LM_IDX_LEFT_HAND_F_MIDLE,
    LM_IDX_LEFT_HAND_F_RING,
    LM_IDX_LEFT_HAND_F_PINKY,
)
LM_LEFT_HAND_NOT_TIP: tuple= (
    44, 45, 46, 47,
    49, 50, 51,
    53, 54, 55,
    57, 58, 59,
    61, 62, 63,
)
LM_RIGHT_HAND_TIP: tuple= (
    LM_IDX_RIGHT_HAND_F_THUMB,
    LM_IDX_RIGHT_HAND_F_INDEX,
    LM_IDX_RIGHT_HAND_F_MIDLE,
    LM_IDX_RIGHT_HAND_F_RING,
    LM_IDX_RIGHT_HAND_F_PINKY,
)
LM_RIGHT_HAND_NOT_TIP: tuple= (
    65, 66, 67, 68,
    70, 71, 72,
    74, 75, 76,
    78, 79, 80,
    82, 83, 84,
)
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)
QUANTITY_HAND_LMARK: int= 21
GLOSS_ONE_HAND: tuple[bool]= (
    False, # -- book
    True,  # -- drink
    False, # -- computer
    False, # -- before
    False, # -- chair
    False, # -- go
    False, # -- clothes
    True,  # -- who
    True,  # -- candy
    True,  # -- cousin
    True,  # -- mine_my
    True,  # -- me_i
    True,  # -- stomach
    False, # -- have
    True,  # -- need
    True,  # -- see
    True,  # -- feel
    False, # -- hurt
    True,  # -- fever
    False, # -- dizzy
    False, # -- headache
    False, # -- doctor
)
