from json import load as loadjson
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent

EPOCHS: int= 12
ON_TRAINING_BATCH: int= 2
QUANTITY_FRAME: int= 18
LANDMARK_SHAPE: tuple= (36 +8 +21*2, 2) # ie. (86, 2)
# SKELETON_SHAPE: tuple= (158, 158, 3)
# PART4_MOD2USE: tuple= (3,4,5,6,7,8,9)                 # 1st try
# PART4_MOD2USE: tuple= (4,5,6,7,8,9,10,11,12,13,14,15) # 2nd try
# PART4_MOD2USE: tuple= (3,5,7,9)                       # 3rd try
PART4_MOD2USE: tuple= (3,4,5,6,7,8,9,10,11,12)  # 4st ( try this said.2026.7.22 4PM UTC+8 )DONE✔ 2026/7/26
# ✔ 2026/7/26 DONE: data cleaning added new GLASL
# ✔ 2026/7/26 DONE: add `right_hand_mandatory` key on an images gloss
# ---- eg. who, drink, fever, ...
IMG_SIZE: int= 158 # on G10 mandatory be 158x158x3
GLASL_LANDMARK_DIR: Path= PROJ_ROOT /"dataset" /"glasl" /"landmark"
GLASL_SKELETON_DIR: Path= PROJ_ROOT /"dataset" /"glasl" /"skeleton"

glasl_landmark: dict= {}
with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl.annotation.landmark.json"}", "r") as f:
    glasl_landmark= loadjson(f)

glasl_skeleton: dict= {}
with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl.annotation.skeleton.json"}", "r") as f:
    glasl_skeleton= loadjson(f)

KEY_TRAIN: str= 'train'   # landmark is face, then pose, then left_had, then right hand
KEY_VAL: str= 'val'       # face full is (468, 2) --> face worthy is (36, 2)
KEY_TEST: str= 'test'     # pose full is (33, 2) --> pose worthy is (8, 2)
KEY_ID2G: str= 'id2gloss' # left_hand is (21, 2)
KEY_G2ID: str= 'gloss2id' # right_hand is (21, 2)
KEY_RH_MANDATORY: str= "right_hand_mandatory"
KEY_GLOSS: str= 'gloss_id'
KEY_VIDEO: str= 'video_id'
KEY_LMARK: str= 'landmark'
KEY_FILE: str= 'file'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'

LEN_GLOSS: int= int(len(glasl_skeleton[KEY_ID2G]))


FACE_CONNECTIONS_FULL: tuple= (
    # oval face
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),

    # left eyebrow
    (276, 283), (283, 282), (282, 295),
    (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
    (276, 300), (285, 336),
    # left eye
    (263, 249), (249, 390), (390, 373), (373, 374),
    (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    # right eyebrow
    (46, 53), (53, 52), (52, 65),
    (65, 55), (70, 63), (63, 105),
    (105, 66), (66, 107),
    (46, 70), (55, 107),
    # right eye
    (33, 7), (7, 163), (163, 144), (144, 145),
    (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),

    # nose
    (168, 6), (6, 197), (197, 195), (195, 5),
    (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97),
    (97, 2), (2, 326), (326, 327), (327, 294),
    (294, 278), (278, 344), (344, 440), (440, 275),
    (275, 4), (4, 45), (45, 220), (220, 115), (115, 48),
    (48, 64), (64, 98),
    # lips
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375),
    (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310),
    (310, 415), (415, 308)
)
FACE_CONNECTIONS: tuple= (
    (3, 28), (28, 34), (34, 27), (27, 35), (35, 17), # left oval face
    (3, 12), (12, 19), (19, 11), (11, 21), (21, 17), # right oval face

    (26, 29), (29, 30), # left eyebrow

    (23, 32), (32, 31), # left eye down
    (31, 33), (33, 23), # left eye up

    (10, 13), (13, 14), # right eyebrow

    (7, 16), (16, 15), # right eye down
    (15, 18), (18, 7), # rght eye up

    (20, 22), (22, 2), # nose vertical line
    (2, 25), (25, 1), # left half nose
    (1, 9), (9, 2), # rigth half nose

    # mouth
    (8, 6), (6, 24), # down lip edge down
    (24, 0), (0, 8), # up lip edge up
    (8, 5), (5, 24), # up/down lip inner a
    (24, 4), (4, 8), # up/down lip inner b
)
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)

# before use of POSE_CONNECTIONS modify landmark 1st
# modify to use index to be used only: 11,12,13,14,15,16,23,24
# so new index: 0,1,2,3,4,5,6,7
POSE_CONNECTIONS_FULL: tuple= ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32))
POSE_CONNECTIONS: tuple= ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6), (1, 7))
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)

HAND_CONNECTIONS: tuple= (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), # palm connections
    (1, 2), (2, 3), (3, 4),           # thumb finger connections
    (5, 6), (6, 7), (7, 8),           # index finger connections
    (9, 10), (10, 11), (11, 12),      # middle finger connections
    (13, 14), (14, 15), (15, 16),     # ring finger connections
    (17, 18), (18, 19), (19, 20)      # pinky finger connections
)
