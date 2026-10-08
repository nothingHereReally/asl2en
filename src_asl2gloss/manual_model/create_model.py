from json import load as loadJson, dump as writeJson
import numpy as np
import jax.numpy as jnp
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
LANDMARK_DIR: Path= DS_DIR /"ds_landmark"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
QUANTITY_FRAME: int= 8
# ---------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
# -----------------------------------------------------
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
QUANTITY_HAND_LMARK: int= 21
IMGS_LM_SHAPE: tuple= (
    QUANTITY_FRAME, # 8
    len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK*2, # 86
    2
)
LM_IDX_TOP_LIP: int        =  0
LM_IDX_TIP_OF_FOREHEAD: int=  3
LM_IDX_RIGHT_MOST_EYE: int =  7
LM_IDX_RIGHT_MOST_LIP: int =  8
LM_IDX_RIGHT_EAR: int      = 19
LM_IDX_LEFT_MOST_EYE: int  = 23
LM_IDX_LEFT_MOST_LIP: int  = 24
LM_IDX_LEFT_EAR: int       = 34
LM_IDX_LEFT_ELBOW: int     = 38
LM_IDX_RIGHT_ELBOW: int    = 39
# -----------------------------
LM_IDX_LEFT_INDEXF_TIP: int= 52
LM_IDX_RIGHT_INDEXF_TIP:int= 73
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
WORTHY_LM_IMPORTANT_IDX: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_RIGHT_MOST_EYE,
    LM_IDX_RIGHT_MOST_LIP,
    LM_IDX_RIGHT_EAR,
    LM_IDX_LEFT_MOST_EYE,
    LM_IDX_LEFT_MOST_LIP,
    LM_IDX_LEFT_EAR,
    LM_IDX_LEFT_ELBOW,
    LM_IDX_RIGHT_ELBOW,
)
# -----------------------------------------------------
VERIFIED_DS_DIR: list= [
    {
        KEY_G: 'book',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'book_07071_00002_001_018_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_07071_00004_001_018_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_68011_00112_006_040_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_68011_00122_006_040_from_gt_p3',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_07076_00254_019_067_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_07076_00264_019_067_from_gt_p3',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_07097_00218_019_043_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'book_07097_00220_019_043_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'drink',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'drink_17724_00020_001_030_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_68538_00030_032_055_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_69302_00065_022_057_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_17729_00107_020_048_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_17725_00162_020_045_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_17726_00177_023_062_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_17727_00195_019_053_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
            {
                KEY_PFOLDER: 'drink_17727_00198_019_053_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                    3,
                    6,
                ],
            },
        ]
    },
    {
        KEY_G: 'computer',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'computer_12306_00002_001_052_from_gt_p2',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_68028_00055_023_063_from_gt_p2',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_12331_00183_024_051_from_gt_p1',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_12331_00189_024_051_from_gt_p2',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_12312_00237_047_081_from_gt_p1',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_12326_00262_013_073_from_gt_p1',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_computer_0004_00335_019_086_from_gt_p1',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'computer_computer_0005_00413_013_082_from_gt_p1',
                KEY_LHAND: [
                    11,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'before',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'before_05730_00001_008_025_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_05733_00013_025_083_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_05729_00068_013_036_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_05727_00077_024_055_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_05739_00089_001_052_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_05750_00178_036_058_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_05744_00200_024_047_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'before_65167_00209_015_032_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'chair',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'chair_09853_00002_001_020_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_09848_00018_018_060_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_09869_00051_040_065_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_09847_00114_001_052_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_68580_00166_013_040_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_70263_00188_033_081_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_68019_00219_017_047_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'chair_09866_00280_028_064_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'go',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'go_24948_00001_001_018_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_68292_00022_019_038_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_24965_00044_017_037_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_24954_00062_001_061_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_24857_00135_001_035_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_69345_00160_015_039_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_24941_00172_016_032_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'go_24971_00221_021_039_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'clothes_11326_00001_002_023_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_11311_00021_027_053_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_11314_00040_022_033_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_11316_00051_022_049_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_11305_00073_001_052_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_68870_00128_012_035_from_gt_p2',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_11328_00190_001_029_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'clothes_11329_00215_001_028_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'who',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'who_63240_00001_025_044_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_63228_00025_013_029_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_63229_00035_019_032_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_63234_00048_025_043_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_63232_00109_065_116_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_70380_00161_001_027_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_who_0003_00180_020_072_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
            {
                KEY_PFOLDER: 'who_who_0005_00238_018_068_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ],
            },
        ]
    },
    {
        KEY_G: 'candy',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'candy_08923_00001_001_039_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_70326_00045_067_099_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_08929_00062_020_041_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_68790_00095_074_105_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_08916_00140_035_062_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_08919_00162_027_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_08921_00171_006_035_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
            {
                KEY_PFOLDER: 'candy_08927_00218_015_046_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    0,
                ]
            },
        ]
    },
    {
        KEY_G: 'cousin',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'cousin_13640_00047_001_059_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_65415_00102_025_043_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_68592_00116_005_031_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_13647_00135_022_045_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_13648_00144_021_040_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_13632_00161_027_052_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_13633_00175_043_067_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'cousin_67535_00187_017_032_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'mine_my_37472_00001_001_026_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'mine_my_37474_00016_010_093_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [],
            },
            {
                KEY_PFOLDER: 'mine_my_37477_00123_031_049_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'mine_my_36139_00137_012_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'mine_my_36145_00214_018_048_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'mine_my_36146_00246_029_049_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'mine_my_37464_00264_004_033_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'mine_my_69404_00293_015_043_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [],
            },
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'me_i_35267_00001_001_035_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_35267_00003_001_035_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_35544_00026_018_042_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_35545_00048_024_041_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_35546_00060_004_058_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_35547_00133_021_055_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_35548_00254_001_030_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'me_i_me_i_0001_00283_018_097_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'stomach_54870_00001_017_049_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_54885_00018_009_042_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_66560_00039_015_034_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_54878_00055_031_063_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_54880_00072_001_051_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_54881_00117_006_019_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_54883_00130_001_060_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
            {
                KEY_PFOLDER: 'stomach_54884_00196_013_040_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    11,
                ],
            },
        ]
    },
    {
        KEY_G: 'have',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'have_69088_00030_001_032_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_26776_00064_012_045_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_26757_00085_001_035_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_68069_00124_030_064_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_70221_00149_027_055_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_68984_00174_013_040_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_26775_00196_014_043_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'have_26766_00225_011_035_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'need',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'need_37891_00026_024_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_37892_00045_017_046_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_68544_00075_004_036_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_37885_00091_001_055_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_37887_00164_017_056_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_37889_00222_017_036_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_37881_00239_028_057_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'need_67925_00267_013_039_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'see',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'see_50125_00001_001_030_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_68444_00030_023_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_50128_00061_020_033_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_50107_00074_017_053_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_50120_00117_039_059_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_67178_00135_010_025_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_50123_00143_001_055_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'see_50126_00216_034_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
        ]
    },
    {
        KEY_G: 'feel',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'feel_67653_00001_012_029_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_21434_00013_018_059_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_69319_00042_023_056_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_21425_00063_001_049_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_21438_00094_021_033_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_21439_00106_023_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_21441_00153_013_048_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'feel_21436_00202_001_053_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'hurt_28450_00039_029_058_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28445_00068_001_055_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28441_00141_016_042_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28443_00160_001_053_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28444_00218_001_053_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28454_00276_025_045_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28452_00294_026_047_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
            {
                KEY_PFOLDER: 'hurt_28438_00314_001_033_from_gt_p1',
                KEY_LHAND: [
                    10,
                ],
                KEY_RHAND: [
                    10,
                ],
            },
        ]
    },
    {
        KEY_G: 'fever',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'fever_78219_00001_007_043_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'fever_78220_00036_007_028_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'fever_78221_00056_002_019_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'fever_78222_00068_004_014_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'fever_78223_00079_005_089_from_gt_p2',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'fever_78224_00196_023_116_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'fever_78225_00339_001_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
        {
                KEY_PFOLDER: 'fever_78226_00377_024_050_from_gt_p1',
                KEY_LHAND: [],
                KEY_RHAND: [
                    1,
                ],
            },
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_LANDMARK: [
            {
                KEY_PFOLDER: 'dizzy_16980_00001_001_091_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'dizzy_16982_00109_017_049_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'dizzy_16986_00126_001_057_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'dizzy_16987_00167_023_058_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'dizzy_16988_00198_017_053_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'dizzy_78227_00233_022_052_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
            {
                KEY_PFOLDER: 'dizzy_78228_00265_048_160_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
        {
                KEY_PFOLDER: 'dizzy_78229_00380_019_130_from_gt_p1',
                KEY_LHAND: [
                    1,
                ],
                KEY_RHAND: [
                    1,
                ],
            },
        ]
    },
    {
        KEY_G: 'headache',
        KEY_LANDMARK: [
            'headache_26832_00001_014_038_from_gt_p1',
            'headache_26835_00013_031_061_from_gt_p1',
            'headache_67747_00045_014_028_from_gt_p1',
            'headache_26846_00082_029_061_from_gt_p1',
            'headache_26839_00099_001_046_from_gt_p1',
            'headache_26838_00176_001_041_from_gt_p1',
            'headache_26841_00199_020_038_from_gt_p1',
            'headache_65881_00213_014_032_from_gt_p1',
        ]
    },
    {
        KEY_G: 'doctor',
        KEY_LANDMARK: [
            'doctor_17007_00035_001_043_from_gt_p2',
            'doctor_70049_00067_026_054_from_gt_p1',
            'doctor_17017_00096_024_058_from_gt_p2',
            'doctor_65504_00119_015_041_from_gt_p2',
            'doctor_17023_00136_022_048_from_gt_p1',
            'doctor_17014_00155_029_053_from_gt_p1',
            'doctor_17020_00171_001_055_from_gt_p2',
            'doctor_17022_00242_028_064_from_gt_p2',
        ]
    },
]
def flip_vertical_x_axis(landmarks) -> np.ndarray:
    '''
    landmarks be of shape (INT, ..., INT, 2)
    '''
    new_landmark= np.array(landmarks).copy()
    new_landmark[..., 0]= 1 -new_landmark[..., 0]
    lhand2rhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ,:].copy()
    rhand2lhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ,:].copy()
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ,:]= rhand2lhand # ie. past was right hand, now be left hand
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ,:]= lhand2rhand # ie. past was left hand, now be right hand
    return new_landmark
def loadnpfiles(parent_folder: str, landmarks: list) -> np.ndarray:
    out: list= []
    for a_lm in landmarks:
        with open(f"{LANDMARK_DIR /parent_folder /a_lm[KEY_FILE]}", 'rb') as f:
            out.append(np.load(f))
    return np.array(out)
def load_ds_samples() -> tuple:
    '''
    return (manual_model, labels)
    manual_model has shape (22, 8, 8, 86, 2)
    --> 22 gloss categories
    -->  8 samples per gloss
    -->  8 images to take
    --> 86 landmarks per image
    -->  2 coordinates per landmarks
    labels is a list[str], the list of actual gloss themselves
    eg. labels: list[str]= ['book', 'drink', ...]
    '''
    ds_landmark: list= []
    with open(f"{DS_DIR /'ds_landmark.json'}", 'r') as f:
        ds_landmark= loadJson(f)
    manual_model: list= []
    annotations: list= []
    for idx, a_gloss in enumerate(VERIFIED_DS_DIR):
        assert ds_landmark[idx][KEY_G]==a_gloss[KEY_G]
        annotations.append({
            KEY_G: ds_landmark[idx][KEY_G],
            KEY_LANDMARK: list(
                {
                    KEY_PFOLDER: el[KEY_PFOLDER],
                    KEY_LANDMARK: el[KEY_LANDMARK],
                }
                for el in ds_landmark[idx][KEY_LANDMARK]
                if el[KEY_PFOLDER] in a_gloss[KEY_LANDMARK]
            )
        })
        manual_model.append([
            loadnpfiles(el[KEY_PFOLDER], el[KEY_LANDMARK])
            for el in annotations[-1][KEY_LANDMARK]
        ])
        # annotations[-1][KEY_LANDMARK]= [{
        #     KEY_PFOLDER: el[KEY_PFOLDER],
        #     KEY_LANDMARK: [{
        #         KEY_FACE: lm_img_el[KEY_FACE],
        #         KEY_POSE: lm_img_el[KEY_POSE],
        #         KEY_LHAND: lm_img_el[KEY_LHAND],
        #         KEY_RHAND: lm_img_el[KEY_RHAND],
        #     } for lm_img_el in el[KEY_LANDMARK]]
        # } for el in annotations[-1][KEY_LANDMARK]]
        annotations[-1][KEY_LANDMARK].extend([
        {
            KEY_PFOLDER: f"{many_imgs_el[KEY_PFOLDER]}__manual_flip",
            KEY_LANDMARK: [{
                KEY_FILE: lm_img_el[KEY_FILE],
                KEY_FACE: lm_img_el[KEY_FACE],
                KEY_POSE: lm_img_el[KEY_POSE],
                KEY_LHAND: lm_img_el[KEY_RHAND],
                KEY_RHAND: lm_img_el[KEY_LHAND],
            } for lm_img_el in many_imgs_el[KEY_LANDMARK]],
        } for many_imgs_el in annotations[-1][KEY_LANDMARK]])
        manual_model[-1].extend(flip_vertical_x_axis(manual_model[-1]))
    annotations[10][KEY_G]= 'mine'
    annotations[11][KEY_G]= 'i'
    return (manual_model, annotations)
def get_left_hand(input: jnp.ndarray) -> jnp.ndarray:
    if input.shape[-1]!=2:
        return input[
            ...,
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
        ].copy()
    return input[
        ...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK,
        :
    ].copy()
def get_right_hand(input: jnp.ndarray) -> jnp.ndarray:
    if input.shape[-1]!=2:
        return input[
            ...,
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
        ].copy()
    return input[
        ...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:,
        :
    ].copy()
def get_worthy10landmarks(input: jnp.ndarray) -> jnp.ndarray:
    '''
    input can be of shape (..., 86, 2) or (..., 86)
    ------------------------------
    if input shape is (..., 86, 2)
    then returned shape is (..., 10, 2)
    else returned shape is (..., 10)
    '''
    if input.shape[-1]!=2:
        return input[
            ...,
            WORTHY_LM_IMPORTANT_IDX
        ]
    return input[
        ...,
        WORTHY_LM_IMPORTANT_IDX,
        :
    ]
def get2dots_frombody(input: jnp.ndarray) -> jnp.ndarray:
    '''
    input -- shape of (..., 86, 2)
    -------------------------------
    returned shape is (...,  2, 2)
    '''
    shoulder_center: jnp.ndarray= (
        input[..., LM_IDX_LEFT_SHOULDER, :] +
        input[..., LM_IDX_RIGHT_SHOULDER, :]
    )/2.0
    hip_center: jnp.ndarray= (
        input[..., LM_IDX_LEFT_HIP, :] +
        input[..., LM_IDX_RIGHT_HIP, :]
    )/2.0
    shoulder_center= jnp.expand_dims(shoulder_center, axis=-2)
    hip_center= jnp.expand_dims(hip_center, axis=-2)
    shoulder_hip_center: jnp.ndarray= (shoulder_center +hip_center)/2.0
    '''
    shoulder_center and hip_center, both has shape (..., 1, 2)
    so as shoulder_hip_center has shape (..., 1, 2)
    '''
    return jnp.concatenate([
        (shoulder_center+shoulder_hip_center)/2.0,
        (shoulder_hip_center+hip_center)/2.0,
    ], axis=-2)
def mean_ignore_zero(input: jnp.ndarray, keepdims: bool=False) -> jnp.ndarray:
    out: jnp.ndarray= jnp.sum(input, axis=-1, keepdims=keepdims)
    count: jnp.ndarray= jnp.sum(input!=0.0, axis=-1, keepdims=keepdims)
    # count= jnp.where(count==0, 1, count)
    count= jnp.maximum(count, 1)
    return out/count
def weights_edges(input: jnp.ndarray) -> jnp.ndarray:
    return input *jnp.ones(input.shape[-1]) \
                             .at[0].set(0.05) \
                             .at[1].set(0.1) \
                             .at[-2].set(0.1) \
                             .at[-1].set(0.05)
def hand_mse(
    input: jnp.ndarray,
    model: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    '''
    input -- shape of ( BATCH_SIZE,              1,                1, QUANTITY_FRAME, 86, 2)
    model -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86)
    86 landmarks mandatory to follow this order
    [...face_landmarks_36..., ...pose_landmarks_8..., ...left_hand_landmarks_21..., ...right_hand_landmarks_21...]
    --------------------------------------------------------------------------------------------------------------
    output, has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    left_hand_diff: jnp.ndarray= (
        get_left_hand(model) -get_left_hand(input)
    )**2
    right_hand_diff: jnp.ndarray= (
        get_right_hand(model) -get_right_hand(input)
    )**2
    left_hand_diff= jnp.sum( # due2now c^2, from c^2= a^2 +b^2
        left_hand_diff,
        axis=-1,
        keepdims=False
    ) # now be shape (..., QUANTITY_FRAME, 21), ie. removed X, Y cords axis
    right_hand_diff= jnp.sum(
        right_hand_diff,
        axis=-1,
        keepdims=False
    ) # now be shape (..., QUANTITY_FRAME, 21), ie. removed X, Y cords axis
    left_hand_diff= left_hand_diff *get_left_hand(mask)
    right_hand_diff= right_hand_diff *get_right_hand(mask)
    '''
    now ----
    left_hand_diff
    right_hand_diff
    ---------------
    both has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 21)
    '''
    left_hand_diff= jnp.mean(left_hand_diff, axis=-1, keepdims=False)
    right_hand_diff= jnp.mean(right_hand_diff, axis=-1, keepdims=False)
    '''
    now ----
    left_hand_diff
    right_hand_diff
    ---------------
    both has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME)
    '''
    left_hand_diff= weights_edges(left_hand_diff)
    right_hand_diff= weights_edges(right_hand_diff)

    # left_hand_diff= jnp.mean(left_hand_diff, axis=-1, keepdims=True, where=left_hand_diff!=0.0)
    # right_hand_diff= jnp.mean(right_hand_diff, axis=-1, keepdims=True, where=right_hand_diff!=0.0)
    left_hand_diff= mean_ignore_zero(left_hand_diff, keepdims=True)
    right_hand_diff= mean_ignore_zero(right_hand_diff, keepdims=True)
    '''
    now ----
    left_hand_diff
    right_hand_diff
    ---------------
    both has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, 1)
    '''''
    out: jnp.ndarray= jnp.concatenate([left_hand_diff, right_hand_diff], axis=-1)
    out= mean_ignore_zero(out, keepdims=False)
    '''
    out, has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    return out
def relativity_mse(
    input: jnp.ndarray,
    model: jnp.ndarray,
    mask: jnp.ndarray,
    model2dots: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    '''
    input -- shape of ( BATCH_SIZE,              1,                1, QUANTITY_FRAME, 86,  2)
    model -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86,  2)
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86)
    body2dots, shape  (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2,  2)
    weights, shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    -------------------------------------------------------------
    output shape is (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    input_lhand_index: jnp.ndarray= input[..., LM_IDX_LEFT_INDEXF_TIP, :]  # shape (..., 1, 2)
    input_rhand_index: jnp.ndarray= input[..., LM_IDX_RIGHT_INDEXF_TIP, :] # shape (..., 1, 2)
    input_12landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy10landmarks(input),
        get2dots_frombody(input),
    ], axis=-2) # shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 12, 2)
    model_lhand_index: jnp.ndarray= model[..., LM_IDX_LEFT_INDEXF_TIP, :]  # shape (..., 1, 2)
    model_rhand_index: jnp.ndarray= model[..., LM_IDX_RIGHT_INDEXF_TIP, :] # shape (..., 1, 2)
    model_12landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy10landmarks(model),
        model2dots,
    ], axis=-2) # shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12, 2)
    '''
    below are
    ---------
    now do difference/subtraction on left/right hand on 12 landmarks
    then multiply to weights
    then multiply to mask4left/right hand
    -------------------------------------
    distance( c^2= a^2 +b^2 ) on left/right hand( index finger tip only ) to 12 landmarks chosen
    '''
    distance_input_lhand: jnp.ndarray= (input_12landmarks -input_lhand_index)**2
    distance_input_rhand: jnp.ndarray= (input_12landmarks -input_rhand_index)**2
    distance_model_lhand: jnp.ndarray= (model_12landmarks -model_lhand_index)**2
    distance_model_rhand: jnp.ndarray= (model_12landmarks -model_rhand_index)**2
    '''
    distance_input_lhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 12, 2)
    distance_input_rhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 12, 2)
    distance_model_lhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12, 2)
    distance_model_rhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12, 2)
    '''
    distance_input_lhand= jnp.sum(distance_input_lhand, axis=-1, keepdims=False)
    distance_input_rhand= jnp.sum(distance_input_rhand, axis=-1, keepdims=False)
    distance_model_lhand= jnp.sum(distance_model_lhand, axis=-1, keepdims=False)
    distance_model_rhand= jnp.sum(distance_model_rhand, axis=-1, keepdims=False)
    '''
    distance_input_lhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 12)
    distance_input_rhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 12)
    distance_model_lhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    distance_model_rhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    '''
    model_input_lhand: jnp.ndarray= (distance_input_lhand -distance_model_lhand)**2
    model_input_rhand: jnp.ndarray= (distance_input_rhand -distance_model_rhand)**2
    lhand_mask: jnp.ndarray= get_left_hand(mask)[..., 0]
    rhand_mask: jnp.ndarray= get_right_hand(mask)[..., 0]
    '''
    lhand_mask        shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  1)
    rhand_mask        shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  1)
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    weights, shape of          (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    '''
    model_input_lhand= model_input_lhand *weights[..., 0, :] *lhand_mask
    model_input_rhand= model_input_rhand *weights[..., 1, :] *rhand_mask
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    '''
    model_input_lhand=mean_ignore_zero( # result shape (..., QUANTITY_FRAME, 12)
        input=jnp.stack([
            model_input_lhand,
            model_input_rhand,
        ], axis=-1),
        keepdims=False,
    )
    del model_input_rhand
    model_input_lhand=mean_ignore_zero( # result shape (..., QUANTITY_FRAME)
        input=model_input_lhand,
        keepdims=False,
    )
    model_input_lhand=mean_ignore_zero( # result shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
        input=model_input_lhand,
        keepdims=False,
    )

    return model_input_lhand
def comparison_score(
    input: np.ndarray,
    model: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    '''
    input, shape of -------------------------- (BATCH_SIZE, QUANTITY_FRAME, 86, 2)
    model, shape of - (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86)
    body2dots, shape  (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    -- on weights axis=-2 of value 2, for left and right hand repsectively
    -- on weights axis=-1 of value 12 for 12 landmarks
    -------------------------------------------------------------
    output shape is (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    assert len(input.shape)==(len(IMGS_LM_SHAPE)+1)
    assert len(model.shape)==(len(IMGS_LM_SHAPE)+3)
    assert input.shape[1:]==IMGS_LM_SHAPE
    assert model.shape[-1*len(IMGS_LM_SHAPE):]==IMGS_LM_SHAPE
    assert model.shape[:-1]==mask.shape
    input_exp: jnp.ndarray= jnp.expand_dims(input, axis=(1, 2))
    left_right_hand_mse: jnp.ndarray= hand_mse(
        input=input_exp,
        model=model,
        mask=mask,
    ) # shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    hand2body_mse: jnp.ndarray= relativity_mse(
        input=input_exp,
        model=model,
        mask=mask,
        model2dots=body2dots,
        weights=weights,
    ) # shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    return mean_ignore_zero(
        input=jnp.stack([left_right_hand_mse, hand2body_mse], axis=-1),
        keepdims=False,
    )
def create_mask(annotations: list) -> list:
    '''
    returns a list of shape (QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 2)
    '''
    output: list= []
    for a_gloss in annotations:
        output.append([]) # --> represents an element from QUANTTIY_GLOSS
        for a_start_end in a_gloss[KEY_LANDMARK]:
            output[-1].append([]) # --> represents 1 ds element, SAMPLE_PER_GLOSS
            for an_img in a_start_end[KEY_LANDMARK]:
                output[-1][-1].append([
                    1 if an_img[KEY_LHAND] else 0,
                    1 if an_img[KEY_RHAND] else 0,
                ]) # --> represents an element from QUANTITY_FRAME
    return output
def main() -> None:
    manual_model, annotations= load_ds_samples()
    with open("/tmp/asdf.json", "w") as f:
        writeJson(annotations, f, indent=4)
    print(f"shape --> {np.array(manual_model).shape}")
    print("\n".join([el[KEY_G] for el in annotations]))
if __name__=="__main__":
    main()
