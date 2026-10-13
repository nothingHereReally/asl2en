from json import load as loadJson, dump as writeJson
import numpy as np
import jax.numpy as jnp
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
LANDMARK_DIR: Path= DS_DIR /"ds_landmark"
MODEL2SAVE_DIR: Path= PROJ_ROOT /"src_asl2gloss" /"manual_model" /"model"
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
LM_IDX_UP_TIP_NOSE: int    = 20
LM_IDX_LEFT_MOST_EYE: int  = 23
LM_IDX_LEFT_MOST_LIP: int  = 24
LM_IDX_LEFT_EAR: int       = 34
LM_IDX_LEFT_ELBOW: int     = 38
LM_IDX_RIGHT_ELBOW: int    = 39
LM_IDX_LEFT_ARM_TIP: int   = 40
LM_IDX_RIGHT_ARM_TIP: int  = 41
# -----------------------------
LM_IDX_LEFT_INDEXF_TIP: int= 52
LM_IDX_RIGHT_INDEXF_TIP:int= 73
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
WORTHY_LM_IMPORTANT_IDX_V1: tuple= (
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
WORTHY_LM_IMPORTANT_IDX: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_UP_TIP_NOSE,
    # -- on later process here to insert up and left body dots
    LM_IDX_LEFT_ARM_TIP,
    LM_IDX_RIGHT_ARM_TIP,
)
# -----------------------------------------------------
VERIFIED_DS_DIR: list= [
    {
        KEY_G: 'book',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'book_07071_00002_001_018_from_gt_p2',
            'book_07071_00004_001_018_from_gt_p2',
            'book_68011_00112_006_040_from_gt_p2',
            'book_68011_00122_006_040_from_gt_p3',
            'book_07076_00254_019_067_from_gt_p2',
            'book_07076_00264_019_067_from_gt_p3',
            'book_07097_00218_019_043_from_gt_p2',
            'book_07097_00220_019_043_from_gt_p2',
        ]
    },
    {
        KEY_G: 'drink',
        KEY_LHAND: [0],
        KEY_RHAND: [0],
        KEY_LANDMARK: [
            'drink_17724_00020_001_030_from_gt_p2',
            'drink_68538_00030_032_055_from_gt_p1',
            'drink_69302_00065_022_057_from_gt_p2',
            'drink_17729_00107_020_048_from_gt_p1',
            'drink_17725_00162_020_045_from_gt_p1',
            'drink_17726_00177_023_062_from_gt_p1',
            'drink_17727_00195_019_053_from_gt_p1',
            'drink_17727_00198_019_053_from_gt_p2',
        ]
    },
    {
        KEY_G: 'computer',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'computer_12306_00002_001_052_from_gt_p2',
            'computer_68028_00055_023_063_from_gt_p2',
            'computer_12331_00183_024_051_from_gt_p1',
            'computer_12331_00189_024_051_from_gt_p2',
            'computer_12312_00237_047_081_from_gt_p1',
            'computer_12326_00262_013_073_from_gt_p1',
            'computer_computer_0004_00335_019_086_from_gt_p1',
            'computer_computer_0005_00413_013_082_from_gt_p1',
        ]
    },
    {
        KEY_G: 'computer',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'computer_68028_00059_023_063_from_gt_p2',
            'computer_69054_00077_020_117_from_gt_p2',
            'computer_69054_00079_020_117_from_gt_p2',
            'computer_12331_00204_024_051_from_gt_p3',
            'computer_12336_00206_021_053_from_gt_p2',
            'computer_12336_00210_021_053_from_gt_p2',
            'computer_12314_00224_014_039_from_gt_p2',
            'computer_12326_00268_013_073_from_gt_p2',
        ]
    },
    {
        KEY_G: 'computer',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'computer_12326_00267_013_073_from_gt_p2',
            'computer_12326_00291_013_073_from_gt_p2',
            'computer_12326_00306_013_073_from_gt_p3',
            'computer_12326_00312_013_073_from_gt_p3',
            'computer_computer_0004_00385_019_086_from_gt_p3',
            'computer_computer_0004_00390_019_086_from_gt_p3',
            'computer_computer_0005_00471_013_082_from_gt_p3',
            'computer_computer_0005_00485_013_082_from_gt_p3',
        ]
    },
    {
        KEY_G: 'before',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'before_05730_00001_008_025_from_gt_p1',
            'before_05733_00013_025_083_from_gt_p1',
            'before_05729_00068_013_036_from_gt_p1',
            'before_05727_00077_024_055_from_gt_p1',
            'before_05739_00089_001_052_from_gt_p1',
            'before_05750_00178_036_058_from_gt_p1',
            'before_05744_00200_024_047_from_gt_p1',
            'before_65167_00209_015_032_from_gt_p1',
        ]
    },
    {
        KEY_G: 'before',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'before_05730_00012_008_025_from_gt_p3',
            'before_05729_00075_013_036_from_gt_p3',
            'before_05727_00084_024_055_from_gt_p3',
            'before_05727_00086_024_055_from_gt_p3',
            'before_05739_00103_001_052_from_gt_p2',
            'before_05739_00109_001_052_from_gt_p2',
            'before_05739_00110_001_052_from_gt_p2',
            'before_05747_00141_022_046_from_gt_p1',
        ]
    },
    {
        KEY_G: 'before',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'before_05747_00142_022_046_from_gt_p2',
            'before_05747_00143_022_046_from_gt_p2',
            'before_05747_00144_022_046_from_gt_p2',
            'before_05748_00153_021_055_from_gt_p1',
            'before_05748_00154_021_055_from_gt_p2',
            'before_05748_00155_021_055_from_gt_p2',
            'before_05748_00156_021_055_from_gt_p2',
            'before_05750_00185_036_058_from_gt_p2',
        ]
    },
    {
        KEY_G: 'chair',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'chair_09853_00002_001_020_from_gt_p2',
            'chair_09848_00018_018_060_from_gt_p2',
            'chair_09869_00051_040_065_from_gt_p1',
            'chair_09847_00114_001_052_from_gt_p1',
            'chair_68580_00166_013_040_from_gt_p1',
            'chair_70263_00188_033_081_from_gt_p1',
            'chair_68019_00219_017_047_from_gt_p1',
            'chair_09866_00280_028_064_from_gt_p1',
        ]
    },
    {
        KEY_G: 'chair',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'chair_09862_00066_021_078_from_gt_p1',
            'chair_09862_00067_021_078_from_gt_p2',
            'chair_09862_00068_021_078_from_gt_p2',
            'chair_09862_00069_021_078_from_gt_p2',
            'chair_68580_00169_013_040_from_gt_p2',
            'chair_68580_00170_013_040_from_gt_p2',
            'chair_68580_00172_013_040_from_gt_p2',
            'chair_70263_00190_033_081_from_gt_p2',
        ]
    },
    {
        KEY_G: 'chair',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'chair_70263_00191_033_081_from_gt_p2',
            'chair_70263_00192_033_081_from_gt_p2',
            'chair_68019_00223_017_047_from_gt_p2',
            'chair_68019_00226_017_047_from_gt_p2',
            'chair_68019_00227_017_047_from_gt_p2',
            'chair_09865_00251_007_048_from_gt_p1',
            'chair_09865_00252_007_048_from_gt_p2',
            'chair_09865_00253_007_048_from_gt_p2',
        ]
    },
    {
        KEY_G: 'chair',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'chair_68580_00179_013_040_from_gt_p2',
            'chair_68580_00184_013_040_from_gt_p3',
            'chair_70263_00193_033_081_from_gt_p2',
            'chair_70263_00201_033_081_from_gt_p3',
            'chair_70263_00205_033_081_from_gt_p3',
            'chair_68019_00247_017_047_from_gt_p3',
            'chair_09865_00263_007_048_from_gt_p2',
            'chair_09865_00272_007_048_from_gt_p3',
        ]
    },
    {
        KEY_G: 'go',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'go_24948_00001_001_018_from_gt_p1',
            'go_68292_00022_019_038_from_gt_p2',
            'go_24965_00044_017_037_from_gt_p1',
            'go_24954_00062_001_061_from_gt_p1',
            'go_24857_00135_001_035_from_gt_p1',
            'go_69345_00160_015_039_from_gt_p1',
            'go_24941_00172_016_032_from_gt_p1',
            'go_24971_00221_021_039_from_gt_p2',
        ]
    },
    {
        KEY_G: 'go',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'go_68292_00033_073_098_from_gt_p2',
            'go_68292_00036_073_098_from_gt_p2',
            'go_68292_00037_073_098_from_gt_p2',
            'go_24965_00054_017_037_from_gt_p2',
            'go_24965_00055_017_037_from_gt_p2',
            'go_24965_00056_017_037_from_gt_p2',
            'go_24954_00070_001_061_from_gt_p2',
            'go_24954_00077_001_061_from_gt_p2',
        ]
    },
    {
        KEY_G: 'go',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'go_24954_00078_001_061_from_gt_p2',
            'go_69345_00164_015_039_from_gt_p2',
            'go_69345_00170_015_039_from_gt_p3',
            'go_69345_00171_015_039_from_gt_p3',
            'go_24969_00182_027_060_from_gt_p1',
            'go_24969_00183_027_060_from_gt_p2',
            'go_24969_00184_027_060_from_gt_p2',
            'go_24969_00185_027_060_from_gt_p2',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'clothes_11326_00001_002_023_from_gt_p1',
            'clothes_11311_00021_027_053_from_gt_p1',
            'clothes_11314_00040_022_033_from_gt_p1',
            'clothes_11316_00051_022_049_from_gt_p1',
            'clothes_11305_00073_001_052_from_gt_p1',
            'clothes_68870_00128_012_035_from_gt_p2',
            'clothes_11328_00190_001_029_from_gt_p1',
            'clothes_11329_00215_001_028_from_gt_p1',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'clothes_11326_00007_002_023_from_gt_p2',
            'clothes_11326_00008_002_023_from_gt_p2',
            'clothes_11326_00009_002_023_from_gt_p2',
            'clothes_11311_00029_027_053_from_gt_p2',
            'clothes_11311_00032_027_053_from_gt_p2',
            'clothes_11311_00033_027_053_from_gt_p2',
            'clothes_11314_00043_022_033_from_gt_p2',
            'clothes_11314_00044_022_033_from_gt_p2',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'clothes_11314_00045_022_033_from_gt_p2',
            'clothes_68870_00130_012_035_from_gt_p3',
            'clothes_68870_00132_012_035_from_gt_p3',
            'clothes_68870_00133_012_035_from_gt_p3',
            'clothes_68024_00134_012_050_from_gt_p1',
            'clothes_68024_00135_012_050_from_gt_p2',
            'clothes_68024_00136_012_050_from_gt_p2',
            'clothes_68024_00137_012_050_from_gt_p2',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'clothes_11327_00178_018_049_from_gt_p1',
            'clothes_11327_00179_018_049_from_gt_p2',
            'clothes_11327_00180_018_049_from_gt_p2',
            'clothes_11327_00181_018_049_from_gt_p2',
            'clothes_11328_00194_001_029_from_gt_p2',
            'clothes_11328_00197_001_029_from_gt_p2',
            'clothes_11328_00198_001_029_from_gt_p2',
            'clothes_11329_00219_001_028_from_gt_p2',
        ]
    },
    {
        KEY_G: 'who',
        KEY_LHAND: [0],
        KEY_RHAND: [0],
        KEY_LANDMARK: [
            'who_63240_00001_025_044_from_gt_p1',
            'who_63228_00025_013_029_from_gt_p1',
            'who_63229_00035_019_032_from_gt_p1',
            'who_63234_00048_025_043_from_gt_p1',
            'who_63232_00109_065_116_from_gt_p1',
            'who_70380_00161_001_027_from_gt_p1',
            'who_who_0003_00180_020_072_from_gt_p1',
            'who_who_0005_00238_018_068_from_gt_p1',
        ]
    },
    {
        KEY_G: 'who',
        KEY_LHAND: [0],
        KEY_RHAND: [0],
        KEY_LANDMARK: [
            'who_63242_00017_024_032_from_gt_p1',
            'who_63242_00018_024_032_from_gt_p2',
            'who_63242_00019_024_032_from_gt_p2',
            'who_63242_00020_024_032_from_gt_p3',
            'who_63229_00038_019_032_from_gt_p2',
            'who_63229_00039_019_032_from_gt_p2',
            'who_63229_00040_019_032_from_gt_p2',
            'who_63232_00112_065_116_from_gt_p2',
        ]
    },
    {
        KEY_G: 'candy',
        KEY_LHAND: [0],
        KEY_RHAND: [0],
        KEY_LANDMARK: [
            'candy_08923_00001_001_039_from_gt_p1',
            'candy_70326_00045_067_099_from_gt_p1',
            'candy_08929_00062_020_041_from_gt_p1',
            'candy_68790_00095_074_105_from_gt_p2',
            'candy_08916_00140_035_062_from_gt_p1',
            'candy_08919_00162_027_050_from_gt_p1',
            'candy_08921_00171_006_035_from_gt_p1',
            'candy_08927_00218_015_046_from_gt_p1',
        ]
    },
    {
        KEY_G: 'cousin',
        KEY_LHAND: [0, 1],
        KEY_RHAND: [0, 1],
        KEY_LANDMARK: [
            'cousin_13640_00047_001_059_from_gt_p1',
            'cousin_65415_00102_025_043_from_gt_p1',
            'cousin_68592_00116_005_031_from_gt_p1',
            'cousin_13647_00135_022_045_from_gt_p1',
            'cousin_13648_00144_021_040_from_gt_p1',
            'cousin_13632_00161_027_052_from_gt_p2',
            'cousin_13633_00175_043_067_from_gt_p1',
            'cousin_67535_00187_017_032_from_gt_p1',
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_LHAND: [3, 5, 6],
        KEY_RHAND: [3, 5, 6],
        KEY_LANDMARK: [
            'mine_my_37472_00001_001_026_from_gt_p1',
            'mine_my_37474_00016_010_093_from_gt_p1',
            'mine_my_37477_00123_031_049_from_gt_p1',
            'mine_my_36139_00137_012_050_from_gt_p1',
            'mine_my_36145_00214_018_048_from_gt_p1',
            'mine_my_36146_00246_029_049_from_gt_p1',
            'mine_my_37464_00264_004_033_from_gt_p1',
            'mine_my_69404_00293_015_043_from_gt_p1',
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_LHAND: [3, 5, 6],
        KEY_RHAND: [3, 5, 6],
        KEY_LANDMARK: [
            'me_i_35267_00001_001_035_from_gt_p1',
            'me_i_35267_00003_001_035_from_gt_p2',
            'me_i_35544_00026_018_042_from_gt_p1',
            'me_i_35545_00048_024_041_from_gt_p1',
            'me_i_35546_00060_004_058_from_gt_p1',
            'me_i_35547_00133_021_055_from_gt_p1',
            'me_i_35548_00254_001_030_from_gt_p1',
            'me_i_me_i_0001_00283_018_097_from_gt_p1',
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_LHAND: [3, 5, 6],
        KEY_RHAND: [3, 5, 6],
        KEY_LANDMARK: [
            'me_i_35549_00158_034_107_from_gt_p1',
            'me_i_35549_00159_034_107_from_gt_p2',
            'me_i_35549_00160_034_107_from_gt_p2',
            'me_i_35549_00161_034_107_from_gt_p2',
            'me_i_35550_00229_014_042_from_gt_p1',
            'me_i_35550_00230_014_042_from_gt_p2',
            'me_i_35550_00231_014_042_from_gt_p2',
            'me_i_35550_00232_014_042_from_gt_p2',
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_LHAND: [4],
        KEY_RHAND: [4],
        KEY_LANDMARK: [
            'stomach_54870_00001_017_049_from_gt_p1',
            'stomach_54885_00018_009_042_from_gt_p1',
            'stomach_66560_00039_015_034_from_gt_p1',
            'stomach_54878_00055_031_063_from_gt_p1',
            'stomach_54880_00072_001_051_from_gt_p1',
            'stomach_54881_00117_006_019_from_gt_p1',
            'stomach_54883_00130_001_060_from_gt_p1',
            'stomach_54884_00196_013_040_from_gt_p1',
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_LHAND: [4],
        KEY_RHAND: [4],
        KEY_LANDMARK: [
            'stomach_stomach_0001_00218_028_095_from_gt_p1',
            'stomach_stomach_0001_00219_028_095_from_gt_p2',
            'stomach_stomach_0001_00220_028_095_from_gt_p2',
            'stomach_stomach_0001_00221_028_095_from_gt_p2',
            'stomach_stomach_0002_00296_044_095_from_gt_p1',
            'stomach_stomach_0002_00297_044_095_from_gt_p2',
            'stomach_stomach_0002_00298_044_095_from_gt_p2',
            'stomach_stomach_0002_00299_044_095_from_gt_p2',
        ]
    },
    {
        KEY_G: 'have',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'have_69088_00030_001_032_from_gt_p1',
            'have_26776_00064_012_045_from_gt_p1',
            'have_26757_00085_001_035_from_gt_p1',
            'have_68069_00124_030_064_from_gt_p1',
            'have_70221_00149_027_055_from_gt_p1',
            'have_68984_00174_013_040_from_gt_p1',
            'have_26775_00196_014_043_from_gt_p1',
            'have_26766_00225_011_035_from_gt_p1',
        ]
    },
    {
        KEY_G: 'need',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'need_37891_00026_024_050_from_gt_p1',
            'need_37892_00045_017_046_from_gt_p1',
            'need_68544_00075_004_036_from_gt_p2',
            'need_37885_00091_001_055_from_gt_p1',
            'need_37887_00164_017_056_from_gt_p1',
            'need_37889_00222_017_036_from_gt_p1',
            'need_37881_00239_028_057_from_gt_p2',
            'need_67925_00267_013_039_from_gt_p1',
        ]
    },
    {
        KEY_G: 'see',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'see_50125_00001_001_030_from_gt_p1',
            'see_68444_00030_023_050_from_gt_p1',
            'see_50128_00061_020_033_from_gt_p1',
            'see_50107_00074_017_053_from_gt_p1',
            'see_50120_00117_039_059_from_gt_p1',
            'see_67178_00135_010_025_from_gt_p1',
            'see_50123_00143_001_055_from_gt_p1',
            'see_50126_00216_034_050_from_gt_p1',
        ]
    },
    {
        KEY_G: 'feel',
        KEY_LHAND: [3],
        KEY_RHAND: [3],
        KEY_LANDMARK: [
            'feel_67653_00001_012_029_from_gt_p1',
            'feel_21434_00013_018_059_from_gt_p1',
            'feel_69319_00042_023_056_from_gt_p1',
            'feel_21425_00063_001_049_from_gt_p1',
            'feel_21438_00094_021_033_from_gt_p1',
            'feel_21439_00106_023_050_from_gt_p1',
            'feel_21441_00153_013_048_from_gt_p1',
            'feel_21436_00202_001_053_from_gt_p1',
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'hurt_28450_00039_029_058_from_gt_p1',
            'hurt_28445_00068_001_055_from_gt_p1',
            'hurt_28441_00141_016_042_from_gt_p1',
            'hurt_28443_00160_001_053_from_gt_p1',
            'hurt_28444_00218_001_053_from_gt_p1',
            'hurt_28454_00276_025_045_from_gt_p1',
            'hurt_28452_00294_026_047_from_gt_p1',
            'hurt_28438_00314_001_033_from_gt_p1',
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'hurt_70147_00001_052_072_from_gt_p1',
            'hurt_70147_00002_052_072_from_gt_p2',
            'hurt_70147_00003_052_072_from_gt_p2',
            'hurt_70147_00004_052_072_from_gt_p2',
            'hurt_28453_00019_016_037_from_gt_p1',
            'hurt_28453_00020_016_037_from_gt_p2',
            'hurt_28453_00021_016_037_from_gt_p2',
            'hurt_28453_00022_016_037_from_gt_p2',
        ]
    },
    {
        KEY_G: 'fever',
        KEY_LHAND: [1],
        KEY_RHAND: [1],
        KEY_LANDMARK: [
            'fever_78219_00001_007_043_from_gt_p1',
            'fever_78220_00036_007_028_from_gt_p1',
            'fever_78221_00056_002_019_from_gt_p1',
            'fever_78222_00068_004_014_from_gt_p1',
            'fever_78223_00079_005_089_from_gt_p2',
            'fever_78224_00196_023_116_from_gt_p1',
            'fever_78225_00339_001_050_from_gt_p1',
            'fever_78226_00377_024_050_from_gt_p1',
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'dizzy_16980_00001_001_091_from_gt_p1',
            'dizzy_16982_00109_017_049_from_gt_p1',
            'dizzy_16986_00126_001_057_from_gt_p1',
            'dizzy_16987_00167_023_058_from_gt_p1',
            'dizzy_16988_00198_017_053_from_gt_p1',
            'dizzy_78227_00233_022_052_from_gt_p1',
            'dizzy_78228_00265_048_160_from_gt_p1',
            'dizzy_78229_00380_019_130_from_gt_p1',
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'dizzy_16982_00113_017_049_from_gt_p2',
            'dizzy_16982_00114_017_049_from_gt_p2',
            'dizzy_16982_00117_017_049_from_gt_p2',
            'dizzy_78228_00279_048_160_from_gt_p2',
            'dizzy_78228_00280_048_160_from_gt_p2',
            'dizzy_78228_00293_048_160_from_gt_p2',
            'dizzy_78229_00398_019_130_from_gt_p3',
            'dizzy_78229_00399_019_130_from_gt_p3',
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_LHAND: [],
        KEY_RHAND: [],
        KEY_LANDMARK: [
            'dizzy_78229_00400_019_130_from_gt_p3',
            'dizzy_65501_00483_030_050_from_gt_p2',
            'dizzy_65501_00485_030_050_from_gt_p2',
            'dizzy_65501_00488_030_050_from_gt_p2',
            'dizzy_65501_00490_030_050_from_gt_p2',
            'dizzy_16985_00499_001_028_from_gt_p1',
            'dizzy_16985_00500_001_028_from_gt_p2',
            'dizzy_16985_00501_001_028_from_gt_p2',
        ]
    },
    {
        KEY_G: 'headache',
        KEY_LHAND: [1, 2],
        KEY_RHAND: [1, 2],
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
        KEY_LHAND: [],
        KEY_RHAND: [],
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

    # -- flip index pair face
    # -- ( 12, 28 )
    tmp= new_landmark[..., 12:13, :].copy()
    new_landmark[..., 12:13, :]= new_landmark[..., 28:29, :].copy()
    new_landmark[..., 28:29, :]= tmp

    # -- ( 19, 34 )
    tmp = new_landmark[..., 19:20, :].copy()
    new_landmark[..., 19:20, :] = new_landmark[..., 34:35, :].copy()
    new_landmark[..., 34:35, :] = tmp

    # -- ( 11, 27 )
    tmp = new_landmark[..., 11:12, :].copy()
    new_landmark[..., 11:12, :] = new_landmark[..., 27:28, :].copy()
    new_landmark[..., 27:28, :] = tmp

    # -- ( 21, 35 )
    tmp = new_landmark[..., 21:22, :].copy()
    new_landmark[..., 21:22, :] = new_landmark[..., 35:36, :].copy()
    new_landmark[..., 35:36, :] = tmp

    # -- ( 14, 30 )
    tmp = new_landmark[..., 14:15, :].copy()
    new_landmark[..., 14:15, :] = new_landmark[..., 30:31, :].copy()
    new_landmark[..., 30:31, :] = tmp

    # -- ( 13, 29 )
    tmp = new_landmark[..., 13:14, :].copy()
    new_landmark[..., 13:14, :] = new_landmark[..., 29:30, :].copy()
    new_landmark[..., 29:30, :] = tmp

    # -- ( 10, 26 )
    tmp = new_landmark[..., 10:11, :].copy()
    new_landmark[..., 10:11, :] = new_landmark[..., 26:27, :].copy()
    new_landmark[..., 26:27, :] = tmp

    # -- ( 15, 31 )
    tmp = new_landmark[..., 15:16, :].copy()
    new_landmark[..., 15:16, :] = new_landmark[..., 31:32, :].copy()
    new_landmark[..., 31:32, :] = tmp

    # -- ( 18, 33 )
    tmp = new_landmark[..., 18:19, :].copy()
    new_landmark[..., 18:19, :] = new_landmark[..., 33:34, :].copy()
    new_landmark[..., 33:34, :] = tmp

    # -- (  7, 23 )
    tmp = new_landmark[..., 7:8, :].copy()
    new_landmark[..., 7:8, :] = new_landmark[..., 23:24, :].copy()
    new_landmark[..., 23:24, :] = tmp

    # -- ( 16, 32 )
    tmp = new_landmark[..., 16:17, :].copy()
    new_landmark[..., 16:17, :] = new_landmark[..., 32:33, :].copy()
    new_landmark[..., 32:33, :] = tmp

    # -- (  9, 25 )
    tmp = new_landmark[..., 9:10, :].copy()
    new_landmark[..., 9:10, :] = new_landmark[..., 25:26, :].copy()
    new_landmark[..., 25:26, :] = tmp

    # -- (  8, 24 )
    tmp = new_landmark[..., 8:9, :].copy()
    new_landmark[..., 8:9, :] = new_landmark[..., 24:25, :].copy()
    new_landmark[..., 24:25, :] = tmp




    # -- flip index pair pose
    # -- ( 37, 36 )
    tmp = new_landmark[..., 37:38, :].copy()
    new_landmark[..., 37:38, :] = new_landmark[..., 36:37, :].copy()
    new_landmark[..., 36:37, :] = tmp

    # -- ( 43, 42 )
    tmp = new_landmark[..., 43:44, :].copy()
    new_landmark[..., 43:44, :] = new_landmark[..., 42:43, :].copy()
    new_landmark[..., 42:43, :] = tmp

    # -- ( 39, 38 )
    tmp = new_landmark[..., 39:40, :].copy()
    new_landmark[..., 39:40, :] = new_landmark[..., 38:39, :].copy()
    new_landmark[..., 38:39, :] = tmp

    # -- ( 41, 40 )
    tmp = new_landmark[..., 41:42, :].copy()
    new_landmark[..., 41:42, :] = new_landmark[..., 40:41, :].copy()
    new_landmark[..., 40:41, :] = tmp


    return new_landmark
def loadnpfiles(parent_folder: str, landmarks: list) -> np.ndarray:
    out: list= []
    for a_lm in landmarks:
        with open(f"{LANDMARK_DIR /parent_folder /a_lm[KEY_FILE]}", 'rb') as f:
            out.append(np.load(f))
    return np.array(out)
def get_model_gloss(gloss: str) -> str:
    if gloss=='mine_my':
        return 'mine'
    if gloss=='me_i':
        return 'i'
    return gloss
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
    ds_landmark_by_gloss: dict= {
        el[KEY_G]: el # el is dict of keys( 'gloss': str, 'landmark': list[dict] )
        for el in ds_landmark
    }
    for a_gloss in VERIFIED_DS_DIR:
        ds_gloss: dict= ds_landmark_by_gloss[a_gloss[KEY_G]] # dict of keys( 'gloss': str, 'landmark': list[dict] )
        list_pfolders: list[str]= a_gloss[KEY_LANDMARK]
        assert len(list_pfolders)==len(VERIFIED_DS_DIR[0][KEY_LANDMARK])
        annotations.append({
            KEY_G: get_model_gloss(ds_gloss[KEY_G]),
            KEY_LANDMARK: list(
                {
                    KEY_PFOLDER: el[KEY_PFOLDER],
                    KEY_LHAND: a_gloss[KEY_LHAND],
                    KEY_RHAND: a_gloss[KEY_RHAND],
                    KEY_LANDMARK: el[KEY_LANDMARK],
                }
                for el in ds_gloss[KEY_LANDMARK]
                if el[KEY_PFOLDER] in list_pfolders
            )
        })
        manual_model.append([
            loadnpfiles(el[KEY_PFOLDER], el[KEY_LANDMARK])
            for el in annotations[-1][KEY_LANDMARK]
        ])
        annotations[-1][KEY_LANDMARK].extend([
        {
            KEY_PFOLDER: f"{many_imgs_el[KEY_PFOLDER]}_flip",
            KEY_LHAND: a_gloss[KEY_RHAND],
            KEY_RHAND: a_gloss[KEY_LHAND],
            KEY_LANDMARK: [{
                KEY_FILE: lm_img_el[KEY_FILE],
                KEY_FACE: lm_img_el[KEY_FACE],
                KEY_POSE: lm_img_el[KEY_POSE],
                KEY_LHAND: lm_img_el[KEY_RHAND],
                KEY_RHAND: lm_img_el[KEY_LHAND],
            } for lm_img_el in many_imgs_el[KEY_LANDMARK]],
        } for many_imgs_el in annotations[-1][KEY_LANDMARK]])
        manual_model[-1].extend(flip_vertical_x_axis(manual_model[-1]))
    for a_gloss in ds_landmark:
        a_gloss[KEY_G]= get_model_gloss(a_gloss[KEY_G])
    return (manual_model, annotations, ds_landmark)
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
        input[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
        input[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
    )/2.0
    hip_center: jnp.ndarray= (
        input[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
        input[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
    )/2.0
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
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    86 landmarks mandatory to follow this order
    [
        ...face_landmarks_36...,
        ...pose_landmarks_8...,
        ...left_hand_landmarks_21...,
        ...right_hand_landmarks_21...,
    ]
    ----------------------------------
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
    left_hand_diff= left_hand_diff *jnp.expand_dims(mask[..., 0], axis=-1)
    right_hand_diff= right_hand_diff *jnp.expand_dims(mask[..., 1], axis=-1)
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
    out: jnp.ndarray= mean_ignore_zero(
        jnp.concatenate([left_hand_diff, right_hand_diff], axis=-1),
        keepdims=False,
    )
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
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2,  2)
    weights, shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    --------------------------------------------------------------
    output shape is (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    input_lhand_index: jnp.ndarray= jnp.expand_dims(
        input[..., LM_IDX_LEFT_INDEXF_TIP, :],
        axis=-2,
    )  # shape (..., 1, 2)
    input_rhand_index: jnp.ndarray= jnp.expand_dims(
        input[..., LM_IDX_RIGHT_INDEXF_TIP, :],
        axis=-2,
    ) # shape (..., 1, 2)
    input_12landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy10landmarks(input),
        get2dots_frombody(input),
    ], axis=-2) # shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 12, 2)
    # ----------------------------------------------
    model_lhand_index: jnp.ndarray= jnp.expand_dims(
        model[..., LM_IDX_LEFT_INDEXF_TIP, :],
        axis=-2
    )  # shape (..., 1, 2)
    model_rhand_index: jnp.ndarray= jnp.expand_dims(
        model[..., LM_IDX_RIGHT_INDEXF_TIP, :],
        axis=-2
    ) # shape (..., 1, 2)
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
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    weights, shape of          (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    mask, shape of             (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    '''
    model_input_lhand= model_input_lhand *weights[..., 0, :] *jnp.expand_dims(
        mask[..., 0],
        axis=-1,
    )
    model_input_rhand= model_input_rhand *weights[..., 1, :] *jnp.expand_dims(
        mask[..., 1],
        axis=-1,
    )
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 12)
    '''
    model_input_lhand= mean_ignore_zero(
        input=jnp.stack([
            model_input_lhand,
            model_input_rhand,
        ], axis=-1),
        keepdims=False,
    ) # result shape (..., QUANTITY_FRAME, 12)
    del model_input_rhand
    model_input_lhand= mean_ignore_zero(
        input=model_input_lhand,
        keepdims=False,
    ) # result shape (..., QUANTITY_FRAME)
    model_input_lhand= weights_edges(model_input_lhand)
    model_input_lhand=mean_ignore_zero(
        input=model_input_lhand,
        keepdims=False,
    ) # result shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)

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
    mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
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
    assert model.shape[:-2]==mask.shape[:-1]
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
def create_weights(annotations: list) -> list:
    '''
    returns a list of shape (QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    on axis=-2 represents left and right hand
    on axis=-1 represents WORTHY_LM_IMPORTANT_IDX and 2 dots on body
    '''
    output: list= []
    for a_gloss in annotations:
        output.append([]) # --> represents an element from QUANTTIY_GLOSS
        for a_start_end in a_gloss[KEY_LANDMARK]:
            output[-1].append([]) # --> represents 1 ds element, SAMPLE_PER_GLOSS
            for an_img in a_start_end[KEY_LANDMARK]:
                output[-1][-1].append([
                    [.02 for _ in range(12)],
                    [.02 for _ in range(12)],
                    # 1 if an_img[KEY_RHAND] else 0,
                ]) # --> represents an element from QUANTITY_FRAME
                if an_img[KEY_LHAND]:
                    for idx_be_98_val in a_start_end[KEY_LHAND]:
                        output[-1][-1][-1][0][idx_be_98_val]= 0.98
                if an_img[KEY_RHAND]:
                    for idx_be_98_val in a_start_end[KEY_RHAND]:
                        output[-1][-1][-1][1][idx_be_98_val]= 0.98
    return output
def sort_score(score: jnp.ndarray) -> list:
    '''
    score has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    ----------
    return is a list of shape (BATCH_SIZE, QUANTITY_GLOSS)
    '''
    result: jnp.ndarray= jnp.min(score, axis=-1)
    return jnp.argsort(result, axis=-1, stable=True).tolist()
def test_model_on_self(
    model: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
    annotations: list,
) -> None:
    '''
    model, shape of - (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    -- on weights axis=-2 of value 2, for left and right hand repsectively
    -- on weights axis=-1 of value 12 for 12 landmarks
    '''
    result= comparison_score(
        input=np.array(jnp.reshape(model, (-1, *model.shape[-3:]))),
        model=model,
        mask=mask,
        body2dots=body2dots,
        weights=weights,
    ) # returns jnp.ndarray shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    index_result: list= sort_score(result) # returns list shape (BATCH_SIZE, QUANTITY_GLOSS)
    correct: int= 0
    count: int= 0
    for a_gloss_idx in range(model.shape[1]):
        for a_sample_per_gloss_idx in range(model.shape[2]):
            ds_element_result: list= index_result[a_gloss_idx*model.shape[2] +a_sample_per_gloss_idx]
            if annotations[ds_element_result[0]][KEY_G]==annotations[a_gloss_idx][KEY_G]:
                top3: list= jnp.argsort(
                    result[
                        a_gloss_idx*model.shape[2] +a_sample_per_gloss_idx,
                        index_result[a_gloss_idx*model.shape[2] +a_sample_per_gloss_idx][0],
                        :
                    ],
                    axis=-1
                )[:3].tolist()
                print(f"correct -- {ds_element_result[0]} -- {annotations[a_gloss_idx][KEY_G]} -- {result[
                    a_gloss_idx*model.shape[2] +a_sample_per_gloss_idx,
                    index_result[a_gloss_idx*model.shape[2] +a_sample_per_gloss_idx][0],
                    top3
                ]}")
                correct+= 1
            else:
                print(f"INcorrect -- {ds_element_result[0]} -- {annotations[a_gloss_idx][KEY_G]}")
            count+= 1
    print(f"correct: {correct/count *100} = {correct}/{count}")
def test_all_data(
    model: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
    model_annotations: list[str],
    annotations: list,
) -> None:
    '''
    model, shape of - (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
    -- on weights axis=-2 of value 2, for left and right hand repsectively
    -- on weights axis=-1 of value 12 for 12 landmarks
    '''
    batch_size: int= 32
    ds_elements: list= [] # be of shape (INT, QUANTITY_FRAME, 86, 2)
    ds_labels: list[int]= [] # loadnpfiles(..)
    ds_pfolders: list[str]= []
    result_details: list= []
    for a_gloss in annotations:
        result_details.append({
            KEY_G: a_gloss[KEY_G],
            'correct': 0,
            'count': 0,
            'incorrect_pfolder': [],
        })

    def test_batch(
        batch_elements: list,
        batch_labels: list[int],
        batch_pfolders: list[str],
    ) -> None:
        result= comparison_score(
            input=np.array(batch_elements),
            model=model,
            mask=mask,
            body2dots=body2dots,
            weights=weights,
        ) # returns jnp.ndarray shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
        index_result: list= sort_score(result) # returns list shape (BATCH_SIZE, QUANTITY_GLOSS)
        for idx_batch, ds_element_result in enumerate(index_result):
            expected_idx: int= batch_labels[idx_batch]
            predicted_idx: int= ds_element_result[0]
            result_details[expected_idx]['count']+= 1
            if model_annotations[predicted_idx]==annotations[expected_idx][KEY_G]:
                result_details[expected_idx]['correct']+= 1
            else:
                result_details[expected_idx]['incorrect_pfolder'].append(batch_pfolders[idx_batch])

    for idx_gloss, a_gloss in enumerate(annotations):
        for a_start_end in a_gloss[KEY_LANDMARK]:
            while batch_size<=len(ds_labels):
                test_batch(
                    batch_elements=ds_elements[:batch_size],
                    batch_labels=ds_labels[:batch_size],
                    batch_pfolders=ds_pfolders[:batch_size],
                )
                ds_elements= ds_elements[batch_size:]
                ds_labels=   ds_labels[batch_size:]
                ds_pfolders= ds_pfolders[batch_size:]
            ds_elements.append(loadnpfiles(
                parent_folder=a_start_end[KEY_PFOLDER],
                landmarks=a_start_end[KEY_LANDMARK],
            ))
            ds_labels.append(idx_gloss)
            ds_pfolders.append(a_start_end[KEY_PFOLDER])
    if len(ds_labels)!=0:
        test_batch(
            batch_elements=ds_elements,
            batch_labels=ds_labels,
            batch_pfolders=ds_pfolders,
        )
    del ds_elements
    del ds_labels
    del ds_pfolders

    total_correct: int= 0
    total_count: int= 0
    for a_result in result_details:
        total_correct+= a_result['correct']
        total_count+= a_result['count']
        incorrect: int= a_result['count'] -a_result['correct']
        print(
            f"\n{a_result[KEY_G]}\n" +
            f"    correct: {a_result['correct']}/{a_result['count']} -- {a_result['correct']/a_result['count']*100}%\n" +
            f"    incorrect: {incorrect}/{a_result['count']} -- {incorrect/a_result['count']*100}%"
        )
        if incorrect!=0:
            print(f"{a_result[KEY_G]} incorrect_pfolder")
            for an_incorrect_pfolder in a_result['incorrect_pfolder']:
                print(f"    {an_incorrect_pfolder}")
    total_incorrect: int= total_count -total_correct
    print(f"total correct: {total_correct}/{total_count} = {total_correct/total_count *100}%")
    print(f"total incorrect: {total_incorrect}/{total_count} = {total_incorrect/total_count *100}%")
    for a_result in result_details:
        incorrect: int= a_result['count'] -a_result['correct']
        print(
            f"\n{a_result[KEY_G]}\n" +
            f"    correct: {a_result['correct']}/{a_result['count']} -- {a_result['correct']/a_result['count']*100}%\n" +
            f"    incorrect: {incorrect}/{a_result['count']} -- {incorrect/a_result['count']*100}%"
        )
def show_shapes(
    ds_samples: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
    model_annotations: list[str],
) -> None:
    print(f"ds_samples shape {ds_samples.shape}")
    print(f"mask shape       {mask.shape}")
    print(f"body2dots shape  {body2dots.shape}")
    print(f"weights shape    {weights.shape}")
    print(f"model_annotations length {len(model_annotations)}")
def init_folders_files() -> None:
    assert (DS_DIR /'ds_landmark.json').exists()
    assert LANDMARK_DIR.exists()
    if MODEL2SAVE_DIR.exists():
        print(
            f"-------------- error --------------" +
            f"Delete: {MODEL2SAVE_DIR}\n" +
            f"Please delete this folder and its contents.\n" +
            f"This folder will be created for you."
        )
        raise FileNotFoundError('Please read above.')
    MODEL2SAVE_DIR.mkdir()
def save_model(
    ds_samples: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
    labels: list[str],
) -> None:
    with open(f"{MODEL2SAVE_DIR /"ds_samples.npy"}", 'wb') as f:
        np.save(f, np.array(ds_samples.tolist()))
    with open(f"{MODEL2SAVE_DIR /"mask.npy"}", 'wb') as f:
        np.save(f, np.array(mask.tolist()))
    with open(f"{MODEL2SAVE_DIR /"body2dots.npy"}", 'wb') as f:
        np.save(f, np.array(body2dots.tolist()))
    with open(f"{MODEL2SAVE_DIR /"weights.npy"}", 'wb') as f:
        np.save(f, np.array(weights.tolist()))
    with open(f"{MODEL2SAVE_DIR /"labels.json"}", 'w') as f:
        writeJson(labels, f, indent=4)
def main() -> None:
    init_folders_files()
    ds_samples, annotations, origin_annotations= load_ds_samples()
    ds_samples= jnp.expand_dims(jnp.array(ds_samples), axis=0)
    '''
    comparison_score(
        input, shape of -------------------------- (BATCH_SIZE, QUANTITY_FRAME, 86, 2)
        model, shape of - (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
        mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
        body2dots, shape  (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
        weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 12)
        -- on weights axis=-2 of value 2, for left and right hand repsectively
        -- on weights axis=-1 of value 12 for 12 landmarks
    ) output shape is (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    mask: jnp.ndarray= jnp.array(create_mask(annotations), dtype=jnp.uint8)
    ds_samples_body2dots: jnp.ndarray= get2dots_frombody(ds_samples)
    weights: jnp.ndarray= jnp.array(create_weights(annotations))
    mask= jnp.expand_dims(mask, axis=0)
    weights= jnp.expand_dims(weights, axis=0)
    labels: list[str]= [el[KEY_G] for el in annotations]
    # test_model_on_self(
    #     model=ds_samples,
    #     mask=mask,
    #     body2dots=ds_samples_body2dots,
    #     weights=weights,
    #     annotations=annotations,
    # )
    test_all_data(
        model=ds_samples,
        mask=mask,
        body2dots=ds_samples_body2dots,
        weights=weights,
        model_annotations=labels,
        annotations=origin_annotations,
    )
    show_shapes(
        ds_samples=ds_samples,
        mask=mask,
        body2dots=ds_samples_body2dots,
        weights=weights,
        model_annotations=labels,
    )
    save_model(
        ds_samples=ds_samples,
        mask=mask,
        body2dots=ds_samples_body2dots,
        weights=weights,
        labels=labels,
    )
    # with open("/tmp/asdf.json", "w") as f:
    #     writeJson(annotations, f, indent=4)
    # print("\n".join([el[KEY_G] for el in annotations]))
if __name__=="__main__":
    main()
