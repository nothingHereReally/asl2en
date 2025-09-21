from math import ceil
from json import load as loadjson


# PROJ_ROOT: str= f"/absolute/dir/to/project/"
PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-2])}/"

EPOCHS: int= 12
# TRAIN_BATCH: int= 32
TRAIN_BATCH: int= 2
# QUANTITY_FRAME: int= 48
QUANTITY_FRAME: int= 22
# IMG_SIZE: int= 480
IMG_SIZE: int= 158 # on G10 mandatory be 158x158x3
# MIN_FRAMES_HAS_HANDS, meaning on a single video file
# where only QUANTITY_FRAME img will be included, then
# mandatory that atleast MIN_FRAMES_HAS_HANDS out of
# QUANTITY_FRAME has at least 1 hand( ie. either left
# or right hand )
# ie. current is at least 12 or 20 has hand/s 20*0.6= 12
MIN_FRAMES_HAS_HANDS: int= int(QUANTITY_FRAME*0.7)
WLASL_SKELETON_DIR: str= f"{PROJ_ROOT}dataset/wlasl/skeleton_image/"
WLASL_LANDMARK_DIR: str= f"{PROJ_ROOT}dataset/wlasl/landmark_numpy/"




tmp_ready: dict= {}
with open(f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.skeleton_image.train_val_test.json", 'r') as f:
    tmp_ready= loadjson(f)
wlasl_skeleton: dict= tmp_ready.copy()
del tmp_ready
SKELETON_IMG_SHAPE: tuple= (158, 158, 3)

tmp_ready: dict= {}
with open(f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.landmark_numpy.train_val_test.json", 'r') as f:
    tmp_ready= loadjson(f)
wlasl_landmark: dict= tmp_ready.copy()
del tmp_ready
LANDMARK_SHAPE: tuple= (21*2+8, 2)
# landmark is face, then pose, then left_had, then right hand
# face is (468, 2)
# pose is (8, 2)
# left_hand is (21, 2)
# right_hand is (21, 2)

K_TRAIN: str= 'train'
K_VAL: str= 'val'
K_TEST: str= 'test'
K_ID2G: str= 'label_id2gloss'
K_G2ID: str= 'label_gloss2id'

T_TRAIN: int= int(len(wlasl_skeleton[K_TRAIN]))
T_VAL: int= int(len(wlasl_skeleton[K_VAL]))
T_TEST: int= int(len(wlasl_skeleton[K_TEST]))
T_GLOSS: int= int(len(wlasl_skeleton[K_ID2G]))
TRAIN_GLOSS: int= 10
if T_GLOSS<TRAIN_GLOSS:
    # TRAIN_GLOSS to be use as to how many gloss be
    # on training, starting from very 1st gloss till
    # ${TRAIN_GLOSS}th gloss classifiction on
    # wlasl_skeleton[K_ID2G] or wlasl_skeleton[K_ID2G]
    # due to both represent same dataset really
    raise ValueError({
        'TRAIN_GLOSS': 'should be less than or equal to T_GLOSS',
        'value': {
            'T_GLOSS': T_GLOSS,
            'TRAIN_GLOSS': TRAIN_GLOSS
        }
    })
# dataset that has only data according to TRAIN_GLOSS
wlasl_skeleton_TG: dict= {}
wlasl_landmark_TG: dict= {}
T_TRAIN_TG: int= 0
T_VAL_TG: int= 0
T_TEST_TG: int= 0
if T_GLOSS!=TRAIN_GLOSS:
    wlasl_skeleton_TG: dict= {
        K_TRAIN: [],
        K_VAL: [],
        K_TEST: [],
        K_ID2G: wlasl_skeleton[K_ID2G],
        K_G2ID: wlasl_skeleton[K_G2ID]
    }
    wlasl_landmark_TG: dict= {
        K_TRAIN: [],
        K_VAL: [],
        K_TEST: [],
        K_ID2G: wlasl_landmark[K_ID2G],
        K_G2ID: wlasl_landmark[K_G2ID]
    }
    for tvt_idv in (K_TRAIN, K_VAL, K_TEST):
        tillWhat: int= 0
        while wlasl_landmark[tvt_idv][tillWhat]['gloss_id']<TRAIN_GLOSS:
            tillWhat+= 1
        wlasl_skeleton_TG[tvt_idv]= wlasl_skeleton[tvt_idv][0:tillWhat]
        wlasl_landmark_TG[tvt_idv]= wlasl_landmark[tvt_idv][0:tillWhat]
    T_TRAIN_TG= len(wlasl_skeleton_TG[K_TRAIN])
    T_VAL_TG= len(wlasl_skeleton_TG[K_VAL])
    T_TEST_TG= len(wlasl_skeleton_TG[K_TEST])

G10_T_TRAIN: int= 11598 # for getGreaterThan_np_initHasHand
G10_T_VAL: int= 2727 # for getGreaterThan_np_initHasHand

TRAIN_STEPS: int= int(ceil((T_TRAIN*70)/TRAIN_BATCH)) if T_TRAIN_TG==0 else int(ceil(G10_T_TRAIN/TRAIN_BATCH))
VAL_STEPS: int= int(ceil((T_VAL*70)/TRAIN_BATCH)) if T_VAL_TG==0 else int(ceil(G10_T_VAL/TRAIN_BATCH))


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
