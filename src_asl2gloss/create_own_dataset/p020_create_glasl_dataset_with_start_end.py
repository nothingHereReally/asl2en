from json import dump as writeJson, load as loadJson
from pathlib import Path
import cv2
import numpy as np


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"glasl_verified"
VIDEO_DIR: Path= DS_DIR /"videos"
IN_LANDMARK_DIR: Path= DS_DIR /"investigate_landmark"
IN_SKELETON_DIR: Path= DS_DIR /"investigate_skeleton"
OUT_LANDMARK_DIR: Path= DS_DIR /"ds_landmark"
OUT_SKELETON_DIR: Path= DS_DIR /"ds_skeleton"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
# ---------------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
RATIO4TRAIN: float= 0.8
# ---------------------------
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_IMG_VALID: str= 'valid_images'
KEY_IMGSTART: str= 'start' # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_IMGEND: str= 'end'     # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_PFOLDER_FROM: str= 'parent_folder_from'
KEY_PFOLDER: str= 'parent_folder'
IMG_SIZE: int= 158
QUANTITY_FRAME: int= 8
MOD_PART_3: tuple= (2, 3, 4, 5, 6)
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
QUANTITY_HAND_LMARK: int= 21
VIDEO_LIST: list= [
    {
        KEY_G: 'book',
        KEY_VIDS: [
            {
                KEY_FILE: 'book_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 45,
                    },
                    {
                        KEY_IMGSTART: 46,
                        KEY_IMGEND: 76,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 27,
                    },
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 54,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 24,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 4,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 50,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 51,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 26,
                    },
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 55,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 3,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 56,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 41,
                        KEY_IMGEND: 56,
                    },
                ]
            },
            {
                KEY_FILE: 'book_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 40,
                        KEY_IMGEND: 61,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'drink',
        KEY_VIDS: [
            {
                KEY_FILE: 'drink_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 38,
                    },
                    {
                        KEY_IMGSTART: 39,
                        KEY_IMGEND: 81,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 32,
                    },
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 68,
                    },
                    {
                        KEY_IMGSTART: 71,
                        KEY_IMGEND: 104,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 55,
                    },
                    {
                        KEY_IMGSTART: 56,
                        KEY_IMGEND: 119,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 55,
                    },
                    {
                        KEY_IMGSTART: 56,
                        KEY_IMGEND: 102,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 96,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 99,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 60,
                    },
                    {
                        KEY_IMGSTART: 61,
                        KEY_IMGEND: 120,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 31,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 82,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 32,
                    },
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 99,
                    },
                ]
            },
            {
                KEY_FILE: 'drink_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 26,
                    },
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 75,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'computer',
        KEY_VIDS: [
            {
                KEY_FILE: 'computer_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 54,
                    },
                    {
                        KEY_IMGSTART: 55,
                        KEY_IMGEND: 72,
                    },
                    {
                        KEY_IMGSTART: 73,
                        KEY_IMGEND: 93,
                    },
                    {
                        KEY_IMGSTART: 94,
                        KEY_IMGEND: 119,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 46,
                    },
                    {
                        KEY_IMGSTART: 47,
                        KEY_IMGEND: 68,
                    },
                    {
                        KEY_IMGSTART: 69,
                        KEY_IMGEND: 92,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 26,
                    },
                    {
                        KEY_IMGSTART: 45,
                        KEY_IMGEND: 66,
                    },
                    {
                        KEY_IMGSTART: 67,
                        KEY_IMGEND: 87,
                    },
                    {
                        KEY_IMGSTART: 88,
                        KEY_IMGEND: 108,
                    },
                    {
                        KEY_IMGSTART: 109,
                        KEY_IMGEND: 123,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 28,
                    },
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 48,
                    },
                    {
                        KEY_IMGSTART: 49,
                        KEY_IMGEND: 67,
                    },
                    {
                        KEY_IMGSTART: 68,
                        KEY_IMGEND: 91,
                    },
                    {
                        KEY_IMGSTART: 92,
                        KEY_IMGEND: 109,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 43,
                    },
                    {
                        KEY_IMGSTART: 44,
                        KEY_IMGEND: 64,
                    },
                    {
                        KEY_IMGSTART: 65,
                        KEY_IMGEND: 86,
                    },
                    {
                        KEY_IMGSTART: 87,
                        KEY_IMGEND: 110,
                    },
                    {
                        KEY_IMGSTART: 111,
                        KEY_IMGEND: 121,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 26,
                    },
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 48,
                    },
                    {
                        KEY_IMGSTART: 49,
                        KEY_IMGEND: 70,
                    },
                    {
                        KEY_IMGSTART: 71,
                        KEY_IMGEND: 89,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 22,
                    },
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 45,
                    },
                    {
                        KEY_IMGSTART: 46,
                        KEY_IMGEND: 68,
                    },
                    {
                        KEY_IMGSTART: 69,
                        KEY_IMGEND: 91,
                    },
                    {
                        KEY_IMGSTART: 92,
                        KEY_IMGEND: 112,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 29,
                    },
                    {
                        KEY_IMGSTART: 30,
                        KEY_IMGEND: 53,
                    },
                    {
                        KEY_IMGSTART: 54,
                        KEY_IMGEND: 73,
                    },
                    {
                        KEY_IMGSTART: 74,
                        KEY_IMGEND: 97,
                    },
                    {
                        KEY_IMGSTART: 98,
                        KEY_IMGEND: 109,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 59,
                    },
                    {
                        KEY_IMGSTART: 60,
                        KEY_IMGEND: 84,
                    },
                    {
                        KEY_IMGSTART: 85,
                        KEY_IMGEND: 109,
                    },
                ]
            },
            {
                KEY_FILE: 'computer_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 27,
                    },
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 54,
                    },
                    {
                        KEY_IMGSTART: 55,
                        KEY_IMGEND: 76,
                    },
                    {
                        KEY_IMGSTART: 77,
                        KEY_IMGEND: 101,
                    },
                    {
                        KEY_IMGSTART: 102,
                        KEY_IMGEND: 124,
                    },
                    {
                        KEY_IMGSTART: 125,
                        KEY_IMGEND: 135,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'before',
        KEY_VIDS: [
            {
                KEY_FILE: 'before_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 52,
                    },
                    {
                        KEY_IMGSTART: 53,
                        KEY_IMGEND: 78,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 69,
                    },
                    {
                        KEY_IMGSTART: 70,
                        KEY_IMGEND: 92,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 32,
                    },
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 56,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    },
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 58,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 42,
                    },
                    {
                        KEY_IMGSTART: 43,
                        KEY_IMGEND: 65,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 41,
                    },
                    {
                        KEY_IMGSTART: 42,
                        KEY_IMGEND: 69,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 53,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 41,
                    },
                    {
                        KEY_IMGSTART: 42,
                        KEY_IMGEND: 55,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 81,
                    },
                    {
                        KEY_IMGSTART: 82,
                        KEY_IMGEND: 98,
                    },
                ]
            },
            {
                KEY_FILE: 'before_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 42,
                    },
                    {
                        KEY_IMGSTART: 43,
                        KEY_IMGEND: 69,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'chair',
        KEY_VIDS: [
            {
                KEY_FILE: 'chair_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 17,
                    },
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 50,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 41,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 51,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 34,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 41,
                    },
                ]
            },
            {
                KEY_FILE: 'chair_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 50,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'go',
        KEY_VIDS: [
            {
                KEY_FILE: 'go_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 17,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 32,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 52,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 19,
                    },
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 57,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 44,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 56,
                    },
                ]
            },
            {
                KEY_FILE: 'go_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 43,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_VIDS: [
            {
                KEY_FILE: 'clothes_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 48,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 47,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 7,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 30,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 39,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 61,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 10,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 37,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'clothes_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 10,
                    },
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 53,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'who',
        KEY_VIDS: [
            {
                KEY_FILE: 'who_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 61,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 22,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 59,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 48,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 4,
                        KEY_IMGEND: 45,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 61,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 62,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 64,
                    },
                ]
            },
            {
                KEY_FILE: 'who_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 55,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'candy',
        KEY_VIDS: [
            {
                KEY_FILE: 'candy_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 10,
                    },
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 36,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 49,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 42,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 37,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 10,
                    },
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 34,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 8,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 53,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                ]
            },
            {
                KEY_FILE: 'candy_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 31,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'cousin',
        KEY_VIDS: [
            {
                KEY_FILE: 'cousin_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 50,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 46,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 17,
                    },
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 36,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 28,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 31,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 37,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 31,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 34,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'cousin_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 36,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_VIDS: [
            {
                KEY_FILE: 'mine_my_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 46,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 34,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 33,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 31,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 28,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 28,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 28,
                    },
                ]
            },
            {
                KEY_FILE: 'mine_my_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 28,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_VIDS: [
            {
                KEY_FILE: 'me_i_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 27,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 84,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 68,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 87,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 46,
                        KEY_IMGEND: 72,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 87,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 69,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 17,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 78,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 33,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                ]
            },
            {
                KEY_FILE: 'me_i_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 80,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_VIDS: [
            {
                KEY_FILE: 'stomach_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 57,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 44,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 45,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 41,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 19,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'stomach_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 36,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'have',
        KEY_VIDS: [
            {
                KEY_FILE: 'have_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 64,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 64,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 31,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 42,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 22,
                    },
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 45,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 47,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 19,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 37,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 37,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 37,
                    },
                ]
            },
            {
                KEY_FILE: 'have_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 36,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'need',
        KEY_VIDS: [
            {
                KEY_FILE: 'need_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 19,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 75,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 10,
                    },
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 50,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 24,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 38,
                    },
                    {
                        KEY_IMGSTART: 39,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 34,
                    },
                    {
                        KEY_IMGSTART: 35,
                        KEY_IMGEND: 63,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 22,
                    },
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 43,
                    },
                    {
                        KEY_IMGSTART: 44,
                        KEY_IMGEND: 62,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 38,
                    },
                    {
                        KEY_IMGSTART: 39,
                        KEY_IMGEND: 55,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 41,
                    },
                    {
                        KEY_IMGSTART: 42,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 42,
                    },
                    {
                        KEY_IMGSTART: 43,
                        KEY_IMGEND: 58,
                    },
                ]
            },
            {
                KEY_FILE: 'need_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 17,
                    },
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 68,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'see',
        KEY_VIDS: [
            {
                KEY_FILE: 'see_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 42,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 46,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 56,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 24,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 55,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 25,
                    },
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 60,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 23,
                    },
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 53,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 27,
                    },
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 59,
                    },
                ]
            },
            {
                KEY_FILE: 'see_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 24,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 57,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'feel',
        KEY_VIDS: [
            {
                KEY_FILE: 'feel_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 25,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 45,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 48,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 52,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 54,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 54,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 52,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 50,
                    },
                ]
            },
            {
                KEY_FILE: 'feel_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 43,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_VIDS: [
            {
                KEY_FILE: 'hurt_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 45,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 34,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 32,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 34,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 32,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 42,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 39,
                    },
                ]
            },
            {
                KEY_FILE: 'hurt_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 36,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'fever',
        KEY_VIDS: [
            {
                KEY_FILE: 'fever_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 63,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 31,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 49,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 38,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 58,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 41,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 57,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 65,
                    },
                ]
            },
            {
                KEY_FILE: 'fever_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 62,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_VIDS: [
            {
                KEY_FILE: 'dizzy_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 49,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 36,
                    },
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 51,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 36,
                    },
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 52,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 38,
                    },
                    {
                        KEY_IMGSTART: 39,
                        KEY_IMGEND: 55,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 21,
                    },
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 54,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 19,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 52,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    },
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 53,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 17,
                    },
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 34,
                    },
                    {
                        KEY_IMGSTART: 35,
                        KEY_IMGEND: 54,
                    },
                    {
                        KEY_IMGSTART: 55,
                        KEY_IMGEND: 71,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 19,
                    },
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 37,
                    },
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 46,
                    },
                ]
            },
            {
                KEY_FILE: 'dizzy_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    },
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 33,
                    },
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 51,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'headache',
        KEY_VIDS: [
            {
                KEY_FILE: 'headache_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 38,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 35,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 31,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 29,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 15,
                    },
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 39,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 13,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 38,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 14,
                    },
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 43,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 46,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 40,
                    },
                ]
            },
            {
                KEY_FILE: 'headache_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 12,
                    },
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 38,
                    },
                ]
            },
        ]
    },
    {
        KEY_G: 'doctor',
        KEY_VIDS: [
            {
                KEY_FILE: 'doctor_0001.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 11,
                    },
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 41,
                    },
                    {
                        KEY_IMGSTART: 42,
                        KEY_IMGEND: 67,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0002.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 16,
                    },
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 49,
                    },
                    {
                        KEY_IMGSTART: 50,
                        KEY_IMGEND: 83,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0003.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 10,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 36,
                    },
                    {
                        KEY_IMGSTART: 41,
                        KEY_IMGEND: 63,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0004.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 9,
                    },
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 41,
                    },
                    {
                        KEY_IMGSTART: 44,
                        KEY_IMGEND: 61,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0005.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 26,
                    },
                    {
                        KEY_IMGSTART: 30,
                        KEY_IMGEND: 48,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0006.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 36,
                    },
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 60,
                    },
                    {
                        KEY_IMGSTART: 61,
                        KEY_IMGEND: 81,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0007.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 31,
                    },
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 80,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0008.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    },
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 71,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0009.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    },
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 70,
                    },
                ]
            },
            {
                KEY_FILE: 'doctor_0010.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 32,
                    },
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 64,
                    },
                ]
            },
        ]
    }
]
def get_face(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., :36, :].copy()
def get_pose(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., 36:36+8, :].copy()
def get_left_hand(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., 36+8:36+8+21, :].copy()
def get_right_hand(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., -21:, :].copy()
def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: np.ndarray, \
                    lmark_coordinates: list, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> np.ndarray:
    img_wh: dict= {"wx": image.shape[1], "hy": image.shape[0]}


    # drawing the lines between 2 landmark connections
    if color_line!=None or color_dot!=None:
        for lmark_idx_pair in connections_idxs:
            pA: tuple= (
                lmark_coordinates[  lmark_idx_pair[0]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[0]  ][1]  # y
            )
            pB: tuple= (
                lmark_coordinates[  lmark_idx_pair[1]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[1]  ][1]  # y
            )
            if isOkPlot(pA) and isOkPlot(pB):
                if color_dot!=None:
                    cv2.circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    cv2.circle(
                        img=image,
                        center=(
                            int(pB[0]*img_wh['wx']),
                            int(pB[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                if color_line!=None:
                    cv2.line(
                        img=image,
                        pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                        pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                        color=color_line,
                        thickness=thick
                    )
            else:
                raise NotImplementedError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0, on both x and y coordinates")
            del pA
            del pB
    return image
def drawFacePoseHandFromLM(img_write_to: np.ndarray, landmark: list[tuple[float, float]], hasLandmark: dict) -> np.ndarray:
    # ---- face landmarks ----
    if hasLandmark[KEY_FACE]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_face(np.array(landmark)).tolist(),
            connections_idxs=FACE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 153, 0), # 153/255= 0.6
        )

    # ---- pose landmarks ----
    if hasLandmark[KEY_POSE]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_pose(np.array(landmark)).tolist(),
            connections_idxs=POSE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 0, 153), # 153/255= 0.6
        )

    # ---- left hand landmarks ----
    if hasLandmark[KEY_LHAND]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_left_hand(np.array(landmark)).tolist(),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(255, 255, 255)
        )

    # ---- right hand landmarks ----
    if hasLandmark[KEY_RHAND]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_right_hand(np.array(landmark)).tolist(),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(153, 204, 204), # 204/255= 0.8
        )

    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to




def init_dirs() -> None:
    err: list[str]= []
    if not IN_LANDMARK_DIR.exists():
        err.append(f"Missing {IN_LANDMARK_DIR} folder.")
    if not IN_SKELETON_DIR.exists():
        err.append(f"Missing {IN_SKELETON_DIR} folder.")

    if not OUT_LANDMARK_DIR.exists():
        err.append(f"Please delete {OUT_LANDMARK_DIR}, this will be created for you.")
    if not OUT_SKELETON_DIR.exists():
        err.append(f"Please delete {OUT_SKELETON_DIR}, this will be created for you.")
    OUT_LANDMARK_DIR.mkdir()
    OUT_SKELETON_DIR.mkdir()
def get_past_annotations() -> list:
    with open(DS_DIR /"investigate_landmark.json", 'r') as f:
        ds_landmark= loadJson(f)
    return ds_landmark
def has_atleast1hand(annotations: dict) -> bool:
    return annotations[KEY_FACE] and \
           annotations[KEY_POSE] and \
           (annotations[KEY_LHAND] or \
           annotations[KEY_RHAND])
# video_landmarks, video_annotations= (
#     IN_LANDMARK_DIR,
#     a_video[KEY_FILE][:-4],
#     past_landmark[idx_gloss][KEY_VIDS][idx_video][KEY_LANDMARK],
# )
def get_past_landmarks(
    abs_root_path: Path,
    parent_folder: str,
    landmarks: list[dict],
) -> list:
    '''
    returns (landmarks, annotations)
    landmarks is a list[numpy.ndarray]
    annotations is a list[dict]
    '''
    out_landmarks: list= []
    for idx, an_img_landmark in enumerate(landmarks):
        if has_atleast1hand(an_img_landmark) or idx == 0:
            with open(abs_root_path /parent_folder /an_img_landmark[KEY_FILE], 'rb') as f:
                out_landmarks.append( np.load(f) )
        else:
            out_landmarks.append(out_landmarks[-1].copy())

    return out_landmarks
def process_videos_ds() -> tuple:
    ds_landmark: list= []
    ds_skeleton: list= []
    '''
    TODO: later blah blah
    del left hand
        drink
        who
        candy
        cousin
        feel
        mine_my
        me_i
        stomach
        need
        see
        feel
        fever
    '''
    past_landmark: list= get_past_annotations()
    for idx_gloss, a_gloss in enumerate(VIDEO_LIST):
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
            video_annotations: dict= past_landmark[idx_gloss][KEY_VIDS][idx_video]
            video_landmarks: list= get_past_landmarks(
                IN_LANDMARK_DIR,
                a_video[KEY_FILE][:-4],
                video_annotations[KEY_LANDMARK],
            )
            assert a_video[KEY_FILE] == video_annotations[KEY_FILE]
            assert a_video[KEY_IMG_VALID][-1][KEY_IMGEND] <= len(video_annotations[KEY_LANDMARK])
            for a_start_end in a_video[KEY_IMG_VALID]:
                pfolder_name: str= f"{a_video[KEY_FILE][:-4]}_{str(a_start_end[KEY_IMGSTART]).zfill(4)}_{str(a_start_end[KEY_IMGEND]).zfill(4)}"
                landmark_abs_folder: Path= OUT_LANDMARK_DIR /pfolder_name
                skeleton_abs_folder: Path= OUT_SKELETON_DIR /pfolder_name
                landmark_abs_folder.mkdir()
                skeleton_abs_folder.mkdir()
                ds_landmark[-1][KEY_VIDS].append({
                    KEY_PFOLDER: pfolder_name,
                    KEY_SPLIT: video_annotations[KEY_SPLIT],
                    KEY_LANDMARK: [],
                })
                ds_skeleton[-1][KEY_VIDS].append({
                    KEY_PFOLDER: pfolder_name,
                    KEY_SPLIT: video_annotations[KEY_SPLIT],
                    KEY_SKELETON: [],
                })
                for idx_init0 in range(a_start_end[KEY_IMGEND] -a_start_end[KEY_IMGSTART]+1):
                    landmark_skeleton_prefix: str= str(idx_init0+1).zfill(8)
                    landmark= video_landmarks[     a_start_end[KEY_IMGSTART]-1   +idx_init0]
                    annotation= video_annotations[KEY_LANDMARK][ a_start_end[KEY_IMGSTART]-1   +idx_init0]
                    with open(f"{landmark_abs_folder /landmark_skeleton_prefix}.npy", 'wb') as f:
                        np.save(f, landmark)
                    cv2.imwrite(
                        filename=f"{skeleton_abs_folder /landmark_skeleton_prefix}.jpg",
                        img=drawFacePoseHandFromLM(
                            img_write_to=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
                            landmark=landmark,
                            hasLandmark=annotation,
                        ),
                    )
                    ds_landmark[-1][KEY_VIDS][-1][KEY_LANDMARK].append({
                        KEY_FILE: f"{landmark_skeleton_prefix}.npy",
                        KEY_FACE: annotation[KEY_FACE],
                        KEY_POSE: annotation[KEY_POSE],
                        KEY_LHAND: annotation[KEY_LHAND],
                        KEY_RHAND: annotation[KEY_RHAND],
                    })
                    ds_skeleton[-1][KEY_VIDS][-1][KEY_SKELETON].append({
                        KEY_FILE: f"{landmark_skeleton_prefix}.jpg",
                        KEY_FACE: annotation[KEY_FACE],
                        KEY_POSE: annotation[KEY_POSE],
                        KEY_LHAND: annotation[KEY_LHAND],
                        KEY_RHAND: annotation[KEY_RHAND],
                    })
    return (ds_landmark, ds_skeleton)
def main() -> None:
    init_dirs()
    ds_landmark, ds_skeleton= process_videos_ds()
    with open(f"{DS_DIR /"ds_landmark.json"}", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{DS_DIR /"ds_skeleton.json"}", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    main()
