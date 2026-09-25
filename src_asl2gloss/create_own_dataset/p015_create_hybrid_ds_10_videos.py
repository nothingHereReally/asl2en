from cv2 import circle, imwrite, line
from json import dump as writeJson, load as loadJson
from numpy import array, ndarray, uint8, zeros, float32, load as numpyload, save as numpysave
from pathlib import Path


# ------------------------
# ------------------------
# ---- contants start ----
PROJ_ROOT= Path(__file__).resolve().parent.parent.parent
HYBRID_DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
LM_RECENT_DIR: Path= HYBRID_DS_DIR /"fix_most_recent_landmark"
LM_NVSTGT_DIR: Path= HYBRID_DS_DIR /"investigate_landmark"
SKLTN_RECENT_DIR: Path= HYBRID_DS_DIR /"fix_most_recent_skeleton"
SKLTN_NVSTGT_DIR: Path= HYBRID_DS_DIR /"investigate_skeleton"
LANDMARK_dir: Path= HYBRID_DS_DIR /"ds_landmark"
SKELETON_dir: Path= HYBRID_DS_DIR /"ds_skeleton"
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
IMG_SIZE: int= 158
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
# -------------------
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
# -------------------
KEY_DSFROM: str= 'get_from'
VAL_RECENT: str= 'fill_via_recent'
VAL_CLASSIC: str= 'classic'
DEL_LEFT_HAND: str= 'remove_left_hand'
# ---------------------
KEY_IMG_VALID: str= 'valid_images'
KEY_IMGSTART: str= 'start' # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_IMGEND: str= 'end'     # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
HYBRID_DATA_SRC: list= [
    {
        KEY_G: 'book',
        KEY_VIDS: [
            {
                KEY_VFILE: "07071.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART:  1,
                        KEY_IMGEND:   18,
                    }
                ]
            },
            {
                KEY_VFILE: "68208.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND:   89,
                    }
                ]
            },
            {
                KEY_VFILE: "07072.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART:  1,
                        KEY_IMGEND:   18,
                    }
                ]
            },
            {
                KEY_VFILE: "68011.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART:  6,
                        KEY_IMGEND:   40,
                    }
                ]
            },
            {
                KEY_VFILE: "65225.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND:   43,
                    }
                ]
            },
            {
                KEY_VFILE: "68012.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND:   64,
                    }
                ]
            },
            {
                KEY_VFILE: "70212.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND:   65,
                    }
                ]
            },
            {
                KEY_VFILE: "07097.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND:   43,
                    }
                ]
            },
            {
                KEY_VFILE: "07099.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND:   52,
                    }
                ]
            },
            {
                KEY_VFILE: "07076.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND:   67,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'drink',
        KEY_VIDS: [
            {
                KEY_VFILE: "17724.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART:  1,
                        KEY_IMGEND:   30,
                    }
                ]
            },
            {
                KEY_VFILE: "68538.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 55,
                    },
                    {
                        KEY_IMGSTART: 84,
                        KEY_IMGEND: 112,
                    }
                ]
            },
            {
                KEY_VFILE: "69302.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 57,
                    }
                ]
            },
            {
                KEY_VFILE: "68042.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 51,
                    }
                ]
            },
            {
                KEY_VFILE: "17729.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 48,
                    }
                ]
            },
            {
                KEY_VFILE: "68660.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 5,
                        KEY_IMGEND: 30,
                    }
                ]
            },
            {
                KEY_VFILE: "68041.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 47,
                    }
                ]
            },
            {
                KEY_VFILE: "17725.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 45,
                    }
                ]
            },
            {
                KEY_VFILE: "17726.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 62,
                    }
                ]
            },
            {
                KEY_VFILE: "17727.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 53,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'computer',
        KEY_VIDS: [
            {
                KEY_VFILE: "12306.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 52,
                    }
                ]
            },
            {
                KEY_VFILE: "68028.mp4",
                KEY_DSFROM: VAL_CLASSIC,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 63,
                    }
                ]
            },
            {
                KEY_VFILE: "69054.mp4",
                KEY_DSFROM: VAL_CLASSIC,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 117,
                    }
                ]
            },
            {
                KEY_VFILE: "12331.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 51,
                    }
                ]
            },
            {
                KEY_VFILE: "12336.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "12314.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 39,
                    }
                ]
            },
            {
                KEY_VFILE: "12312.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 47,
                        KEY_IMGEND: 81,
                    }
                ]
            },
            {
                KEY_VFILE: "12326.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 73,
                    }
                ]
            },
            {
                KEY_VFILE: "computer_0004.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 86,
                    }
                ]
            },
            {
                KEY_VFILE: "computer_0005.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 82,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'before',
        KEY_VIDS: [
            {
                KEY_VFILE: "05730.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 8,
                        KEY_IMGEND: 25,
                    }
                ]
            },
            {
                KEY_VFILE: "05733.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 83,
                    }
                ]
            },
            {
                KEY_VFILE: "05744.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 47,
                    }
                ]
            },
            {
                KEY_VFILE: "05729.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 36,
                    }
                ]
            },
            {
                KEY_VFILE: "65167.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 32,
                    }
                ]
            },
            {
                KEY_VFILE: "05727.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "05739.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 52,
                    }
                ]
            },
            {
                KEY_VFILE: "05747.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 46,
                    }
                ]
            },
            {
                KEY_VFILE: "05748.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "05750.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 58,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'chair',
        KEY_VIDS: [
            {
                KEY_VFILE: "09853.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 20,
                    }
                ]
            },
            {
                KEY_VFILE: "09848.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 60,
                    }
                ]
            },
            {
                KEY_VFILE: "09869.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 40,
                        KEY_IMGEND: 65,
                    }
                ]
            },
            {
                KEY_VFILE: "09862.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 78,
                    }
                ]
            },
            {
                KEY_VFILE: "09847.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 52,
                    }
                ]
            },
            {
                KEY_VFILE: "68580.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 40,
                    }
                ]
            },
            {
                KEY_VFILE: "70263.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 81,
                    }
                ]
            },
            {
                KEY_VFILE: "68019.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 47,
                    }
                ]
            },
            {
                KEY_VFILE: "09865.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 7,
                        KEY_IMGEND: 48,
                    }
                ]
            },
            {
                KEY_VFILE: "09866.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 64,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'go',
        KEY_VIDS: [
            {
                KEY_VFILE: "24948.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 18,
                    }
                ]
            },
            {
                KEY_VFILE: "68292.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 38,
                    },
                    {
                        KEY_IMGSTART: 73,
                        KEY_IMGEND: 98,
                    }
                ]
            },
            {
                KEY_VFILE: "24965.mp4",
                KEY_DSFROM: VAL_CLASSIC,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 37,
                    }
                ]
            },
            {
                KEY_VFILE: "24954.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 61,
                    }
                ]
            },
            {
                KEY_VFILE: "24857.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    }
                ]
            },
            {
                KEY_VFILE: "69345.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 39,
                    }
                ]
            },
            {
                KEY_VFILE: "24941.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 32,
                    }
                ]
            },
            {
                KEY_VFILE: "24969.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 60,
                    }
                ]
            },
            {
                KEY_VFILE: "24970.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 60,
                    }
                ]
            },
            {
                KEY_VFILE: "24971.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 39,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_VIDS: [
            {
                KEY_VFILE: "11326.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 23,
                    }
                ]
            },
            {
                KEY_VFILE: "11311.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "11314.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "11316.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "11305.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 52,
                    }
                ]
            },
            {
                KEY_VFILE: "68870.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 35,
                    }
                ]
            },
            {
                KEY_VFILE: "68024.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "11327.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "11328.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 29,
                    }
                ]
            },
            {
                KEY_VFILE: "11329.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 28,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'who',
        KEY_VIDS: [
            {
                KEY_VFILE: "63240.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 44,
                    }
                ]
            },
            {
                KEY_VFILE: "63242.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 32,
                    }
                ]
            },
            {
                KEY_VFILE: "63228.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 29,
                    }
                ]
            },
            {
                KEY_VFILE: "63229.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 32,
                    }
                ]
            },
            {
                KEY_VFILE: "63234.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 43,
                    }
                ]
            },
            {
                KEY_VFILE: "63237.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 46,
                    }
                ]
            },
            {
                KEY_VFILE: "63232.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 65,
                        KEY_IMGEND: 116,
                    }
                ]
            },
            {
                KEY_VFILE: "70380.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 27,
                    }
                ]
            },
            {
                KEY_VFILE: "who_0003.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 72,
                    }
                ]
            },
            {
                KEY_VFILE: "who_0005.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 68,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'candy',
        KEY_VIDS: [
            {
                KEY_VFILE: "08923.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 39,
                    }
                ]
            },
            {
                KEY_VFILE: "70326.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 67,
                        KEY_IMGEND: 99,
                    }
                ]
            },
            {
                KEY_VFILE: "08929.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 41,
                    }
                ]
            },
            {
                KEY_VFILE: "68790.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 51,
                    },
                    {
                        KEY_IMGSTART: 74,
                        KEY_IMGEND: 105,
                    }
                ]
            },
            {
                KEY_VFILE: "68018.mp4",
                KEY_DSFROM: VAL_CLASSIC,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 65,
                    }
                ]
            },
            {
                KEY_VFILE: "08916.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 35,
                        KEY_IMGEND: 62,
                    }
                ]
            },
            {
                KEY_VFILE: "08919.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "08921.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 6,
                        KEY_IMGEND: 35,
                    }
                ]
            },
            {
                KEY_VFILE: "65299.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 45,
                    }
                ]
            },
            {
                KEY_VFILE: "08927.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 46,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'cousin',
        KEY_VIDS: [
            {
                KEY_VFILE: "13638.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 29,
                    }
                ]
            },
            {
                KEY_VFILE: "13636.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 66,
                        KEY_IMGEND: 99,
                    }
                ]
            },
            {
                KEY_VFILE: "13640.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 59,
                    }
                ]
            },
            {
                KEY_VFILE: "65415.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 43,
                    }
                ]
            },
            {
                KEY_VFILE: "68592.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 5,
                        KEY_IMGEND: 31,
                    }
                ]
            },
            {
                KEY_VFILE: "13647.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 45,
                    }
                ]
            },
            {
                KEY_VFILE: "13648.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 40,
                    }
                ]
            },
            {
                KEY_VFILE: "13632.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 52,
                    }
                ]
            },
            {
                KEY_VFILE: "13633.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 43,
                        KEY_IMGEND: 67,
                    }
                ]
            },
            {
                KEY_VFILE: "67535.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 32,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_VIDS: [
            {
                KEY_VFILE: "37472.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 26,
                    }
                ]
            },
            {
                KEY_VFILE: "37474.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 93,
                    }
                ]
            },
            {
                KEY_VFILE: "37477.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "36139.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "36141.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 27,
                    }
                ]
            },
            {
                KEY_VFILE: "36144.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "36145.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 48,
                    }
                ]
            },
            {
                KEY_VFILE: "36146.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "37464.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 4,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "69404.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 43,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_VIDS: [
            {
                KEY_VFILE: "35267.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    }
                ]
            },
            {
                KEY_VFILE: "35544.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 42,
                    }
                ]
            },
            {
                KEY_VFILE: "67879.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 7,
                        KEY_IMGEND: 23,
                    }
                ]
            },
            {
                KEY_VFILE: "35545.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 41,
                    }
                ]
            },
            {
                KEY_VFILE: "35546.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 4,
                        KEY_IMGEND: 58,
                    }
                ]
            },
            {
                KEY_VFILE: "35547.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "35549.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 107,
                    }
                ]
            },
            {
                KEY_VFILE: "35550.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 42,
                    }
                ]
            },
            {
                KEY_VFILE: "35548.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    }
                ]
            },
            {
                KEY_VFILE: "me_i_0001.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 97,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_VIDS: [
            {
                KEY_VFILE: "54870.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "54885.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 42,
                    }
                ]
            },
            {
                KEY_VFILE: "66560.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 34,
                    }
                ]
            },
            {
                KEY_VFILE: "54878.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 63,
                    }
                ]
            },
            {
                KEY_VFILE: "54880.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 51,
                    }
                ]
            },
            {
                KEY_VFILE: "54881.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 6,
                        KEY_IMGEND: 19,
                    }
                ]
            },
            {
                KEY_VFILE: "54883.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 60,
                    }
                ]
            },
            {
                KEY_VFILE: "54884.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 40,
                    }
                ]
            },
            {
                KEY_VFILE: "stomach_0001.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 95,
                    }
                ]
            },
            {
                KEY_VFILE: "stomach_0002.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 44,
                        KEY_IMGEND: 95,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'have',
        KEY_VIDS: [
            {
                KEY_VFILE: "26773.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    }
                ]
            },
            {
                KEY_VFILE: "69088.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 32,
                    },
                    {
                        KEY_IMGSTART: 64,
                        KEY_IMGEND: 86,
                    }
                ]
            },
            {
                KEY_VFILE: "26776.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 45,
                    }
                ]
            },
            {
                KEY_VFILE: "26757.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    }
                ]
            },
            {
                KEY_VFILE: "69360.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 42,
                    }
                ]
            },
            {
                KEY_VFILE: "68069.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 30,
                        KEY_IMGEND: 64,
                    }
                ]
            },
            {
                KEY_VFILE: "70221.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "68984.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 40,
                    }
                ]
            },
            {
                KEY_VFILE: "26775.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 43,
                    }
                ]
            },
            {
                KEY_VFILE: "26766.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 35,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'need',
        KEY_VIDS: [
            {
                KEY_VFILE: "37879.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 35,
                    }
                ]
            },
            {
                KEY_VFILE: "37891.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "37892.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 46,
                    }
                ]
            },
            {
                KEY_VFILE: "68544.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 4,
                        KEY_IMGEND: 36,
                    }
                ]
            },
            {
                KEY_VFILE: "37885.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "37887.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 56,
                    }
                ]
            },
            {
                KEY_VFILE: "37888.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 60,
                    }
                ]
            },
            {
                KEY_VFILE: "37889.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 36,
                    }
                ]
            },
            {
                KEY_VFILE: "37881.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 57,
                    }
                ]
            },
            {
                KEY_VFILE: "67925.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 39,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'see',
        KEY_VIDS: [
            {
                KEY_VFILE: "50125.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 30,
                    }
                ]
            },
            {
                KEY_VFILE: "68444.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 50,
                    },
                    {
                        KEY_IMGSTART: 91,
                        KEY_IMGEND: 114,
                    }
                ]
            },
            {
                KEY_VFILE: "50128.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "50107.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "69460.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "50120.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 39,
                        KEY_IMGEND: 59,
                    }
                ]
            },
            {
                KEY_VFILE: "67178.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 25,
                    }
                ]
            },
            {
                KEY_VFILE: "50123.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "50126.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "50127.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 34,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'feel',
        KEY_VIDS: [
            {
                KEY_VFILE: "67653.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 29,
                    }
                ]
            },
            {
                KEY_VFILE: "21434.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 59,
                    }
                ]
            },
            {
                KEY_VFILE: "69319.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 56,
                    }
                ]
            },
            {
                KEY_VFILE: "21425.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "21438.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "21439.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "21440.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "21441.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 48,
                    }
                ]
            },
            {
                KEY_VFILE: "65696.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "21436.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 53,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_VIDS: [
            {
                KEY_VFILE: "28438.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 33,
                    }
                ]
            },
            {
                KEY_VFILE: "70147.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 52,
                        KEY_IMGEND: 72,
                    }
                ]
            },
            {
                KEY_VFILE: "28453.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 37,
                    }
                ]
            },
            {
                KEY_VFILE: "28450.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 58,
                    }
                ]
            },
            {
                KEY_VFILE: "28445.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "28441.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 42,
                    }
                ]
            },
            {
                KEY_VFILE: "28443.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "28444.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "28452.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 47,
                    }
                ]
            },
            {
                KEY_VFILE: "28454.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 45,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'fever',
        KEY_VIDS: [
            {
                KEY_VFILE: "78219.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 7,
                        KEY_IMGEND: 43,
                    }
                ]
            },
            {
                KEY_VFILE: "78220.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 7,
                        KEY_IMGEND: 28,
                    }
                ]
            },
            {
                KEY_VFILE: "78221.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 2,
                        KEY_IMGEND: 19,
                    }
                ]
            },
            {
                KEY_VFILE: "78222.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: True,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 4,
                        KEY_IMGEND: 14,
                    }
                ]
            },
            {
                KEY_VFILE: "78223.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 5,
                        KEY_IMGEND: 89,
                    }
                ]
            },
            {
                KEY_VFILE: "78224.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 116,
                    }
                ]
            },
            {
                KEY_VFILE: "78225.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "78226.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "fever_0001.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 98,
                    }
                ]
            },
            {
                KEY_VFILE: "fever_0002.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 35,
                        KEY_IMGEND: 87,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_VIDS: [
            {
                KEY_VFILE: "16980.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 91,
                    }
                ]
            },
            {
                KEY_VFILE: "65501.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 30,
                        KEY_IMGEND: 50,
                    }
                ]
            },
            {
                KEY_VFILE: "16982.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 49,
                    }
                ]
            },
            {
                KEY_VFILE: "16985.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 28,
                    }
                ]
            },
            {
                KEY_VFILE: "16986.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 57,
                    }
                ]
            },
            {
                KEY_VFILE: "16987.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 58,
                    }
                ]
            },
            {
                KEY_VFILE: "16988.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "78227.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 52,
                    }
                ]
            },
            {
                KEY_VFILE: "78228.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 48,
                        KEY_IMGEND: 160,
                    }
                ]
            },
            {
                KEY_VFILE: "78229.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 130,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'headache',
        KEY_VIDS: [
            {
                KEY_VFILE: "26832.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 38,
                    }
                ]
            },
            {
                KEY_VFILE: "26835.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 61,
                    }
                ]
            },
            {
                KEY_VFILE: "67747.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 28,
                    }
                ]
            },
            {
                KEY_VFILE: "26837.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 53,
                    }
                ]
            },
            {
                KEY_VFILE: "26846.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 61,
                    }
                ]
            },
            {
                KEY_VFILE: "26839.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 46,
                    }
                ]
            },
            {
                KEY_VFILE: "26836.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 54,
                    }
                ]
            },
            {
                KEY_VFILE: "26838.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 41,
                    }
                ]
            },
            {
                KEY_VFILE: "26841.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 38,
                    }
                ]
            },
            {
                KEY_VFILE: "65881.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 32,
                    }
                ]
            },
        ]
    },
    {
        KEY_G: 'doctor',
        KEY_VIDS: [
            {
                KEY_VFILE: "67579.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 30,
                    }
                ]
            },
            {
                KEY_VFILE: "17015.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 48,
                        KEY_IMGEND: 72,
                    }
                ]
            },
            {
                KEY_VFILE: "17020.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 55,
                    }
                ]
            },
            {
                KEY_VFILE: "17007.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 1,
                        KEY_IMGEND: 43,
                    }
                ]
            },
            {
                KEY_VFILE: "70049.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 54,
                    }
                ]
            },
            {
                KEY_VFILE: "17017.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 58,
                    }
                ]
            },
            {
                KEY_VFILE: "17022.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 64,
                    }
                ]
            },
            {
                KEY_VFILE: "65504.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 41,
                    }
                ]
            },
            {
                KEY_VFILE: "17023.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 48,
                    }
                ]
            },
            {
                KEY_VFILE: "17014.mp4",
                KEY_DSFROM: VAL_RECENT,
                DEL_LEFT_HAND: False,
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 53,
                    }
                ]
            },
        ]
    },
]
# ---- contants end ------
# ------------------------
# ------------------------




def part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # xs, ys= zip(*landmarks)
    xs= list(map(lambda el: el[0], landmarks))
    ys= list(map(lambda el: el[1], landmarks))
    min_x, min_y= min(xs), min(ys)
    xNeedForward: bool= min_x<0.0
    yNeedForward: bool= min_y<0.0
    if xNeedForward or yNeedForward:
        landmarks= [(
            x -min_x    if xNeedForward    else x,
            y -min_y    if yNeedForward    else y
        ) for x, y in landmarks]
    del xs, ys, min_x, min_y, xNeedForward, yNeedForward

    max_xy: float= max(
        max(x for x, _ in landmarks),
        max(y for _, y in landmarks)
    )
    if max_xy>1:
        landmarks= [(
            x/max_xy,
            y/max_xy
        ) for x, y in landmarks]

    return landmarks
def part2_beSquareRatioOnImage(landmarks: list[tuple[float, float]], original_shape: tuple[int, int]) -> list[tuple[float, float]]:
    height, width= original_shape
    if height==width:
        return landmarks

    if width<height: # portrait, change2withRespect2Height
        scale: float= width/height
        return [(
            x*scale,
            y
        ) for x, y in landmarks]

    # landscape, change2withRespect2Width
    scale: float= height/width
    return [(
        x,
        y*scale
    ) for x, y in landmarks]
def part3_zoomInOutForPadding(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    # zoom in/out for padding be 10% each side with respect to original aspect ratio
    # ie.:
    # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
    # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
    # pad: float= 0.05
    pad: float= 4.0/158.0
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    # xs= list(filter(lambda el: el!=0, xs))
    # ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    min_x, min_y=    min(xs), min(ys)
    max_x, max_y=    max(xs), max(ys)
    scale: float= (1  -2*pad)/max(
        max_x -min_x,
        max_y -min_y
    )
    return [(
        # (x -min_x)    *scale    +pad  if x!=0 else x,
        # (y -min_y)    *scale    +pad  if y!=0 else y
        (x -min_x)    *scale    +pad,
        (y -min_y)    *scale    +pad
    ) for x, y in landmarks]
def part4_centerLandmarkVerticallyHorizontally(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 3) center landmark with same aspect ratio as original
    # center horizontally and vertically, since done padding then just
    # move to right/down
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    # xs= list(filter(lambda el: el!=0, xs))
    # ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    shift_x: float=  0.5    -(min(xs) +max(xs))  /2
    shift_y: float=  0.5    -(min(ys) +max(ys))  /2

    return [(
        # x +shift_x  if x!=0 else x,
        # y +shift_y  if y!=0 else y
        x +shift_x,
        y +shift_y
    ) for x, y in landmarks]
def normalizeLandmarks(landmarks: list[tuple[float, float]], original_shape: tuple) -> list[tuple[float, float]]:
    '''
    landmarks is an array eg. of shape (86, 2)
    original_shape is tuple (HEIGHT, WIDTH)
    '''
    # lmark_fph.face_landmarks.landmark
    # lmark_fph.pose_landmarks.landmark
    # lmark_fph.left_hand_landmarks.landmark
    # lmark_fph.right_hand_landmarks.landmark
    # logic resize to --> 480 x 480 x 3
    #     0) all coords be greater than|= 0.0 and less than|= 1.0
    #         a) lmarks x,y overwrite to [0.0, 1.0] only
    #         b) has x < 0.0 then ALL_x+abs(min(x_neg)), ie. move right
    #         b) has y < 0.0 then ALL_y+abs(min(y_neg)), ie. move down
    #         c) ALL_coords_x_y/highest_value
    #         d) eg. 1.74 then ALL_coords_x_y/1.74
    #         e) for all be scaled down with same aspect ratio as orig
    #         f) NOW all( x, y ) are 0.0 to 1.0 value only
    #     1) from old img ratio to new square img ratio
    #         a) if owx < ohy: all_x= all_x* (480*owx/ohy)/480
    #         b) if ohy < owx: all_y= all_y* (480*ohy/owx)/480
    #     2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    #         a) if too far zoom in
    #         b) if too close zoom out
    #         c) goal lowest val 0.05 both(x,y) .ie padding
    #         d) goal highest val 0.95 both(x,y)
    #         e) ie. max(lm_wx, lm_hy) == 0.9
    #     3) center landmark with same aspect ratio as original
    #         a) min_wx_hy= min( wx, hy ); max_wx_hy= max( wx, hy )
    #         b) min_wx_hy as mn; max_wx_hy as mx
    #         c) if mn is wx, all X +( (mx-mn)/(mx*2) )
    #         d) if mn is hy, all Y +( (mx-mn)/(mx*2) )
    assert 1<len(original_shape) # incorrect use of normalizeLandmarks(...), mandatory 1<len(original_shape)
    landmarks= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks)
    landmarks= part2_beSquareRatioOnImage(
        landmarks,
        (original_shape[0], original_shape[1])
    )
    landmarks= part3_zoomInOutForPadding(landmarks)
    landmarks= part4_centerLandmarkVerticallyHorizontally(landmarks)


    return landmarks
def normalizeWorthyLandmarkWrapper(
    landmarks: list[tuple[float, float]], # order --> face_pose_left_right_hand
    original_shape: tuple,
    hasLandmarks: dict
) -> list[tuple[float, float]]:
    if not (hasLandmarks["face"] or hasLandmarks["pose"] or hasLandmarks["left_hand"] or hasLandmarks["right_hand"]):
        return landmarks
    norm: list= []
    if hasLandmarks['face']:
        norm.extend(landmarks[
            :len(WORTHY_FACE_IDX)
        ])
    if hasLandmarks['pose']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
        ])
    if hasLandmarks['left_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
        ])
    if hasLandmarks['right_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
        ])
    norm= normalizeLandmarks(landmarks=norm, original_shape=original_shape)

    out_landmark: list= []
    # -- face landmarks --
    if hasLandmarks["face"]:
        out_landmark.extend(norm[:len(WORTHY_FACE_IDX)])
    else:
        out_landmark.extend(zeros(
            (
                len(WORTHY_FACE_IDX),
                2
            ), dtype=float32).tolist()
        )

    # -- pose landmarks --
    if hasLandmarks["pose"]:
        if hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
            ])
        else:
            out_landmark.extend(norm[
                :len(WORTHY_POSE_IDX)
            ])
    else:
        out_landmark.extend(zeros(
            (
                len(WORTHY_POSE_IDX),
                2
            ), dtype=float32).tolist())

    # -- left hand landmarks --
    if hasLandmarks["left_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +QUANTITY_HAND_LMARK
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
                len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ])
        else:
            out_landmark.extend(norm[
                :QUANTITY_HAND_LMARK
            ])
    else:
        out_landmark.extend(zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            ), dtype=float32).tolist())

    # -- right hand landmarks --
    if hasLandmarks["right_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["face"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                QUANTITY_HAND_LMARK:
            ])
        else:
            out_landmark.extend(norm)
    else:
        out_landmark.extend(zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            ), dtype=float32).tolist())

    # out_landmark shape:
    # (
    #     len(WORTHY_FACE_IDX)
    #     +len(WORTHY_POSE_IDX)
    #     +QUANTITY_HAND_LMARK *2,
    #     2
    # )
    return out_landmark # ie. output shape (86, 2)
def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: ndarray, \
                    lmark_coordinates: list, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> ndarray:
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
                    circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    circle(
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
                    line(
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
def get_lmark_face(landmark):
    return landmark[:len(WORTHY_FACE_IDX)]
def get_lmark_pose(landmark):
    return landmark[
        len(WORTHY_FACE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
    ]
def get_lmark_lhand(landmark):
    return landmark[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ]
def get_lmark_rhand(landmark):
    return landmark[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ]
def drawFacePoseHand(img_write_to: ndarray, landmarks: ndarray, hasLandmarks: dict) -> ndarray:
    if hasLandmarks[KEY_FACE] \
        or hasLandmarks[KEY_POSE] \
        or hasLandmarks[KEY_LHAND] \
        or hasLandmarks[KEY_RHAND]:

        # ---- face landmarks ----
        if hasLandmarks[KEY_FACE]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_face(landmarks).tolist(),
                connections_idxs=FACE_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(0, 153, 0), # 153/255= 0.6
            )

        # ---- pose landmarks ----
        if hasLandmarks[KEY_POSE]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_pose(landmarks).tolist(),
                connections_idxs=POSE_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(0, 0, 153), # 153/255= 0.6
            )

        # ---- left hand landmarks ----
        if hasLandmarks[KEY_LHAND]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_lhand(landmarks).tolist(),
                connections_idxs=HAND_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(255, 255, 255)
            )

        # ---- right hand landmarks ----
        if hasLandmarks[KEY_RHAND]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_rhand(landmarks).tolist(),
                connections_idxs=HAND_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(153, 204, 204), # 204/255= 0.8
            )

        # HERE ORDER OF LANDMARKS
        # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to
def init_directories() -> None:
    msg_err: list= []
    if not LM_RECENT_DIR.exists():
        msg_err.append(f"Folder Not Found( Please provide this folder ): {LM_RECENT_DIR}")
    if not LM_NVSTGT_DIR.exists():
        msg_err.append(f"Folder Not Found( Please provide this folder ): {LM_NVSTGT_DIR}")
    if not SKLTN_RECENT_DIR.exists():
        msg_err.append(f"Folder Not Found( Please provide this folder ): {SKLTN_RECENT_DIR}")
    if not SKLTN_NVSTGT_DIR.exists():
        msg_err.append(f"Folder Not Found( Please provide this folder ): {SKLTN_NVSTGT_DIR}")

    if LANDMARK_dir.exists():
        msg_err.append(f"Please delete this folder( this will automatically be created for you ): {LANDMARK_dir}")
    if SKELETON_dir.exists():
        msg_err.append(f"Please delete this folder( this will automatically be created for you ): {SKELETON_dir}")

    if 0<len(msg_err):
        for msg in msg_err:
            print(msg)
        raise FileNotFoundError('Please provide the appropriate files, as said above.')
    else:
        LANDMARK_dir.mkdir()
        SKELETON_dir.mkdir()
        print('Initialized and checked needed directories.')
def get_annotations() -> tuple:
    ds_classic_landmark: list= []
    # ds_classic_skeleton: list= []
    ds_use_recent_landmark: list= []
    # ds_use_recent_skeleton: list= []
    with open(f"{HYBRID_DS_DIR /"hybrid_data.investigate.landmark.json"}", 'r') as f:
        ds_classic_landmark= loadJson(f)
    # with open(f"{HYBRID_DS_DIR /"hybrid_data.investigate.skeleton.json"}", 'r') as f:
    #     ds_classic_skeleton= loadJson(f)
    with open(f"{HYBRID_DS_DIR /"fix_most_recent.investigate.landmark.json"}", 'r') as f:
        ds_use_recent_landmark= loadJson(f)
    # with open(f"{HYBRID_DS_DIR /"fix_most_recent.investigate.skeleton.json"}", 'r') as f:
    #     ds_use_recent_skeleton= loadJson(f)
    return (
        ds_classic_landmark,
        # ds_classic_skeleton,
        ds_use_recent_landmark,
        # ds_use_recent_skeleton,
    )
def rm_lhand(lm_data_npy: ndarray, hasLandmarks: dict):
    lm_data_npy[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ]= zeros((
        QUANTITY_HAND_LMARK,
        2
    ), dtype=float32)
    lm_data_npy= array(normalizeWorthyLandmarkWrapper(
        landmarks=lm_data_npy.tolist(),
        original_shape=(300, 300, 3),
        hasLandmarks={
            KEY_FACE: hasLandmarks[KEY_FACE],
            KEY_POSE: hasLandmarks[KEY_POSE],
            KEY_LHAND: False,
            KEY_RHAND: hasLandmarks[KEY_RHAND],
        },
    ))
    return lm_data_npy
def process_lm_video(
    folder_video_from: str,
    folder_video_to: str,
    video_details: dict,
    del_lhand: bool,
    which_images: list,
    where_from: str,
    ) -> tuple:
    '''
    returns tuple --> corrected_landmarks, new_annotations

    LM_RECENT_DIR: Path= HYBRID_DS_DIR /"fix_most_recent_landmark"
    LM_NVSTGT_DIR: Path= HYBRID_DS_DIR /"investigate_landmark"
    '''
    landmark_annotations: list= []
    skeleton_annotations: list= []
    abs_landmark_dir_from: Path= LM_RECENT_DIR if where_from==VAL_RECENT else LM_NVSTGT_DIR
    abs_landmark_dir_from= abs_landmark_dir_from /folder_video_from
    for a_start_end in which_images:
        parent_folder_a_start_end: str= f"{folder_video_to}_{str(
            a_start_end[KEY_IMGSTART]
        ).zfill(3)}_{str(a_start_end[KEY_IMGEND]).zfill(3)}"
        abs_landmark_dir_to: Path= LANDMARK_dir /parent_folder_a_start_end
        abs_landmark_dir_to.mkdir()
        abs_skeleton_dir_to: Path= SKELETON_dir /parent_folder_a_start_end
        abs_skeleton_dir_to.mkdir()
        landmark_annotations.append({
            KEY_PFOLDER: parent_folder_a_start_end,
            KEY_LANDMARK: []
        })
        skeleton_annotations.append({
            KEY_PFOLDER: parent_folder_a_start_end,
            KEY_SKELETON: []
        })
        for idx_init_1, idx in zip(
            range(
                1,
                a_start_end[KEY_IMGEND] -(a_start_end[KEY_IMGSTART]-1) +1,
            ),
            range(
                a_start_end[KEY_IMGSTART] -1,
                a_start_end[KEY_IMGEND],
            )
        ):
            lm_data_npy: ndarray
            with open(f"{abs_landmark_dir_from /video_details[KEY_LANDMARK][idx][KEY_FILE]}", 'rb') as f:
                lm_data_npy= numpyload(f)
            assert lm_data_npy.shape==(
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK*2,
                2
            )
            filename2save: str= str(idx_init_1).zfill(8)
            landmark_annotations[-1][KEY_LANDMARK].append({
                KEY_FILE: f"{filename2save}.npy",
                KEY_FACE: bool(video_details[KEY_LANDMARK][idx][KEY_FACE]),
                KEY_POSE: bool(video_details[KEY_LANDMARK][idx][KEY_POSE]),
                KEY_LHAND: bool(video_details[KEY_LANDMARK][idx][KEY_LHAND]),
                KEY_RHAND: bool(video_details[KEY_LANDMARK][idx][KEY_RHAND]),
            })
            skeleton_annotations[-1][KEY_SKELETON].append({
                KEY_FILE: f"{filename2save}.jpg",
                KEY_FACE: bool(video_details[KEY_LANDMARK][idx][KEY_FACE]),
                KEY_POSE: bool(video_details[KEY_LANDMARK][idx][KEY_POSE]),
                KEY_LHAND: bool(video_details[KEY_LANDMARK][idx][KEY_LHAND]),
                KEY_RHAND: bool(video_details[KEY_LANDMARK][idx][KEY_RHAND]),
            })
            if del_lhand:
                lm_data_npy= rm_lhand(
                    lm_data_npy=lm_data_npy,
                    hasLandmarks=video_details[KEY_LANDMARK][idx],
                )
                landmark_annotations[-1][KEY_LANDMARK][-1][KEY_LHAND]= False
                skeleton_annotations[-1][KEY_SKELETON][-1][KEY_LHAND]= False
            with open(f"{abs_landmark_dir_to /filename2save}.npy", 'wb') as f:
                numpysave(f, lm_data_npy)
            imwrite(
                f"{abs_skeleton_dir_to /filename2save}.jpg",
                drawFacePoseHand(
                    img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    landmarks=lm_data_npy,
                    hasLandmarks=video_details[KEY_LANDMARK][idx],
                )
            )
    return (
        landmark_annotations,
        skeleton_annotations,
    )
def process_dataset() -> tuple:
    ds_c_landmark, ds_ur_landmark= get_annotations()
    ds_landmark: list= []
    ds_skeleton: list= []
    for a_gloss in HYBRID_DATA_SRC:
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS]:
            ds_landmark[-1][KEY_VIDS].append({
                KEY_VFILE: a_video[KEY_VFILE],
                KEY_LANDMARK: []
            })
            ds_skeleton[-1][KEY_VIDS].append({
                KEY_VFILE: a_video[KEY_VFILE],
                KEY_SKELETON: []
            })
            video_details_from: dict|list= []
            lm_folder: str
            if a_video[KEY_DSFROM]==VAL_CLASSIC:
                video_details_from= list(filter(
                    lambda el: el[KEY_G][3:]==a_gloss[KEY_G],
                    ds_c_landmark
                ))
                lm_folder= video_details_from[0][KEY_G]
                video_details_from= list(filter(
                    lambda el: el[KEY_VFILE]==a_video[KEY_VFILE],
                    video_details_from[0][KEY_VIDS]
                ))
            # elif a_video[KEY_DSFROM]==VAL_RECENT:
            else:
                video_details_from= list(filter(
                    lambda el: el[KEY_G][3:]==a_gloss[KEY_G],
                    ds_ur_landmark
                ))
                lm_folder= video_details_from[0][KEY_G][3:]
                video_details_from= list(filter(
                    lambda el: el[KEY_VFILE]==a_video[KEY_VFILE],
                    video_details_from[0][KEY_VIDS]
                ))
            video_details_from= dict(video_details_from[0])
            lm_folder= f"{lm_folder}_{Path(video_details_from[KEY_VFILE]).stem}"
            new_ann_landmark, new_ann_skeleton= process_lm_video(
                folder_video_from=lm_folder,
                folder_video_to=f"{a_gloss[KEY_G]}_{Path(a_video[KEY_VFILE]).stem}",
                video_details=video_details_from,
                del_lhand=a_video[DEL_LEFT_HAND],
                which_images=a_video[KEY_IMG_VALID],
                where_from=a_video[KEY_DSFROM],
            )
            ds_landmark[-1][KEY_VIDS][-1][KEY_LANDMARK]= new_ann_landmark
            ds_skeleton[-1][KEY_VIDS][-1][KEY_SKELETON]= new_ann_skeleton
    return ds_landmark, ds_skeleton
def main():
    init_directories()
    ds_landmark, ds_skeleton= process_dataset()
    with open(f"{HYBRID_DS_DIR /'ds_landmark'}.json", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{HYBRID_DS_DIR /'ds_skeleton'}.json", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    main()
