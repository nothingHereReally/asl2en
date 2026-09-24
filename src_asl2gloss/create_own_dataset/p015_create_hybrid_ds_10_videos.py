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
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_FILE: str= 'file'
# -------------------
KEY_DSFROM: str= 'get_from'
VAL_RECENT: str= 'fill_via_recent'
VAL_CLASSIC: str= 'classic'
DEL_LEFT_HAND: str= 'remove_left_hand'
# ---------------------
KEY_SPLIT: str= 'split'
KEY_TRAIN: str= 'train'
KEY_TEST: str= 'test'
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




def main():
    print(len(HYBRID_DATA_SRC))
    for idx, a_gloss in enumerate(HYBRID_DATA_SRC):
        print(f"{idx} -- {a_gloss[KEY_G]} -- {len(a_gloss[KEY_VIDS])}")
        for a_video in a_gloss[KEY_VIDS]:
            print(f"    {a_video[KEY_VFILE]}         ", end='')
            for a_start_end in a_video[KEY_IMG_VALID]:
                print(f"s( {a_start_end[KEY_IMGSTART]} )   e( {a_start_end[KEY_IMGEND]} ) ,  ", end='')
            if a_video[KEY_DSFROM]==VAL_CLASSIC:
                print(' -- useClassic', end='')
            if a_video[DEL_LEFT_HAND]:
                print(' -- delete left hand', end='')
            print()
        print()
if __name__=="__main__":
    main()
