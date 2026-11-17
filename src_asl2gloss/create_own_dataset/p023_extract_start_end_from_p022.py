from json import dump as writeJson, load as loadJson
from pathlib import Path
from shutil import copy as copyFile


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"asl_citizen"
VIDEO_DIR: Path= DS_DIR /"video"
IN_LANDMARK_DIR: Path= DS_DIR /"investigate_landmark"
IN_SKELETON_DIR: Path= DS_DIR /"investigate_skeleton"
OUT_LANDMARK_DIR: Path= DS_DIR /"selected_start_end_landmark"
OUT_SKELETON_DIR: Path= DS_DIR /"selected_start_end_skeleton"
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
KEY_PFOLDER: str= 'parent_folder'
IMG_SIZE: int= 158
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
VIDEO_LIST: list=[
    {
        KEY_G: 'body',
        KEY_VIDS: [
            {
                KEY_FILE: 'body_012509162398409268-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'body_023036288555289275-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'body_14988750046183918-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'body_18110186267814066-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 46,
                    },
                ],
            },
            {
                KEY_FILE: 'body_18235319744310297-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 22,
                    },
                ],
            },
            {
                KEY_FILE: 'body_2179874717207788-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'body_22658344717761292-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'body_23152470677862436-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'body_3003852815347694-seedBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 28,
                    },
                ],
            },
            {
                KEY_FILE: 'body_31812874442193184-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 39,
                        KEY_IMGEND: 57,
                    },
                ],
            },
            {
                KEY_FILE: 'body_3219216092683568-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'body_3470900852703691-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'body_38673461744862414-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'body_5487980874423544-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'body_5857440141605472-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 22,
                    },
                ],
            },
            {
                KEY_FILE: 'body_6459896690781501-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'body_7111482896558119-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 25,
                    },
                ],
            },
            {
                KEY_FILE: 'body_7381642274284421-BODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 31,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'ear',
        KEY_VIDS: [
            {
                KEY_FILE: 'ear_004887131733942285-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 58,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_015143605161060059-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_06251353179823571-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_156747591554246-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_1677354883522144-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 50,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_1856278575298309-seedEAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_22124278737999714-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 48,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_23747369865179402-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 89,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_24470512221089535-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_3550234595479076-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 50,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_3597923622170567-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 30,
                        KEY_IMGEND: 59,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_393295857925948-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 47,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_4402580501014326-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 51,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_4496497579994534-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_468644667228318-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 19,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_48472331371206123-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_4886596064274962-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 42,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_4980479911230302-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_6904017055031177-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'ear_7453873486095204-EAR.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 33,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'eyes',
        KEY_VIDS: [
            {
                KEY_FILE: 'eyes_015907943367508448-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_024859836566530236-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_0259965808147955-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_06163237568266666-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_0825936687116755-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_09936405951023719-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_11765090581788384-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 36,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_1417646504520429-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_1710485018536383-seedEYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_18542508287605441-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_23137361727286043-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 46,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_2960811103126946-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_31593048576495075-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_32332228660933526-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_32695601160288645-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_32913239320491705-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_3703962814561417-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_4324249049090094-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_4798839631383074-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'eyes_4799020574000763-EYES.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 34,
                        KEY_IMGEND: 56,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'head',
        KEY_VIDS: [
            {
                KEY_FILE: 'head_04384724186611-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'head_09712564478415864-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 51,
                    },
                ],
            },
            {
                KEY_FILE: 'head_1434058469566546-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'head_14486467848103057-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'head_18306126220160568-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'head_209949774531083-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'head_22456983792819685-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'head_24673566947760395-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'head_24680684027267863-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 41,
                        KEY_IMGEND: 61,
                    },
                ],
            },
            {
                KEY_FILE: 'head_2820341113576026-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'head_3011506857442785-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'head_42134820560428166-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'head_49019167048619505-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'head_5670994066372224-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'head_6435251723193283-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'head_6723633822967117-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'head_6931138867364373-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'head_708769017680714-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 22,
                    },
                ],
            },
            {
                KEY_FILE: 'head_7216744097791985-HEAD.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 57,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'nose',
        KEY_VIDS: [
            {
                KEY_FILE: 'nose_024011464372713398-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_05530025137611427-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_08859205707077078-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 18,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_09217547618470956-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_11100447586620077-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_11213721786275399-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_12789189992227112-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 63,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_2649632981009986-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_2691787200603868-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 22,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_27103930782829466-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 48,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_27763818780548877-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_33334259968320534-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_3712619010031373-seedNOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_37138484525407467-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_388068410054895-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_388468441362549-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 6,
                        KEY_IMGEND: 19,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_6032656220146979-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 36,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_6652814499755777-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_6657895793756088-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 49,
                    },
                ],
            },
            {
                KEY_FILE: 'nose_6746586649060085-NOSE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 28,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'chin',
        KEY_VIDS: [
            {
                KEY_FILE: 'chin_028431370391587052-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_09101309522440815-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_09176723865512693-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_11642380273775732-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_17378200161913648-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 20,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_1921109044689493-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 53,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_22956006243428684-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_25405637892250943-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_30852673069728254-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 30,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_3145184748224372-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_3969659858373005-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_41965703854599123-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_45079112918929276-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 49,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_45798047890015803-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_5804474557706663-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 51,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_6126017782493465-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 20,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_6388058030166044-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_7475699875205186-seedCHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'chin_7489397758558725-CHIN_2.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 34,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'gum',
        KEY_VIDS: [
            {
                KEY_FILE: 'gum_018213254785876654-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_0663439230334546-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_07050828584534741-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 47,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_07804494694411823-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 62,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_15670060196174318-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 18,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_17151697988510128-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 48,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_3704551309457076-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_41310640938491505-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 22,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_49233665334372345-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 56,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_5169364593751733-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_6083769831208186-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_6458203226414239-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_6525706206661304-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 47,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_6774779032894782-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_6833405506171966-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 69,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_7307539259823796-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_7553281453781038-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 62,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_7567845252066006-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_7959535258775348-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 62,
                    },
                ],
            },
            {
                KEY_FILE: 'gum_8098218886424609-GUM.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 43,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'mouth',
        KEY_VIDS: [
            {
                KEY_FILE: 'mouth_03816967766642465-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_07770960749445921-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_08629381582280149-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_11034432898849245-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 48,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_16288519431904458-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 59,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_17189033119247688-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_27934471953770945-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_3108143937298642-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_36090957279237323-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_36938644349320904-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 51,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_3755885673667929-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 68,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_38196977602640914-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_38571484397842015-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_43438530362822236-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 53,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_5022024196847912-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_5193451899614112-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_5432467311144609-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'mouth_5548976355283721-MOUTH.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 52,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'neck',
        KEY_VIDS: [
            {
                KEY_FILE: 'neck_002159742772197104-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_010111338865819608-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_029915112474971917-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_04541132470344267-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 66,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_10197900855285669-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 51,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_1105210028899144-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_11253959737739772-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_2117672377435793-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 50,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_2914346788925697-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_392847143355763-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_40869836172089014-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 58,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_46050718150682335-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 36,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_5912486425716992-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 56,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_6444945863834219-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 53,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_7132418799120459-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'neck_7432560177781751-NECK.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 32,
                        KEY_IMGEND: 54,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'tongue',
        KEY_VIDS: [
            {
                KEY_FILE: 'tongue_10952554539506631-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 22,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_11388872998746447-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_16538366877078858-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_1731074372958592-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_18793572427530347-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 8,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_25040182027203084-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_2928599376925225-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 36,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_29312176513514654-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_2995978175034413-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 25,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_3180818311436544-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_3766203415307263-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_43750843655956917-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_44185907261489477-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_4578141062450063-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_4716794408227516-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_5125705967911591-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 28,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_5231084233910046-seedTONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_5322853783203081-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'tongue_7378948905149243-TONGUE.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 24,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'arm',
        KEY_VIDS: [
            {
                KEY_FILE: 'arm_06875405886440444-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_13014078898215797-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_1535086938572987-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_1703736894296095-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 42,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_1766551266387144-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_18252662803841702-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_18436257304546988-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_20151143656361903-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_252084110246515-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_3280914345213024-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 26,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_3532171049708259-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 25,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_3875903439603874-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 26,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_38957968308015767-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 48,
                        KEY_IMGEND: 63,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_40026841881315-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_44775996014584885-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 46,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_46398054082560547-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 27,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_46904377801196495-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 28,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_5617325649329079-seedARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_6252908560318753-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 28,
                    },
                ],
            },
            {
                KEY_FILE: 'arm_6451133306899228-ARM_3.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 30,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'finger',
        KEY_VIDS: [
            {
                KEY_FILE: 'finger_006007887322393879-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 58,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_06889282709145239-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_19511824774736541-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_2196507384016626-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 7,
                        KEY_IMGEND: 13,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_30123812533804895-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_3017633228385066-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 24,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_31835717268957353-seedFINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_3237225336922218-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 42,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_33057973168830923-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_35791960976845894-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_3781013762752725-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 37,
                        KEY_IMGEND: 63,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_3825663392118379-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 31,
                        KEY_IMGEND: 49,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_4033702656472984-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 16,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_42851478328968695-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 30,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_47727751077906433-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_6325675928394079-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 42,
                    },
                ],
            },
            {
                KEY_FILE: 'finger_6537381381903058-FINGER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 31,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'hands',
        KEY_VIDS: [
            {
                KEY_FILE: 'hands_04268194927860791-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_09820261529418906-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 20,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_13923543055899334-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_24253465795861295-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_31487135960578416-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 28,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_3802234939313722-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 26,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_4141697998179248-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 37,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_5315418020402818-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 33,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_5548063463829158-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_6368495873121254-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 8,
                        KEY_IMGEND: 26,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_6719352826133227-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'hands_6720783140394533-HANDS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 45,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'heart',
        KEY_VIDS: [
            {
                KEY_FILE: 'heart_0596738632892615-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 43,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_06823095187885575-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 61,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_14629201741355402-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_20132491671886066-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 64,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_20146972673867425-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_21189655894981962-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 10,
                        KEY_IMGEND: 38,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_23476630068245297-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_2898262557497693-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 44,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_32138134663150675-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 50,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_4070974316820264-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_47938919974598626-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 42,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_5145420536026162-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 9,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_5396509619343579-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 62,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_561963088245941-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 34,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_5710813665222216-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 53,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_5754764115128781-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_5885245957009604-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 48,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_5977460365096976-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_6466497454416145-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'heart_6597977577129788-HEARTBODY.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 37,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'lungs',
        KEY_VIDS: [
            {
                KEY_FILE: 'lungs_03926378975519795-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 15,
                        KEY_IMGEND: 36,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_07014496526883707-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 12,
                        KEY_IMGEND: 27,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_1361388675596844-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 40,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_17927180292408784-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_25611976222926836-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 49,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_2960784973349957-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_3156394499603383-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 62,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5064445532351614-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5351840003920145-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 20,
                        KEY_IMGEND: 29,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5647565128465428-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 22,
                        KEY_IMGEND: 52,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5830051383767008-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 36,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5847004858651264-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 17,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5897532389009943-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'lungs_5915817261595504-LUNGS.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 14,
                        KEY_IMGEND: 40,
                    },
                ],
            }
                  ]
    },
    {
        KEY_G: 'shoulder',
        KEY_VIDS: [
            {
                KEY_FILE: 'shoulder_0015823648451052108-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 24,
                        KEY_IMGEND: 49,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_0752008538127451-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 18,
                        KEY_IMGEND: 41,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_08082334261631097-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 29,
                        KEY_IMGEND: 57,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_10026272000810366-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 23,
                        KEY_IMGEND: 32,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_1674492320377452-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 47,
                        KEY_IMGEND: 54,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_286056050097667-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 69,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_33477858047544085-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 28,
                        KEY_IMGEND: 39,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_33812397275866224-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 21,
                        KEY_IMGEND: 45,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_372842197823239-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 33,
                        KEY_IMGEND: 58,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_5030770445869783-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 13,
                        KEY_IMGEND: 23,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_5366284286897927-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 47,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_5375547392845288-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 16,
                        KEY_IMGEND: 31,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_564853635851184-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 38,
                        KEY_IMGEND: 60,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_5688080856457567-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 11,
                        KEY_IMGEND: 26,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_6045076691946099-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 36,
                        KEY_IMGEND: 51,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_6101256568316258-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 25,
                        KEY_IMGEND: 35,
                    },
                ],
            },
            {
                KEY_FILE: 'shoulder_620816222847451-SHOULDER.mp4',
                KEY_IMG_VALID: [
                    {
                        KEY_IMGSTART: 19,
                        KEY_IMGEND: 32,
                    },
                ],
            }
                  ]
    },
]




def init_dirs() -> None:
    if not IN_LANDMARK_DIR.exists():
        raise NotImplementedError(f"Please provide this {IN_LANDMARK_DIR}.")
    if not IN_SKELETON_DIR.exists():
        raise NotImplementedError(f"Please provide this {IN_SKELETON_DIR}.")

    if OUT_LANDMARK_DIR.exists():
        raise NotImplementedError(f"Please delete {OUT_LANDMARK_DIR}, this will be created for you.")
    if OUT_SKELETON_DIR.exists():
        raise NotImplementedError(f"Please delete {OUT_SKELETON_DIR}, this will be created for you.")
    OUT_LANDMARK_DIR.mkdir()
    OUT_SKELETON_DIR.mkdir()
def get_past_dataset_landmarks() -> list:
    with open(DS_DIR /'investigate_landmark.json', 'r') as f:
        out: list= loadJson(f)
    return out
def process_videos_ds() -> tuple:
    input_ds_landmark: list= get_past_dataset_landmarks()
    ds_landmark: list= []
    ds_skeleton: list= []
    for idx_gloss, a_gloss in enumerate(VIDEO_LIST):
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        input_ds_landmark[idx_gloss][KEY_VIDS]= list(filter(
            lambda x: x[KEY_FILE] in tuple(
                el[KEY_FILE] for el in a_gloss[KEY_VIDS]
            ),
            input_ds_landmark[idx_gloss][KEY_VIDS]
        ))
        for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
            prev_video_ds: dict= input_ds_landmark[idx_gloss][KEY_VIDS][idx_video]
            assert a_video[KEY_FILE] == prev_video_ds[KEY_FILE]
            for a_start_end in a_video[KEY_IMG_VALID]:
                assert a_start_end[KEY_IMGSTART] <= a_start_end[KEY_IMGEND]
                assert a_start_end[KEY_IMGEND] <= len(prev_video_ds[KEY_LANDMARK])
                from_pfolder: str= a_video[KEY_FILE][:-4]
                pfolder_name: str= f"{from_pfolder}_{str(a_start_end[KEY_IMGSTART]).zfill(4)}"
                pfolder_name= f"{pfolder_name}_{str(a_start_end[KEY_IMGEND]).zfill(4)}"
                landmark_abs_folder: Path= OUT_LANDMARK_DIR /pfolder_name
                skeleton_abs_folder: Path= OUT_SKELETON_DIR /pfolder_name
                landmark_abs_folder.mkdir()
                skeleton_abs_folder.mkdir()
                split_what: str= VAL_TRAIN if idx_video<int(RATIO4TRAIN*len(a_gloss[KEY_VIDS])) else VAL_TEST
                ds_landmark[-1][KEY_VIDS].append({
                    KEY_PFOLDER: pfolder_name,
                    KEY_SPLIT: split_what,
                    KEY_LANDMARK: [],
                })
                ds_skeleton[-1][KEY_VIDS].append({
                    KEY_PFOLDER: pfolder_name,
                    KEY_SPLIT: split_what,
                    KEY_SKELETON: [],
                })
                for idx_img_init0 in range(a_start_end[KEY_IMGEND]-a_start_end[KEY_IMGSTART]+1):
                    assert a_start_end[KEY_IMGEND] <= len(prev_video_ds[KEY_LANDMARK])
                    landmark_skeleton_prefix: str= str(idx_img_init0+1).zfill(8)
                    idx4prev_what: int= idx_img_init0 +a_start_end[KEY_IMGSTART] -1
                    img_annotation= prev_video_ds[KEY_LANDMARK][ idx4prev_what ]
                    from_landmark: Path= IN_LANDMARK_DIR /from_pfolder /prev_video_ds[KEY_LANDMARK][idx4prev_what][KEY_FILE]
                    from_skeleton: Path= IN_SKELETON_DIR /from_pfolder /f"{prev_video_ds[KEY_LANDMARK][idx4prev_what][KEY_FILE][:-4]}.jpg"
                    to_landmark: Path= landmark_abs_folder /f"{landmark_skeleton_prefix}.npy"
                    to_skeleton: Path= skeleton_abs_folder /f"{landmark_skeleton_prefix}.jpg"
                    assert from_landmark.exists()
                    assert from_skeleton.exists()
                    copyFile(
                        src=from_landmark,
                        dst=to_landmark,
                        follow_symlinks=False,
                    )
                    copyFile(
                        src=from_skeleton,
                        dst=to_skeleton,
                        follow_symlinks=False,
                    )
                    ds_landmark[-1][KEY_VIDS][-1][KEY_LANDMARK].append({
                        KEY_FILE: f"{landmark_skeleton_prefix}.npy",
                        KEY_FACE:  img_annotation[KEY_FACE],
                        KEY_POSE:  img_annotation[KEY_POSE],
                        KEY_LHAND: img_annotation[KEY_LHAND],
                        KEY_RHAND: img_annotation[KEY_RHAND],
                    })
                    ds_skeleton[-1][KEY_VIDS][-1][KEY_SKELETON].append({
                        KEY_FILE: f"{landmark_skeleton_prefix}.jpg",
                        KEY_FACE:  img_annotation[KEY_FACE],
                        KEY_POSE:  img_annotation[KEY_POSE],
                        KEY_LHAND: img_annotation[KEY_LHAND],
                        KEY_RHAND: img_annotation[KEY_RHAND],
                    })
    return (ds_landmark, ds_skeleton)
def main() -> None:
    init_dirs()
    ds_landmark, ds_skeleton= process_videos_ds()
    with open(f"{DS_DIR /"selected_start_end_landmark.json"}", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{DS_DIR /"selected_start_end_skeleton.json"}", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    main()
