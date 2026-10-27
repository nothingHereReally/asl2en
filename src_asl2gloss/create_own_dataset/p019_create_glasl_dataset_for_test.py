from json import dump as writeJson
from mediapipe.python.solutions.holistic import Holistic
from pathlib import Path
from typing import Any
import cv2
import numpy as np


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"glasl_verified"
VIDEO_DIR: Path= DS_DIR /"videos"
LANDMARK_DIR: Path= DS_DIR /"investigate_landmark"
SKELETON_DIR: Path= DS_DIR /"investigate_skeleton"
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
MPH_fph: Holistic= Holistic(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
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
QUANTITY_HAND_LMARK: int= 21
VIDEO_LIST: list= [
    {
        KEY_G: 'book',
        KEY_VIDS: [
            'book_0001.mp4',
            'book_0002.mp4',
            'book_0003.mp4',
            'book_0004.mp4',
            'book_0005.mp4',
            'book_0006.mp4',
            'book_0007.mp4',
            'book_0008.mp4',
            'book_0009.mp4',
            'book_0010.mp4',
        ]
    },
    {
        KEY_G: 'drink',
        KEY_VIDS: [
            'drink_0001.mp4',
            'drink_0002.mp4',
            'drink_0003.mp4',
            'drink_0004.mp4',
            'drink_0005.mp4',
            'drink_0006.mp4',
            'drink_0007.mp4',
            'drink_0008.mp4',
            'drink_0009.mp4',
            'drink_0010.mp4',
        ]
    },
    {
        KEY_G: 'computer',
        KEY_VIDS: [
            'computer_0001.mp4',
            'computer_0002.mp4',
            'computer_0003.mp4',
            'computer_0004.mp4',
            'computer_0005.mp4',
            'computer_0006.mp4',
            'computer_0007.mp4',
            'computer_0008.mp4',
            'computer_0009.mp4',
            'computer_0010.mp4',
        ]
    },
    {
        KEY_G: 'before',
        KEY_VIDS: [
            'before_0001.mp4',
            'before_0002.mp4',
            'before_0003.mp4',
            'before_0004.mp4',
            'before_0005.mp4',
            'before_0006.mp4',
            'before_0007.mp4',
            'before_0008.mp4',
            'before_0009.mp4',
            'before_0010.mp4',
        ]
    },
    {
        KEY_G: 'chair',
        KEY_VIDS: [
            'chair_0001.mp4',
            'chair_0002.mp4',
            'chair_0003.mp4',
            'chair_0004.mp4',
            'chair_0005.mp4',
            'chair_0006.mp4',
            'chair_0007.mp4',
            'chair_0008.mp4',
            'chair_0009.mp4',
            'chair_0010.mp4',
        ]
    },
    {
        KEY_G: 'go',
        KEY_VIDS: [
            'go_0001.mp4',
            'go_0002.mp4',
            'go_0003.mp4',
            'go_0004.mp4',
            'go_0005.mp4',
            'go_0006.mp4',
            'go_0007.mp4',
            'go_0008.mp4',
            'go_0009.mp4',
            'go_0010.mp4',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_VIDS: [
            'clothes_0001.mp4',
            'clothes_0002.mp4',
            'clothes_0003.mp4',
            'clothes_0004.mp4',
            'clothes_0005.mp4',
            'clothes_0006.mp4',
            'clothes_0007.mp4',
            'clothes_0008.mp4',
            'clothes_0009.mp4',
            'clothes_0010.mp4',
        ]
    },
    {
        KEY_G: 'who',
        KEY_VIDS: [
            'who_0001.mp4',
            'who_0002.mp4',
            'who_0003.mp4',
            'who_0004.mp4',
            'who_0005.mp4',
            'who_0006.mp4',
            'who_0007.mp4',
            'who_0008.mp4',
            'who_0009.mp4',
            'who_0010.mp4',
        ]
    },
    {
        KEY_G: 'candy',
        KEY_VIDS: [
            'candy_0001.mp4',
            'candy_0002.mp4',
            'candy_0003.mp4',
            'candy_0004.mp4',
            'candy_0005.mp4',
            'candy_0006.mp4',
            'candy_0007.mp4',
            'candy_0008.mp4',
            'candy_0009.mp4',
            'candy_0010.mp4',
        ]
    },
    {
        KEY_G: 'cousin',
        KEY_VIDS: [
            'cousin_0001.mp4',
            'cousin_0002.mp4',
            'cousin_0003.mp4',
            'cousin_0004.mp4',
            'cousin_0005.mp4',
            'cousin_0006.mp4',
            'cousin_0007.mp4',
            'cousin_0008.mp4',
            'cousin_0009.mp4',
            'cousin_0010.mp4',
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_VIDS: [
            'mine_my_0001.mp4',
            'mine_my_0002.mp4',
            'mine_my_0003.mp4',
            'mine_my_0004.mp4',
            'mine_my_0005.mp4',
            'mine_my_0006.mp4',
            'mine_my_0007.mp4',
            'mine_my_0008.mp4',
            'mine_my_0009.mp4',
            'mine_my_0010.mp4',
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_VIDS: [
            'me_i_0001.mp4',
            'me_i_0002.mp4',
            'me_i_0003.mp4',
            'me_i_0004.mp4',
            'me_i_0005.mp4',
            'me_i_0006.mp4',
            'me_i_0007.mp4',
            'me_i_0008.mp4',
            'me_i_0009.mp4',
            'me_i_0010.mp4',
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_VIDS: [
            'stomach_0001.mp4',
            'stomach_0002.mp4',
            'stomach_0003.mp4',
            'stomach_0004.mp4',
            'stomach_0005.mp4',
            'stomach_0006.mp4',
            'stomach_0007.mp4',
            'stomach_0008.mp4',
            'stomach_0009.mp4',
            'stomach_0010.mp4',
        ]
    },
    {
        KEY_G: 'have',
        KEY_VIDS: [
            'have_0001.mp4',
            'have_0002.mp4',
            'have_0003.mp4',
            'have_0004.mp4',
            'have_0005.mp4',
            'have_0006.mp4',
            'have_0007.mp4',
            'have_0008.mp4',
            'have_0009.mp4',
            'have_0010.mp4',
        ]
    },
    {
        KEY_G: 'need',
        KEY_VIDS: [
            'need_0001.mp4',
            'need_0002.mp4',
            'need_0003.mp4',
            'need_0004.mp4',
            'need_0005.mp4',
            'need_0006.mp4',
            'need_0007.mp4',
            'need_0008.mp4',
            'need_0009.mp4',
            'need_0010.mp4',
        ]
    },
    {
        KEY_G: 'see',
        KEY_VIDS: [
            'see_0001.mp4',
            'see_0002.mp4',
            'see_0003.mp4',
            'see_0004.mp4',
            'see_0005.mp4',
            'see_0006.mp4',
            'see_0007.mp4',
            'see_0008.mp4',
            'see_0009.mp4',
            'see_0010.mp4',
        ]
    },
    {
        KEY_G: 'feel',
        KEY_VIDS: [
            'feel_0001.mp4',
            'feel_0002.mp4',
            'feel_0003.mp4',
            'feel_0004.mp4',
            'feel_0005.mp4',
            'feel_0006.mp4',
            'feel_0007.mp4',
            'feel_0008.mp4',
            'feel_0009.mp4',
            'feel_0010.mp4',
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_VIDS: [
            'hurt_0001.mp4',
            'hurt_0002.mp4',
            'hurt_0003.mp4',
            'hurt_0004.mp4',
            'hurt_0005.mp4',
            'hurt_0006.mp4',
            'hurt_0007.mp4',
            'hurt_0008.mp4',
            'hurt_0009.mp4',
            'hurt_0010.mp4',
        ]
    },
    {
        KEY_G: 'fever',
        KEY_VIDS: [
            'fever_0001.mp4',
            'fever_0002.mp4',
            'fever_0003.mp4',
            'fever_0004.mp4',
            'fever_0005.mp4',
            'fever_0006.mp4',
            'fever_0007.mp4',
            'fever_0008.mp4',
            'fever_0009.mp4',
            'fever_0010.mp4',
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_VIDS: [
            'dizzy_0001.mp4',
            'dizzy_0002.mp4',
            'dizzy_0003.mp4',
            'dizzy_0004.mp4',
            'dizzy_0005.mp4',
            'dizzy_0006.mp4',
            'dizzy_0007.mp4',
            'dizzy_0008.mp4',
            'dizzy_0009.mp4',
            'dizzy_0010.mp4',
        ]
    },
    {
        KEY_G: 'headache',
        KEY_VIDS: [
            'headache_0001.mp4',
            'headache_0002.mp4',
            'headache_0003.mp4',
            'headache_0004.mp4',
            'headache_0005.mp4',
            'headache_0006.mp4',
            'headache_0007.mp4',
            'headache_0008.mp4',
            'headache_0009.mp4',
            'headache_0010.mp4',
        ]
    },
    {
        KEY_G: 'doctor',
        KEY_VIDS: [
            'doctor_0001.mp4',
            'doctor_0002.mp4',
            'doctor_0003.mp4',
            'doctor_0004.mp4',
            'doctor_0005.mp4',
            'doctor_0006.mp4',
            'doctor_0007.mp4',
            'doctor_0008.mp4',
            'doctor_0009.mp4',
            'doctor_0010.mp4',
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
        out_landmark.extend(np.zeros(
            (
                len(WORTHY_FACE_IDX),
                2
            )).tolist()
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
        out_landmark.extend(np.zeros(
            (
                len(WORTHY_POSE_IDX),
                2
            )).tolist())

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
        out_landmark.extend(np.zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            )).tolist())

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
        out_landmark.extend(np.zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            )).tolist())

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
def extract_lm_from_mph(lmark_mph, orig_shape: tuple) -> list[tuple[float, float]]:
    landmark__face_pose_left_right_hand: list= np.zeros((
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +(QUANTITY_HAND_LMARK*2),
        2
    )).tolist()
    if lmark_mph.face_landmarks!=None \
        or lmark_mph.pose_landmarks!=None \
        or lmark_mph.left_hand_landmarks!=None \
        or lmark_mph.right_hand_landmarks!=None:
        landmark__face_pose_left_right_hand= list()
        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0

        # ---- face landmarks ----
        if lmark_mph.face_landmarks != None:
            for idx, el in enumerate(lmark_mph.face_landmarks.landmark):
                if idx in WORTHY_FACE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((len(WORTHY_FACE_IDX), 2)).tolist())

        # ---- pose landmarks ----
        if lmark_mph.pose_landmarks != None:
            for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
                if idx in WORTHY_POSE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((len(WORTHY_POSE_IDX), 2)).tolist())

        # ---- left hand landmarks ----
        if lmark_mph.left_hand_landmarks != None:
            for el in lmark_mph.left_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((QUANTITY_HAND_LMARK, 2)).tolist())

        # ---- right hand landmarks ----
        if lmark_mph.right_hand_landmarks != None:
            for el in lmark_mph.right_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((QUANTITY_HAND_LMARK, 2)).tolist())


        landmark__face_pose_left_right_hand= normalizeWorthyLandmarkWrapper(
            landmark__face_pose_left_right_hand,
            orig_shape,
            {
                'face': lmark_mph.face_landmarks != None,
                'pose': lmark_mph.pose_landmarks != None,
                'left_hand': lmark_mph.left_hand_landmarks != None,
                'right_hand': lmark_mph.right_hand_landmarks != None,
            }
        )
    # return list of length 86, each is tuple of length 2
    return landmark__face_pose_left_right_hand
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
    if LANDMARK_DIR.exists():
        raise NotImplementedError(f"Please delete {LANDMARK_DIR}, this will be created for you.")
    if SKELETON_DIR.exists():
        raise NotImplementedError(f"Please delete {SKELETON_DIR}, this will be created for you.")
    LANDMARK_DIR.mkdir()
    SKELETON_DIR.mkdir()
def has3landmarks(mph_result) -> bool:
    return mph_result.face_landmarks != None and \
           mph_result.pose_landmarks != None and \
           mph_result.left_hand_landmarks != None and \
           mph_result.right_hand_landmarks != None
def extract_lm_from_video(video_filename: str) -> tuple:
    '''
    video_filename, eg. file.mp4
    --------------
    return a tuple of (out_landmarks, out_annotations)
    '''
    video_data: cv2.VideoCapture= cv2.VideoCapture(f"{VIDEO_DIR /video_filename}")
    out_landmarks: list= []
    out_annotations: list= []
    for _ in range(int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))):
        has_img, image= video_data.read()
        if has_img:
            image= cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            mph_result: Any= MPH_fph.process(image)
            norm_landmarks: list[tuple[float, float]]= extract_lm_from_mph(mph_result, orig_shape=image.shape) # list of shape (86, 2)
            if len(out_landmarks) == 0 or has3landmarks(mph_result):
                out_landmarks.append(norm_landmarks)
                out_annotations.append({
                    KEY_FACE: mph_result.face_landmarks != None,
                    KEY_POSE: mph_result.pose_landmarks != None,
                    KEY_LHAND: mph_result.left_hand_landmarks != None,
                    KEY_RHAND: mph_result.right_hand_landmarks != None,
                })
            elif not has3landmarks(mph_result):
                new_landmark: list= []
                new_annotation: dict= {
                    KEY_FACE: False,
                    KEY_POSE: False,
                    KEY_LHAND: False,
                    KEY_RHAND: False,
                }
                if mph_result.face_landmarks == None:
                    new_landmark.extend(get_face(np.array(out_landmarks[-1])).tolist())
                    new_annotation[KEY_FACE]= out_annotations[-1][KEY_FACE]
                else:
                    new_landmark.extend(get_face(np.array(norm_landmarks)).tolist())
                    new_annotation[KEY_FACE]= True

                if mph_result.pose_landmarks == None:
                    new_landmark.extend(get_pose(np.array(out_landmarks[-1])).tolist())
                    new_annotation[KEY_POSE]= out_annotations[-1][KEY_POSE]
                else:
                    new_landmark.extend(get_pose(np.array(norm_landmarks)).tolist())
                    new_annotation[KEY_POSE]= True

                if mph_result.left_hand_landmarks == None:
                    new_landmark.extend(get_left_hand(np.array(out_landmarks[-1])).tolist())
                    new_annotation[KEY_LHAND]= out_annotations[-1][KEY_LHAND]
                else:
                    new_landmark.extend(get_left_hand(np.array(norm_landmarks)).tolist())
                    new_annotation[KEY_LHAND]= True

                if mph_result.right_hand_landmarks == None:
                    new_landmark.extend(get_right_hand(np.array(out_landmarks[-1])).tolist())
                    new_annotation[KEY_RHAND]= out_annotations[-1][KEY_RHAND]
                else:
                    new_landmark.extend(get_right_hand(np.array(norm_landmarks)).tolist())
                    new_annotation[KEY_RHAND]= True
                out_landmarks.append(new_landmark)
                out_annotations.append(new_annotation)

    return (out_landmarks, out_annotations)
def process_videos_ds() -> tuple:
    ds_landmark: list= []
    ds_skeleton: list= []
    for a_gloss in VIDEO_LIST:
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
            video_landmarks, video_annotations= extract_lm_from_video(a_video[KEY_FILE])
            for a_start_end in a_video[KEY_IMG_VALID]:
                pfolder_name: str= f"{a_video[KEY_FILE][:-4]}_{str(a_start_end[KEY_IMGSTART]).zfill(4)}_{str(a_start_end[KEY_IMGEND]).zfill(4)}"
                landmark_abs_folder: Path= LANDMARK_DIR /pfolder_name
                skeleton_abs_folder: Path= SKELETON_DIR /pfolder_name
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
                for idx_init0 in range(a_start_end[KEY_IMGEND] -a_start_end[KEY_IMGSTART]+1):
                    landmark_skeleton_prefix: str= str(idx_init0+1).zfill(8)
                    landmark= video_landmarks[     a_start_end[KEY_IMGSTART]-1   +idx_init0]
                    annotation= video_annotations[ a_start_end[KEY_IMGSTART]-1   +idx_init0]
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
    with open(f"{DS_DIR /"investigate_landmark.json"}", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{DS_DIR /"investigate_skeleton.json"}", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    main()
