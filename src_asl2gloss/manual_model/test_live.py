import cv2 as cv
import jax.numpy as jnp
from json import load as loadJson
from mediapipe.python.solutions.holistic import Holistic
import numpy as np
from pathlib import Path
import pygame


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
MODEL_DIR: Path= PROJ_ROOT /"src_asl2gloss" /"manual_model" /"model"
MPH_fph: Holistic= Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.68,
    min_tracking_confidence=0.68,
)
IMG_SIZE: int= 1_000
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
QUANTITY_FRAME: int= 8
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
WORTHY_LM_IMPORTANT_IDX: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_UP_TIP_NOSE,
    # -- on later process here to insert up and left body dots
    LM_IDX_LEFT_ARM_TIP,
    LM_IDX_RIGHT_ARM_TIP,
)


def load_model() -> tuple:
    with open(f"{MODEL_DIR /"ds_samples.npy"}", 'rb') as f:
        ds_samples: np.ndarray= np.load(f)
    with open(f"{MODEL_DIR /"mask.npy"}", 'rb') as f:
        mask: np.ndarray= np.load(f)
    with open(f"{MODEL_DIR /"body2dots.npy"}", 'rb') as f:
        body2dots: np.ndarray= np.load(f)
    with open(f"{MODEL_DIR /"weights.npy"}", 'rb') as f:
        weights: np.ndarray= np.load(f)
    with open(f"{MODEL_DIR /"labels.json"}", 'r') as f:
        labels: list[str]= loadJson(f)
    return (
        ds_samples,
        mask,
        body2dots,
        weights,
        labels,
    )
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
                    cv.circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    cv.circle(
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
                    cv.line(
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
def landmarks_force_0_to_1(landmarks: list) -> list:
    return [
        (
            max(0.0, min(x, 1.0)),
            max(0.0, min(y, 1.0)),
        ) for x, y in landmarks
    ]
def drawFacePoseHand(img_write_to: np.ndarray, lmark_mph, orig_shape: tuple) -> tuple:
    assert lmark_mph.face_landmarks!=None
    assert lmark_mph.pose_landmarks!=None
    assert lmark_mph.left_hand_landmarks!=None
    assert lmark_mph.right_hand_landmarks!=None


    landmark__face_pose_left_right_hand: list= list()
    # ---- face landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            [(el.x, el.y) for el in lmark_mph.face_landmarks.landmark]
        ),
        connections_idxs=FACE_CONNECTIONS,
        thick=1,
        color_dot=None,
        color_line=(0, 153, 0), # 153/255= 0.6
    )
    for idx, el in enumerate(lmark_mph.face_landmarks.landmark):
        if idx in WORTHY_FACE_IDX:
            landmark__face_pose_left_right_hand.append((  el.x, el.y  ))

    # ---- pose landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            [(el.x, el.y) for el in lmark_mph.pose_landmarks.landmark]
        ),
        connections_idxs=POSE_CONNECTIONS,
        thick=1,
        color_dot=None,
        color_line=(0, 0, 153), # 153/255= 0.6
    )
    for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
        if idx in WORTHY_POSE_IDX:
            landmark__face_pose_left_right_hand.append((  el.x, el.y  ))

    # ---- left hand landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            [(el.x, el.y) for el in lmark_mph.left_hand_landmarks.landmark]
        ),
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=None,
        color_line=(255, 255, 255)
    )
    for el in lmark_mph.left_hand_landmarks.landmark:
        landmark__face_pose_left_right_hand.append((  el.x, el.y  ))

    # ---- right hand landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            [(el.x, el.y) for el in lmark_mph.right_hand_landmarks.landmark]
        ),
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=None,
        color_line=(153, 204, 204), # 204/255= 0.8
    )
    for el in lmark_mph.right_hand_landmarks.landmark:
        landmark__face_pose_left_right_hand.append((  el.x, el.y  ))




    landmark__face_pose_left_right_hand= normalizeLandmarks(
        landmarks=landmark__face_pose_left_right_hand,
        original_shape=orig_shape
    )


    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return (img_write_to, landmark__face_pose_left_right_hand)
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
def get_worthy5landmarks(input: jnp.ndarray) -> jnp.ndarray:
    '''
    input can be of shape (..., 86, 2) or (..., 86)
    ------------------------------
    if input shape is (..., 86, 2)
    then returned shape is (..., 5, 2)
    else returned shape is (..., 5)
    -------------------------------
    meaning does not include 2 dots
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
    input -- shape of ( BATCH_SIZE,              1,                1, QUANTITY_FRAME, 86, 2)
    model -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 7)
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
    input_7landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy5landmarks(input),
        get2dots_frombody(input),
    ], axis=-2) # shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7, 2)
    input_7landmarks= jnp.concatenate([
        input_7landmarks[..., :3,  :],
        input_7landmarks[..., 5:,  :], # body2dots
        input_7landmarks[..., 3:5, :], # for left and right arm tip
    ], axis=-2)
    # ----------------------------------------------
    model_lhand_index: jnp.ndarray= jnp.expand_dims(
        model[..., LM_IDX_LEFT_INDEXF_TIP, :],
        axis=-2
    )  # shape (..., 1, 2)
    model_rhand_index: jnp.ndarray= jnp.expand_dims(
        model[..., LM_IDX_RIGHT_INDEXF_TIP, :],
        axis=-2
    ) # shape (..., 1, 2)
    model_7landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy5landmarks(model),
        model2dots,
    ], axis=-2) # shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7, 2)
    model_7landmarks= jnp.concatenate([
        model_7landmarks[..., :3,  :],
        model_7landmarks[..., 5:,  :], # body2dots
        model_7landmarks[..., 3:5, :], # for left and right arm tip
    ], axis=-2)
    '''
    below are
    ---------
    now do difference/subtraction on left/right hand on 7 landmarks
    then multiply to weights
    then multiply to mask4left/right hand
    -------------------------------------
    distance( c^2= a^2 +b^2 ) on left/right hand( index finger tip only ) to 7 landmarks chosen
    '''
    distance_input_lhand: jnp.ndarray= (input_7landmarks -input_lhand_index)**2
    distance_input_rhand: jnp.ndarray= (input_7landmarks -input_rhand_index)**2
    distance_model_lhand: jnp.ndarray= (model_7landmarks -model_lhand_index)**2
    distance_model_rhand: jnp.ndarray= (model_7landmarks -model_rhand_index)**2
    '''
    distance_input_lhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7, 2)
    distance_input_rhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7, 2)
    distance_model_lhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7, 2)
    distance_model_rhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7, 2)
    '''
    distance_input_lhand= jnp.sum(distance_input_lhand, axis=-1, keepdims=False)
    distance_input_rhand= jnp.sum(distance_input_rhand, axis=-1, keepdims=False)
    distance_model_lhand= jnp.sum(distance_model_lhand, axis=-1, keepdims=False)
    distance_model_rhand= jnp.sum(distance_model_rhand, axis=-1, keepdims=False)
    '''
    distance_input_lhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7)
    distance_input_rhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7)
    distance_model_lhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    distance_model_rhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    '''
    model_input_lhand: jnp.ndarray= (distance_input_lhand -distance_model_lhand)**2
    model_input_rhand: jnp.ndarray= (distance_input_rhand -distance_model_rhand)**2
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    weights, shape of          (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 2, 7)
    mask, shape of             (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 2)
    '''
    model_input_lhand= model_input_lhand *weights[..., 0, :] *mask[..., 0:1]
    model_input_rhand= model_input_rhand *weights[..., 1, :] *mask[..., 1:2]
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    '''
    model_input_lhand= mean_ignore_zero(
        input=jnp.stack([ # jnp.stack(...) gives shape of (..., 7, 2)
            model_input_lhand,
            model_input_rhand,
        ], axis=-1),
        keepdims=False,
    ) # result shape (..., QUANTITY_FRAME, 7)
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
    weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 7)
    -- on weights axis=-2 of value 2, for left and right hand repsectively
    -- on weights axis=-1 of value 7 for 7 landmarks
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
def sort_score(score: jnp.ndarray) -> list:
    '''
    score has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    ----------
    return is a list of shape (BATCH_SIZE, QUANTITY_GLOSS)
    '''
    result: jnp.ndarray= jnp.min(score, axis=-1)
    return jnp.argsort(result, axis=-1, stable=True).tolist()
def has4parts_landmark(results) -> bool:
    return results.face_landmarks != None and \
           results.pose_landmarks != None and \
           results.left_hand_landmarks != None and \
           results.right_hand_landmarks != None
def main() -> None:
    ds_samples, mask, body2dots, weights, labels= load_model()
    ds_samples= jnp.array(ds_samples)
    mask= jnp.array(mask)
    body2dots= jnp.array(body2dots)
    weights= jnp.array(weights)
    labels= list(labels)

    cap: cv.VideoCapture= cv.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    pygame.init() # init pygame display
    screen= pygame.display.set_mode((IMG_SIZE, IMG_SIZE))
    pygame.display.set_caption("ASL Recognition")

    landmark_buffer: list= []
    no_landmark_count: int= 0
    count_frames_skip: int= 0
    try:
        while True:
            ret, frame= cap.read()
            if not ret:
                break

            count_frames_skip+= 1
            rgb_frame= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results= MPH_fph.process(rgb_frame)

            if has4parts_landmark(results):
                img_with_skeleton, landmarks= drawFacePoseHand(frame, results, frame.shape[:2])
                no_landmark_count= 0

                if count_frames_skip >= 5: # 208ms per frame for a 24FPS webcam
                    count_frames_skip= 0
                    landmark_buffer.append(np.array(landmarks))
                    if len(landmark_buffer) >= QUANTITY_FRAME:
                        recent_landmarks= np.array(landmark_buffer[-QUANTITY_FRAME:])
                        score= comparison_score(
                            input=np.expand_dims(recent_landmarks, axis=0),
                            model=ds_samples,
                            mask=mask,
                            body2dots=body2dots,
                            weights=weights
                        ) # returns jnp.ndarray shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
                        top_indices= sort_score(score)[0] # list of shape (BATCH_SIZE, QUANTITY_GLOSS) then [0]
                        print("Top 5 Sign Classifications:")
                        for i in range(5):
                            idx= top_indices[i]
                            print(f"{i+1}. {labels[idx]} (score: {min(score[0][idx]):.4f})")
                        landmark_buffer= []
            else:
                no_landmark_count += 1
                img_with_skeleton= frame

                if no_landmark_count >= 3:
                    print("No landmarks detected. Please slow down. Or there is no human being in front")
                    no_landmark_count= 0

            # Display the image with skeleton
            display_frame= cv.cvtColor(img_with_skeleton, cv.COLOR_BGR2RGB)
            pygame_surface= pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
            screen.blit(pygame_surface, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Check quit event
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cap.release()
        pygame.quit()

if __name__=="__main__":
    main()
