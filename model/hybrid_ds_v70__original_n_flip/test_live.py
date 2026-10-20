import cv2 as cv
from keras.src.layers import Layer
from keras.src import ops
from keras.src.saving import register_keras_serializable, load_model
from typing import Any
from mediapipe.python.solutions.holistic import Holistic
import numpy as np
from pathlib import Path
import pygame


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
MPH_fph: Holistic= Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)
IMG_SIZE: int= 800
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
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)
QUANTITY_HAND_LMARK: int= 21
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
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
LM_IDX_LEFT_HAND_F_THUMB: int= 48
LM_IDX_LEFT_HAND_F_INDEX: int= 52
LM_IDX_LEFT_HAND_F_MIDLE: int= 56
LM_IDX_LEFT_HAND_F_RING: int = 60
LM_IDX_LEFT_HAND_F_PINKY: int= 64
# -----------------------------
LM_IDX_RIGHT_HAND_F_THUMB: int= 48
LM_IDX_RIGHT_HAND_F_INDEX: int= 52
LM_IDX_RIGHT_HAND_F_MIDLE: int= 56
LM_IDX_RIGHT_HAND_F_RING: int = 60
LM_IDX_RIGHT_HAND_F_PINKY: int= 64
# -----------------------------
LM_IMPORTANT_FACE_IDX: tuple= (
    LM_IDX_TIP_OF_FOREHEAD, # ok

    LM_IDX_LEFT_MOST_EYE,   # ok
    LM_IDX_RIGHT_MOST_EYE,  # ok

    LM_IDX_LEFT_EAR,        # ok
    LM_IDX_RIGHT_EAR,       # ok

    LM_IDX_UP_TIP_NOSE,     # ok

    LM_IDX_LEFT_MOST_LIP,   # ok
    LM_IDX_RIGHT_MOST_LIP,  # ok
    LM_IDX_TOP_LIP,         # ok
)
# -----------------------------
WORTHY_LM_IMPORTANT_IDX: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_UP_TIP_NOSE,
    # -- on later process here to insert up and left body dots
    LM_IDX_LEFT_ARM_TIP,
    LM_IDX_RIGHT_ARM_TIP,
)
LM_LEFT_HAND_TIP: tuple= (
    LM_IDX_LEFT_HAND_F_THUMB,
    LM_IDX_LEFT_HAND_F_INDEX,
    LM_IDX_LEFT_HAND_F_MIDLE,
    LM_IDX_LEFT_HAND_F_RING,
    LM_IDX_LEFT_HAND_F_PINKY,
)
LM_LEFT_HAND_NOT_TIP: tuple= (
    44, 45, 46, 47,
    49, 50, 51,
    53, 54, 55,
    57, 58, 59,
    61, 62, 63,
)
LM_RIGHT_HAND_TIP: tuple= (
    LM_IDX_RIGHT_HAND_F_THUMB,
    LM_IDX_RIGHT_HAND_F_INDEX,
    LM_IDX_RIGHT_HAND_F_MIDLE,
    LM_IDX_RIGHT_HAND_F_RING,
    LM_IDX_RIGHT_HAND_F_PINKY,
)
LM_RIGHT_HAND_NOT_TIP: tuple= (
    65, 66, 67, 68,
    70, 71, 72,
    74, 75, 76,
    78, 79, 80,
    82, 83, 84,
)
@register_keras_serializable()
class SelectLM(Layer):
    def __init__(self, name: str='select49', **kwargs):
        super().__init__(**kwargs)
        self.name= name
    def build(self, input_shape):
        del input_shape
    def call(self, input):
        # left_hand_tip: Any= input[..., LM_LEFT_HAND_TIP, :]
        # right_hand_tip: Any= input[..., LM_RIGHT_HAND_TIP, :]
        left_hand_tip: Any= ops.take(
            input,
            indices=LM_LEFT_HAND_TIP,
            axis=-2,
        )
        right_hand_tip: Any= ops.take(
            input,
            indices=LM_RIGHT_HAND_TIP,
            axis=-2,
        )
        left_hand_tip= ops.expand_dims(left_hand_tip, axis=-3)
        right_hand_tip= ops.expand_dims(right_hand_tip, axis=-3)
        the_rest= ops.concatenate([
            # input[..., LM_IMPORTANT_FACE_IDX, :],
            ops.take(
                input,
                indices=LM_IMPORTANT_FACE_IDX,
                axis=-2,
            ),
            input[..., 36:36+8, :],
            # input[..., LM_LEFT_HAND_NOT_TIP, :],
            ops.take(
                input,
                indices=LM_LEFT_HAND_NOT_TIP,
                axis=-2,
            ),
            # input[..., LM_RIGHT_HAND_NOT_TIP, :],
            ops.take(
                input,
                indices=LM_RIGHT_HAND_NOT_TIP,
                axis=-2,
            ),
        ], axis=-2)
        the_rest= ops.expand_dims(the_rest, axis=-2)

        left_hand_tip= (left_hand_tip - the_rest)**2
        left_hand_tip= ops.sum(left_hand_tip, axis=-1, keepdims=False)
        left_hand_tip= left_hand_tip**0.5
        shape: Any= ops.shape(left_hand_tip)
        left_hand_tip= ops.reshape(left_hand_tip, (-1, shape[1], shape[2]*shape[3]))

        right_hand_tip= (right_hand_tip - the_rest)**2
        right_hand_tip= ops.sum(right_hand_tip, axis=-1, keepdims=False)
        right_hand_tip= right_hand_tip**0.5
        shape: Any= ops.shape(right_hand_tip)
        right_hand_tip= ops.reshape(right_hand_tip, (-1, shape[1], shape[2]*shape[3]))

        out= ops.concatenate([
            left_hand_tip,
            right_hand_tip,
        ], axis=-1)
        return out
        # out= ops.concatenate([
        #     input[..., 44:, :], # -- left and right hand
        #     input[..., 36:36+8, :], # -- pose landmarks
        #     ops.take(
        #         input,
        #         indices=LM_IMPORTANT_FACE_IDX,
        #         axis=-2,
        #     ),
        # ], axis=-2)
        # center_shoulder= (
        #     input[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
        #     input[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
        # )/2.0
        # center_hip= (
        #     input[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
        #     input[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
        # )/2.0
        # center_shoulder_hip= (center_shoulder + center_hip)/2.0
        # out= ops.concatenate([
        #     out,
        #     (center_shoulder + center_shoulder_hip)/2.0,
        #     (center_hip + center_shoulder_hip)/2.0,
        # ], axis=-2)
        # shape: Any= ops.shape(out)
        # out= ops.reshape(out, (-1, shape[1], shape[2]*shape[3]))
        # return out
        # out= input[..., 44:, :]
        # shape: Any= ops.shape(out)
        # out= ops.reshape(out, (-1, shape[1], shape[2]*shape[3]))
        # return out
    def get_config(self):
        return {
            **super().get_config(),
            "name":      self.name,
        }
MODEL: Any= load_model(f"{PROJ_ROOT /"model" /"hybrid_ds_v70__original_n_flip" /"aslvid2gloss_v70.keras"}")


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
    for idx, el in enumerate(lmark_mph.face_landmarks.landmark):
        if idx in WORTHY_FACE_IDX:
            landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            landmark__face_pose_left_right_hand
        ),
        connections_idxs=FACE_CONNECTIONS,
        thick=10,
        color_dot=None,
        color_line=(0, 153, 0), # 153/255= 0.6
    )

    # ---- pose landmarks ----
    for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
        if idx in WORTHY_POSE_IDX:
            landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            landmark__face_pose_left_right_hand[-1*len(WORTHY_POSE_IDX):]
        ),
        connections_idxs=POSE_CONNECTIONS,
        thick=10,
        color_dot=None,
        color_line=(0, 0, 153), # 153/255= 0.6
    )

    # ---- left hand landmarks ----
    for el in lmark_mph.left_hand_landmarks.landmark:
        landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            [(el.x, el.y) for el in lmark_mph.left_hand_landmarks.landmark]
        ),
        connections_idxs=HAND_CONNECTIONS,
        thick=10,
        color_dot=None,
        color_line=(255, 255, 255)
    )

    # ---- right hand landmarks ----
    for el in lmark_mph.right_hand_landmarks.landmark:
        landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks_force_0_to_1(
            [(el.x, el.y) for el in lmark_mph.right_hand_landmarks.landmark]
        ),
        connections_idxs=HAND_CONNECTIONS,
        thick=10,
        color_dot=None,
        color_line=(153, 204, 204), # 204/255= 0.8
    )




    landmark__face_pose_left_right_hand= normalizeLandmarks(
        landmarks=landmark__face_pose_left_right_hand,
        original_shape=orig_shape
    )


    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return (img_write_to, landmark__face_pose_left_right_hand)
def has4parts_landmark(results) -> bool:
    return results.face_landmarks != None and \
           results.pose_landmarks != None and \
           results.left_hand_landmarks != None and \
           results.right_hand_landmarks != None
def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w= image.shape[:2]
    scale= target_size /max(h, w)
    return cv.resize(image, (int(w * scale), int(h * scale)))
def main() -> None:
    labels: list= [
        'book',
        'drink',
        'computer',
        'before',
        'chair',
        'go',
        'clothes',
        'who',
        'candy',
        'cousin',
        'mine_my',
        'me_i',
        'stomach',
        'have',
        'need',
        'see',
        'feel',
        'hurt',
        'fever',
        'dizzy',
        'headache',
        'doctor',
    ]

    cap: cv.VideoCapture= cv.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    pygame.init() # init pygame display
    screen= pygame.display.set_mode((IMG_SIZE, IMG_SIZE))
    pygame.display.set_caption("ASL Recognition")

    landmark_left_right_hands: list= []
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
                    landmark_left_right_hands.append(np.array(landmarks))
                    if len(landmark_left_right_hands) >= QUANTITY_FRAME:
                        recent_lm_left_right= np.array(landmark_left_right_hands[-QUANTITY_FRAME:])
                        pred_result: np.ndarray= np.array(MODEL.predict(
                            x=np.expand_dims(recent_lm_left_right, axis=0),
                            batch_size=1,
                        )[0])
                        top_index: list= np.argsort(pred_result, axis=0).tolist()
                        print("Top 5 Sign Classifications:")
                        for i in [-1, -2, -3, -4, -5]:
                            idx= top_index[i]
                            print(f"    {i+1}. {labels[idx]} (score: {pred_result[idx]:.4f})")
                        print()
                        landmark_left_right_hands= []
            else:
                no_landmark_count += 1
                img_with_skeleton= frame

                if no_landmark_count >= 3:
                    print("No landmarks detected. Please slow down. Or there is no human being in front")
                    no_landmark_count= 0

            # Display the image with skeleton
            display_frame= cv.cvtColor(img_with_skeleton, cv.COLOR_BGR2RGB)
            display_frame= cv.flip( # flip to mirror, left be right, right be left
                resize_image(display_frame, IMG_SIZE),
                1
            )
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
    # print(PROJ_ROOT)
