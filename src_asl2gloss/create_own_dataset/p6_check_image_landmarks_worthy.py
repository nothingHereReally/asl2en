from cv2 import COLOR_BGR2RGB, circle, cvtColor, imread, imwrite, line
from mediapipe.python.solutions.holistic import Holistic
from numpy import array, full, ndarray, uint8
from os import makedirs
from os.path import exists
from pathlib import Path
from random import uniform
from tempfile import gettempdir


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
WHOLE_BODY_FILE_PATH: Path= PROJ_ROOT /"src_asl2gloss" /"pics_others" /"human_whole_body.png"
LANDMARK_Q_FACE_FULL: int= 468
LANDMARK_Q_FACE_WORTHY: int= 36
LANDMARK_Q_POSE_FULL: int= 33
LANDMARK_Q_POSE_WORTHY: int= 8
LANDMARK_Q_EACH_HAND: int= 21
# IMG_SIZE: int= 158
IMG_SIZE: int= 1500
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




MPH_fph: Holistic= Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)




def isOKplt(coord: tuple[float, float]) -> bool:
    '''
    x and y coordinates
    mandatory be greater than or equal to Zero
    and less than or equal to One
    '''
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: ndarray, \
                    lmark_coordinates: tuple, \
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
            if isOKplt(pA) and isOKplt(pB):
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
                raise ValueError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0")
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
    xs= list(filter(lambda el: el!=0, xs))
    ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    min_x, min_y=    min(xs), min(ys)
    max_x, max_y=    max(xs), max(ys)
    scale: float= (1  -2*pad)/max(
        max_x -min_x,
        max_y -min_y
    )
    return [(
        (x -min_x)    *scale    +pad  if x!=0 else x,
        (y -min_y)    *scale    +pad  if y!=0 else y
    ) for x, y in landmarks]
def part4_centerLandmarkVerticallyHorizontally(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 3) center landmark with same aspect ratio as original
    # center horizontally and vertically, since done padding then just
    # move to right/down
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    xs= list(filter(lambda el: el!=0, xs))
    ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    shift_x: float=  0.5    -(min(xs) +max(xs))  /2
    shift_y: float=  0.5    -(min(ys) +max(ys))  /2

    return [(
        x +shift_x  if x!=0 else x,
        y +shift_y  if y!=0 else y
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
def drawFacePoseHand(lmark_mph, original_shape: tuple) -> tuple:
    def recalcDrawFace_full_dots(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_face,
            connections_idxs=FACE_CONNECTIONS_FULL,
            thick=6,
            color_line=None,
            color_dot=(0, 153, 0), # 153/255= 0.6
            # color_line=(255, 255, 255),
            # color_dot=(0, 153, 0), # 153/255= 0.6
        )
    def recalcDrawPose_full_dots(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_pose,
            connections_idxs=POSE_CONNECTIONS_FULL,
            thick=6,
            color_line=None,
            color_dot=(0, 0, 153), # 153/255= 0.6
            # color_line=(255, 255, 255),
            # color_dot=(0, 0, 153), # 153/255= 0.6
        )
    def recalcDrawFace_worthy_dots(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_face,
            connections_idxs=FACE_CONNECTIONS,
            thick=6,
            color_line=None,
            color_dot=(0, 153, 0),  # 153/255= 0.6
            # color_line=(255, 255, 255),
            # color_dot=(0, 153, 0),  # 153/255= 0.6
        )
    def recalcDrawPose_worthy_dots(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_pose,
            connections_idxs=POSE_CONNECTIONS,
            thick=6,
            color_line=None,
            color_dot=(0, 0, 153),  # 153/255= 0.6
            # color_line=(255, 255, 255),
            # color_dot=(0, 0, 153),  # 153/255= 0.6
        )
    def recalcDrawLeftHand_dots(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_lhand,
            connections_idxs=HAND_CONNECTIONS,
            thick=6,
            color_line=None,
            color_dot=(37, 0, 80),
            # color_line=(255, 255, 255),
            # color_dot=(37, 0, 80),
        )
    def recalcDrawRightHand_dots(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_rhand,
            connections_idxs=HAND_CONNECTIONS,
            thick=6,
            color_line=None,
            color_dot=(12, 84, 84),  # 204/255= 0.8
            # color_line=(255, 255, 255),
            # color_dot=(12, 84, 84),  # 204/255= 0.8
        )


    def recalcDrawFace_full_lines(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_face,
            connections_idxs=FACE_CONNECTIONS_FULL,
            thick=6,
            color_line=(0, 153, 0),  # 153/255= 0.6
            color_dot=None,
            # color_line=(0, 153, 0),  # 153/255= 0.6
        )
    def recalcDrawPose_full_lines(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_pose,
            connections_idxs=POSE_CONNECTIONS_FULL,
            thick=6,
            color_line=(0, 0, 153),  # 153/255= 0.6
            color_dot=None,
            # color_line=(0, 0, 153),  # 153/255= 0.6
        )
    def recalcDrawFace_worthy_lines(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_face,
            connections_idxs=FACE_CONNECTIONS,
            thick=6,
            color_line=(0, 153, 0),  # 153/255= 0.6
            color_dot=None,
            # color_line=(0, 153, 0),  # 153/255= 0.6
        )
    def recalcDrawPose_worthy_lines(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_pose,
            connections_idxs=POSE_CONNECTIONS,
            thick=6,
            color_line=(0, 0, 153),  # 153/255= 0.6
            color_dot=None,
            # color_line=(0, 0, 153),  # 153/255= 0.6
        )
    def recalcDrawLeftHand_lines(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_lhand,
            connections_idxs=HAND_CONNECTIONS,
            thick=6,
            color_line=(12, 155, 140),
            color_dot=None,
            # color_line=(12, 155, 140),
        )
    def recalcDrawRightHand_lines(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        return drawSkeletonImg(
            image=img_orig,
            lmark_coordinates=lmark_rhand,
            connections_idxs=HAND_CONNECTIONS,
            thick=6,
            color_line=(12, 230, 12),  # 204/255= 0.8
            color_dot=None,
            # color_line=(12, 230, 12),  # 204/255= 0.8
        )


    # ---- face pose left_hand right_hand full ----
    #  1  ---- image face,pose,left_hand,right_hand full dots
    image__facePoseLeftHandRightHand_full_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    #  2  ---- image face,pose,left_hand,right_hand full lines
    image__facePoseLeftHandRightHand_full_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- face pose left_hand right_hand worthy ----
    #  3  ---- image face,pose,left_hand,right_hand worthy dots
    image__facePoseLeftHandRightHand_worthy_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    #  4  ---- image face,pose,left_hand,right_hand worthy lines
    image__facePoseLeftHandRightHand_worthy_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- face full ----
    #  5  ---- image face full dots
    image__face_full_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    #  6  ---- image face full lines
    image__face_full_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- face worthy ----
    #  7  ---- image face worthy dots
    image__face_worthy_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    #  8 ---- image face worthy lines
    image__face_worthy_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- pose full ----
    #  9 ---- image pose full dots
    image__pose_full_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 10 ---- image pose full lines
    image__pose_full_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- pose worthy ----
    # 11 ---- image pose worthy dots
    image__pose_worthy_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 12 ---- image pose worthy lines
    image__pose_worthy_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- left hand ----
    # 13 ---- image left hand dots
    image__left_hand_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 14 ---- image left hand lines
    image__left_hand_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- right hand ----
    # 15 ---- image right hand dots
    image__right_hand_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 16 ---- image right hand lines
    image__right_hand_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)


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


    if lmark_mph.face_landmarks!=None or \
        lmark_mph.pose_landmarks!=None or \
        lmark_mph.left_hand_landmarks!=None or \
        lmark_mph.right_hand_landmarks!=None:

        tmp__face_full: list= []
        tmp__pose_full: list= []
        tmp__left_hand_full: list= []
        tmp__right_hand_full: list= []

        tmp__face_worthy: list= []
        tmp__pose_worthy: list= []

        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0
        if lmark_mph.face_landmarks != None:
            for idx, el in enumerate(lmark_mph.face_landmarks.landmark):
                tmp__face_full.append((el.x, el.y))
                if idx in WORTHY_FACE_IDX:
                    tmp__face_worthy.append((el.x, el.y))
        else:
            raise ValueError("Please provide an image file where the Face is clearly visible")
        if lmark_mph.pose_landmarks != None:
            for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
                tmp__pose_full.append((el.x, el.y))
                if idx in WORTHY_POSE_IDX:
                    tmp__pose_worthy.append((el.x, el.y))
        else:
            raise ValueError("Please provide an image file where the Body is clearly visible")
        if lmark_mph.left_hand_landmarks != None:
            for el in lmark_mph.left_hand_landmarks.landmark:
                tmp__left_hand_full.append((el.x, el.y))
        else:
            raise ValueError("Please provide an image file where the Left Hand is clearly visible")
        if lmark_mph.right_hand_landmarks != None:
            for el in lmark_mph.right_hand_landmarks.landmark:
                tmp__right_hand_full.append((el.x, el.y))
        else:
            raise ValueError("Please provide an image file where the Right Hand is clearly visible")


        original_face_full: tuple= tuple((
            el[0],
            el[1]
        ) for el in tmp__face_full )
        original_pose_full: tuple= tuple((
            el[0],
            el[1]
        ) for el in tmp__pose_full)
        original_left_hand_full: tuple= tuple((
            el[0],
            el[1]
        ) for el in tmp__left_hand_full)
        original_right_hand_full: tuple= tuple((
            el[0],
            el[1]
        ) for el in tmp__right_hand_full)
        # --------------------------------------------
        original_face_worthy: tuple= tuple((
            el[0],
            el[1]
        ) for el in tmp__face_worthy)
        original_pose_worthy: tuple= tuple((
            el[0],
            el[1]
        ) for el in tmp__pose_worthy)
        del tmp__face_full
        del tmp__pose_full
        del tmp__left_hand_full
        del tmp__right_hand_full

        del tmp__face_worthy
        del tmp__pose_worthy


        # 18 main pics
        # ---------------------------------------------------------------
        # 1  ---- image face,pose,left_hand,right_hand full dots
        # 2  ---- image face,pose,left_hand,right_hand full lines

        # 3  ---- image face,pose,left_hand,right_hand worthy dots
        # 4  ---- image face,pose,left_hand,right_hand worthy lines

        # 5  ---- image face full dots
        # 6  ---- image face full lines

        # 7  ---- image face worthy dots
        # 8 ---- image face worthy lines

        # 9 ---- image pose full dots
        # 10 ---- image pose full lines

        # 11 ---- image pose worthy dots
        # 12 ---- image pose worthy lines

        # 13 ---- image left hand worthy dots
        # 14 ---- image left hand worthy lines

        # 15 ---- image right hand worthy dots
        # 16 ---- image right hand worthy lines


        # --------------------------------
        # -- landmarks transformation( from what to how many )
        # 468 --> 36 landmarks on face
        # 33  --> 8  landmarks on pose
        # 21  --> 21 landmarks on left hand
        # 21  --> 21 landmarks on right hand
        # --------------------------------
        # ---- face pose left_hand right_hand full ----
        #  1  ---- landmark face,pose,left_hand,right_hand full dots
        landmark__facePoseLeftHandRightHand_full_dots: tuple= tuple(normalizeLandmarks([
            *original_face_full,
            *original_pose_full,
            *original_left_hand_full,
            *original_right_hand_full
        ], original_shape=original_shape[:2]))

        #  2  ---- landmark face,pose,left_hand,right_hand full lines
        landmark__facePoseLeftHandRightHand_full_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__facePoseLeftHandRightHand_full_dots
        )

        # ---- face pose left_hand right_hand worthy ----
        #  3  ---- landmark face,pose,left_hand,right_hand worthy dots
        landmark__facePoseLeftHandRightHand_worthy_dots: tuple= tuple(normalizeLandmarks([
            *original_face_worthy,
            *original_pose_worthy,
            *original_left_hand_full,
            *original_right_hand_full
        ], original_shape=original_shape[:2]))

        #  4  ---- landmark face,pose,left_hand,right_hand worthy lines
        landmark__facePoseLeftHandRightHand_worthy_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__facePoseLeftHandRightHand_worthy_dots
        )

        # ---- face full ----
        #  5  ---- landmark face full dots
        landmark__face_full_dots: tuple= tuple(normalizeLandmarks([
            *original_face_full
        ], original_shape=original_shape[:2]))

        #  6  ---- landmark face full lines
        landmark__face_full_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__face_full_dots
        )

        # ---- face worthy ----
        #  7  ---- landmark face worthy dots
        landmark__face_worthy_dots: tuple= tuple(normalizeLandmarks([
            *original_face_worthy
        ], original_shape=original_shape[:2]))

        #  8 ---- landmark face worthy lines
        landmark__face_worthy_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__face_worthy_dots
        )

        # ---- pose full ----
        #  9 ---- landmark pose full dots
        landmark__pose_full_dots: tuple= tuple(normalizeLandmarks([
            *original_pose_full
        ], original_shape=original_shape[:2]))

        # 10 ---- landmark pose full lines
        landmark__pose_full_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__pose_full_dots
        )

        # ---- pose worthy ----
        # 11 ---- landmark pose worthy dots
        landmark__pose_worthy_dots: tuple= tuple(normalizeLandmarks([
            *original_pose_worthy
        ], original_shape=original_shape[:2]))

        # 12 ---- landmark pose worthy lines
        landmark__pose_worthy_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__pose_worthy_dots
        )

        # ---- left hand ----
        # 13 ---- landmark left hand dots
        landmark__left_hand_dots: tuple= tuple(normalizeLandmarks([
            *original_left_hand_full
        ], original_shape=original_shape[:2]))

        # 14 ---- landmark left hand lines
        landmark__left_hand_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__left_hand_dots
        )

        # ---- right hand ----
        # 15 ---- landmark right hand dots
        landmark__right_hand_dots: tuple= tuple(normalizeLandmarks([
            *original_right_hand_full
        ], original_shape=original_shape[:2]))

        # 16 ---- landmark right hand lines
        landmark__right_hand_lines: tuple= tuple(
            (el[0], el[1]) for el in landmark__right_hand_dots
        )


        # ---- face lanmark if exist ----
        if lmark_mph.face_landmarks != None:
            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawFace_full_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__facePoseLeftHandRightHand_full_dots[:LANDMARK_Q_FACE_FULL]
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawFace_full_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__facePoseLeftHandRightHand_full_lines[:LANDMARK_Q_FACE_FULL]
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_worthy_dots= recalcDrawFace_worthy_dots(
                image__facePoseLeftHandRightHand_worthy_dots,
                landmark__facePoseLeftHandRightHand_worthy_dots[:LANDMARK_Q_FACE_WORTHY]
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_worthy_lines= recalcDrawFace_worthy_lines(
                image__facePoseLeftHandRightHand_worthy_lines,
                landmark__facePoseLeftHandRightHand_worthy_lines[:LANDMARK_Q_FACE_WORTHY]
            )

            # 5  ---- image face full dots
            image__face_full_dots= recalcDrawFace_full_dots(
                image__face_full_dots,
                landmark__face_full_dots
            )
            # 6  ---- image face full lines
            image__face_full_lines= recalcDrawFace_full_lines(
                image__face_full_lines,
                landmark__face_full_lines
            )

            # 7  ---- image face worthy dots
            image__face_worthy_dots= recalcDrawFace_worthy_dots(
                image__face_worthy_dots,
                landmark__face_worthy_dots
            )
            # 8 ---- image face worthy lines
            image__face_worthy_lines= recalcDrawFace_worthy_lines(
                image__face_worthy_lines,
                landmark__face_worthy_lines
            )
        # ---- pose lanmark if exist ----
        if lmark_mph.pose_landmarks != None:
            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawPose_full_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__facePoseLeftHandRightHand_full_dots[LANDMARK_Q_FACE_FULL:LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL]
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawPose_full_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__facePoseLeftHandRightHand_full_lines[LANDMARK_Q_FACE_FULL:LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL]
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_worthy_dots= recalcDrawPose_worthy_dots(
                image__facePoseLeftHandRightHand_worthy_dots,
                landmark__facePoseLeftHandRightHand_worthy_dots[LANDMARK_Q_FACE_WORTHY:LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY]
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_worthy_lines= recalcDrawPose_worthy_lines(
                image__facePoseLeftHandRightHand_worthy_lines,
                landmark__facePoseLeftHandRightHand_worthy_lines[LANDMARK_Q_FACE_WORTHY:LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY]
            )


            # 11 ---- image pose full dots
            image__pose_full_dots= recalcDrawPose_full_dots(
                image__pose_full_dots,
                landmark__pose_full_dots
            )
            # 12 ---- image pose full lines
            image__pose_full_lines= recalcDrawPose_full_lines(
                image__pose_full_lines,
                landmark__pose_full_lines
            )

            # 13 ---- image pose worthy dots
            image__pose_worthy_dots= recalcDrawPose_worthy_dots(
                image__pose_worthy_dots,
                landmark__pose_worthy_dots
            )
            # 14 ---- image pose worthy lines
            image__pose_worthy_lines= recalcDrawPose_worthy_lines(
                image__pose_worthy_lines,
                landmark__pose_worthy_lines
            )
        # ---- left hand lanmark if exist ----
        if lmark_mph.left_hand_landmarks != None:
            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawLeftHand_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__facePoseLeftHandRightHand_full_dots[LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL:LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL+LANDMARK_Q_EACH_HAND]
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawLeftHand_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__facePoseLeftHandRightHand_full_lines[LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL:LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL+LANDMARK_Q_EACH_HAND]
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_worthy_dots= recalcDrawLeftHand_dots(
                image__facePoseLeftHandRightHand_worthy_dots,
                landmark__facePoseLeftHandRightHand_worthy_dots[LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY:LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY+LANDMARK_Q_EACH_HAND]
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_worthy_lines= recalcDrawLeftHand_lines(
                image__facePoseLeftHandRightHand_worthy_lines,
                landmark__facePoseLeftHandRightHand_worthy_lines[LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY:LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY+LANDMARK_Q_EACH_HAND]
            )


            # 15 ---- image left hand worthy dots
            image__left_hand_dots= recalcDrawLeftHand_dots(
                image__left_hand_dots,
                landmark__left_hand_dots
            )
            # 16 ---- image left hand worthy lines
            image__left_hand_lines= recalcDrawLeftHand_lines(
                image__left_hand_lines,
                landmark__left_hand_lines
            )
        # ---- right hand lanmark if exist ----
        if lmark_mph.right_hand_landmarks != None:
            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawRightHand_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__facePoseLeftHandRightHand_full_dots[LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL+LANDMARK_Q_EACH_HAND:]
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawRightHand_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__facePoseLeftHandRightHand_full_lines[LANDMARK_Q_FACE_FULL+LANDMARK_Q_POSE_FULL+LANDMARK_Q_EACH_HAND:]
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_worthy_dots= recalcDrawRightHand_dots(
                image__facePoseLeftHandRightHand_worthy_dots,
                landmark__facePoseLeftHandRightHand_worthy_dots[LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY+LANDMARK_Q_EACH_HAND:]
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_worthy_lines= recalcDrawRightHand_lines(
                image__facePoseLeftHandRightHand_worthy_lines,
                landmark__facePoseLeftHandRightHand_worthy_lines[LANDMARK_Q_FACE_WORTHY+LANDMARK_Q_POSE_WORTHY+LANDMARK_Q_EACH_HAND:]
            )


            # 15 ---- image right hand worthy dots
            image__right_hand_dots= recalcDrawRightHand_dots(
                image__right_hand_dots,
                landmark__right_hand_dots
            )
            # 16 ---- image right hand worthy lines
            image__right_hand_lines= recalcDrawRightHand_lines(
                image__right_hand_lines,
                landmark__right_hand_lines
            )


        # 1 ---- face pose left_hand right_hand full ----
        # 2 ---- face pose left_hand right_hand worthy ----
        # 3 ---- face full ----
        # landmark__face_solo_full, done process
        # 4 ---- face worthy ----
        # landmark__face_solo, done process
        # 5 ---- pose full ----
        # landmark__pose_solo_full, done process
        # 6 ---- pose worthy ----
        # landmark__pose_solo, done process
        # 7 ---- left hand solo ----
        # landmark__left_hand_solo, done process
        # 8 ---- right hand solo ----
        # landmark__right_hand_solo, done process

    return (
        image__facePoseLeftHandRightHand_full_dots,    #  1
        image__facePoseLeftHandRightHand_full_lines,   #  2

        image__facePoseLeftHandRightHand_worthy_dots,  #  3
        image__facePoseLeftHandRightHand_worthy_lines, #  4

        image__face_full_dots,                         #  5
        image__face_full_lines,                        #  6
        image__face_worthy_dots,                       #  7
        image__face_worthy_lines,                      #  8

        image__pose_full_dots,                         #  9
        image__pose_full_lines,                        # 10
        image__pose_worthy_dots,                       # 11
        image__pose_worthy_lines,                      # 12

        image__left_hand_dots,                         # 13
        image__left_hand_lines,                        # 14

        image__right_hand_dots,                        # 15
        image__right_hand_lines,                       # 16
    )


def main():
    if exists(WHOLE_BODY_FILE_PATH):
        image_origin= imread(str(WHOLE_BODY_FILE_PATH))
        image_origin= cvtColor(src=image_origin, code=COLOR_BGR2RGB).copy()
        image_origin= array(image_origin, dtype=uint8)
        img_fplhrh_full_dots, img_fplhrh_full_lines, \
            img_fplhrh_worthy_dots, img_fplhrh_worthy_lines, \
            img_face_full_dots, img_face_full_lines, img_face_worthy_dots, img_face_worthy_lines, \
            img_pose_full_dots, img_pose_full_lines, img_pose_worthy_dots, img_pose_worthy_lines, \
            img_left_hand_dots, img_left_hand_lines, \
            img_right_hand_dots, img_right_hand_lines= drawFacePoseHand(
                lmark_mph=MPH_fph.process(image_origin),
                original_shape=image_origin.shape
        )


        folder_dir: Path= Path(gettempdir()).resolve() /f"p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"
        # f"/tmp/p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"
        while exists(folder_dir):
            folder_dir= Path(gettempdir()).resolve() /f"p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"


        makedirs(folder_dir)
        # 18 main pics
        # ---------------------------------------------------------------
        # 1  ---- image face,pose,left_hand,right_hand full dots
        #        --> img_fplhrh_full_dots
        imwrite(
            filename=f"{folder_dir /"01_image_face_pose_left_hand_right_hand_full_dots.png"}",
            img=img_fplhrh_full_dots
        )
        # 2  ---- image face,pose,left_hand,right_hand full lines
        #        --> img_fplhrh_full_lines
        imwrite(
            filename=f"{folder_dir /"02_image_face_pose_left_hand_right_hand_full_lines.png"}",
            img=img_fplhrh_full_lines
        )

        # 3  ---- image face,pose,left_hand,right_hand worthy dots
        #        --> img_fplhrh_dots
        imwrite(
            filename=f"{folder_dir /"03_image_face_pose_left_hand_right_hand_worthy_dots.png"}",
            img=img_fplhrh_worthy_dots
        )
        # 4  ---- image face,pose,left_hand,right_hand worthy lines
        #        --> img_fplhrh_lines
        imwrite(
            filename=f"{folder_dir /"04_image_face_pose_left_hand_right_hand_worthy_lines.png"}",
            img=img_fplhrh_worthy_lines
        )

        # 5  ---- image face full dots
        #        --> img_face_full_dots
        imwrite(
            filename=f"{folder_dir /"05_image_face_full_dots.png"}",
            img=img_face_full_dots
        )
        # 6  ---- image face full lines
        #        --> img_face_full_lines
        imwrite(
            filename=f"{folder_dir /"06_image_face_full_lines.png"}",
            img=img_face_full_lines
        )

        # 9  ---- image face worthy dots
        #        --> img_face_dots
        imwrite(
            filename=f"{folder_dir /"07_image_face_worthy_dots.png"}",
            img=img_face_worthy_dots
        )
        # 10 ---- image face worthy lines
        #        --> img_face_lines
        imwrite(
            filename=f"{folder_dir /"08_image_face_worthy_lines.png"}",
            img=img_face_worthy_lines
        )

        # 11 ---- image pose full dots
        #        --> img_pose_full_dots
        imwrite(
            filename=f"{folder_dir /"09_image_pose_full_dots.png"}",
            img=img_pose_full_dots
        )
        # 12 ---- image pose full lines
        #        --> img_pose_full_lines
        imwrite(
            filename=f"{folder_dir /"10_image_pose_full_lines.png"}",
            img=img_pose_full_lines
        )

        # 13 ---- image pose worthy dots
        #        --> img_pose_dots
        imwrite(
            filename=f"{folder_dir /"11_image_pose_worthy_dots.png"}",
            img=img_pose_worthy_dots
        )
        # 14 ---- image pose worthy lines
        #        --> img_pose_lines
        imwrite(
            filename=f"{folder_dir /"12_image_pose_worthy_lines.png"}",
            img=img_pose_worthy_lines
        )

        # 15 ---- image left hand worthy dots
        #        --> img_left_hand_dots
        imwrite(
            filename=f"{folder_dir /"13_image_left_hand_dots.png"}",
            img=img_left_hand_dots
        )
        # 16 ---- image left hand worthy lines
        #        --> img_left_hand_lines
        imwrite(
            filename=f"{folder_dir /"14_image_left_hand_lines.png"}",
            img=img_left_hand_lines
        )

        # 17 ---- image right hand worthy dots
        #        --> img_right_hand_dots
        imwrite(
            filename=f"{folder_dir /"15_image_right_hand_dots.png"}",
            img=img_right_hand_dots
        )
        # 18 ---- image right hand worthy lines
        #        --> img_right_hand_lines
        imwrite(
            filename=f"{folder_dir /"16_image_right_hand_lines.png"}",
            img=img_right_hand_lines
        )
        print(f"images at {folder_dir}")
    else:
        print(f"file does not exist: {WHOLE_BODY_FILE_PATH}")


if __name__=='__main__':
    main()
