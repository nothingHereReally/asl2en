from mediapipe.python.solutions.holistic import Holistic
from numpy import array, float32, full, ndarray, uint8, zeros
from os.path import exists
from os import makedirs
from random import uniform
from cv2 import COLOR_BGR2RGB, circle, cvtColor, imread, imwrite, line


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
WHOLE_BODY_FILE_STR: str= f"{PROJ_ROOT}src_asl2gloss/pics_others/human_whole_body.png"
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




def drawSkeletonImg(img_orig: ndarray, \
                    lmark_cords: tuple, \
                    conn_idxs_list: tuple, \
                    thick: int=2, \
                    color_conn: tuple=(255,0,255), \
                    color_lmark: tuple=(255,255,0), \
                    drawJoint: bool=True) -> ndarray:
    def isOKplt(coord: tuple) -> bool:
        # x and y coordinates
        # mandatory be greater than or equal to Zero
        # and less than or equal to One
        return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
    img: ndarray= img_orig.copy()
    img_wh: dict= {"wx": img.shape[1], "hy": img.shape[0]}


    # drawing the lines between 2 landmark connections
    for l in conn_idxs_list:
        pA: tuple= (
            lmark_cords[  l[0]  ][0], # x
            lmark_cords[  l[0]  ][1]  # y
        )
        pB: tuple= (
            lmark_cords[  l[1]  ][0], # x
            lmark_cords[  l[1]  ][1]  # y
        )
        if isOKplt(pA) and isOKplt(pB):
            line(
                img=img,
                pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                color=color_conn,
                thickness=thick
            )
        else:
            raise ValueError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0")
        del pA
        del pB


    # drawing joints as cricles
    if drawJoint:
        for o in lmark_cords:
            if isOKplt(o):
                circle(
                    img=img,
                    center=(
                        int(o[0]*img_wh['wx']),
                        int(o[1]*img_wh['hy'])
                    ),
                    radius=0,
                    color=color_lmark,
                    thickness=thick*2
                )
            else:
                raise ValueError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0")
    return img
def drawFacePoseHand(lmark_mph, orig_shape: tuple) -> tuple:
    def part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
        all_x: list,
        all_y: list,
        landmark_face: list,
        landmark_pose: list,
        landmark_left_hand: list,
        landmark_right_hand: list
    ) -> tuple:
        ### 0) all coords be greater than|= 0.0 and less than|= 1.0
        # force all be greater than or = to 0.0, ie. move right/down
        min_x: float= float(min(all_x))
        min_y: float= float(min(all_y))
        if min_x<0.0: # move right
            all_x= []
            if 0<len(landmark_face):
                landmark_face= [(i[0]+abs(min_x), i[1])
                                    for i in landmark_face]
                all_x.extend([i[0] for i in landmark_face])
            if 0<len(landmark_pose):
                landmark_pose= [(i[0]+abs(min_x), i[1])
                                    for i in landmark_pose]
                all_x.extend([i[0] for i in landmark_pose])
            if 0<len(landmark_left_hand):
                landmark_left_hand= [(i[0]+abs(min_x), i[1])
                                    for i in landmark_left_hand]
                all_x.extend([i[0] for i in landmark_left_hand])
            if 0<len(landmark_right_hand):
                landmark_right_hand= [(i[0]+abs(min_x), i[1])
                                    for i in landmark_right_hand]
                all_x.extend([i[0] for i in landmark_right_hand])
            min_x= 0.0
        if min_y<0.0: # move down
            all_y= []
            if 0<len(landmark_face):
                landmark_face= [(i[0], i[1]+abs(min_y))
                                    for i in landmark_face]
                all_y.extend([i[1] for i in landmark_face])
            if 0<len(landmark_pose):
                landmark_pose= [(i[0], i[1]+abs(min_y))
                                    for i in landmark_pose]
                all_y.extend([i[1] for i in landmark_pose])
            if 0<len(landmark_left_hand):
                landmark_left_hand= [(i[0], i[1]+abs(min_y))
                                    for i in landmark_left_hand]
                all_y.extend([i[1] for i in landmark_left_hand])
            if 0<len(landmark_right_hand):
                landmark_right_hand= [(i[0], i[1]+abs(min_y))
                                    for i in landmark_right_hand]
                all_y.extend([i[1] for i in landmark_right_hand])
            min_y= 0.0
        # force all be less than or = to 1.0
        # makes maximum be 1.0, due to max/max= 1.0
        max_xy: float= max([float(max(all_x)), float(max(all_y))])
        if 1.0<max_xy:
            all_x= []
            all_y= []
            if 0<len(landmark_face):
                landmark_face= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in landmark_face]
                all_x.extend([i[0] for i in landmark_face])
                all_y.extend([i[1] for i in landmark_face])
            if 0<len(landmark_pose):
                landmark_pose= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in landmark_pose]
                all_x.extend([i[0] for i in landmark_pose])
                all_y.extend([i[1] for i in landmark_pose])
            if 0<len(landmark_left_hand):
                landmark_left_hand= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in landmark_left_hand]
                all_x.extend([i[0] for i in landmark_left_hand])
                all_y.extend([i[1] for i in landmark_left_hand])
            if 0<len(landmark_right_hand):
                landmark_right_hand= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in landmark_right_hand]
                all_x.extend([i[0] for i in landmark_right_hand])
                all_y.extend([i[1] for i in landmark_right_hand])
            min_x= min(all_x)
            min_y= min(all_y)
        del max_xy

        return (
            all_x,
            all_y,
            landmark_face,
            landmark_pose,
            landmark_left_hand,
            landmark_right_hand
        )
    def part2_beSquareRatioOnImage(
        orig_shape: tuple,
        all_x: list,
        all_y: list,
        landmark_face: list,
        landmark_pose: list,
        landmark_left_hand: list,
        landmark_right_hand: list
    ) -> tuple:
        ### 1) from old img ratio to new ratio(ie. square img )
        # remap coords( x,y ) to rescale( same ratio as orig ) on square
        # and also center orig img to New img sqaure
        if orig_shape[0]!=orig_shape[1]: # else equal, then don't touch it
            owx: int= int(orig_shape[1])
            ohy: int= int(orig_shape[0])
            wx_hy: int= 200
            if owx<ohy: # just overwrite x with respect to now on square
                all_x= []
                ccc: float= (wx_hy*owx/ohy)/wx_hy # rescale
                if 0<len(landmark_face):
                    landmark_face= [(i[0]*ccc, i[1])
                                        for i in landmark_face]
                    all_x.extend([i[0] for i in landmark_face])
                if 0<len(landmark_pose):
                    landmark_pose= [(i[0]*ccc, i[1])
                                        for i in landmark_pose]
                    all_x.extend([i[0] for i in landmark_pose])
                if 0<len(landmark_left_hand):
                    landmark_left_hand= [(i[0]*ccc, i[1])
                                        for i in landmark_left_hand]
                    all_x.extend([i[0] for i in landmark_left_hand])
                if 0<len(landmark_right_hand):
                    landmark_right_hand= [(i[0]*ccc, i[1])
                                        for i in landmark_right_hand]
                    all_x.extend([i[0] for i in landmark_right_hand])
            else: # ohy < owx, just overwrite y with respect to now on square
                all_y= []
                ccc: float= (wx_hy*ohy/owx)/wx_hy # rescale
                if 0<len(landmark_face):
                    landmark_face= [(i[0], i[1]*ccc)
                                        for i in landmark_face]
                    all_y.extend([i[1] for i in landmark_face])
                if 0<len(landmark_pose):
                    landmark_pose= [(i[0], i[1]*ccc)
                                        for i in landmark_pose]
                    all_y.extend([i[1] for i in landmark_pose])
                if 0<len(landmark_left_hand):
                    landmark_left_hand= [(i[0], i[1]*ccc)
                                        for i in landmark_left_hand]
                    all_y.extend([i[1] for i in landmark_left_hand])
                if 0<len(landmark_right_hand):
                    landmark_right_hand= [(i[0], i[1]*ccc)
                                        for i in landmark_right_hand]
                    all_y.extend([i[1] for i in landmark_right_hand])
        return (
            all_x,
            all_y,
            landmark_face,
            landmark_pose,
            landmark_left_hand,
            landmark_right_hand
        )
    def part3_zoomInOutForPadding(
        all_x: list,
        all_y: list,
        landmark_face: list,
        landmark_pose: list,
        landmark_left_hand: list,
        landmark_right_hand: list
    ) -> tuple:
        ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
        # zoom in/out for padding be 10% each side with respect to original aspect ratio
        # ie.:
        # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
        # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
        # pad: float= 0.05
        min_x: float= float(min(all_x))
        min_y: float= float(min(all_y))
        pad: float= 4.0/IMG_SIZE
        # scale: float= (1.0 -2.0*pad)/max_wy_hy, 0.0< max_wy_hy <=1.0
        # scale: float= (whole -pad_leftRight_upDown)/max_wy_hy, 0.0< max_wy_hy <=1.0
        scale: float= (1.0 -2.0*pad)/max((  max(all_x)-min_x, max(all_y)-min_y  ))
        all_x= []
        all_y= []
        if 0<len(landmark_face):
            landmark_face= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in landmark_face]
            all_x.extend([i[0] for i in landmark_face])
            all_y.extend([i[1] for i in landmark_face])
        if 0<len(landmark_pose):
            landmark_pose= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in landmark_pose]
            all_x.extend([i[0] for i in landmark_pose])
            all_y.extend([i[1] for i in landmark_pose])
        if 0<len(landmark_left_hand):
            landmark_left_hand= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in landmark_left_hand]
            all_x.extend([i[0] for i in landmark_left_hand])
            all_y.extend([i[1] for i in landmark_left_hand])
        if 0<len(landmark_right_hand):
            landmark_right_hand= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in landmark_right_hand]
            all_x.extend([i[0] for i in landmark_right_hand])
            all_y.extend([i[1] for i in landmark_right_hand])
        return (
            all_x,
            all_y,
            landmark_face,
            landmark_pose,
            landmark_left_hand,
            landmark_right_hand
        )
    def part4_centerLandmarkVerticallyHorizontally(
        all_x: list,
        all_y: list,
        landmark_face: list,
        landmark_pose: list,
        landmark_left_hand: list,
        landmark_right_hand: list
    ) -> tuple:
        ### 3) center landmark with same aspect ratio as original
        # center horizontally and vertically, since done padding then just
        # move to right/down
        min_x: float= float(min(all_x))
        min_y: float= float(min(all_y))
        lm_wx: float= max(all_x)-min_x
        lm_hy: float= max(all_y)-min_y
        if lm_wx < lm_hy:
            # all_x= []
            shift_x_right= (1.0 -lm_wx) /2.0 -min_x
            landmark_face= [(i[0]+shift_x_right, i[1])
                                for i in landmark_face]
            # all_x.extend([i[0] for i in landmark_face])
            landmark_pose= [(i[0]+shift_x_right, i[1])
                                for i in landmark_pose]
            # all_x.extend([i[0] for i in landmark_pose])
            landmark_left_hand= [(i[0]+shift_x_right, i[1])
                                for i in landmark_left_hand]
            # all_x.extend([i[0] for i in landmark_left_hand])
            landmark_right_hand= [(i[0]+shift_x_right, i[1])
                                for i in landmark_right_hand]
            # all_x.extend([i[0] for i in landmark_right_hand])
            # all_x= tuple(all_x)
            # min_x= min(all_x)
        elif lm_hy < lm_wx:
            # all_y= []
            shift_y_down= (1.0 -lm_hy) /2.0 -min_y
            landmark_face= [(i[0], i[1]+shift_y_down)
                                for i in landmark_face]
            # all_y.extend([i[1] for i in landmark_face])
            landmark_pose= [(i[0], i[1]+shift_y_down)
                                for i in landmark_pose]
            # all_y.extend([i[1] for i in landmark_pose])
            landmark_left_hand= [(i[0], i[1]+shift_y_down)
                                for i in landmark_left_hand]
            # all_y.extend([i[1] for i in landmark_left_hand])
            landmark_right_hand= [(i[0], i[1]+shift_y_down)
                                for i in landmark_right_hand]
            # all_y.extend([i[1] for i in landmark_right_hand])
            # all_y= tuple(all_y)
            # min_y= min(all_y)
        # shift_x= 0.5 -(max(all_x)+min_x)/2
        # shift_y= 0.5 -(max(all_y)+min_y)/2
        # if 0<len(landmark_face):
        #     landmark_face= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in landmark_face]
        # if 0<len(landmark_pose):
        #     landmark_pose= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in landmark_pose]
        # if 0<len(landmark_left_hand):
        #     landmark_left_hand= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in landmark_left_hand]
        # if 0<len(landmark_right_hand):
        #     landmark_right_hand= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in landmark_right_hand]
        # print(f"len(all_x) {len(all_x)}")
        # print(f"len(all_y) {len(all_y)}")
        # print(f"min_x {min_x} ---- max x {max(all_x)}")
        # print(f"min_y {min_y} ---- max y {max(all_y)}")
        return (
            all_x,
            all_y,
            landmark_face,
            landmark_pose,
            landmark_left_hand,
            landmark_right_hand
        )


    def recalcDrawFace_full_dots(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS_FULL,
            thick=1,
            color_conn=(0, 0, 0),
            color_lmark=(0, 153, 0), # 153/255= 0.6
            drawJoint=True
        )
    def recalcDrawPose_full_dots(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS_FULL,
            thick=1,
            color_conn=(0, 0, 0),
            color_lmark=(0, 0, 153), # 153/255= 0.6
            drawJoint=True
        )
    def recalcDrawFace_worthy_dots(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 0),
            color_lmark=(0, 153, 0), # 153/255= 0.6
            drawJoint=True
        )
    def recalcDrawPose_worthy_dots(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 0),
            color_lmark=(0, 0, 153), # 153/255= 0.6
            drawJoint=True
        )
    def recalcDrawLeftHand_dots(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_lhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 0),
            color_lmark=(255, 255, 255),
            drawJoint=True
        )
    def recalcDrawRightHand_dots(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_rhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 0),
            color_lmark=(204, 204, 204), # 204/255= 0.8
            drawJoint=True
        )


    def recalcDrawFace_full_lines(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS_FULL,
            thick=1,
            color_conn=(0, 153, 0), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawPose_full_lines(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS_FULL,
            thick=1,
            color_conn=(0, 0, 153), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawFace_worthy_lines(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS,
            thick=1,
            color_conn=(0, 153, 0), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawPose_worthy_lines(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 153), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawLeftHand_lines(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_lhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(255, 255, 255),
            drawJoint=False
        )
    def recalcDrawRightHand_lines(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_rhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(204, 204, 204), # 204/255= 0.8
            drawJoint=False
        )
    # 18 main pics
    # ---------------------------------------------------------------
    # 1  ---- image face,pose,left_hand,right_hand full dots
    # 2  ---- image face,pose,left_hand,right_hand full lines

    # 3  ---- image face,pose,left_hand,right_hand worthy dots
    # 4  ---- image face,pose,left_hand,right_hand worthy lines

    # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
    # 6  ---- image face,pose,left_hand,right_hand worthy lines HD

    # 7  ---- image face full dots
    # 8  ---- image face full lines

    # 9  ---- image face worthy dots
    # 10 ---- image face worthy lines

    # 11 ---- image pose full dots
    # 12 ---- image pose full lines

    # 13 ---- image pose worthy dots
    # 14 ---- image pose worthy lines

    # 15 ---- image left hand worthy dots
    # 16 ---- image left hand worthy lines

    # 17 ---- image right hand worthy dots
    # 18 ---- image right hand worthy lines


    # ---- face pose left_hand right_hand full ----
    # 1  ---- image face,pose,left_hand,right_hand full dots
    image__facePoseLeftHandRightHand_full_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 2  ---- image face,pose,left_hand,right_hand full lines
    image__facePoseLeftHandRightHand_full_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- face pose left_hand right_hand worthy ----
    # 3  ---- image face,pose,left_hand,right_hand worthy dots
    image__facePoseLeftHandRightHand_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 4  ---- image face,pose,left_hand,right_hand worthy lines
    image__facePoseLeftHandRightHand_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
    image__facePoseLeftHandRightHand_dots_hd: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 6  ---- image face,pose,left_hand,right_hand worthy lines HD
    image__facePoseLeftHandRightHand_lines_hd: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- face full ----
    # 7  ---- image face full dots
    image__face_solo_full_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 8  ---- image face full lines
    image__face_solo_full_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- face worthy ----
    # 9  ---- image face worthy dots
    image__face_solo_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 10 ---- image face worthy lines
    image__face_solo_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- pose full ----
    # 11 ---- image pose full dots
    image__pose_solo_full_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 12 ---- image pose full lines
    image__pose_solo_full_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- pose worthy ----
    # 13 ---- image pose worthy dots
    image__pose_solo_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 14 ---- image pose worthy lines
    image__pose_solo_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- left hand solo ----
    # 15 ---- image left hand worthy dots
    image__left_hand_solo_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 16 ---- image left hand worthy lines
    image__left_hand_solo_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)

    # ---- right hand solo ----
    # 17 ---- image right hand worthy dots
    image__right_hand_solo_dots: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)
    # 18 ---- image right hand worthy lines
    image__right_hand_solo_lines: ndarray= full((IMG_SIZE, IMG_SIZE, 3), 255, dtype=uint8)


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


    # 1 ---- face pose left_hand right_hand full ----
    landmark__facePoseLeftHandRightHand_full= zeros(((
        LANDMARK_Q_FACE_FULL +LANDMARK_Q_POSE_FULL +(LANDMARK_Q_EACH_HAND*2)
    ), 2), dtype=float32)
    # 2 ---- face pose left_hand right_hand worthy ----
    landmark__facePoseLeftHandRightHand= zeros(((
        LANDMARK_Q_FACE_WORTHY +LANDMARK_Q_POSE_WORTHY +(LANDMARK_Q_EACH_HAND*2)
    ), 2), dtype=float32)
    # 3 ---- face full ----
    landmark__face_solo_full= zeros((LANDMARK_Q_FACE_FULL, 2), dtype=float32)
    # 4 ---- face worthy ----
    landmark__face_solo= zeros((LANDMARK_Q_FACE_WORTHY, 2), dtype=float32)
    # 5 ---- pose full ----
    landmark__pose_solo_full= zeros((LANDMARK_Q_POSE_FULL, 2), dtype=float32)
    # 6 ---- pose worthy ----
    landmark__pose_solo= zeros((LANDMARK_Q_POSE_WORTHY, 2), dtype=float32)
    # 7 ---- left hand solo ----
    landmark__left_hand_solo= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)
    # 8 ---- right hand solo ----
    landmark__right_hand_solo= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)


    # 1 ---- face pose left_hand right_hand full ----
    landmark__facePoseLeftHandRightHand_full= landmark__facePoseLeftHandRightHand_full.tolist()
    # 2 ---- face pose left_hand right_hand worthy ----
    landmark__facePoseLeftHandRightHand= landmark__facePoseLeftHandRightHand.tolist()
    # 3 ---- face full ----
    landmark__face_solo_full= landmark__face_solo_full.tolist()
    # 4 ---- face worthy ----
    landmark__face_solo= landmark__face_solo.tolist()
    # 5 ---- pose full ----
    landmark__pose_solo_full= landmark__pose_solo_full.tolist()
    # 6 ---- pose worthy ----
    landmark__pose_solo= landmark__pose_solo.tolist()
    # 7 ---- left hand solo ----
    landmark__left_hand_solo= landmark__left_hand_solo.tolist()
    # 8 ---- right hand solo ----
    landmark__right_hand_solo= landmark__right_hand_solo.tolist()
    if lmark_mph.face_landmarks!=None \
        or lmark_mph.pose_landmarks!=None \
        or lmark_mph.left_hand_landmarks!=None \
        or lmark_mph.right_hand_landmarks!=None:

        recalc_face_full= []
        recalc_pose_full= []
        recalc_left_hand_full= []
        recalc_right_hand_full= []
        all_x_full= []
        all_y_full= []

        recalc_face= []
        recalc_pose= []
        recalc_left_hand= []
        recalc_right_hand= []
        all_x= []
        all_y= []

        recalc_face_solo_full= []
        all_x_face_solo_full= []
        all_y_face_solo_full= []

        recalc_face_solo= []
        all_x_face_solo= []
        all_y_face_solo= []

        recalc_pose_solo_full= []
        all_x_pose_solo_full= []
        all_y_pose_solo_full= []

        recalc_pose_solo= []
        all_x_pose_solo= []
        all_y_pose_solo= []

        recalc_left_hand_solo= []
        all_x_left_hand_solo= []
        all_y_left_hand_solo= []

        recalc_right_hand_solo= []
        all_x_right_hand_solo= []
        all_y_right_hand_solo= []

        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0
        if lmark_mph.face_landmarks != None:
            for i in enumerate(lmark_mph.face_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_face_full.append((  (i[1]).x, (i[1]).y  ))

                all_x_face_solo_full.append( (i[1]).x )
                all_y_face_solo_full.append( (i[1]).y )
                recalc_face_solo_full.append((  (i[1]).x, (i[1]).y  ))

                if int(i[0]) in WORTHY_FACE_IDX:
                    all_x.append( (i[1]).x )
                    all_y.append( (i[1]).y )
                    recalc_face.append((  (i[1]).x, (i[1]).y  ))

                    all_x_face_solo.append( (i[1]).x )
                    all_y_face_solo.append( (i[1]).y )
                    recalc_face_solo.append((  (i[1]).x, (i[1]).y  ))
        if lmark_mph.pose_landmarks != None:
            for i in enumerate(lmark_mph.pose_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_pose_full.append((  (i[1]).x, (i[1]).y  ))

                all_x_pose_solo_full.append( (i[1]).x )
                all_y_pose_solo_full.append( (i[1]).y )
                recalc_pose_solo_full.append((  (i[1]).x, (i[1]).y  ))

                if int(i[0]) in WORTHY_POSE_IDX:
                    all_x.append( (i[1]).x )
                    all_y.append( (i[1]).y )
                    recalc_pose.append((  (i[1]).x, (i[1]).y  ))

                    all_x_pose_solo.append( (i[1]).x )
                    all_y_pose_solo.append( (i[1]).y )
                    recalc_pose_solo.append((  (i[1]).x, (i[1]).y  ))
        if lmark_mph.left_hand_landmarks != None:
            for i in enumerate(lmark_mph.left_hand_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_left_hand_full.append((  (i[1]).x, (i[1]).y  ))

                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )
                recalc_left_hand.append((  (i[1]).x, (i[1]).y  ))

                all_x_left_hand_solo.append( (i[1]).x )
                all_y_left_hand_solo.append( (i[1]).y )
                recalc_left_hand_solo.append((  (i[1]).x, (i[1]).y  ))
        if lmark_mph.right_hand_landmarks != None:
            for i in enumerate(lmark_mph.right_hand_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_right_hand_full.append((  (i[1]).x, (i[1]).y  ))

                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )
                recalc_right_hand.append((  (i[1]).x, (i[1]).y  ))

                all_x_right_hand_solo.append( (i[1]).x )
                all_y_right_hand_solo.append( (i[1]).y )
                recalc_right_hand_solo.append((  (i[1]).x, (i[1]).y  ))


        ### 0) all coords be greater than|= 0.0 and less than|= 1.0
        # force all be greater than or = to 0.0, ie. move right/down
        # ---- face pose left_hand right_hand full ----
        all_x_full, all_y_full, recalc_face_full, recalc_pose_full, recalc_left_hand_full, recalc_right_hand_full= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_full,
            all_y=all_y_full,
            landmark_face=recalc_face_full,
            landmark_pose=recalc_pose_full,
            landmark_left_hand=recalc_left_hand_full,
            landmark_right_hand=recalc_right_hand_full
        )
        # ---- face pose left_hand right_hand worthy ----
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )
        # ---- face full ----
        all_x_face_solo_full, all_y_face_solo_full, recalc_face_solo_full, tmp1, tmp2, tmp3= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_face_solo_full,
            all_y=all_y_face_solo_full,
            landmark_face=recalc_face_solo_full,
            landmark_pose=recalc_face_solo_full,
            landmark_left_hand=recalc_face_solo_full,
            landmark_right_hand=recalc_face_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- face worthy ----
        all_x_face_solo, all_y_face_solo, recalc_face_solo, tmp1, tmp2, tmp3= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_face_solo,
            all_y=all_y_face_solo,
            landmark_face=recalc_face_solo,
            landmark_pose=recalc_face_solo,
            landmark_left_hand=recalc_face_solo,
            landmark_right_hand=recalc_face_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose full ----
        all_x_pose_solo_full, all_y_pose_solo_full, recalc_pose_solo_full, tmp1, tmp2, tmp3= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_pose_solo_full,
            all_y=all_y_pose_solo_full,
            landmark_face=recalc_pose_solo_full,
            landmark_pose=recalc_pose_solo_full,
            landmark_left_hand=recalc_pose_solo_full,
            landmark_right_hand=recalc_pose_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose worthy ----
        all_x_pose_solo, all_y_pose_solo, recalc_pose_solo, tmp1, tmp2, tmp3= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_pose_solo,
            all_y=all_y_pose_solo,
            landmark_face=recalc_pose_solo,
            landmark_pose=recalc_pose_solo,
            landmark_left_hand=recalc_pose_solo,
            landmark_right_hand=recalc_pose_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- left hand solo ----
        all_x_left_hand_solo, all_y_left_hand_solo, tmp1, tmp2, recalc_left_hand_solo, tmp3= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_left_hand_solo,
            all_y=all_y_left_hand_solo,
            landmark_face=recalc_left_hand_solo,
            landmark_pose=recalc_left_hand_solo,
            landmark_left_hand=recalc_left_hand_solo,
            landmark_right_hand=recalc_left_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- right hand solo ----
        all_x_right_hand_solo, all_y_right_hand_solo, tmp1, tmp2, tmp3, recalc_right_hand_solo= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x_right_hand_solo,
            all_y=all_y_right_hand_solo,
            landmark_face=recalc_right_hand_solo,
            landmark_pose=recalc_right_hand_solo,
            landmark_left_hand=recalc_right_hand_solo,
            landmark_right_hand=recalc_right_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3


        ### 1) from old img ratio to new ratio(ie. square img )
        # remap coords( x,y ) to rescale( same ratio as orig ) on square
        # and also center orig img to New img sqaure
        # ---- face pose left_hand right_hand full ----
        all_x_full, all_y_full, recalc_face_full, recalc_pose_full, recalc_left_hand_full, recalc_right_hand_full= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_full,
            all_y=all_y_full,
            landmark_face=recalc_face_full,
            landmark_pose=recalc_pose_full,
            landmark_left_hand=recalc_left_hand_full,
            landmark_right_hand=recalc_right_hand_full
        )
        # ---- face pose left_hand right_hand worthy ----
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )
        # ---- face full ----
        all_x_face_solo_full, all_y_face_solo_full, recalc_face_solo_full, tmp1, tmp2, tmp3= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_face_solo_full,
            all_y=all_y_face_solo_full,
            landmark_face=recalc_face_solo_full,
            landmark_pose=recalc_face_solo_full,
            landmark_left_hand=recalc_face_solo_full,
            landmark_right_hand=recalc_face_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- face worthy ----
        all_x_face_solo, all_y_face_solo, recalc_face_solo, tmp1, tmp2, tmp3= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_face_solo,
            all_y=all_y_face_solo,
            landmark_face=recalc_face_solo,
            landmark_pose=recalc_face_solo,
            landmark_left_hand=recalc_face_solo,
            landmark_right_hand=recalc_face_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose full ----
        all_x_pose_solo_full, all_y_pose_solo_full, recalc_pose_solo_full, tmp1, tmp2, tmp3= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_pose_solo_full,
            all_y=all_y_pose_solo_full,
            landmark_face=recalc_pose_solo_full,
            landmark_pose=recalc_pose_solo_full,
            landmark_left_hand=recalc_pose_solo_full,
            landmark_right_hand=recalc_pose_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose worthy ----
        all_x_pose_solo, all_y_pose_solo, recalc_pose_solo, tmp1, tmp2, tmp3= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_pose_solo,
            all_y=all_y_pose_solo,
            landmark_face=recalc_pose_solo,
            landmark_pose=recalc_pose_solo,
            landmark_left_hand=recalc_pose_solo,
            landmark_right_hand=recalc_pose_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- left hand solo ----
        all_x_left_hand_solo, all_y_left_hand_solo, tmp1, tmp2, recalc_left_hand_solo, tmp3= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_left_hand_solo,
            all_y=all_y_left_hand_solo,
            landmark_face=recalc_left_hand_solo,
            landmark_pose=recalc_left_hand_solo,
            landmark_left_hand=recalc_left_hand_solo,
            landmark_right_hand=recalc_left_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- right hand solo ----
        all_x_right_hand_solo, all_y_right_hand_solo, tmp1, tmp2, tmp3, recalc_right_hand_solo= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x_right_hand_solo,
            all_y=all_y_right_hand_solo,
            landmark_face=recalc_right_hand_solo,
            landmark_pose=recalc_right_hand_solo,
            landmark_left_hand=recalc_right_hand_solo,
            landmark_right_hand=recalc_right_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3


        ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
        # zoom in/out for padding be 10% each side with respect to original aspect ratio
        # ie.:
        # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
        # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
        # pad: float= 0.05
        # ---- face pose left_hand right_hand full ----
        all_x_full, all_y_full, recalc_face_full, recalc_pose_full, recalc_left_hand_full, recalc_right_hand_full= part3_zoomInOutForPadding(
            all_x=all_x_full,
            all_y=all_y_full,
            landmark_face=recalc_face_full,
            landmark_pose=recalc_pose_full,
            landmark_left_hand=recalc_left_hand_full,
            landmark_right_hand=recalc_right_hand_full
        )
        # ---- face pose left_hand right_hand worthy ----
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part3_zoomInOutForPadding(
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )
        # ---- face full ----
        all_x_face_solo_full, all_y_face_solo_full, recalc_face_solo_full, tmp1, tmp2, tmp3= part3_zoomInOutForPadding(
            all_x=all_x_face_solo_full,
            all_y=all_y_face_solo_full,
            landmark_face=recalc_face_solo_full,
            landmark_pose=recalc_face_solo_full,
            landmark_left_hand=recalc_face_solo_full,
            landmark_right_hand=recalc_face_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- face worthy ----
        all_x_face_solo, all_y_face_solo, recalc_face_solo, tmp1, tmp2, tmp3= part3_zoomInOutForPadding(
            all_x=all_x_face_solo,
            all_y=all_y_face_solo,
            landmark_face=recalc_face_solo,
            landmark_pose=recalc_face_solo,
            landmark_left_hand=recalc_face_solo,
            landmark_right_hand=recalc_face_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose full ----
        all_x_pose_solo_full, all_y_pose_solo_full, recalc_pose_solo_full, tmp1, tmp2, tmp3= part3_zoomInOutForPadding(
            all_x=all_x_pose_solo_full,
            all_y=all_y_pose_solo_full,
            landmark_face=recalc_pose_solo_full,
            landmark_pose=recalc_pose_solo_full,
            landmark_left_hand=recalc_pose_solo_full,
            landmark_right_hand=recalc_pose_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose worthy ----
        all_x_pose_solo, all_y_pose_solo, recalc_pose_solo, tmp1, tmp2, tmp3= part3_zoomInOutForPadding(
            all_x=all_x_pose_solo,
            all_y=all_y_pose_solo,
            landmark_face=recalc_pose_solo,
            landmark_pose=recalc_pose_solo,
            landmark_left_hand=recalc_pose_solo,
            landmark_right_hand=recalc_pose_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- left hand solo ----
        all_x_left_hand_solo, all_y_left_hand_solo, tmp1, tmp2, recalc_left_hand_solo, tmp3= part3_zoomInOutForPadding(
            all_x=all_x_left_hand_solo,
            all_y=all_y_left_hand_solo,
            landmark_face=recalc_left_hand_solo,
            landmark_pose=recalc_left_hand_solo,
            landmark_left_hand=recalc_left_hand_solo,
            landmark_right_hand=recalc_left_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- right hand solo ----
        all_x_right_hand_solo, all_y_right_hand_solo, tmp1, tmp2, tmp3, recalc_right_hand_solo= part3_zoomInOutForPadding(
            all_x=all_x_right_hand_solo,
            all_y=all_y_right_hand_solo,
            landmark_face=recalc_right_hand_solo,
            landmark_pose=recalc_right_hand_solo,
            landmark_left_hand=recalc_right_hand_solo,
            landmark_right_hand=recalc_right_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3


        ### 3) center landmark with same aspect ratio as original
        # center horizontally and vertically, since done padding then just
        # move to right/down
        # ---- face pose left_hand right_hand full ----
        all_x_full, all_y_full, recalc_face_full, recalc_pose_full, recalc_left_hand_full, recalc_right_hand_full= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_full,
            all_y=all_y_full,
            landmark_face=recalc_face_full,
            landmark_pose=recalc_pose_full,
            landmark_left_hand=recalc_left_hand_full,
            landmark_right_hand=recalc_right_hand_full
        )
        # ---- face pose left_hand right_hand worthy ----
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )
        # ---- face full ----
        all_x_face_solo_full, all_y_face_solo_full, recalc_face_solo_full, tmp1, tmp2, tmp3= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_face_solo_full,
            all_y=all_y_face_solo_full,
            landmark_face=recalc_face_solo_full,
            landmark_pose=recalc_face_solo_full,
            landmark_left_hand=recalc_face_solo_full,
            landmark_right_hand=recalc_face_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- face worthy ----
        all_x_face_solo, all_y_face_solo, recalc_face_solo, tmp1, tmp2, tmp3= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_face_solo,
            all_y=all_y_face_solo,
            landmark_face=recalc_face_solo,
            landmark_pose=recalc_face_solo,
            landmark_left_hand=recalc_face_solo,
            landmark_right_hand=recalc_face_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose full ----
        all_x_pose_solo_full, all_y_pose_solo_full, recalc_pose_solo_full, tmp1, tmp2, tmp3= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_pose_solo_full,
            all_y=all_y_pose_solo_full,
            landmark_face=recalc_pose_solo_full,
            landmark_pose=recalc_pose_solo_full,
            landmark_left_hand=recalc_pose_solo_full,
            landmark_right_hand=recalc_pose_solo_full
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- pose worthy ----
        all_x_pose_solo, all_y_pose_solo, recalc_pose_solo, tmp1, tmp2, tmp3= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_pose_solo,
            all_y=all_y_pose_solo,
            landmark_face=recalc_pose_solo,
            landmark_pose=recalc_pose_solo,
            landmark_left_hand=recalc_pose_solo,
            landmark_right_hand=recalc_pose_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- left hand solo ----
        all_x_left_hand_solo, all_y_left_hand_solo, tmp1, tmp2, recalc_left_hand_solo, tmp3= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_left_hand_solo,
            all_y=all_y_left_hand_solo,
            landmark_face=recalc_left_hand_solo,
            landmark_pose=recalc_left_hand_solo,
            landmark_left_hand=recalc_left_hand_solo,
            landmark_right_hand=recalc_left_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3
        # ---- right hand solo ----
        all_x_right_hand_solo, all_y_right_hand_solo, tmp1, tmp2, tmp3, recalc_right_hand_solo= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x_right_hand_solo,
            all_y=all_y_right_hand_solo,
            landmark_face=recalc_right_hand_solo,
            landmark_pose=recalc_right_hand_solo,
            landmark_left_hand=recalc_right_hand_solo,
            landmark_right_hand=recalc_right_hand_solo
        )
        del tmp1
        del tmp2
        del tmp3


        # ---- face pose left_hand right_hand full ----
        del all_x_full
        del all_y_full

        # ---- face pose left_hand right_hand worthy ----
        del all_x
        del all_y

        # ---- face full ----
        del all_x_face_solo_full
        del all_y_face_solo_full

        # ---- face worthy ----
        del all_x_face_solo
        del all_y_face_solo

        # ---- pose full ----
        del all_x_pose_solo_full
        del all_y_pose_solo_full

        # ---- pose worthy ----
        del all_x_pose_solo
        del all_y_pose_solo

        # ---- left hand solo ----
        del all_x_left_hand_solo
        del all_y_left_hand_solo

        # ---- right hand solo ----
        del all_x_right_hand_solo
        del all_y_right_hand_solo


        # done calcuation now mandatory all landamrks be constant
        # ---- face pose left_hand right_hand full ----
        recalc_face_full= tuple(tuple(i) for i in recalc_face_full)
        recalc_pose_full= tuple(tuple(i) for i in recalc_pose_full)
        recalc_left_hand_full= tuple(tuple(i) for i in recalc_left_hand_full)
        recalc_right_hand_full= tuple(tuple(i) for i in recalc_right_hand_full)

        # ---- face pose left_hand right_hand worthy ----
        recalc_face= tuple(tuple(i) for i in recalc_face)
        recalc_pose= tuple(tuple(i) for i in recalc_pose)
        recalc_left_hand= tuple(tuple(i) for i in recalc_left_hand)
        recalc_right_hand= tuple(tuple(i) for i in recalc_right_hand)

        # ---- face full ----
        recalc_face_solo_full= tuple(tuple(i) for i in recalc_face_solo_full)

        # ---- face worthy ----
        recalc_face_solo= tuple(tuple(i) for i in recalc_face_solo)

        # ---- pose full ----
        recalc_pose_solo_full= tuple(tuple(i) for i in recalc_pose_solo_full)

        # ---- pose worthy ----
        recalc_pose_solo= tuple(tuple(i) for i in recalc_pose_solo)

        # ---- left hand solo ----
        recalc_left_hand_solo= tuple(tuple(i) for i in recalc_left_hand_solo)

        # ---- right hand solo ----
        recalc_right_hand_solo= tuple(tuple(i) for i in recalc_right_hand_solo)


        # 18 main pics
        # ---------------------------------------------------------------
        # 1  ---- image face,pose,left_hand,right_hand full dots
        # 2  ---- image face,pose,left_hand,right_hand full lines

        # 3  ---- image face,pose,left_hand,right_hand worthy dots
        # 4  ---- image face,pose,left_hand,right_hand worthy lines

        # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
        # 6  ---- image face,pose,left_hand,right_hand worthy lines HD

        # 7  ---- image face full dots
        # 8  ---- image face full lines

        # 9  ---- image face worthy dots
        # 10 ---- image face worthy lines

        # 11 ---- image pose full dots
        # 12 ---- image pose full lines

        # 13 ---- image pose worthy dots
        # 14 ---- image pose worthy lines

        # 15 ---- image left hand worthy dots
        # 16 ---- image left hand worthy lines

        # 17 ---- image right hand worthy dots
        # 18 ---- image right hand worthy lines


        # --------------------------------
        # -- landmarks transformation( from what to how many )
        # 468 --> 36 landmarks on face
        # 33  --> 8  landmarks on pose
        # 21  --> 21 landmarks on left hand
        # 21  --> 21 landmarks on right hand
        # --------------------------------


        # ---- face pose left_hand right_hand full ----
        landmark__face_full= zeros((LANDMARK_Q_FACE_FULL, 2), dtype=float32)
        landmark__pose_full= zeros((LANDMARK_Q_POSE_FULL, 2), dtype=float32)
        landmark__left_hand_full= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)
        landmark__right_hand_full= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)

        # ---- face pose left_hand right_hand worthy ----
        landmark__face= zeros((LANDMARK_Q_FACE_WORTHY, 2), dtype=float32)
        landmark__pose= zeros((LANDMARK_Q_POSE_WORTHY, 2), dtype=float32)
        landmark__left_hand= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)
        landmark__right_hand= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)

        # ---- face full ----
        landmark__face_solo_full= zeros((LANDMARK_Q_FACE_FULL, 2), dtype=float32)

        # ---- face worthy ----
        landmark__face_solo= zeros((LANDMARK_Q_FACE_WORTHY, 2), dtype=float32)

        # ---- pose full ----
        landmark__pose_solo_full= zeros((LANDMARK_Q_POSE_FULL, 2), dtype=float32)

        # ---- pose worthy ----
        landmark__pose_solo= zeros((LANDMARK_Q_POSE_WORTHY, 2), dtype=float32)

        # ---- left hand solo ----
        landmark__left_hand_solo= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)

        # ---- right hand solo ----
        landmark__right_hand_solo= zeros((LANDMARK_Q_EACH_HAND, 2), dtype=float32)




        # ---- face pose left_hand right_hand full ----
        landmark__face_full= tuple(landmark__face_full.tolist())
        landmark__pose_full= tuple(landmark__pose_full.tolist())
        landmark__left_hand_full= tuple(landmark__left_hand_full.tolist())
        landmark__right_hand_full= tuple(landmark__right_hand_full.tolist())

        # ---- face pose left_hand right_hand worthy ----
        landmark__face= tuple(landmark__face.tolist())
        landmark__pose= tuple(landmark__pose.tolist())
        landmark__left_hand= tuple(landmark__left_hand.tolist())
        landmark__right_hand= tuple(landmark__right_hand.tolist())

        # ---- face full ----
        landmark__face_solo_full= tuple(landmark__face_solo_full.tolist())

        # ---- face worthy ----
        landmark__face_solo= tuple(landmark__face_solo.tolist())

        # ---- pose full ----
        landmark__pose_solo_full= tuple(landmark__pose_solo_full.tolist())

        # ---- pose worthy ----
        landmark__pose_solo= tuple(landmark__pose_solo.tolist())

        # ---- left hand solo ----
        landmark__left_hand_solo= tuple(landmark__left_hand_solo.tolist())

        # ---- right hand solo ----
        landmark__right_hand_solo= tuple(landmark__right_hand_solo.tolist())


        # ---- face lanmark if exist ----
        if lmark_mph.face_landmarks != None:
            landmark__face_full= tuple(recalc_face_full)
            landmark__face= tuple(recalc_face)
            landmark__face_solo_full= tuple(recalc_face_solo_full)
            landmark__face_solo= tuple(recalc_face_solo)

            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawFace_full_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__face_full
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawFace_full_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__face_full
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_dots= recalcDrawFace_worthy_dots(
                image__facePoseLeftHandRightHand_dots,
                landmark__face
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_lines= recalcDrawFace_worthy_lines(
                image__facePoseLeftHandRightHand_lines,
                landmark__face
            )

            # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
            image__facePoseLeftHandRightHand_dots_hd= recalcDrawFace_worthy_dots(
                image__facePoseLeftHandRightHand_dots_hd,
                landmark__face
            )
            # 6  ---- image face,pose,left_hand,right_hand worthy lines HD
            image__facePoseLeftHandRightHand_lines_hd= recalcDrawFace_worthy_lines(
                image__facePoseLeftHandRightHand_lines_hd,
                landmark__face
            )

            # 7  ---- image face full dots
            image__face_solo_full_dots= recalcDrawFace_full_dots(
                image__face_solo_full_dots,
                landmark__face_solo_full
            )
            # 8  ---- image face full lines
            image__face_solo_full_lines= recalcDrawFace_full_lines(
                image__face_solo_full_lines,
                landmark__face_solo_full
            )

            # 9  ---- image face worthy dots
            image__face_solo_dots= recalcDrawFace_worthy_dots(
                image__face_solo_dots,
                landmark__face_solo
            )
            # 10 ---- image face worthy lines
            image__face_solo_lines= recalcDrawFace_worthy_lines(
                image__face_solo_lines,
                landmark__face_solo
            )
        # ---- pose lanmark if exist ----
        if lmark_mph.pose_landmarks != None:
            landmark__pose_full= tuple(recalc_pose_full)
            landmark__pose= tuple(recalc_pose)
            landmark__pose_solo_full= tuple(recalc_pose_solo_full)
            landmark__pose_solo= tuple(recalc_pose_solo)

            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawPose_full_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__pose_full
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawPose_full_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__pose_full
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_dots= recalcDrawPose_worthy_dots(
                image__facePoseLeftHandRightHand_dots,
                landmark__pose
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_lines= recalcDrawPose_worthy_lines(
                image__facePoseLeftHandRightHand_lines,
                landmark__pose
            )

            # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
            image__facePoseLeftHandRightHand_dots_hd= recalcDrawPose_worthy_dots(
                image__facePoseLeftHandRightHand_dots_hd,
                landmark__pose
            )
            # 6  ---- image face,pose,left_hand,right_hand worthy lines HD
            image__facePoseLeftHandRightHand_lines_hd= recalcDrawPose_worthy_lines(
                image__facePoseLeftHandRightHand_lines_hd,
                landmark__pose
            )

            # 11 ---- image pose full dots
            image__pose_solo_full_dots= recalcDrawPose_full_dots(
                image__pose_solo_full_dots,
                landmark__pose_solo_full
            )
            # 12 ---- image pose full lines
            image__pose_solo_full_lines= recalcDrawPose_full_lines(
                image__pose_solo_full_lines,
                landmark__pose_solo_full
            )

            # 13 ---- image pose worthy dots
            image__pose_solo_dots= recalcDrawPose_worthy_dots(
                image__pose_solo_dots,
                landmark__pose_solo
            )
            # 14 ---- image pose worthy lines
            image__pose_solo_lines= recalcDrawPose_worthy_lines(
                image__pose_solo_lines,
                landmark__pose_solo
            )
        # ---- left hand lanmark if exist ----
        if lmark_mph.left_hand_landmarks != None:
            landmark__left_hand_full= tuple(recalc_left_hand_full)
            landmark__left_hand= tuple(recalc_left_hand)
            landmark__left_hand_solo= tuple(recalc_left_hand_solo)

            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawLeftHand_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__left_hand_full
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawLeftHand_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__left_hand_full
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_dots= recalcDrawLeftHand_dots(
                image__facePoseLeftHandRightHand_dots,
                landmark__left_hand
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_lines= recalcDrawLeftHand_lines(
                image__facePoseLeftHandRightHand_lines,
                landmark__left_hand
            )

            # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
            image__facePoseLeftHandRightHand_dots_hd= recalcDrawLeftHand_dots(
                image__facePoseLeftHandRightHand_dots_hd,
                landmark__left_hand
            )
            # 6  ---- image face,pose,left_hand,right_hand worthy lines HD
            image__facePoseLeftHandRightHand_lines_hd= recalcDrawLeftHand_lines(
                image__facePoseLeftHandRightHand_lines_hd,
                landmark__left_hand
            )

            # 15 ---- image left hand worthy dots
            image__left_hand_solo_dots= recalcDrawLeftHand_dots(
                image__left_hand_solo_dots,
                landmark__left_hand_solo
            )
            # 16 ---- image left hand worthy lines
            image__left_hand_solo_lines= recalcDrawLeftHand_lines(
                image__left_hand_solo_lines,
                landmark__left_hand_solo
            )
        # ---- right hand lanmark if exist ----
        if lmark_mph.right_hand_landmarks != None:
            landmark__right_hand_full= tuple(recalc_right_hand_full)
            landmark__right_hand= tuple(recalc_right_hand)
            landmark__right_hand_solo= tuple(recalc_right_hand_solo)

            # 1  ---- image face,pose,left_hand,right_hand full dots
            image__facePoseLeftHandRightHand_full_dots= recalcDrawRightHand_dots(
                image__facePoseLeftHandRightHand_full_dots,
                landmark__right_hand_full
            )
            # 2  ---- image face,pose,left_hand,right_hand full lines
            image__facePoseLeftHandRightHand_full_lines= recalcDrawRightHand_lines(
                image__facePoseLeftHandRightHand_full_lines,
                landmark__right_hand_full
            )

            # 3  ---- image face,pose,left_hand,right_hand worthy dots
            image__facePoseLeftHandRightHand_dots= recalcDrawRightHand_dots(
                image__facePoseLeftHandRightHand_dots,
                landmark__right_hand
            )
            # 4  ---- image face,pose,left_hand,right_hand worthy lines
            image__facePoseLeftHandRightHand_lines= recalcDrawRightHand_lines(
                image__facePoseLeftHandRightHand_lines,
                landmark__right_hand
            )

            # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
            image__facePoseLeftHandRightHand_dots_hd= recalcDrawRightHand_dots(
                image__facePoseLeftHandRightHand_dots_hd,
                landmark__right_hand
            )
            # 6  ---- image face,pose,left_hand,right_hand worthy lines HD
            image__facePoseLeftHandRightHand_lines_hd= recalcDrawRightHand_lines(
                image__facePoseLeftHandRightHand_lines_hd,
                landmark__right_hand
            )

            # 17 ---- image right hand worthy dots
            image__right_hand_solo_dots= recalcDrawRightHand_dots(
                image__right_hand_solo_dots,
                landmark__right_hand_solo
            )
            # 18 ---- image right hand worthy lines
            image__right_hand_solo_lines= recalcDrawRightHand_lines(
                image__right_hand_solo_lines,
                landmark__right_hand_solo
            )


        # 1 ---- face pose left_hand right_hand full ----
        landmark__facePoseLeftHandRightHand_full= []
        landmark__facePoseLeftHandRightHand_full.extend(landmark__face_full)
        landmark__facePoseLeftHandRightHand_full.extend(landmark__pose_full)
        landmark__facePoseLeftHandRightHand_full.extend(landmark__left_hand_full)
        landmark__facePoseLeftHandRightHand_full.extend(landmark__right_hand_full)
        # 2 ---- face pose left_hand right_hand worthy ----
        landmark__facePoseLeftHandRightHand= []
        landmark__facePoseLeftHandRightHand.extend(landmark__face)
        landmark__facePoseLeftHandRightHand.extend(landmark__pose)
        landmark__facePoseLeftHandRightHand.extend(landmark__left_hand)
        landmark__facePoseLeftHandRightHand.extend(landmark__right_hand)
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
        (
            image__facePoseLeftHandRightHand_full_dots,  #  1
            image__facePoseLeftHandRightHand_full_lines, #  2

            image__facePoseLeftHandRightHand_dots,       #  3
            image__facePoseLeftHandRightHand_lines,      #  4
            image__facePoseLeftHandRightHand_dots_hd,    #  5
            image__facePoseLeftHandRightHand_lines_hd,   #  6

            image__face_solo_full_dots,                  #  7
            image__face_solo_full_lines,                 #  8
            image__face_solo_dots,                       #  9
            image__face_solo_lines,                      # 10

            image__pose_solo_full_dots,                  # 11
            image__pose_solo_full_lines,                 # 12
            image__pose_solo_dots,                       # 13
            image__pose_solo_lines,                      # 14

            image__left_hand_solo_dots,                  # 15
            image__left_hand_solo_lines,                 # 16

            image__right_hand_solo_dots,                 # 17
            image__right_hand_solo_lines,                # 18
        ),
        landmark__facePoseLeftHandRightHand_full, # 1
        landmark__facePoseLeftHandRightHand,      # 2
        landmark__face_solo_full,                 # 3
        landmark__face_solo,                      # 4
        landmark__pose_solo_full,                 # 5
        landmark__pose_solo,                      # 6
        landmark__left_hand_solo,                 # 7
        landmark__right_hand_solo,                # 8
    )


if __name__=='__main__':
    if exists(WHOLE_BODY_FILE_STR):
        image_origin= imread(WHOLE_BODY_FILE_STR)
        image_origin= cvtColor(src=image_origin, code=COLOR_BGR2RGB).copy()
        image_origin= array(image_origin, dtype=uint8)
        (img_fplhrh_full_dots, img_fplhrh_full_lines, \
            img_fplhrh_dots, img_fplhrh_lines, img_fplhrh_dots_hd, img_fplhrh_lines_hd, \
            img_face_full_dots, img_face_full_lines, img_face_dots, img_face_lines, \
            img_pose_full_dots, img_pose_full_lines, img_pose_dots, img_pose_lines, \
            img_left_hand_dots, img_left_hand_lines, \
            img_right_hand_dots, img_right_hand_lines, \
        ), lm__fplhrh_full, lm__fplhrh, \
        lm__face_full, lm__face, \
        lm__pose_full, lm__pose, \
        lm__left_hand, lm__right_hand= drawFacePoseHand(
            lmark_mph=MPH_fph.process(image_origin),
            orig_shape=image_origin.shape
        )
        del lm__fplhrh_full
        del lm__fplhrh
        del lm__face_full
        del lm__face
        del lm__pose_full
        del lm__pose
        del lm__left_hand
        del lm__right_hand


        folder_dir: str= f"/tmp/p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"
        while exists(folder_dir):
            folder_dir: str= f"/tmp/p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"


        makedirs(folder_dir)
        # 18 main pics
        # ---------------------------------------------------------------
        # 1  ---- image face,pose,left_hand,right_hand full dots
        #        --> img_fplhrh_full_dots
        imwrite(
            filename=f"{folder_dir}/01_image_face_pose_left_hand_right_hand_full_dots.png",
            img=img_fplhrh_full_dots
        )
        # 2  ---- image face,pose,left_hand,right_hand full lines
        #        --> img_fplhrh_full_lines
        imwrite(
            filename=f"{folder_dir}/02_image_face_pose_left_hand_right_hand_full_lines.png",
            img=img_fplhrh_full_lines
        )

        # 3  ---- image face,pose,left_hand,right_hand worthy dots
        #        --> img_fplhrh_dots
        imwrite(
            filename=f"{folder_dir}/03_image_face_pose_left_hand_right_hand_worthy_dots.png",
            img=img_fplhrh_dots
        )
        # 4  ---- image face,pose,left_hand,right_hand worthy lines
        #        --> img_fplhrh_lines
        imwrite(
            filename=f"{folder_dir}/04_image_face_pose_left_hand_right_hand_worthy_lines.png",
            img=img_fplhrh_lines
        )

        # 5  ---- image face,pose,left_hand,right_hand worthy dots HD
        #        --> img_fplhrh_dots_hd
        imwrite(
            filename=f"{folder_dir}/05_image_face_pose_left_hand_right_hand_worthy_dots_hd.png",
            img=img_fplhrh_dots_hd
        )
        # 6  ---- image face,pose,left_hand,right_hand worthy lines HD
        #        --> img_fplhrh_lines_hd
        imwrite(
            filename=f"{folder_dir}/06_image_face_pose_left_hand_right_hand_worthy_lines_hd.png",
            img=img_fplhrh_lines_hd
        )

        # 7  ---- image face full dots
        #        --> img_face_full_dots
        imwrite(
            filename=f"{folder_dir}/07_image_face_full_dots.png",
            img=img_face_full_dots
        )
        # 8  ---- image face full lines
        #        --> img_face_full_lines
        imwrite(
            filename=f"{folder_dir}/08_image_face_full_lines.png",
            img=img_face_full_lines
        )

        # 9  ---- image face worthy dots
        #        --> img_face_dots
        imwrite(
            filename=f"{folder_dir}/09_image_face_worthy_dots.png",
            img=img_face_dots
        )
        # 10 ---- image face worthy lines
        #        --> img_face_lines
        imwrite(
            filename=f"{folder_dir}/10_image_face_worthy_lines.png",
            img=img_face_lines
        )

        # 11 ---- image pose full dots
        #        --> img_pose_full_dots
        imwrite(
            filename=f"{folder_dir}/11_image_pose_full_dots.png",
            img=img_pose_full_dots
        )
        # 12 ---- image pose full lines
        #        --> img_pose_full_lines
        imwrite(
            filename=f"{folder_dir}/12_image_pose_full_lines.png",
            img=img_pose_full_lines
        )

        # 13 ---- image pose worthy dots
        #        --> img_pose_dots
        imwrite(
            filename=f"{folder_dir}/13_image_pose_worthy_dots.png",
            img=img_pose_dots
        )
        # 14 ---- image pose worthy lines
        #        --> img_pose_lines
        imwrite(
            filename=f"{folder_dir}/14_image_pose_worthy_lines.png",
            img=img_pose_lines
        )

        # 15 ---- image left hand worthy dots
        #        --> img_left_hand_dots
        imwrite(
            filename=f"{folder_dir}/15_image_left_hand_dots.png",
            img=img_left_hand_dots
        )
        # 16 ---- image left hand worthy lines
        #        --> img_left_hand_lines
        imwrite(
            filename=f"{folder_dir}/16_image_left_hand_lines.png",
            img=img_left_hand_lines
        )

        # 17 ---- image right hand worthy dots
        #        --> img_right_hand_dots
        imwrite(
            filename=f"{folder_dir}/17_image_right_hand_dots.png",
            img=img_right_hand_dots
        )
        # 18 ---- image right hand worthy lines
        #        --> img_right_hand_lines
        imwrite(
            filename=f"{folder_dir}/18_image_right_hand_lines.png",
            img=img_right_hand_lines
        )
    else:
        print(f"file does not exist: {WHOLE_BODY_FILE_STR}")
