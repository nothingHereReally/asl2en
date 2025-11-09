from mediapipe.python.solutions.holistic import Holistic
from numpy import array, float32, ndarray, uint8, zeros
from os.path import exists
from os import makedirs
from random import uniform
from cv2 import COLOR_BGR2RGB, circle, cvtColor, imread, imwrite, line


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
WHOLE_BODY_FILE_STR: str= f"{PROJ_ROOT}src_asl2gloss/pics_others/human_whole_body.png"
IMG_SIZE: int= 158
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
def drawFacePoseHand(img_write_to: ndarray, lmark_mph, orig_shape: tuple) -> tuple:
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
            wx_hy: int= img_write_to.shape[0]
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


    def recalcDrawFace_full_connections(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS_FULL,
            thick=1,
            color_conn=(0, 153, 0), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawPose_full_connections(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS_FULL,
            thick=1,
            color_conn=(0, 0, 153), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawFace_worthy_connections(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS,
            thick=1,
            color_conn=(0, 153, 0), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawPose_worthy_connections(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 153), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawLeftHand_connections(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_lhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(255, 255, 255),
            drawJoint=False
        )
    def recalcDrawRightHand_connections(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_rhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(204, 204, 204), # 204/255= 0.8
            drawJoint=False
        )
    img_dots_full: ndarray= img_write_to.copy()
    img_dots_worthy: ndarray= img_write_to.copy()
    img_connections_full: ndarray= img_write_to.copy()
    img_connections_worthy: ndarray= img_write_to.copy()


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
    landmark__face_pose_left_right_hand= zeros(((36 +8 +(21*2)), 2), dtype=float32)
    landmark__face_pose_left_right_hand= landmark__face_pose_left_right_hand.tolist()
    landmark__face_pose_left_right_hand_full= zeros(((468 +33 +(21*2)), 2), dtype=float32)
    landmark__face_pose_left_right_hand_full= landmark__face_pose_left_right_hand_full.tolist()
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
        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0
        if lmark_mph.face_landmarks != None:
            for i in enumerate(lmark_mph.face_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_face_full.append((  (i[1]).x, (i[1]).y  ))
                if int(i[0]) in WORTHY_FACE_IDX:
                    recalc_face.append((  (i[1]).x, (i[1]).y  ))
                    all_x.append( (i[1]).x )
                    all_y.append( (i[1]).y )
        if lmark_mph.pose_landmarks != None:
            for i in enumerate(lmark_mph.pose_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_pose_full.append((  (i[1]).x, (i[1]).y  ))
                if int(i[0]) in WORTHY_POSE_IDX:
                    recalc_pose.append((  (i[1]).x, (i[1]).y  ))
                    all_x.append( (i[1]).x )
                    all_y.append( (i[1]).y )
        if lmark_mph.left_hand_landmarks != None:
            for i in enumerate(lmark_mph.left_hand_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_left_hand_full.append((  (i[1]).x, (i[1]).y  ))
                recalc_left_hand.append((  (i[1]).x, (i[1]).y  ))
                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )
        if lmark_mph.right_hand_landmarks != None:
            for i in enumerate(lmark_mph.right_hand_landmarks.landmark):
                all_x_full.append( (i[1]).x )
                all_y_full.append( (i[1]).y )
                recalc_right_hand_full.append((  (i[1]).x, (i[1]).y  ))
                recalc_right_hand.append((  (i[1]).x, (i[1]).y  ))
                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )


        ### 0) all coords be greater than|= 0.0 and less than|= 1.0
        # force all be greater than or = to 0.0, ie. move right/down
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )


        ### 1) from old img ratio to new ratio(ie. square img )
        # remap coords( x,y ) to rescale( same ratio as orig ) on square
        # and also center orig img to New img sqaure
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part2_beSquareRatioOnImage(
            orig_shape=orig_shape,
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )


        ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
        # zoom in/out for padding be 10% each side with respect to original aspect ratio
        # ie.:
        # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
        # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
        # pad: float= 0.05
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part3_zoomInOutForPadding(
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )


        ### 3) center landmark with same aspect ratio as original
        # center horizontally and vertically, since done padding then just
        # move to right/down
        all_x, all_y, recalc_face, recalc_pose, recalc_left_hand, recalc_right_hand= part4_centerLandmarkVerticallyHorizontally(
            all_x=all_x,
            all_y=all_y,
            landmark_face=recalc_face,
            landmark_pose=recalc_pose,
            landmark_left_hand=recalc_left_hand,
            landmark_right_hand=recalc_right_hand
        )
        del all_x
        del all_y


        # done calcuation now mandatory all landamrks be constant
        recalc_face= tuple(tuple(i) for i in recalc_face)
        recalc_pose= tuple(tuple(i) for i in recalc_pose)
        recalc_left_hand= tuple(tuple(i) for i in recalc_left_hand)
        recalc_right_hand= tuple(tuple(i) for i in recalc_right_hand)


        # 4 main pics
        # ---- image full connections
        # ---- image full dots
        # ---- image worthy connections
        # ---- image worthy dots
        # --------------------------------
        # -- landmarks transformation( from what to how many )
        # 468 --> 36 on face
        # 33  --> 8  on pose
        # 21 --> 21 on left hand
        # 21 --> 21 on right hand
        # --------------------------------
        # -- face
        landmark__face_full= zeros((468, 2), dtype=float32)
        landmark__face= zeros((36, 2), dtype=float32)
        if lmark_mph.face_landmarks != None:
            landmark__face_full= tuple(recalc_face)
            landmark__face= tuple(landmark__face_full[i] for i in WORTHY_FACE_IDX) # shape is (36, 2)

            img_dots_full= recalcDrawFace_full_dots(img_dots_full, landmark__face_full)
            img_dots_worthy= recalcDrawFace_worthy_dots(img_dots_worthy, landmark__face)

            img_connections_full= recalcDrawFace_full_connections(img_connections_full, landmark__face_full)
            img_connections_worthy= recalcDrawFace_worthy_connections(img_connections_worthy, landmark__face)
        # -------------------------------------------------------------------
        # -- pose
        landmark__pose_full= zeros((33, 2), dtype=float32)
        landmark__pose= zeros((8, 2), dtype=float32)
        if lmark_mph.pose_landmarks != None:
            landmark__pose_full= tuple(recalc_pose)
            landmark__pose= tuple(landmark__pose_full[i] for i in WORTHY_POSE_IDX) # shape is (8, 2)

            img_dots_full= recalcDrawPose_full_dots(img_dots_full, landmark__pose_full)
            img_dots_worthy= recalcDrawPose_worthy_dots(img_dots_worthy, landmark__pose)

            img_connections_full= recalcDrawPose_full_connections(img_connections_full, landmark__pose_full)
            img_connections_worthy= recalcDrawPose_worthy_connections(img_connections_worthy, landmark__pose)
        # -------------------------------------------------------------------
        # -- left hand
        landmark__left_hand= zeros((21, 2), dtype=float32)
        landmark__left_hand= tuple(tuple(i) for i in landmark__left_hand.tolist())
        if lmark_mph.left_hand_landmarks != None:
            landmark__left_hand= tuple(recalc_left_hand)

            img_dots_full= recalcDrawLeftHand_dots(img_dots_full, landmark__left_hand)
            img_dots_worthy= recalcDrawLeftHand_dots(img_dots_worthy, landmark__left_hand)

            img_connections_full= recalcDrawLeftHand_connections(img_connections_full, landmark__left_hand)
            img_connections_worthy= recalcDrawLeftHand_connections(img_connections_worthy, landmark__left_hand)
        # -------------------------------------------------------------------
        # -- right hand
        landmark__right_hand= zeros((21, 2), dtype=float32)
        landmark__right_hand= tuple(tuple(i) for i in landmark__right_hand.tolist())
        if lmark_mph.right_hand_landmarks != None:
            landmark__right_hand= tuple(recalc_right_hand)

            img_dots_full= recalcDrawRightHand_dots(img_dots_full, landmark__right_hand)
            img_dots_worthy= recalcDrawRightHand_dots(img_dots_worthy, landmark__right_hand)

            img_connections_full= recalcDrawRightHand_connections(img_connections_full, landmark__right_hand)
            img_connections_worthy= recalcDrawRightHand_connections(img_connections_worthy, landmark__right_hand)
        # -------------------------------------------------------------------

        landmark__face_pose_left_right_hand= []
        landmark__face_pose_left_right_hand.extend(landmark__face)
        landmark__face_pose_left_right_hand.extend(landmark__pose)
        landmark__face_pose_left_right_hand.extend(landmark__left_hand)
        landmark__face_pose_left_right_hand.extend(landmark__right_hand)
        landmark__face_pose_left_right_hand_full= []
        landmark__face_pose_left_right_hand_full.extend(landmark__face_full)
        landmark__face_pose_left_right_hand_full.extend(landmark__pose_full)
        landmark__face_pose_left_right_hand_full.extend(landmark__left_hand)
        landmark__face_pose_left_right_hand_full.extend(landmark__right_hand)

    return (
        (img_dots_full, img_dots_worthy, img_connections_full, img_connections_worthy),
        landmark__face_pose_left_right_hand_full,
        landmark__face_pose_left_right_hand,
    )


if __name__=='__main__':
    if exists(WHOLE_BODY_FILE_STR):
        image_origin= imread(WHOLE_BODY_FILE_STR)
        image_origin= cvtColor(src=image_origin, code=COLOR_BGR2RGB).copy()
        image_origin= array(image_origin, dtype=uint8)
        (img_dots_f, img_dots_w, img_con_f, img_con_w), lm_full, lm_worthy= drawFacePoseHand(
            img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
            lmark_mph=MPH_fph.process(image_origin),
            orig_shape=image_origin.shape
        )
        folder_dir: str= f"/tmp/p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"
        while exists(folder_dir):
            folder_dir: str= f"/tmp/p6_check_image_landmarks_worthy_{int(uniform(0,1)*1000)}"
        makedirs(folder_dir)
        imwrite(
            filename=f"{folder_dir}/image_dots_full.png",
            img=img_dots_f
        )
        imwrite(
            filename=f"{folder_dir}/image_dots_worthy.png",
            img=img_dots_w
        )
        imwrite(
            filename=f"{folder_dir}/image_connections_full.png",
            img=img_con_f
        )
        imwrite(
            filename=f"{folder_dir}/image_connections_worthy.png",
            img=img_con_w
        )
    else:
        print(f"file does not exist: {WHOLE_BODY_FILE_STR}")
