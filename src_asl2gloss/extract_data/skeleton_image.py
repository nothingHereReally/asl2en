from json import dump
from os.path import exists
from os import listdir, makedirs
from typing import Any
from cv2 import (
    imread,
    imwrite,
    COLOR_BGR2RGB,
    cvtColor,
    line,
    circle
)
from mediapipe.python.solutions.holistic import Holistic
from numpy import float32, load as loadnpy, ndarray, array, save, uint8, zeros
from json import load as loadjson


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
FACE_CONNECTIONS: tuple= (
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


IMG_SIZE: int= 158




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
def drawFacePoseHand(img_orig: ndarray, lmark_mph, orig_shape: tuple) -> tuple:
    def recalcDrawFace(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS,
            thick=1,
            color_conn=(0, 153, 0), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawPose(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 153), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawLeftHands(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_lhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(255, 255, 255),
            drawJoint=False
        )
    def recalcDrawRightHands(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_rhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(204, 204, 204), # 204/255= 0.8
            drawJoint=False
        )
    img: ndarray= img_orig.copy()


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
    landmark__face_pose_left_right_hand= zeros(((468+8+(21*2)), 2), dtype=float32)
    landmark__face_pose_left_right_hand= landmark__face_pose_left_right_hand.tolist()
    if lmark_mph.face_landmarks!=None \
        or lmark_mph.pose_landmarks!=None \
        or lmark_mph.left_hand_landmarks!=None \
        or lmark_mph.right_hand_landmarks!=None:
        recalc_lmark_face= []
        recalc_lmark_pose= []
        recalc_lmark_left_hand= []
        recalc_lmark_right_hand= []
        all_x= []
        all_y= []
        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0
        if lmark_mph.face_landmarks != None:
            for i in enumerate(lmark_mph.face_landmarks.landmark):
                recalc_lmark_face.append((  (i[1]).x, (i[1]).y  ))
                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )
        if lmark_mph.pose_landmarks != None:
            for i in enumerate(lmark_mph.pose_landmarks.landmark):
                if int(i[0]) in WORTHY_POSE_IDX:
                    recalc_lmark_pose.append((  (i[1]).x, (i[1]).y  ))
                    all_x.append( (i[1]).x )
                    all_y.append( (i[1]).y )
        if lmark_mph.left_hand_landmarks != None:
            for i in enumerate(lmark_mph.left_hand_landmarks.landmark):
                recalc_lmark_left_hand.append((  (i[1]).x, (i[1]).y  ))
                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )
        if lmark_mph.right_hand_landmarks != None:
            for i in enumerate(lmark_mph.right_hand_landmarks.landmark):
                recalc_lmark_right_hand.append((  (i[1]).x, (i[1]).y  ))
                all_x.append( (i[1]).x )
                all_y.append( (i[1]).y )
        all_x= tuple(all_x)
        all_y= tuple(all_y)
        min_x= float(min(all_x))
        min_y= float(min(all_y))


        ### 0) all coords be greater than|= 0.0 and less than|= 1.0
        # force all be greater than or = to 0.0, ie. move right/down
        if min_x<0.0: # move right
            all_x= []
            if 0<len(recalc_lmark_face):
                recalc_lmark_face= [(i[0]+abs(min_x), i[1])
                                    for i in recalc_lmark_face]
                all_x.extend([i[0] for i in recalc_lmark_face])
            if 0<len(recalc_lmark_pose):
                recalc_lmark_pose= [(i[0]+abs(min_x), i[1])
                                    for i in recalc_lmark_pose]
                all_x.extend([i[0] for i in recalc_lmark_pose])
            if 0<len(recalc_lmark_left_hand):
                recalc_lmark_left_hand= [(i[0]+abs(min_x), i[1])
                                    for i in recalc_lmark_left_hand]
                all_x.extend([i[0] for i in recalc_lmark_left_hand])
            if 0<len(recalc_lmark_right_hand):
                recalc_lmark_right_hand= [(i[0]+abs(min_x), i[1])
                                    for i in recalc_lmark_right_hand]
                all_x.extend([i[0] for i in recalc_lmark_right_hand])
            min_x= 0.0
            all_x= tuple(all_x)
        if min_y<0.0: # move down
            all_y= []
            if 0<len(recalc_lmark_face):
                recalc_lmark_face= [(i[0], i[1]+abs(min_y))
                                    for i in recalc_lmark_face]
                all_y.extend([i[1] for i in recalc_lmark_face])
            if 0<len(recalc_lmark_pose):
                recalc_lmark_pose= [(i[0], i[1]+abs(min_y))
                                    for i in recalc_lmark_pose]
                all_y.extend([i[1] for i in recalc_lmark_pose])
            if 0<len(recalc_lmark_left_hand):
                recalc_lmark_left_hand= [(i[0], i[1]+abs(min_y))
                                    for i in recalc_lmark_left_hand]
                all_y.extend([i[1] for i in recalc_lmark_left_hand])
            if 0<len(recalc_lmark_right_hand):
                recalc_lmark_right_hand= [(i[0], i[1]+abs(min_y))
                                    for i in recalc_lmark_right_hand]
                all_y.extend([i[1] for i in recalc_lmark_right_hand])
            min_y= 0.0
            all_y= tuple(all_y)
        # force all be less than or = to 1.0
        # makes maximum be 1.0, due to max/max= 1.0
        max_xy= max([float(max(all_x)), float(max(all_y))])
        if 1.0<max_xy:
            all_x= []
            all_y= []
            if 0<len(recalc_lmark_face):
                recalc_lmark_face= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in recalc_lmark_face]
                all_x.extend([i[0] for i in recalc_lmark_face])
                all_y.extend([i[1] for i in recalc_lmark_face])
            if 0<len(recalc_lmark_pose):
                recalc_lmark_pose= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in recalc_lmark_pose]
                all_x.extend([i[0] for i in recalc_lmark_pose])
                all_y.extend([i[1] for i in recalc_lmark_pose])
            if 0<len(recalc_lmark_left_hand):
                recalc_lmark_left_hand= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in recalc_lmark_left_hand]
                all_x.extend([i[0] for i in recalc_lmark_left_hand])
                all_y.extend([i[1] for i in recalc_lmark_left_hand])
            if 0<len(recalc_lmark_right_hand):
                recalc_lmark_right_hand= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in recalc_lmark_right_hand]
                all_x.extend([i[0] for i in recalc_lmark_right_hand])
                all_y.extend([i[1] for i in recalc_lmark_right_hand])
            all_x= tuple(all_x)
            all_y= tuple(all_y)
            min_x= min(all_x)
            min_y= min(all_y)
        del max_xy


        ### 1) from old img ratio to new ratio(ie. square img )
        # remap coords( x,y ) to rescale( same ratio as orig ) on square
        # and also center orig img to New img sqaure
        if orig_shape[0]!=orig_shape[1]: # else equal, then don't touch it
            owx: int= int(orig_shape[1])
            ohy: int= int(orig_shape[0])
            wx_hy: int= img.shape[0]
            if owx<ohy: # just overwrite x with respect to now on square
                all_x= []
                ccc: float= (wx_hy*owx/ohy)/wx_hy # rescale
                if 0<len(recalc_lmark_face):
                    recalc_lmark_face= [(i[0]*ccc, i[1])
                                        for i in recalc_lmark_face]
                    all_x.extend([i[0] for i in recalc_lmark_face])
                if 0<len(recalc_lmark_pose):
                    recalc_lmark_pose= [(i[0]*ccc, i[1])
                                        for i in recalc_lmark_pose]
                    all_x.extend([i[0] for i in recalc_lmark_pose])
                if 0<len(recalc_lmark_left_hand):
                    recalc_lmark_left_hand= [(i[0]*ccc, i[1])
                                        for i in recalc_lmark_left_hand]
                    all_x.extend([i[0] for i in recalc_lmark_left_hand])
                if 0<len(recalc_lmark_right_hand):
                    recalc_lmark_right_hand= [(i[0]*ccc, i[1])
                                        for i in recalc_lmark_right_hand]
                    all_x.extend([i[0] for i in recalc_lmark_right_hand])
                all_x= tuple(all_x)
                min_x= min(all_x)
            else: # ohy < owx, just overwrite y with respect to now on square
                all_y= []
                ccc: float= (wx_hy*ohy/owx)/wx_hy # rescale
                if 0<len(recalc_lmark_face):
                    recalc_lmark_face= [(i[0], i[1]*ccc)
                                        for i in recalc_lmark_face]
                    all_y.extend([i[1] for i in recalc_lmark_face])
                if 0<len(recalc_lmark_pose):
                    recalc_lmark_pose= [(i[0], i[1]*ccc)
                                        for i in recalc_lmark_pose]
                    all_y.extend([i[1] for i in recalc_lmark_pose])
                if 0<len(recalc_lmark_left_hand):
                    recalc_lmark_left_hand= [(i[0], i[1]*ccc)
                                        for i in recalc_lmark_left_hand]
                    all_y.extend([i[1] for i in recalc_lmark_left_hand])
                if 0<len(recalc_lmark_right_hand):
                    recalc_lmark_right_hand= [(i[0], i[1]*ccc)
                                        for i in recalc_lmark_right_hand]
                    all_y.extend([i[1] for i in recalc_lmark_right_hand])
                all_y= tuple(all_y)
                min_y= min(all_y)
            del owx
            del ohy
            del wx_hy


        ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
        # zoom in/out for padding be 10% each side with respect to original aspect ratio
        # ie.:
        # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
        # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
        # pad: float= 0.05
        pad: float= 4.0/IMG_SIZE
        # scale: float= (1.0 -2.0*pad)/max_wy_hy, 0.0< max_wy_hy <=1.0
        # scale: float= (whole -pad_leftRight_upDown)/max_wy_hy, 0.0< max_wy_hy <=1.0
        scale: float= (1.0 -2.0*pad)/max((  max(all_x)-min_x, max(all_y)-min_y  ))
        all_x= []
        all_y= []
        if 0<len(recalc_lmark_face):
            recalc_lmark_face= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in recalc_lmark_face]
            all_x.extend([i[0] for i in recalc_lmark_face])
            all_y.extend([i[1] for i in recalc_lmark_face])
        if 0<len(recalc_lmark_pose):
            recalc_lmark_pose= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in recalc_lmark_pose]
            all_x.extend([i[0] for i in recalc_lmark_pose])
            all_y.extend([i[1] for i in recalc_lmark_pose])
        if 0<len(recalc_lmark_left_hand):
            recalc_lmark_left_hand= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in recalc_lmark_left_hand]
            all_x.extend([i[0] for i in recalc_lmark_left_hand])
            all_y.extend([i[1] for i in recalc_lmark_left_hand])
        if 0<len(recalc_lmark_right_hand):
            recalc_lmark_right_hand= [((i[0]-min_x)*scale +pad, (i[1]-min_y)*scale +pad)
                                for i in recalc_lmark_right_hand]
            all_x.extend([i[0] for i in recalc_lmark_right_hand])
            all_y.extend([i[1] for i in recalc_lmark_right_hand])
        del pad
        del scale
        all_x= tuple(all_x)
        all_y= tuple(all_y)
        min_x= min(all_x)
        min_y= min(all_y)


        ### 3) center landmark with same aspect ratio as original
        # center horizontally and vertically, since done padding then just
        # move to right/down
        lm_wx: float= max(all_x)-min_x
        lm_hy: float= max(all_y)-min_y
        if lm_wx < lm_hy:
            # all_x= []
            shift_x_right= (1.0 -lm_wx) /2.0 -min_x
            recalc_lmark_face= [(i[0]+shift_x_right, i[1])
                                for i in recalc_lmark_face]
            # all_x.extend([i[0] for i in recalc_lmark_face])
            recalc_lmark_pose= [(i[0]+shift_x_right, i[1])
                                for i in recalc_lmark_pose]
            # all_x.extend([i[0] for i in recalc_lmark_pose])
            recalc_lmark_left_hand= [(i[0]+shift_x_right, i[1])
                                for i in recalc_lmark_left_hand]
            # all_x.extend([i[0] for i in recalc_lmark_left_hand])
            recalc_lmark_right_hand= [(i[0]+shift_x_right, i[1])
                                for i in recalc_lmark_right_hand]
            # all_x.extend([i[0] for i in recalc_lmark_right_hand])
            # all_x= tuple(all_x)
            # min_x= min(all_x)
        elif lm_hy < lm_wx:
            # all_y= []
            shift_y_down= (1.0 -lm_hy) /2.0 -min_y
            recalc_lmark_face= [(i[0], i[1]+shift_y_down)
                                for i in recalc_lmark_face]
            # all_y.extend([i[1] for i in recalc_lmark_face])
            recalc_lmark_pose= [(i[0], i[1]+shift_y_down)
                                for i in recalc_lmark_pose]
            # all_y.extend([i[1] for i in recalc_lmark_pose])
            recalc_lmark_left_hand= [(i[0], i[1]+shift_y_down)
                                for i in recalc_lmark_left_hand]
            # all_y.extend([i[1] for i in recalc_lmark_left_hand])
            recalc_lmark_right_hand= [(i[0], i[1]+shift_y_down)
                                for i in recalc_lmark_right_hand]
            # all_y.extend([i[1] for i in recalc_lmark_right_hand])
            # all_y= tuple(all_y)
            # min_y= min(all_y)
        del lm_wx
        del lm_hy
        # shift_x= 0.5 -(max(all_x)+min_x)/2
        # shift_y= 0.5 -(max(all_y)+min_y)/2
        # if 0<len(recalc_lmark_face):
        #     recalc_lmark_face= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in recalc_lmark_face]
        # if 0<len(recalc_lmark_pose):
        #     recalc_lmark_pose= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in recalc_lmark_pose]
        # if 0<len(recalc_lmark_left_hand):
        #     recalc_lmark_left_hand= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in recalc_lmark_left_hand]
        # if 0<len(recalc_lmark_right_hand):
        #     recalc_lmark_right_hand= [(i[0]+shift_x, i[1]+shift_y)
        #                         for i in recalc_lmark_right_hand]
        # print(f"len(all_x) {len(all_x)}")
        # print(f"len(all_y) {len(all_y)}")
        # print(f"min_x {min_x} ---- max x {max(all_x)}")
        # print(f"min_y {min_y} ---- max y {max(all_y)}")
        del all_x
        del all_y
        del min_x
        del min_y


        landmark__face= zeros((468, 2), dtype=float32)
        if lmark_mph.face_landmarks != None:
            img= recalcDrawFace(img, tuple(recalc_lmark_face))
            landmark__face= array(recalc_lmark_face, dtype=float32)
        landmark__face= landmark__face.tolist()

        landmark__pose= zeros((8, 2), dtype=float32)
        if lmark_mph.pose_landmarks != None:
            img= recalcDrawPose(img, tuple(recalc_lmark_pose))
            landmark__pose= array(recalc_lmark_pose, dtype=float32)
        landmark__pose= landmark__pose.tolist()

        landmark__left_hand= zeros((21, 2), dtype=float32)
        if lmark_mph.left_hand_landmarks != None:
            img= recalcDrawLeftHands(img, tuple(recalc_lmark_left_hand))
            landmark__left_hand= array(recalc_lmark_left_hand, dtype=float32)
        landmark__left_hand= landmark__left_hand.tolist()

        landmark__right_hand= zeros((21, 2), dtype=float32)
        if lmark_mph.right_hand_landmarks != None:
            img= recalcDrawRightHands(img, tuple(recalc_lmark_right_hand))
            landmark__right_hand= array(recalc_lmark_right_hand, dtype=float32)
        landmark__right_hand= landmark__right_hand.tolist()

        landmark__face_pose_left_right_hand= []
        landmark__face_pose_left_right_hand.extend(landmark__face)
        landmark__face_pose_left_right_hand.extend(landmark__pose)
        landmark__face_pose_left_right_hand.extend(landmark__left_hand)
        landmark__face_pose_left_right_hand.extend(landmark__right_hand)

    return (img, landmark__face_pose_left_right_hand)
def getSkeletonFrames(fpath_vid: str, mpH: Any=None) -> tuple:
    def getAllImg_frames(vidpath: str) -> list:
        image_files: list= sorted(listdir(vidpath))
        out= []

        for img_file in image_files:
            out.append(array(cvtColor(
                src=imread(f"{vidpath}/{img_file}"),
                code=COLOR_BGR2RGB
            ).copy(), dtype=uint8))
        return out
    allImg_human: list= getAllImg_frames(fpath_vid)
    allImg_skeleton: list= []
    allImg_skeleton_ann: list= []
    allLandMark_ann: list= []
    oqFRAMES: int= int(len(allImg_human))
    if oqFRAMES<=0:
        raise FileExistsError(f"image folder {fpath_vid.rsplit("/")[-1]} no image file exist")
    for img in allImg_human:
        fph_lmark= mpH.process(img)
        skeleton__image, landmark__fph= drawFacePoseHand(
            img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
            lmark_mph=fph_lmark,
            orig_shape=img.shape
        )
        allImg_skeleton.append(skeleton__image)
        allLandMark_ann.append(landmark__fph)
        allImg_skeleton_ann.append({
            'face': fph_lmark.face_landmarks != None,
            'pose': fph_lmark.face_landmarks != None,
            'left_hand': fph_lmark.left_hand_landmarks != None,
            'right_hand': fph_lmark.right_hand_landmarks != None,
            'width': IMG_SIZE,
            'height': IMG_SIZE
        })
    del allImg_human
    del oqFRAMES
    return (
        array(allImg_skeleton, dtype=uint8),
        allImg_skeleton_ann,
        allLandMark_ann
    )




if __name__=='__main__':
    WLASL_tvt_file: str= f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.train_val_test.json"
    WLASL_raw_img: str= f"{PROJ_ROOT}dataset/wlasl/raw_image/"
    wlasl_annotation: dict= {}
    with open(WLASL_tvt_file, 'r') as f:
        wlasl_annotation= loadjson(f)

    wlasl_root: str= f"{PROJ_ROOT}dataset/wlasl/"
    write_to_skeleton: str= f"{wlasl_root}skeleton_image/"
    write_to_landmark: str= f"{wlasl_root}landmark_numpy/"
    mpH_fph: Holistic= Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    TrainValTest: tuple= ("train", "val", "test")
    if exists(wlasl_root) and not exists(write_to_skeleton):
        makedirs(f"{write_to_skeleton}")
        print(f"writing to {write_to_skeleton}")
        skeleton_annotation: dict= {
            'train': [],
            'val': [],
            'test': [],
            'label_id2gloss': wlasl_annotation['label_id2gloss'],
            'label_gloss2id': wlasl_annotation['label_gloss2id']
        }
        landmark_annotation: dict= {
            'train': [],
            'val': [],
            'test': [],
            'label_id2gloss': wlasl_annotation['label_id2gloss'],
            'label_gloss2id': wlasl_annotation['label_gloss2id']
        }
        for tvt in TrainValTest:
            print(f"processing {tvt}...")
            total_qvideo: int= len(wlasl_annotation[tvt])
            for trainValTest_ins, video_idx in zip(wlasl_annotation[tvt], range(total_qvideo)):
                if video_idx in (int(total_qvideo*.1*iii) for iii in range(1, 10)):
                    print(f"processing at {(video_idx+1)/total_qvideo*100}%")
                imgFolder4singleVideo: str= f"{WLASL_raw_img}{trainValTest_ins['video_id']}"
                if exists(imgFolder4singleVideo) and 0<len(listdir(imgFolder4singleVideo)):
                    images, sktn_ann, landmarks_on_video= getSkeletonFrames(
                        fpath_vid=imgFolder4singleVideo,
                        mpH=mpH_fph
                    )
                    if 0<len(images):
                        makedirs(f"{write_to_skeleton}{trainValTest_ins['video_id']}")
                        makedirs(f"{write_to_landmark}{trainValTest_ins['video_id']}")
                        skeleton_annotation[tvt].append({
                            'gloss_id': int(trainValTest_ins['gloss_id']),
                            'video_id': str(trainValTest_ins['video_id']),
                            'image': []
                        })
                        landmark_annotation[tvt].append({
                            'gloss_id': int(trainValTest_ins['gloss_id']),
                            'video_id': str(trainValTest_ins['video_id']),
                            'landmark': []
                        })
                        for i in range(len(images)):
                            file_id: str= ''
                            if i<9:
                                file_id= f"0000{i+1}"
                            elif i<99:
                                file_id= f"000{i+1}"
                            elif i<999:
                                file_id= f"00{i+1}"
                            elif i<9999:
                                file_id= f"0{i+1}"
                            else:
                                file_id= f"{i+1}"
                            imwrite(
                                filename=f"{write_to_skeleton}{trainValTest_ins['video_id']}/{file_id}.png",
                                img=images[i]
                            )
                            skeleton_annotation[tvt][-1]['image'].append({
                                'file': f"{file_id}.png",
                                'face': sktn_ann[i]['face'],
                                'pose': sktn_ann[i]['pose'],
                                'left_hand': sktn_ann[i]['left_hand'],
                                'right_hand': sktn_ann[i]['right_hand'],
                                'width': sktn_ann[i]['width'],
                                'height': sktn_ann[i]['height'],
                            })
                            with open(f"{write_to_landmark}{trainValTest_ins['video_id']}/{file_id}.npy", 'wb') as f:
                                save(f, array(landmarks_on_video[i], dtype=float32))
                            landmark_annotation[tvt][-1]['landmark'].append({
                                'file': f"{file_id}.npy",
                                'face': sktn_ann[i]['face'],
                                'pose': sktn_ann[i]['pose'],
                                'left_hand': sktn_ann[i]['left_hand'],
                                'right_hand': sktn_ann[i]['right_hand'],
                            })
        with open(f"{wlasl_root}wlasl.annotation.skeleton_image.train_val_test.json", 'w') as f:
            dump(skeleton_annotation, f, indent=4)
        with open(f"{wlasl_root}wlasl.annotation.landmark_numpy.train_val_test.json", 'w') as f:
            dump(landmark_annotation, f, indent=4)
        print("creating skeleton images done...")
        landmark_test_shape= None
        with open(f"{write_to_landmark}23199/00032.npy", 'rb') as f:
            landmark_test_shape= loadnpy(f)
        print(f"landmark shape {landmark_test_shape.shape}")
    else:
        print(f"{wlasl_root} doesn't\nexist, please get the dataset 1st")
        print(f"or if do exist, please delete this\ndirectory {write_to_skeleton}")


