from random import shuffle
from typing import Any, Generator
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, circle, cvtColor, imread, line
from numpy import array, concatenate, float32, ndarray, reshape, uint16, uint8, zeros, load as loadnp
# from json import dump as dumpjson, load as loadjson
from math import ceil
from os.path import exists

from .lmark_constant import (
    KEY_TRAIN,
    LANDMARK_SHAPE,
    SKELETON_SHAPE,
    LEN_TRAIN,
    LEN_VAL,
    ON_TRAINING_BATCH,
    IMG_SIZE,
    QUANTITY_FRAME,

    FACE_CONNECTIONS,
    POSE_CONNECTIONS,
    HAND_CONNECTIONS,
    GLASL_LANDMARK_DIR,
    GLASL_SKELETON_DIR,
    WORTHY_FACE_IDX,
    WORTHY_POSE_IDX,

    glasl_landmark,
    glasl_skeleton,
)








def drawSkeletonImg(img_orig: ndarray, \
                    lmark_cords: tuple, \
                    conn_idxs_list: tuple, \
                    thick: int=2, \
                    color_conn: tuple=(255,0,255), \
                    color_lmark: tuple=(255,255,0), \
                    drawJoint: bool=True) -> ndarray:
    def isOKplt(coord: tuple) -> bool:
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








def drawFacePoseHand(img_orig: ndarray, lmark_mph, orig_shape: tuple) -> ndarray:
    def recalcDrawFace(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS,
            thick=1,
            color_conn=(0,255,0), # 255/255= 1.0
            # color_conn=(255,255,255), # blackNwhite
            drawJoint=False
        )
    def recalcDrawPose(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            # thick=4,
            thick=1,
            color_conn=(51, 204, 204), # 204/255= 0.8
            color_lmark=(204, 204, 51), # 51/255= 0.2
            # color_conn=(255,255,255), # blackNwhite
            # color_lmark=(255,255,255), # blackNwhite
            drawJoint=False
        )
    def recalcDrawLeftHands(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_lhand,
            conn_idxs_list=HAND_CONNECTIONS,
            # thick=2,
            thick=1,
            color_conn=(204,0,0), # 204/255= 0.8
            color_lmark=(255,255,255) # 255/255= 1.0
            # color_conn=(204,255,255), # blackNwhite
            # color_lmark=(255,255,255) # blackNwhite
        )
    def recalcDrawRightHands(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_rhand,
            conn_idxs_list=HAND_CONNECTIONS,
            # thick=2,
            thick=1,
            color_conn=(204,0,0), # 204/255= 0.8
            color_lmark=(255,0,255) # 255/255= 1.0
            # color_conn=(255,255,255), # blackNwhite
            # color_lmark=(255,255,255) # blackNwhite
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
        pad: float= 0.02
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


        if lmark_mph.face_landmarks != None:
            img= recalcDrawFace(img, tuple(recalc_lmark_face))
        if lmark_mph.pose_landmarks != None:
            img= recalcDrawPose(img, tuple(recalc_lmark_pose))
        if lmark_mph.left_hand_landmarks != None:
            img= recalcDrawLeftHands(img, tuple(recalc_lmark_left_hand))
        if lmark_mph.right_hand_landmarks != None:
            img= recalcDrawRightHands(img, tuple(recalc_lmark_right_hand))
        # # turn to 1 channel only, to not eat so much memory
        # img= img[:, :, 0]

    return img








def getSkeletonFrames(fpath_vid: str, isSingleImg: bool=False, initGT: int= 0, mpH: Any=None, TqFRAMES: int= QUANTITY_FRAME) -> list:
    '''
    fpath_vid: str, video file path string
    TqFRAMES: int, target quantity of frames on output
    initGT: int, init Get Target at what mod, for when original frame( oqFRAMES )
        is greater than target( TqFRAMES ) then get
        at what mod, say is it mod( 0 ) ie. initGT=0,
        mod( 1 ) ie. initGT=1, ..., due to TqFRAMES<oqFRAMES, then might be that
        oqFRAMES//TqFRAMES > 1, then it would be a waste for other frames to
        not be used
    TqFRAMES: int= QUANTITY_FRAMES, output number of frames which is from the video data

    output is ndarray of image frames, from start of the video till end
    of size (TqFRAMES, IMG_SIZE, IMG_SIZE, 3) of dtype=numpy.uint8
    '''
    def getAllImg_frames(vidpath: str) -> list:
        try:
            vid= VideoCapture(vidpath)
            if vid.isOpened():
                q_images: int= int(vid.get(CAP_PROP_FRAME_COUNT))
                if q_images==0:
                    vid.release()
                    del vid
                    # destroyAllWindows() has bug, due to will make
                    # all vid.read() data prev be gone/disappear
                    # destroyAllWindows()
                    return []
                all_Imgs: list= []
                for _ in range(q_images):
                    isNotEnd, frame= vid.read()
                    frame= array(frame, dtype=uint8)
                    if isNotEnd and 0<len(frame):
                        all_Imgs.append(array(cvtColor(
                            src=frame,
                            code=COLOR_BGR2RGB
                        ).copy(), dtype=uint8))
                vid.release()
                del vid
                # destroyAllWindows() has bug, due to will make
                # all vid.read() data prev be gone/disappear
                # destroyAllWindows()
                return all_Imgs
            vid.release()
            del vid
            # destroyAllWindows() has bug, due to will make
            # all vid.read() data prev be gone/disappear
            # destroyAllWindows()
            return []
        except Exception as e:
            del e
            return []
    allImg_human: list= getAllImg_frames(fpath_vid)
    allImg_skeleton: list= []
    oqFRAMES: int= int(len(allImg_human))
    if oqFRAMES<=0:
        raise FileExistsError("file can't be opened or is corrupted")
    qHands: int= 0
    o2t_ratio: int= oqFRAMES//TqFRAMES
    if oqFRAMES<TqFRAMES:
        t2o_ratio: int= int(ceil(TqFRAMES/oqFRAMES))
        for i in range(oqFRAMES):
            fph_lmark= mpH.process(allImg_human[i])
            for ii in range(t2o_ratio):
                if fph_lmark.left_hand_landmarks!=None or fph_lmark.right_hand_landmarks!=None:
                    qHands+= 1
                # # start checking but on 2nd be opposite
                # # if (TqFRAMES-MIN_FRAMES_HAS_HANDS)<=(i*t2o_ratio +ii) and ((i*t2o_ratio +ii) -(TqFRAMES-MIN_FRAMES_HAS_HANDS))<qHands:
                # if (TqFRAMES-MIN_FRAMES_HAS_HANDS)<=(i*t2o_ratio +ii) and qHands<=((i*t2o_ratio +ii) -(TqFRAMES-MIN_FRAMES_HAS_HANDS)):
                #     del allImg_human
                #     del allImg_skeleton
                #     del oqFRAMES
                #     del t2o_ratio
                #     del fph_lmark
                #     del qHands
                #     raise FileExistsError("video not worthy to be on training due to did not meet MIN_FRAMES_HAS_HANDS")
                if isSingleImg and (i*t2o_ratio +ii)<TqFRAMES:
                    allImg_skeleton.extend(drawFacePoseHand(
                        img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                        lmark_mph=fph_lmark,
                        orig_shape=allImg_human[i].shape
                    ))
                elif (i*t2o_ratio +ii)<TqFRAMES:
                    allImg_skeleton.append(drawFacePoseHand(
                        img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                        lmark_mph=fph_lmark,
                        orig_shape=allImg_human[i].shape
                    ))
    elif oqFRAMES==TqFRAMES:
        for i in range(TqFRAMES):
            fph_lmark= mpH.process(allImg_human[i])
            if fph_lmark.left_hand_landmarks!=None or fph_lmark.right_hand_landmarks!=None:
                qHands+= 1
            # if (TqFRAMES-MIN_FRAMES_HAS_HANDS)<=i and qHands<=(i -(TqFRAMES-MIN_FRAMES_HAS_HANDS)):
            #     del allImg_human
            #     del allImg_skeleton
            #     del oqFRAMES
            #     del fph_lmark
            #     del qHands
            #     raise FileExistsError("video not worthy to be on training due to did not meet MIN_FRAMES_HAS_HANDS")
            if isSingleImg:
                allImg_skeleton.extend(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=fph_lmark,
                    orig_shape=allImg_human[i].shape
                ))
            else:
                allImg_skeleton.append(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=fph_lmark,
                    orig_shape=allImg_human[i].shape
                ))
    else: # TqFRAMES < oqFRAMES
        # o2t_ratio= oqFRAMES//TqFRAMES, due to done calculating it above
        initGT= initGT%o2t_ratio
        for i in range(TqFRAMES):
            fph_lmark= mpH.process(allImg_human[i*o2t_ratio +initGT])
            if fph_lmark.left_hand_landmarks!=None or fph_lmark.right_hand_landmarks!=None:
                qHands+= 1
            # if (TqFRAMES-MIN_FRAMES_HAS_HANDS)<=i and qHands<=(i -(TqFRAMES-MIN_FRAMES_HAS_HANDS)):
            #     del allImg_human
            #     del allImg_skeleton
            #     del oqFRAMES
            #     del o2t_ratio
            #     del fph_lmark
            #     del qHands
            #     raise FileExistsError("video not worthy to be on training due to did not meet MIN_FRAMES_HAS_HANDS")
            if isSingleImg:
                allImg_skeleton.extend(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=fph_lmark,
                    orig_shape=allImg_human[i*o2t_ratio].shape
                ))
            else:
                allImg_skeleton.append(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=fph_lmark,
                    orig_shape=allImg_human[i*o2t_ratio].shape
                ))
    if len(allImg_skeleton)!=TqFRAMES and not isSingleImg:
        raise ValueError(f"frames on single video failed match target( {
        TqFRAMES
        } ) orig( {oqFRAMES} ), but result is {
        allImg_skeleton
        } --> {fpath_vid}")
    del allImg_human
    del oqFRAMES
    del qHands
    return [
        array(allImg_skeleton, dtype=uint8),
        o2t_ratio if 1<o2t_ratio else 0
    ]








def getWorthyFacePoseHand_landmark(lmark_: tuple) -> tuple:
    # order be face then pose then left hand then right hand
    lmark_np: ndarray= array(lmark_, dtype=float32)
    if lmark_np.shape!=(QUANTITY_FRAME, 468+8+21*2, 2):
        raise ValueError(f"incorrect use, getWorthyFacePoseHand_landmark expects input of shape ({QUANTITY_FRAME}, {468+8+21*2}, 2)")

    # for face
    lmark_video= reshape(lmark_np[:, WORTHY_FACE_IDX[0], :], shape=(QUANTITY_FRAME, 1, 2))
    for i in WORTHY_FACE_IDX[1:]:
        lmark_video= concatenate((
            lmark_video,
            reshape(lmark_np[:, i, :], shape=(QUANTITY_FRAME, 1, 2))
        ), axis=1)

    # for pose and hand
    lmark_video= concatenate((lmark_video, lmark_np[:, 468:, :]), axis=1)

    return tuple(lmark_video.tolist())








def getGreaterThan_landmark(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 86, 2 ____)
    start may or maynot has hands, also forward may or maynot has hands
    '''
    lmark_numpy_MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

    lmark_all: list= []
    idx_init_has_hand: int= -1
    for i in range(len(lmark_['landmark'])):
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][i]['file']}", 'rb') as f:
            lmark_all.append(loadnp(f))
        if idx_init_has_hand==-1:
            if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                idx_init_has_hand= i
    if idx_init_has_hand==-1:
        return []

    # part 1, floor at index level, still idx_init_has_hand
    o2t_ratio: float= (len(lmark_['landmark'])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        to_append: int= idx_init_has_hand+int(i*o2t_ratio)
        lmark_numpy_MANY_VIDS[0].append(lmark_all[to_append]) # floor
    if len(lmark_numpy_MANY_VIDS[0])!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_landmark_allHasHand on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del o2t_ratio
    # lmark_numpy_MANY_VIDS[0] is of shape (QUANTITY_FRAME, 518, 2), but
    # here lmark_numpy_MANY_VIDS is of shape (1, QUANTITY_FRAME, 518, 2)

    len_available_images: int= len(lmark_['landmark'])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
        notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        # notIncludedOn_mod, due to on a single video has quantity of images( ie. len(lmark_['landmark']) )
        # then mandatory idx_init_has_hand till last has enough images for QUANTITY_FRAME
        # ie. above --> QUANTITY_FRAME<=len_available_images,
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                         ^^^^^^^^^^^^^^^^^^^^^^^____ total quanitty images on video
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^____ subtract
        # whats to be used on forward index images
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                  ^^^^^^^^^^^^^^^^^____ due below appends starts
        # at idx_init_has_hand
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                                     ^^^^^^^^^^^^^^^^^^^^^^____ due to
        # index below `ii`( represents mod ie. o2t_mod, ie. int(len_available_images/QUANTITY_FRAME) # floor,
        # ie. 0, 1, 2, ..., o2t_mod-1 ) and `iii`( represents 0, 1, 2, ..., QUANTITY_FRAME-1 ), ie.
        # for combo o2t_mod*`iii` on last part of images as (QUANTITY_FRAME, 86, 2)
        for i in range(notIncludedOn_mod+1):
            for ii in range(o2t_mod):
                lmark_numpy_MANY_VIDS.append([])
                for iii in range(QUANTITY_FRAME):
                    to_append: int= idx_init_has_hand +(iii*o2t_mod+ii) +i
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[ to_append ])
        del o2t_mod
        del notIncludedOn_mod

        # part 3, consecutive, mandatory initial has hand
        for i in range((len_available_images-QUANTITY_FRAME)+1):
            lmark_numpy_MANY_VIDS.append([])
            # due to below appends shape (QUANTITY_FRAME, 86, 2)
            for ii in range(QUANTITY_FRAME):
                to_append: int= idx_init_has_hand+ii +i
                lmark_numpy_MANY_VIDS[-1].append(lmark_all[ to_append ])
    elif len_available_images<=QUANTITY_FRAME:
        lmark_numpy_MANY_VIDS.append([])
        t2o_ratio: int= int(ceil(QUANTITY_FRAME/len_available_images)) # ceiling2make QUANTITY_FRAME possible
        for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, idx_init_has_hand+len_available_images), range(len_available_images)):
            # i_0to_t2o_multiplier, for counting later due to target is (QUANTITY_FRAME, 86, 2)
            # i_0to_t2o_multiplier, ie. 0, 1, 2, ..., int(len_available_images-1)
            # i_0to_t2o_multiplier<QUANTITY_FRAME, due to len_available_images<=QUANTITY_FRAME
            for ii in range(t2o_ratio):
                if (i_0to_t2o_multiplier*t2o_ratio +ii)<QUANTITY_FRAME:
                    # i_0to_t2o_multiplier*t2o_ratio, due to since: len_available_images<=QUANTITY_FRAME,
                    # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                    # multiple times ie. int(t2o_ratio) times
                    # then +ii, due to current be added mod of from int(t2o_ratio),
                    # thus i_0to_t2o_multiplier*t2o_ratio+ii
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[  i  ])
        if len(lmark_numpy_MANY_VIDS[-1])!=QUANTITY_FRAME:
            raise ValueError("incorrect implementation on idx idx_init_has_hand!=-1 and len_available_images<QUANTITY_FRAME")
    del len_available_images

    return lmark_numpy_MANY_VIDS


def getGreaterThan_landmark_initHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 86, 2 ____)
    start always has hands but forward may or maynot has hands
    '''
    lmark_numpy_MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

    lmark_all: list= []
    idx_init_has_hand: int= -1
    for i in range(len(lmark_['landmark'])):
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][i]['file']}", 'rb') as f:
            lmark_all.append(loadnp(f))
        if idx_init_has_hand==-1:
            if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                idx_init_has_hand= i
    if idx_init_has_hand==-1:
        return []

    # part 1, floor at index level, still idx_init_has_hand
    o2t_ratio: float= (len(lmark_['landmark'])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        to_append: int= idx_init_has_hand+int(i*o2t_ratio)
        lmark_numpy_MANY_VIDS[0].append(lmark_all[to_append]) # floor
    if len(lmark_numpy_MANY_VIDS[0])!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_landmark_allHasHand on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del o2t_ratio
    # lmark_numpy_MANY_VIDS[0] is of shape (QUANTITY_FRAME, 518, 2), but
    # here lmark_numpy_MANY_VIDS is of shape (1, QUANTITY_FRAME, 518, 2)

    len_available_images: int= len(lmark_['landmark'])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
        notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        # notIncludedOn_mod, due to on a single video has quantity of images( ie. len(lmark_['landmark']) )
        # then mandatory idx_init_has_hand till last has enough images for QUANTITY_FRAME
        # ie. above --> QUANTITY_FRAME<=len_available_images,
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                         ^^^^^^^^^^^^^^^^^^^^^^^____ total quanitty images on video
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^____ subtract
        # whats to be used on forward index images
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                  ^^^^^^^^^^^^^^^^^____ due below appends starts
        # at idx_init_has_hand
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                                     ^^^^^^^^^^^^^^^^^^^^^^____ due to
        # index below `ii`( represents mod ie. o2t_mod, ie. int(len_available_images/QUANTITY_FRAME) # floor,
        # ie. 0, 1, 2, ..., o2t_mod-1 ) and `iii`( represents 0, 1, 2, ..., QUANTITY_FRAME-1 ), ie.
        # for combo o2t_mod*`iii` on last part of images as (QUANTITY_FRAME, 86, 2)
        for i in range(notIncludedOn_mod+1):
            for ii in range(o2t_mod):
                lmark_numpy_MANY_VIDS.append([])
                for iii in range(QUANTITY_FRAME):
                    to_append: int= idx_init_has_hand +(iii*o2t_mod+ii) +i
                    if iii==0 and (lmark_['landmark'][to_append]['left_hand']==False or \
                        lmark_['landmark'][to_append]['right_hand']==False):
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[idx_init_has_hand])
                    else:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[
                            to_append
                        ])
        del o2t_mod
        del notIncludedOn_mod

        # part 3, consecutive, mandatory initial has hand
        for i in range((len_available_images-QUANTITY_FRAME)+1):
            lmark_numpy_MANY_VIDS.append([])
            # due to below appends shape (QUANTITY_FRAME, 86, 2)
            for ii in range(QUANTITY_FRAME):
                to_append: int= idx_init_has_hand+ii +i
                if ii==0 and (lmark_['landmark'][to_append]['left_hand']==False or \
                    lmark_['landmark'][to_append]['right_hand']==False):
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[idx_init_has_hand])
                else:
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[
                        to_append
                    ])
    elif len_available_images<=QUANTITY_FRAME:
        lmark_numpy_MANY_VIDS.append([])
        t2o_ratio: int= int(ceil(QUANTITY_FRAME/len_available_images)) # ceiling2make QUANTITY_FRAME possible
        for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, idx_init_has_hand+len_available_images), range(len_available_images)):
            # i_0to_t2o_multiplier, for counting later due to target is (QUANTITY_FRAME, 86, 2)
            # i_0to_t2o_multiplier, ie. 0, 1, 2, ..., int(len_available_images-1)
            # i_0to_t2o_multiplier<QUANTITY_FRAME, due to len_available_images<=QUANTITY_FRAME
            for ii in range(t2o_ratio):
                if (i_0to_t2o_multiplier*t2o_ratio +ii)<QUANTITY_FRAME:
                    # i_0to_t2o_multiplier*t2o_ratio, due to since: len_available_images<=QUANTITY_FRAME,
                    # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                    # multiple times ie. int(t2o_ratio) times
                    # then +ii, due to current be added mod of from int(t2o_ratio),
                    # thus i_0to_t2o_multiplier*t2o_ratio+ii
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[  i  ])
        if len(lmark_numpy_MANY_VIDS[-1])!=QUANTITY_FRAME:
            raise ValueError("incorrect implementation on idx idx_init_has_hand!=-1 and len_available_images<QUANTITY_FRAME")
    del len_available_images

    return lmark_numpy_MANY_VIDS


def getLessThanOrEqual_landmark_initHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) <= QUANTITY_FRAME
    start always has hands but forward may or maynot has hands
    '''
    def getIdxStartHand(image_list: list) -> int:
        for i in range(len(image_list)):
            if image_list[i]['left_hand'] or image_list[i]['right_hand']:
                return i
        return -1
    idx_init_has_hand: int= getIdxStartHand(image_list=lmark_['landmark'])
    if idx_init_has_hand==-1:
        return []
    lmark_numpy: list= []
    t2o_ratio: int= int(ceil(QUANTITY_FRAME/(len(lmark_['landmark'])-idx_init_has_hand)))
    for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, len(lmark_['landmark'])), range(len(lmark_['landmark'])-idx_init_has_hand)):
        landmark_data_numpy= None
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][  i  ]['file']}", 'rb') as f:
            landmark_data_numpy= loadnp(f)
        for ii in range(t2o_ratio):
            if (i_0to_t2o_multiplier*t2o_ratio+ii)<QUANTITY_FRAME:
                # i_0to_t2o_multiplier*t2o_ratio, due to since: getLessThanOrEqual_landmark,
                # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                # multiple/( or 1 time if equal and idx 0 has hand ) times ie. int(t2o_ratio) times
                # then +ii, due to current be added mod of from int(t2o_ratio),
                # thus i_0to_t2o_multiplier*t2o_ratio+ii
                lmark_numpy.append( landmark_data_numpy )
    if len(lmark_numpy)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getLessThanOrEqual_landmark_allHasHand, due to len(lmark_numpy)!=QUANTITY_FRAME")
    return lmark_numpy


def getGreaterThan_landmark_allHasHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 86, 2 ____)
    '''
    lmark_numpy_MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

    lmark_all: list= []
    idx_init_has_hand: int= -1
    for i in range(len(lmark_['landmark'])):
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][i]['file']}", 'rb') as f:
            lmark_all.append(loadnp(f))
        if idx_init_has_hand==-1:
            if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                idx_init_has_hand= i
    if idx_init_has_hand==-1:
        return []

    # part 1, floor at index level, still idx_init_has_hand
    o2t_ratio: float= (len(lmark_['landmark'])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        append_if_valid: int= idx_init_has_hand+int(i*o2t_ratio)
        if lmark_['landmark'][append_if_valid]['left_hand'] or \
            lmark_['landmark'][append_if_valid]['right_hand']:
            lmark_numpy_MANY_VIDS[0].append(lmark_all[append_if_valid]) # floor
        else:
            lmark_numpy_MANY_VIDS[0].append(lmark_numpy_MANY_VIDS[0][-1])
    if len(lmark_numpy_MANY_VIDS[0])!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_landmark_allHasHand on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del o2t_ratio
    # lmark_numpy_MANY_VIDS[0] is of shape (QUANTITY_FRAME, 518, 2), but
    # here lmark_numpy_MANY_VIDS is of shape (1, QUANTITY_FRAME, 518, 2)

    len_available_images: int= len(lmark_['landmark'])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
        notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        # notIncludedOn_mod, due to on a single video has quantity of images( ie. len(lmark_['landmark']) )
        # then mandatory idx_init_has_hand till last has enough images for QUANTITY_FRAME
        # ie. above --> QUANTITY_FRAME<=len_available_images,
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                         ^^^^^^^^^^^^^^^^^^^^^^^____ total quanitty images on video
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^____ subtract
        # whats to be used on forward index images
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                  ^^^^^^^^^^^^^^^^^____ due below appends starts
        # at idx_init_has_hand
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                                     ^^^^^^^^^^^^^^^^^^^^^^____ due to
        # index below `ii`( represents mod ie. o2t_mod, ie. int(len_available_images/QUANTITY_FRAME) # floor,
        # ie. 0, 1, 2, ..., o2t_mod-1 ) and `iii`( represents 0, 1, 2, ..., QUANTITY_FRAME-1 ), ie.
        # for combo o2t_mod*`iii` on last part of images as (QUANTITY_FRAME, 86, 2)
        for i in range(notIncludedOn_mod+1):
            for ii in range(o2t_mod):
                lmark_numpy_MANY_VIDS.append([])
                for iii in range(QUANTITY_FRAME):
                    append_if_valid: int= idx_init_has_hand +(iii*o2t_mod+ii) +i
                    if lmark_['landmark'][append_if_valid]['left_hand'] or \
                        lmark_['landmark'][append_if_valid]['right_hand']:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[
                            append_if_valid
                        ])
                    elif iii==0:
                        # due to since iii==0 then lmark_numpy_MANY_VIDS[-1][-1] does not exist,
                        # ie. len(lmark_numpy_MANY_VIDS[-1])==0 True, due to prev at
                        # lmark_numpy_MANY_VIDS.append([]) above
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[idx_init_has_hand])
                    else:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
        del o2t_mod
        del notIncludedOn_mod

        # part 3, consecutive, mandatory initial has hand
        for i in range((len_available_images-QUANTITY_FRAME)+1):
            lmark_numpy_MANY_VIDS.append([])
            # due to below appends shape (QUANTITY_FRAME, 86, 2)
            for ii in range(QUANTITY_FRAME):
                append_if_valid: int= idx_init_has_hand+ii +i
                if lmark_['landmark'][append_if_valid]['left_hand'] or \
                    lmark_['landmark'][append_if_valid]['right_hand']:
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[
                        append_if_valid
                    ])
                elif ii==0:
                    # due to since ii==0 then lmark_numpy_MANY_VIDS[-1][-1] does not exist,
                    # ie. len(lmark_numpy_MANY_VIDS[-1])==0 True, due to prev at
                    # lmark_numpy_MANY_VIDS.append([]) above
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[idx_init_has_hand])
                else:
                    lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
    elif len_available_images<=QUANTITY_FRAME:
        lmark_numpy_MANY_VIDS.append([])
        t2o_ratio: int= int(ceil(QUANTITY_FRAME/len_available_images)) # ceiling2make QUANTITY_FRAME possible
        for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, idx_init_has_hand+len_available_images), range(len_available_images)):
            # i_0to_t2o_multiplier, for counting later due to target is (QUANTITY_FRAME, 86, 2)
            # i_0to_t2o_multiplier, ie. 0, 1, 2, ..., int(len_available_images-1)
            # i_0to_t2o_multiplier<QUANTITY_FRAME, due to len_available_images<=QUANTITY_FRAME
            for ii in range(t2o_ratio):
                if (i_0to_t2o_multiplier*t2o_ratio +ii)<QUANTITY_FRAME:
                    # i_0to_t2o_multiplier*t2o_ratio, due to since: len_available_images<=QUANTITY_FRAME,
                    # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                    # multiple times ie. int(t2o_ratio) times
                    # then +ii, due to current be added mod of from int(t2o_ratio),
                    # thus i_0to_t2o_multiplier*t2o_ratio+ii
                    if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[  i  ])
                    else:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
        if len(lmark_numpy_MANY_VIDS[-1])!=QUANTITY_FRAME:
            raise ValueError("incorrect implementation on idx idx_init_has_hand!=-1 and len_available_images<QUANTITY_FRAME")
    del len_available_images

    return lmark_numpy_MANY_VIDS


def getLessThanOrEqual_landmark_allHasHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) <= QUANTITY_FRAME
    '''
    def getIdxStartHand(image_list: list) -> int:
        for i in range(len(image_list)):
            if image_list[i]['left_hand'] or image_list[i]['right_hand']:
                return i
        return -1
    idx_init_has_hand: int= getIdxStartHand(image_list=lmark_['landmark'])
    if idx_init_has_hand==-1:
        return []
    lmark_numpy: list= []
    t2o_ratio: int= int(ceil(QUANTITY_FRAME/(len(lmark_['landmark'])-idx_init_has_hand)))
    for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, len(lmark_['landmark'])), range(len(lmark_['landmark'])-idx_init_has_hand)):
        landmark_data_numpy= None
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][  i  ]['file']}", 'rb') as f:
            landmark_data_numpy= loadnp(f)
        for ii in range(t2o_ratio):
            if (i_0to_t2o_multiplier*t2o_ratio+ii)<QUANTITY_FRAME:
                # i_0to_t2o_multiplier*t2o_ratio, due to since: getLessThanOrEqual_landmark,
                # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                # multiple/( or 1 time if equal and idx 0 has hand ) times ie. int(t2o_ratio) times
                # then +ii, due to current be added mod of from int(t2o_ratio),
                # thus i_0to_t2o_multiplier*t2o_ratio+ii
                if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                    lmark_numpy.append( landmark_data_numpy )
                else:
                    lmark_numpy.append( lmark_numpy[-1] )
    if len(lmark_numpy)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getLessThanOrEqual_landmark_allHasHand, due to len(lmark_numpy)!=QUANTITY_FRAME")
    return lmark_numpy


def getdata_landmark(trainVal: str= 'train', batch: int=ON_TRAINING_BATCH) -> Generator[tuple, None, None]:
    # glasl_READY['train']
    # glasl_READY['val']
    # glasl_READY['test']
    # glasl_READY['id2gloss']
    # glasl_READY['gloss2id']

    # each landmark numpy file is of shape (518, 2)
    shuffle(glasl_landmark[trainVal])
    b_idxINIT: int= 0
    total_q_dataset: int= LEN_TRAIN if trainVal==KEY_TRAIN else LEN_VAL
    past_landmarks: list= [] # to hold for past landmark
    # `while True:` loop runs int(TRAIN_STEPS) for every epoch
    # total_q_count, counts the quantity of video landmarks that was and is training
    # ie. past all batch_vids on instance training, ie. `p -m src_asl2gloss.model_train`, then
    # count glasl_LM[trainVal][  idx_DS  ] including repeated( video but different
    # images, due to video has many images ) due to greater
    # than QUANTITY_FRAME
    total_q_count: int= 0
    i_0toBatchOrMore: int= 0 # for glasl_LM[TrainVal][__ b_idxINIT + i_0toBatchOrMore __]
    while True:
        batch_vids: ndarray= zeros((batch, QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]), dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)


        idx_add2batch: int= 0
        # for batch_vids[__ idx_add2batch __]
        # batch_class[__ idx_add2batch __]
        # below( ie. while idx_add2batch<batch: ) runs 1 time( 1 while loop done ) per batch,
        # below only knows batch NOTHING MORE NOTHING LESS
        # does ----> NOT <---- have control on train steps and epochs
        while idx_add2batch<batch:
            idx_DS: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<total_q_dataset else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-total_q_dataset
            ))
            lmark_nplist: list= [] # at end should be of shape 22, 86, 2
            if len(past_landmarks)==0:
                folder_landmark: str= f"{GLASL_LANDMARK_DIR}{glasl_landmark[trainVal][  idx_DS  ]['video_id']}"
                if exists(folder_landmark):
                    if len(glasl_landmark[trainVal][  idx_DS  ]['landmark'])<=QUANTITY_FRAME:
                        past_landmarks.append([getLessThanOrEqual_landmark_allHasHand(glasl_landmark[trainVal][  idx_DS  ])])
                        # past_landmarks.append([getLessThanOrEqual_landmark_initHand(glasl_landmark[trainVal][  idx_DS  ])])
                    else: # quanity of image landmark is more than QUANTITY_FRAME
                        past_landmarks= getGreaterThan_landmark_allHasHand(glasl_landmark[trainVal][  idx_DS  ])
                        # past_landmarks.extend(getGreaterThan_landmark_initHand(glasl_landmark[trainVal][  idx_DS  ]))
                        # past_landmarks.extend(getGreaterThan_landmark(glasl_landmark[trainVal][  idx_DS  ]))
            if 0<len(past_landmarks):
                lmark_nplist= past_landmarks[0]
                past_landmarks= past_landmarks[1:]
                total_q_count+= 1
            if len(past_landmarks)==0 or len(lmark_nplist)==0:
                i_0toBatchOrMore+= 1
            if len(lmark_nplist)==QUANTITY_FRAME:
                batch_vids[idx_add2batch]= tuple(lmark_nplist) # array of shape(QUANTITY_FRAME, 86, 2)
                batch_class[idx_add2batch]= int(glasl_landmark[trainVal][  idx_DS  ]['gloss_id'])
                idx_add2batch+= 1
            elif len(lmark_nplist)!=0 and len(lmark_nplist)!=QUANTITY_FRAME:
                print(f"len of lmark_nplist: {len(lmark_nplist)}")
                raise ValueError("incorrect implementation on getdata_landmark, due to len(lmark_nplist)!=QUANTITY_FRAME and len(lmark_nplist)!=QUANTITY_FRAME")


            if idx_DS==(total_q_dataset-1) and len(past_landmarks)==0:
                # print(f"________ total_q_count: {total_q_count+len(past_landmarks)} ______ {trainVal}")
                total_q_count= 0


        if len(past_landmarks)==0:
            b_idxINIT= (b_idxINIT+batch) if (b_idxINIT+batch)<total_q_dataset else 0+( (b_idxINIT+batch)-total_q_dataset )
            i_0toBatchOrMore= 0
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))








def getGreaterThan_skeleton_allHasHand(skeleton_: dict) -> list:
    '''
    to be used for when len(skeleton_['skeleton']) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 158, 158, 3 ____)
    '''
    skeleton_image_MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 158, 158, 3 __)

    skeleton_all: list= []
    idx_init_has_hand: int= -1
    for i in range(len(skeleton_['skeleton'])):
        skeleton_single_image= imread( f"{GLASL_SKELETON_DIR}{skeleton_['video_id']}/{skeleton_['skeleton'][i]['file']}" )
        skeleton_single_image= cvtColor(
            src=skeleton_single_image,
            code=COLOR_BGR2RGB
        ).copy()
        skeleton_all.append(skeleton_single_image)
        if idx_init_has_hand==-1:
            if skeleton_['skeleton'][i]['left_hand'] or skeleton_['skeleton'][i]['right_hand']:
                idx_init_has_hand= i
    if idx_init_has_hand==-1:
        return []

    # part 1, floor at index level, still idx_init_has_hand
    o2t_ratio: float= (len(skeleton_['skeleton'])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        append_if_valid: int= idx_init_has_hand+int(i*o2t_ratio)
        if skeleton_['skeleton'][append_if_valid]['left_hand'] or \
            skeleton_['skeleton'][append_if_valid]['right_hand']:
            skeleton_image_MANY_VIDS[0].append(skeleton_all[append_if_valid]) # floor
        else:
            skeleton_image_MANY_VIDS[0].append(skeleton_image_MANY_VIDS[0][-1])
    if len(skeleton_image_MANY_VIDS[0])!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_skeleton_allHasHand on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del o2t_ratio
    # skeleton_image_MANY_VIDS[0] is of shape (QUANTITY_FRAME, 158, 158, 3), but
    # here skeleton_image_MANY_VIDS is of shape (1, QUANTITY_FRAME, 158, 158, 3)

    len_available_images: int= len(skeleton_['skeleton'])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
        notIncludedOn_mod: int= len(skeleton_['skeleton'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        # notIncludedOn_mod, due to on a single video has quantity of images( ie. len(skeleton_['skeleton']) )
        # then mandatory idx_init_has_hand till last has enough images for QUANTITY_FRAME
        # ie. above --> QUANTITY_FRAME<=len_available_images,
        # notIncludedOn_mod: int= len(skeleton_['skeleton'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                         ^^^^^^^^^^^^^^^^^^^^^^^____ total quanitty images on video
        # notIncludedOn_mod: int= len(skeleton_['skeleton'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^____ subtract
        # whats to be used on forward index images
        # notIncludedOn_mod: int= len(skeleton_['skeleton'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                     ^^^^^^^^^^^^^^^^^____ due below appends starts
        # at idx_init_has_hand
        # notIncludedOn_mod: int= len(skeleton_['skeleton'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                                        ^^^^^^^^^^^^^^^^^^^^^^____ due to
        # index below `ii`( represents mod ie. o2t_mod, ie. int(len_available_images/QUANTITY_FRAME) # floor,
        # ie. 0, 1, 2, ..., o2t_mod-1 ) and `iii`( represents 0, 1, 2, ..., QUANTITY_FRAME-1 ), ie.
        # for combo o2t_mod*`iii` on last part of images as (QUANTITY_FRAME, 86, 2)
        for i in range(notIncludedOn_mod+1):
            for ii in range(o2t_mod):
                skeleton_image_MANY_VIDS.append([])
                for iii in range(QUANTITY_FRAME):
                    append_if_valid: int= idx_init_has_hand +(iii*o2t_mod+ii) +i
                    if skeleton_['skeleton'][append_if_valid]['left_hand'] or \
                        skeleton_['skeleton'][append_if_valid]['right_hand']:
                        skeleton_image_MANY_VIDS[-1].append(skeleton_all[
                            append_if_valid
                        ])
                    elif iii==0:
                        # due to since iii==0 then skeleton_image_MANY_VIDS[-1][-1] does not exist,
                        # ie. len(skeleton_image_MANY_VIDS[-1])==0 True, due to prev at
                        # skeleton_image_MANY_VIDS.append([]) above
                        skeleton_image_MANY_VIDS[-1].append(skeleton_all[idx_init_has_hand])
                    else:
                        skeleton_image_MANY_VIDS[-1].append(skeleton_image_MANY_VIDS[-1][-1])
        del o2t_mod
        del notIncludedOn_mod

        # part 3, consecutive, mandatory initial has hand
        for i in range((len_available_images-QUANTITY_FRAME)+1):
            skeleton_image_MANY_VIDS.append([])
            # due to below appends shape (QUANTITY_FRAME, 158, 158, 2)
            for ii in range(QUANTITY_FRAME):
                append_if_valid: int= idx_init_has_hand+ii +i
                if skeleton_['skeleton'][append_if_valid]['left_hand'] or \
                    skeleton_['skeleton'][append_if_valid]['right_hand']:
                    skeleton_image_MANY_VIDS[-1].append(skeleton_all[
                        append_if_valid
                    ])
                elif ii==0:
                    # due to since ii==0 then skeleton_image_MANY_VIDS[-1][-1] does not exist,
                    # ie. len(skeleton_image_MANY_VIDS[-1])==0 True, due to prev at
                    # skeleton_image_MANY_VIDS.append([]) above
                    skeleton_image_MANY_VIDS[-1].append(skeleton_all[idx_init_has_hand])
                else:
                    skeleton_image_MANY_VIDS[-1].append(skeleton_image_MANY_VIDS[-1][-1])
    elif len_available_images<=QUANTITY_FRAME:
        skeleton_image_MANY_VIDS.append([])
        t2o_ratio: int= int(ceil(QUANTITY_FRAME/len_available_images)) # ceiling2make QUANTITY_FRAME possible
        for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, idx_init_has_hand+len_available_images), range(len_available_images)):
            # i_0to_t2o_multiplier, for counting later due to target is (QUANTITY_FRAME, 86, 2)
            # i_0to_t2o_multiplier, ie. 0, 1, 2, ..., int(len_available_images-1)
            # i_0to_t2o_multiplier<QUANTITY_FRAME, due to len_available_images<=QUANTITY_FRAME
            for ii in range(t2o_ratio):
                if (i_0to_t2o_multiplier*t2o_ratio +ii)<QUANTITY_FRAME:
                    # i_0to_t2o_multiplier*t2o_ratio, due to since: len_available_images<=QUANTITY_FRAME,
                    # then mandatory be each image/frame/skeleton/pose_face_lefthand_righthand be used
                    # multiple times ie. int(t2o_ratio) times
                    # then +ii, due to current be added mod of from int(t2o_ratio),
                    # thus i_0to_t2o_multiplier*t2o_ratio+ii
                    if skeleton_['skeleton'][i]['left_hand'] or skeleton_['skeleton'][i]['right_hand']:
                        skeleton_image_MANY_VIDS[-1].append(skeleton_all[  i  ])
                    else:
                        skeleton_image_MANY_VIDS[-1].append(skeleton_image_MANY_VIDS[-1][-1])
        if len(skeleton_image_MANY_VIDS[-1])!=QUANTITY_FRAME:
            raise ValueError("incorrect implementation on idx idx_init_has_hand!=-1 and len_available_images<QUANTITY_FRAME")
    del len_available_images

    return skeleton_image_MANY_VIDS


def getLessThanOrEqual_skeleton_allHasHand(skeleton_: dict) -> list:
    '''
    to be used for when len(skeleton_['skeleton']) <= QUANTITY_FRAME
    '''
    def getIdxStartHand(image_list: list) -> int:
        for i in range(len(image_list)):
            if image_list[i]['left_hand'] or image_list[i]['right_hand']:
                return i
        return -1
    idx_init_has_hand: int= getIdxStartHand(image_list=skeleton_['skeleton'])
    if idx_init_has_hand==-1:
        return []
    skeleton_images: list= []
    t2o_ratio: int= int(ceil(QUANTITY_FRAME/(len(skeleton_['skeleton'])-idx_init_has_hand)))
    for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, len(skeleton_['skeleton'])), range(len(skeleton_['skeleton'])-idx_init_has_hand)):
        skeleton_data_image= imread( f"{GLASL_SKELETON_DIR}{skeleton_['video_id']}/{skeleton_['skeleton'][  i  ]['file']}" )
        skeleton_data_image= cvtColor(
            src=skeleton_data_image,
            code=COLOR_BGR2RGB
        ).copy()
        for ii in range(t2o_ratio):
            if (i_0to_t2o_multiplier*t2o_ratio+ii)<QUANTITY_FRAME:
                # i_0to_t2o_multiplier*t2o_ratio, due to since: getLessThanOrEqual_skeleton,
                # then mandatory be each image/frame/skeleton/pose_face_lefthand_righthand be used
                # multiple/( or 1 time if equal and idx 0 has hand ) times ie. int(t2o_ratio) times
                # then +ii, due to current be added mod of from int(t2o_ratio),
                # thus i_0to_t2o_multiplier*t2o_ratio+ii
                if skeleton_['skeleton'][i]['left_hand'] or skeleton_['skeleton'][i]['right_hand']:
                    skeleton_images.append( skeleton_data_image )
                else:
                    skeleton_images.append( skeleton_images[-1] )
    if len(skeleton_images)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getLessThanOrEqual_skeleton_allHasHand, due to len(skeleton_images)!=QUANTITY_FRAME")
        # due to out be of shape (QUANTITY_FRAME, 158, 158, 3)
    return skeleton_images


def getdata_skeleton_allHasHand(trainVal: str= 'train', batch: int=ON_TRAINING_BATCH) -> Generator[tuple, None, None]:
    # wlasl_READY['train']
    # wlasl_READY['val']
    # wlasl_READY['test']
    # wlasl_READY['label_id2gloss']
    # wlasl_READY['label_gloss2id']

    # each skeleton image be of shape (158, 158, 3)
    shuffle(glasl_skeleton[trainVal])
    b_idxINIT: int= 0
    total_q_dataset: int= LEN_TRAIN if trainVal==KEY_TRAIN else LEN_VAL
    pastSKELETON_GT_QF: list= [] # pastSKELETON_GT_QF --> past skeleton videos on greater than QUANTITY_FRAME
    # while loop runs 1 for every epoch
    # total_q_count, counts the quantity of video skeleton that was and is training
    # ie. past all batch_vids on instance training, ie. `p -m src_asl2gloss.model_train`, then
    # count wlasl_SKELETON[trainVal][  idx_DS  ] including repeated due to greater
    # than QUANTITY_FRAME
    total_q_count: int= 0
    i_0toBatchOrMore: int= 0 # for wlasl_SKELETON[TrainVal][__ b_idxINIT + i_0toBatchOrMore __]
    while True:
        batch_vids: ndarray= zeros((batch, QUANTITY_FRAME, SKELETON_SHAPE[0], SKELETON_SHAPE[1], SKELETON_SHAPE[2]), dtype=uint8)
        batch_class: ndarray= zeros((batch), dtype=uint16)


        idx_add2batch: int= 0
        # for batch_vids[__ idx_add2batch __]
        # batch_class[__ idx_add2batch __]
        # below( ie. while idx_add2batch<batch: ) runs 1 time( 1 while loop done ) per batch,
        # below only knows batch NOTHING MORE NOTHING LESS
        # does ----> NOT <---- have control on train steps and epochs
        while idx_add2batch<batch:
            idx_DS: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<total_q_dataset else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-total_q_dataset
            ))
            skeleton_1vid: list= [] # at end should be of shape 22, 158, 158, 3
            if len(pastSKELETON_GT_QF)==0:
                folder_skeleton: str= f"{GLASL_SKELETON_DIR}{glasl_skeleton[trainVal][  idx_DS  ]['video_id']}"
                if exists(folder_skeleton):
                    if len(glasl_skeleton[trainVal][  idx_DS  ]['skeleton'])<=QUANTITY_FRAME:
                         skeleton_1vid= getLessThanOrEqual_skeleton_allHasHand(glasl_skeleton[trainVal][  idx_DS  ])
                    else: # quanity of image skeleton is more than QUANTITY_FRAME
                        pastSKELETON_GT_QF= getGreaterThan_skeleton_allHasHand(glasl_skeleton[trainVal][  idx_DS  ])
                        if 0<len(pastSKELETON_GT_QF):
                            skeleton_1vid= pastSKELETON_GT_QF[0]
                            pastSKELETON_GT_QF= pastSKELETON_GT_QF[1:]
                    total_q_count+= 1
            else:
                skeleton_1vid= pastSKELETON_GT_QF[0]
                pastSKELETON_GT_QF= pastSKELETON_GT_QF[1:]
                total_q_count+= 1
            if len(pastSKELETON_GT_QF)==0 or len(skeleton_1vid)==0:
                i_0toBatchOrMore+= 1
            if len(skeleton_1vid)==QUANTITY_FRAME:
                batch_vids[idx_add2batch]= array(skeleton_1vid, dtype=uint8)
                batch_class[idx_add2batch]= glasl_skeleton[trainVal][  idx_DS  ]['gloss_id']
                idx_add2batch+= 1
            elif len(skeleton_1vid)!=0 and len(skeleton_1vid)!=QUANTITY_FRAME:
                print(f"len of skeleton_1vid: {len(skeleton_1vid)}")
                raise ValueError("incorrect implementation on getdata_skeleton, due to len(skeleton_1vid)!=QUANTITY_FRAME")


            if idx_DS==(total_q_dataset-1) and len(pastSKELETON_GT_QF)==0:
                # print(f"________ total_q_count: {total_q_count+len(pastSKELETON_GT_QF)} ______ {trainVal}")
                total_q_count= 0


        if len(pastSKELETON_GT_QF)==0:
            b_idxINIT= (b_idxINIT+batch) if (b_idxINIT+batch)<total_q_dataset else 0+( (b_idxINIT+batch)-total_q_dataset )
            i_0toBatchOrMore= 0
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))
