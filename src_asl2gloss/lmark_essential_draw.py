from random import choices, shuffle
from typing import Generator
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, circle, cvtColor, line
from numpy import array, float32, ndarray, uint16, uint8, zeros
from math import ceil
from os.path import exists

from .lmark_constant import FACE_CONNECTIONS, HAND_CONNECTIONS, IMG_SIZE, POSE_CONNECTIONS, QUANTITY_FRAME, TRAIN_BATCH, WLASL_VID_DIR, WORTHY_POSE_IDX, wlasl_READY
from .lmark_constant import mpH


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


def getSkeletonFrames(fpath_vid: str, isSingleImg: bool=False, TqFRAMES: int= QUANTITY_FRAME) -> ndarray:
    '''
    fpath_vid: str, video file path string
    TqFRAMES: int, target quantity of frames on output

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
    if oqFRAMES<TqFRAMES:
        t2o_ratio: int= int(ceil(TqFRAMES/oqFRAMES))
        for i in range(oqFRAMES):
            for ii in range(t2o_ratio):
                if isSingleImg and (i*t2o_ratio +ii)<TqFRAMES:
                    allImg_skeleton.extend(drawFacePoseHand(
                        img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                        lmark_mph=mpH.process(allImg_human[i]),
                        orig_shape=allImg_human[i].shape
                    ))
                elif (i*t2o_ratio +ii)<TqFRAMES:
                    allImg_skeleton.append(drawFacePoseHand(
                        img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                        lmark_mph=mpH.process(allImg_human[i]),
                        orig_shape=allImg_human[i].shape
                    ))
    elif oqFRAMES==TqFRAMES:
        for i in range(TqFRAMES):
            if isSingleImg:
                allImg_skeleton.extend(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=mpH.process(allImg_human[i]),
                    orig_shape=allImg_human[i].shape
                ))
            else:
                allImg_skeleton.append(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=mpH.process(allImg_human[i]),
                    orig_shape=allImg_human[i].shape
                ))
    else: # TqFRAMES < oqFRAMES
        o2t_ratio: int= oqFRAMES//TqFRAMES
        for i in range(TqFRAMES):
            if isSingleImg:
                allImg_skeleton.extend(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=mpH.process(allImg_human[i*o2t_ratio]),
                    orig_shape=allImg_human[i].shape
                ))
            else:
                allImg_skeleton.append(drawFacePoseHand(
                    img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=mpH.process(allImg_human[i*o2t_ratio]),
                    orig_shape=allImg_human[i].shape
                ))
    if len(allImg_skeleton)!=TqFRAMES and not isSingleImg:
        raise ValueError(f"frames on single video failed match target( {
        TqFRAMES
        } ) orig( {oqFRAMES} ), but result is {
        allImg_skeleton
        } --> {fpath_vid}")
    return array(allImg_skeleton, dtype=uint8)


def getdata(isSimg: bool=False, TrainVal: str= 'train', batch: int=TRAIN_BATCH) -> Generator[tuple, None, None]:
    # wlasl_READY['train']
    # wlasl_READY['val']
    # wlasl_READY['test']
    # wlasl_READY['label_id2gloss']
    # wlasl_READY['label_gloss2id']
    tmp_arrChoice: list= [2,3,1]
    tmp_arrChoice= choices(tmp_arrChoice)
    for _ in range(tmp_arrChoice[0]):
        shuffle(wlasl_READY[TrainVal])
    del tmp_arrChoice
    b_idxINIT: int= 0
    shape_vidBatch: tuple= (batch, QUANTITY_FRAME*IMG_SIZE, IMG_SIZE, 3) if isSimg else (batch, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3)
    # while loop runs 1 for every epoch
    while True:
        batch_vids: ndarray= zeros(shape_vidBatch, dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)
        i_0toBatchOrMore: int= 0
        idx_add2batch: int= 0
        while idx_add2batch<batch:
            curr_IDX_USE: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<len(wlasl_READY[TrainVal]) else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-len(wlasl_READY[TrainVal])
            ))
            vidfile_dir: str= f"{WLASL_VID_DIR}{wlasl_READY[TrainVal][  curr_IDX_USE  ]['video_id']}.mp4"
            if exists(vidfile_dir):
                try:
                    batch_vids[idx_add2batch]= getSkeletonFrames(vidfile_dir, isSingleImg=isSimg).astype(float32)/255.0
                    batch_class[idx_add2batch]= int(wlasl_READY[TrainVal][  curr_IDX_USE  ]['gloss_id'])/1.0
                    idx_add2batch+= 1
                except FileExistsError as e:
                    del e
            i_0toBatchOrMore+= 1
        b_idxINIT= (b_idxINIT+batch) if (b_idxINIT+batch)<len(wlasl_READY[TrainVal]) else 0
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))

