from random import shuffle
from typing import Generator
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, circle, cvtColor, destroyAllWindows, line
from numpy import array, float32, ndarray, uint16, uint8, zeros
from json import load as jsonload
from math import ceil

from .lmark_constant import FACE_CONNECTIONS, HAND_CONNECTIONS, IMG_SIZE, POSE_CONNECTIONS, PROJ_ROOT, QUANTITY_FRAME, TRAIN_BATCH, WLASL_VID_DIR, WORTHY_POSE_IDX
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
    del img_orig
    img_wh: dict= {"wx": img.shape[1], "hy": img.shape[0]}
    for l in conn_idxs_list:
        pA: tuple= (
            lmark_cords[l[0]][0], # x
            lmark_cords[l[0]][1]  # y
        )
        pB: tuple= (
            lmark_cords[l[1]][0], # x
            lmark_cords[l[1]][1]  # y
        )
        if isOKplt(pA) and isOKplt(pB):
            line(
                img=img,
                pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                color=color_conn,
                thickness=thick
            )
        del pA
        del pB
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
    return img
def drawFacePoseHand(img_orig: ndarray, lmark_mph, orig_shape: tuple) -> ndarray:
    def recalcDrawFace(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        del img_orig
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
        del img_orig
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            # thick=4,
            thick=1,
            color_conn=(51, 204, 204), # 204/255= 0.8
            color_lmark=(204, 204, 51) # 51/255= 0.2
            # color_conn=(255,255,255), # blackNwhite
            # color_lmark=(255,255,255) # blackNwhite
        )
    def recalcDrawLeftHands(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        del img_orig
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
        del img_orig
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
    del img_orig


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
                all_x.extend([i[0] for i in recalc_lmark_face])
                all_y.extend([i[1] for i in recalc_lmark_face])
            if 0<len(recalc_lmark_left_hand):
                recalc_lmark_left_hand= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in recalc_lmark_left_hand]
                all_x.extend([i[0] for i in recalc_lmark_face])
                all_y.extend([i[1] for i in recalc_lmark_face])
            if 0<len(recalc_lmark_right_hand):
                recalc_lmark_right_hand= [(i[0]/max_xy, i[1]/max_xy)
                                    for i in recalc_lmark_right_hand]
                all_x.extend([i[0] for i in recalc_lmark_face])
                all_y.extend([i[1] for i in recalc_lmark_face])
            all_x= tuple(all_x)
            all_y= tuple(all_y)
            min_x= min(all_x)
            min_y= min(all_y)
        del max_xy


        ### 1) from old img ratio to new square img ratio
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


        ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
        # zoom in/out for padding be 10% each side with respect to original aspect ratio
        pad: float= 0.05
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
            recalc_lmark_face= [(i[0]+shift_y_down, i[1])
                                for i in recalc_lmark_face]
            # all_y.extend([i[1] for i in recalc_lmark_face])
            recalc_lmark_pose= [(i[0]+shift_y_down, i[1])
                                for i in recalc_lmark_pose]
            # all_y.extend([i[1] for i in recalc_lmark_pose])
            recalc_lmark_left_hand= [(i[0]+shift_y_down, i[1])
                                for i in recalc_lmark_left_hand]
            # all_y.extend([i[1] for i in recalc_lmark_left_hand])
            recalc_lmark_right_hand= [(i[0]+shift_y_down, i[1])
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


def getSkeletonFrames(fpath_vid: str, TqFRAMES: int= QUANTITY_FRAME) -> ndarray:
    '''
    fpath_vid: str, video file path string
    TqFRAMES: int, target quantity of frames on output

    output is ndarray of image frames, from start of the video till end
    of size (TqFRAMES, IMG_SIZE, IMG_SIZE, 3) of dtype=numpy.uint8
    '''
    vid= VideoCapture(fpath_vid)
    oqFRAMES: int= int(vid.get(CAP_PROP_FRAME_COUNT))
    all_frames: list= []


    if vid.isOpened():
        isNotEnd, frame= True, zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if oqFRAMES < TqFRAMES:
            # problem, oqFRAMES 33, 46
            # for all target frames have frames from orig frames
            target2orig_ratio: int= int(ceil(TqFRAMES/oqFRAMES))
            for i in range(TqFRAMES):
                if (i%target2orig_ratio)==0 and i<(oqFRAMES-1):
                    isNotEnd, frame= vid.read()
                if isNotEnd:
                    frame= array(cvtColor(src=frame, code=COLOR_BGR2RGB), dtype=uint8)
                    all_frames.append(drawFacePoseHand(
                        img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                        lmark_mph=mpH.process(frame),
                        orig_shape=frame.shape
                    ))
        elif oqFRAMES==TqFRAMES:
            for i in range(TqFRAMES):
                isNotEnd, frame= vid.read()
                if isNotEnd:
                    frame= array(cvtColor(src=frame, code=COLOR_BGR2RGB), dtype=uint8)
                    all_frames.append(drawFacePoseHand(
                        img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                        lmark_mph=mpH.process(frame),
                        orig_shape=frame.shape
                    ))
        else: # TqFRAMES < oqFRAMES
            orig2target_ratio: int= oqFRAMES//TqFRAMES
            for i in range(orig2target_ratio*TqFRAMES):
                isNotEnd, frame= vid.read()
                if 1<orig2target_ratio and (i%orig2target_ratio)==(orig2target_ratio-1): # get last part
                    if isNotEnd:
                        frame= array(cvtColor(src=frame, code=COLOR_BGR2RGB), dtype=uint8)
                        all_frames.append(drawFacePoseHand(
                            img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                            lmark_mph=mpH.process(frame),
                            orig_shape=frame.shape
                        ))
                elif 1==orig2target_ratio:
                    if isNotEnd:
                        frame= array(cvtColor(src=frame, code=COLOR_BGR2RGB), dtype=uint8)
                        all_frames.append(drawFacePoseHand(
                            img_orig=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                            lmark_mph=mpH.process(frame),
                            orig_shape=frame.shape
                        ))
    else:
        raise FileExistsError(f"file {fpath_vid} can't be opened")
    vid.release()
    del vid
    destroyAllWindows()
    if len(all_frames)!=TqFRAMES:
        raise ValueError(f"frames on single video failed match target( {TqFRAMES} ), but result is {len(all_frames)} --> {fpath_vid}")
    return array(all_frames, dtype=uint8)


def getdata(batch: int=TRAIN_BATCH) -> Generator[tuple, None, None]:
    wlasl_ready: dict= {}
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
        wlasl_ready= jsonload(f)
        # wlasl_ready['train']
        # wlasl_ready['val']
        # wlasl_ready['test']
        # wlasl_ready['label_id2gloss']
        # wlasl_ready['label_gloss2id']
    shuffle(wlasl_ready['train'])
    current_idxTRAIN: int= 0
    while current_idxTRAIN<len(wlasl_ready['train']):
        batch_vids: ndarray= zeros((batch, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3), dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)
        for i in range(current_idxTRAIN%batch, (current_idxTRAIN+batch)%batch):
            # batch_vids[i, :, :, :, :]= getSkeletonFrames(f"{WLASL_VID_DIR}{wlasl_ready['train'][i]['video_id']}.mp4")
            batch_vids[i]= getSkeletonFrames(f"{WLASL_VID_DIR}{wlasl_ready['train'][i]['video_id']}.mp4").astype(float32)
            batch_class[i]= int(wlasl_ready['train'][i]['gloss_id'])/1.0
            # print(f"_________type {type(wlasl_ready['train'][i]['gloss_id'])}")
            # print(f"_________{wlasl_ready['train'][i]['gloss_id']}")
        current_idxTRAIN +=batch
        # yield (
        #     {"batch_vid": batch_vids.astype(float32)},
        #     {"batch_class": batch_class.astype(float32)}
        # )
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))

