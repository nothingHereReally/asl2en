from json import dump
from os.path import exists
from typing import Any
from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, COLOR_BGR2RGB, cvtColor
from mediapipe.python.solutions.holistic import Holistic
from numpy import array, uint8


from ..lmark_constant import IMG_SIZE, PROJ_ROOT, WLASL_VID_DIR, wlasl_READY_10, WORTHY_POSE_IDX




def getRecalcLMark(lmark_mph, orig_shape: tuple) -> dict:
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
            wx_hy: int= owx if ohy<owx else ohy
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


        outdata: dict= {}
        if lmark_mph.face_landmarks != None:
            outdata['landmark_face']= recalc_lmark_face
        else:
            outdata['landmark_face']= []
        if lmark_mph.pose_landmarks != None:
            outdata['landmark_pose']= recalc_lmark_pose
        else:
            outdata['landmark_pose']= []
        if lmark_mph.left_hand_landmarks != None:
            outdata['landmark_left_hand']= recalc_lmark_left_hand
        else:
            outdata['landmark_left_hand']= []
        if lmark_mph.right_hand_landmarks != None:
            outdata['landmark_right_hand']= recalc_lmark_right_hand
        else:
            outdata['landmark_right_hand']= []
    else:
        raise FileExistsError("no landmark exist on any face|pose|left_hand|right_hand")

    return outdata


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


def getFramesLMarks(fpath_vid: str, mpH: Any=None) -> tuple:
    allImg_human: list= getAllImg_frames(fpath_vid)
    allImg_lmark: list= []
    oqFRAMES: int= int(len(allImg_human))
    if oqFRAMES<=0:
        raise FileExistsError(f"file {fpath_vid.rsplit("/")[-1]} can't be opened or is corrupted")
    qHands: int= 0
    for img in allImg_human:
        try:
            allImg_lmark.append(getRecalcLMark(
                lmark_mph=mpH.process(img),
                orig_shape=img.shape
            ))
        except FileExistsError as e:
            del e
    del allImg_human
    del oqFRAMES
    del qHands

    # allImg_lmark= (
    #     {
    #         'landmark_face': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_pose': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_left_hand': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_right_hand': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #     },
    #     {
    #         'landmark_face': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_pose': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_left_hand': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_right_hand': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #     },
    #     ...,
    #     {
    #         'landmark_face': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_pose': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_left_hand': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #         'landmark_right_hand': [...ALL_BE_GreaterThan_0_and_LessThan_1_due2padding_see_getRecalcLMark_func...]
    #     },
    # )
    # len(allImg_lmark) (is on video file) is quantity of frames/images itself
    return tuple(allImg_lmark)




if __name__=='__main__':
    init_dir: str= f"{PROJ_ROOT}dataset/wlasl_dataset/"
    mpH_fph: Holistic= Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    TrainValTest: list= ["train", "val", "test"]
    if exists(init_dir):
        lmark_ann: dict= {
            'train': [],
            'val': [],
            'test': [],
            'label_id2gloss': wlasl_READY_10['label_id2gloss'],
            'label_gloss2id': wlasl_READY_10['label_gloss2id']
        }
        for tvt in TrainValTest:
            print(f"processing {tvt}...")
            for trainValTest_ins in wlasl_READY_10[tvt]:
                vidfile: str= f"{WLASL_VID_DIR}{trainValTest_ins['video_id']}.mp4"
                if exists(vidfile):
                    try:
                        vid: VideoCapture= VideoCapture(vidfile)
                        if vid.isOpened():
                            lmarks_onSingleVideo: tuple= getFramesLMarks(
                                fpath_vid=vidfile,
                                mpH=mpH_fph,
                            )
                            if 0<len(lmarks_onSingleVideo):
                                lmark_ann[tvt].append({
                                    'gloss_id': int(trainValTest_ins['gloss_id']),
                                    'video_id': str(trainValTest_ins['video_id']),
                                    'landmark': list(lmarks_onSingleVideo)
                                })
                        vid.release()
                        del vid
                    except Exception as e:
                        print(f"err: {e}")
                        del e
        with open(f"{init_dir}wlasl.annotation.g10_landmarks.json", 'w') as f:
            dump(lmark_ann, f)
        print("testing result( see sample details below ):")
        for i in range(len(lmark_ann['label_id2gloss'])):
            print(f"{lmark_ann['label_id2gloss'][  lmark_ann['train'][i]['gloss_id']  ]}")
            print(f"video_id {lmark_ann['train'][i]['video_id']}")
            print(f"img 1 --> {lmark_ann['train'][i]['landmark'][0]}")
            print(f"img 2 --> {lmark_ann['train'][i]['landmark'][1]}")
        print("creating g10 dataset done...")
    else:
        print(f"{init_dir} doesn't\nexist, please get the dataset 1st")

