from json import dump
from os import makedirs
from os.path import join as pjoin
from numpy import array, float64, ndarray, save, zeros
from math import ceil
from os.path import exists


from ..lmark_constant_v2 import LM_NPZ_DIR, LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME,  wlasl_READY_10




def getFramesLessThanTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
    oqFrames: int= len(vid['landmark'])
    if oqFrames<1 or TqFrames<=oqFrames:
        raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
    t2o_ratio: int= int(ceil(TqFrames/oqFrames))
    multiVids: list= [[]]
    # multiVids in end be shape of (1, TqFrames, 518, 2)
    for i in range(oqFrames):
        for ii in range(t2o_ratio):
            if (i*t2o_ratio +ii)<TqFrames:
                # have slot for landmarks later ie. face, pose, left_hand, right_hand
                multiVids[0].append([])
                lm_face= vid['landmark'][i]['landmark_face']
                if len(lm_face)==LM_Q_FACE:
                    multiVids[0][-1].extend(array(lm_face, dtype=float64))
                else:
                    multiVids[0][-1].extend(zeros((LM_Q_FACE, 2), dtype=float64))


                lm_pose= vid['landmark'][i]['landmark_pose']
                if len(lm_pose)==LM_Q_POSE:
                    multiVids[0][-1].extend(array(lm_pose, dtype=float64))
                else:
                    multiVids[0][-1].extend(zeros((LM_Q_POSE, 2), dtype=float64))


                lm_left_hand= vid['landmark'][i]['landmark_left_hand']
                if len(lm_left_hand)==LM_Q_HAND:
                    multiVids[0][-1].extend(array(lm_left_hand, dtype=float64))
                else:
                    multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
                lm_right_hand= vid['landmark'][i]['landmark_right_hand']
                if len(lm_right_hand)==LM_Q_HAND:
                    multiVids[0][-1].extend(array(lm_right_hand, dtype=float64))
                else:
                    multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
    # # by 2
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
    # beEmpty: list= [i for i in range(0, TqFrames, 2)]
    # for i in beEmpty:
    #     multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+1<TqFrames:
    #         multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #
    #
    # # by 3
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
    #
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
    # beEmpty= [i for i in range(0, TqFrames, 3)]
    # for i in beEmpty:
    #     multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
    #     multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
    #     if i+1<TqFrames:
    #         multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
    #         multiVids[6][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
    #         multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
    #     if i+2<TqFrames:
    #         multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
    #         multiVids[7][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
    #
    #
    # # by 4
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
    # beEmpty= [i for i in range(0, TqFrames, 4)]
    # for i in beEmpty:
    #     multiVids[8][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+1<TqFrames:
    #         multiVids[9][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+2<TqFrames:
    #         multiVids[10][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+3<TqFrames:
    #         multiVids[11][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #
    #
    # # by 5
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 15
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 16
    # beEmpty= [i for i in range(0, TqFrames, 5)]
    # for i in beEmpty:
    #     multiVids[12][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+1<TqFrames:
    #         multiVids[13][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+2<TqFrames:
    #         multiVids[14][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+3<TqFrames:
    #         multiVids[15][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+4<TqFrames:
    #         multiVids[16][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)

    # output shape be (1, TqFrames, 518, 2)
    checkshape= array(multiVids, dtype=float64).shape
    if checkshape[0]!=1 or checkshape[1]!=TqFrames or checkshape[2]!=(LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2)) or \
        checkshape[3]!=2:
        raise ValueError("problem is on less than TqFrames generator")
    return tuple(multiVids)


# def getFramesLessThanTarget_lessEmpty(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
#     if int(len(vid['images']))<1 or TqFrames<=int(len(vid['images'])):
#         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
#     oqFrames: int= len(vid['images'])
#     t2o_ratio: int= int(ceil(TqFrames/oqFrames))
#     multiVids: list= [[]]
#     for i in range(oqFrames):
#         for ii in range(t2o_ratio):
#             if (i*t2o_ratio +ii)<TqFrames:
#                 img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
#                 if exists(img_path):
#                     multiVids[0].append(imread(  img_path  ).astype(uint8))
#                 else:
#                     raise FileExistsError(f"no file exist on {img_path}")
#     # by 2
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
#     beEmpty: list= [i for i in range(0, TqFrames, 2)]
#     for i in beEmpty:
#         multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+1<TqFrames:
#             multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#
#     # by 3
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
#     beEmpty= [i for i in range(0, TqFrames, 3)]
#     for i in beEmpty:
#         multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
#         if i+1<TqFrames:
#             multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
#         if i+2<TqFrames:
#             multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
#
#
#     # by 4
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
#     beEmpty= [i for i in range(0, TqFrames, 4)]
#     for i in beEmpty:
#         multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+1<TqFrames:
#             multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+2<TqFrames:
#             multiVids[8][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+3<TqFrames:
#             multiVids[9][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#
#     # by 5
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
#     beEmpty= [i for i in range(0, TqFrames, 5)]
#     for i in beEmpty:
#         multiVids[10][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+1<TqFrames:
#             multiVids[11][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+2<TqFrames:
#             multiVids[12][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+3<TqFrames:
#             multiVids[13][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+4<TqFrames:
#             multiVids[14][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#     return tuple(multiVids)


def getFramesEqualTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
    if int(len(vid['landmark']))!=TqFrames:
        raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Equal to {TqFrames}( TqFrames )")
    multiVids: list= [[]]
    for i in range(TqFrames):
        # have slot for landmarks later ie. face, pose, left_hand, right_hand
        multiVids[0].append([])
        lm_face= vid['landmark'][i]['landmark_face']
        if len(lm_face)==LM_Q_FACE:
            multiVids[0][-1].extend(array(lm_face, dtype=float64))
        else:
            multiVids[0][-1].extend(zeros((LM_Q_FACE, 2), dtype=float64))


        lm_pose= vid['landmark'][i]['landmark_pose']
        if len(lm_pose)==LM_Q_POSE:
            multiVids[0][-1].extend(array(lm_pose, dtype=float64))
        else:
            multiVids[0][-1].extend(zeros((LM_Q_POSE, 2), dtype=float64))


        lm_left_hand= vid['landmark'][i]['landmark_left_hand']
        if len(lm_left_hand)==LM_Q_HAND:
            multiVids[0][-1].extend(array(lm_left_hand, dtype=float64))
        else:
            multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
        lm_right_hand= vid['landmark'][i]['landmark_right_hand']
        if len(lm_right_hand)==LM_Q_HAND:
            multiVids[0][-1].extend(array(lm_right_hand, dtype=float64))
        else:
            multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
    # # by 2
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
    # beEmpty: list= [i for i in range(0, TqFrames, 2)]
    # for i in beEmpty:
    #     multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+1<TqFrames:
    #         multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #
    #
    # # by 3
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
    #
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
    # beEmpty= [i for i in range(0, TqFrames, 3)]
    # for i in beEmpty:
    #     multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
    #     multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
    #     if i+1<TqFrames:
    #         multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
    #         multiVids[6][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
    #         multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
    #     if i+2<TqFrames:
    #         multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
    #         multiVids[7][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
    #
    #
    # # by 4
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
    # beEmpty= [i for i in range(0, TqFrames, 4)]
    # for i in beEmpty:
    #     multiVids[8][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+1<TqFrames:
    #         multiVids[9][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+2<TqFrames:
    #         multiVids[10][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+3<TqFrames:
    #         multiVids[11][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #
    #
    # # by 5
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 15
    # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 16
    # beEmpty= [i for i in range(0, TqFrames, 5)]
    # for i in beEmpty:
    #     multiVids[12][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+1<TqFrames:
    #         multiVids[13][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+2<TqFrames:
    #         multiVids[14][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+3<TqFrames:
    #         multiVids[15][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #     if i+4<TqFrames:
    #         multiVids[16][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)

    checkshape= array(multiVids, dtype=float64).shape
    if checkshape[0]!=1 or checkshape[1]!=TqFrames or checkshape[2]!=(LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2)) or \
        checkshape[3]!=2:
        raise ValueError("problem is on equal TqFrames generator")
    return tuple(multiVids)


# def getFramesEqualTarget_lessEmpty(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
#     if int(len(vid['images']))!=TqFrames:
#         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Equal to {TqFrames}( TqFrames )")
#     multiVids: list= [[]]
#     for i in range(TqFrames):
#         img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
#         if exists(img_path):
#             multiVids[0].append(imread(  img_path  ).astype(uint8))
#         else:
#             raise FileExistsError(f"no file exist on {img_path}")
#     # by 2
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
#     beEmpty: list= [i for i in range(0, TqFrames, 2)]
#     for i in beEmpty:
#         multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+1<TqFrames:
#             multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#
#     # by 3
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
#     beEmpty= [i for i in range(0, TqFrames, 3)]
#     for i in beEmpty:
#         multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
#         if i+1<TqFrames:
#             multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
#         if i+2<TqFrames:
#             multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
#
#
#     # by 4
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
#     beEmpty= [i for i in range(0, TqFrames, 4)]
#     for i in beEmpty:
#         multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+1<TqFrames:
#             multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+2<TqFrames:
#             multiVids[8][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+3<TqFrames:
#             multiVids[9][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#
#     # by 5
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
#     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
#     beEmpty= [i for i in range(0, TqFrames, 5)]
#     for i in beEmpty:
#         multiVids[10][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+1<TqFrames:
#             multiVids[11][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+2<TqFrames:
#             multiVids[12][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+3<TqFrames:
#             multiVids[13][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#         if i+4<TqFrames:
#             multiVids[14][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#     return tuple(multiVids)


def getFramesGreaterThanTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
    oqFrames: int= len(vid['landmark'])
    if oqFrames<TqFrames:
        raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
    o2t_ratio: int= oqFrames//TqFrames


    # evenly spaced
    multiVids: list= [[] for _ in range(o2t_ratio)]
    for i_mod in range(o2t_ratio):
        for i in range(TqFrames):
            # have slot for landmarks later ie. face, pose, left_hand, right_hand
            multiVids[i_mod].append([])
            lm_face= vid['landmark'][i*o2t_ratio +i_mod]['landmark_face']
            if len(lm_face)==LM_Q_FACE:
                multiVids[i_mod][-1].extend(array(lm_face, dtype=float64))
            else:
                multiVids[i_mod][-1].extend(zeros((LM_Q_FACE, 2), dtype=float64))


            lm_pose= vid['landmark'][i*o2t_ratio +i_mod]['landmark_pose']
            if len(lm_pose)==LM_Q_POSE:
                multiVids[i_mod][-1].extend(array(lm_pose, dtype=float64))
            else:
                multiVids[i_mod][-1].extend(zeros((LM_Q_POSE, 2), dtype=float64))


            lm_left_hand= vid['landmark'][i*o2t_ratio +i_mod]['landmark_left_hand']
            if len(lm_left_hand)==LM_Q_HAND:
                multiVids[i_mod][-1].extend(array(lm_left_hand, dtype=float64))
            else:
                multiVids[i_mod][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
            lm_right_hand= vid['landmark'][i*o2t_ratio +i_mod]['landmark_right_hand']
            if len(lm_right_hand)==LM_Q_HAND:
                multiVids[i_mod][-1].extend(array(lm_right_hand, dtype=float64))
            else:
                multiVids[i_mod][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
    # # len(multiVids) is now o2t_ratio, ie. o2t_ratio>=1
    # # append version with missing frames, each be by 2 and 3
    # for i_mod in range(o2t_ratio):
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
    #     beEmpty: list= [i for i in range(0, TqFrames, 2)]
    #     for i in beEmpty:
    #         multiVids[-2][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #         if i+1<TqFrames:
    #             multiVids[-1][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
    #
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
    #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
    #     beEmpty= [i for i in range(0, TqFrames, 3)]
    #     for i in beEmpty:
    #         multiVids[-5][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
    #         multiVids[-2][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
    #         if i+1<TqFrames:
    #             multiVids[-4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
    #             multiVids[-2][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
    #             multiVids[-1][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
    #         if i+2<TqFrames:
    #             multiVids[-3][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
    #             multiVids[-1][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
    # del o2t_ratio


    # consecutive
    init_HasHand: int= -1
    while init_HasHand==-1:
        i: int= 0
        while init_HasHand==-1 and i<len(vid['landmark']):
            if 0<len(vid['landmark'][i]['landmark_left_hand']) or 0<len(vid['landmark'][i]['landmark_right_hand']):
                init_HasHand= i
            i+= 1
        if init_HasHand==-1:
            init_HasHand= 0
    # to check if has atleast enough frames for 1 TqFrames
    if TqFrames<=(oqFrames -init_HasHand):
        for init_consecutive in range(init_HasHand, oqFrames+1 -TqFrames): # if a>b on range(a,b), then forLoopNotRun
            multiVids.append([]) # append new video for each valid-consecutive
            for i in range(init_consecutive, init_consecutive+TqFrames):
                multiVids[-1].append([]) # append for image(lmark really) of a video
                lm_face= vid['landmark'][i]['landmark_face']
                if len(lm_face)==LM_Q_FACE:
                    multiVids[-1][-1].extend(array(lm_face, dtype=float64))
                else:
                    multiVids[-1][-1].extend(zeros((LM_Q_FACE, 2), dtype=float64))


                lm_pose= vid['landmark'][i]['landmark_pose']
                if len(lm_pose)==LM_Q_POSE:
                    multiVids[-1][-1].extend(array(lm_pose, dtype=float64))
                else:
                    multiVids[-1][-1].extend(zeros((LM_Q_POSE, 2), dtype=float64))


                lm_left_hand= vid['landmark'][i]['landmark_left_hand']
                if len(lm_left_hand)==LM_Q_HAND:
                    multiVids[-1][-1].extend(array(lm_left_hand, dtype=float64))
                else:
                    multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
                lm_right_hand= vid['landmark'][i]['landmark_right_hand']
                if len(lm_right_hand)==LM_Q_HAND:
                    multiVids[-1][-1].extend(array(lm_right_hand, dtype=float64))
                else:
                    multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))


        # use floor in a different way, a different way of evenly splitting to TqFrames
        # be included on checking if has atleast enough frames for 1 TqFrames
        multiVids.append([])
        o2t_rFloat: float= (oqFrames-init_HasHand)/TqFrames
        for i in range(TqFrames):
            idx_onVidImg: int= int(i*o2t_rFloat +init_HasHand)
            multiVids[-1].append([]) # append for image(lmark really) of a video
            lm_face= vid['landmark'][idx_onVidImg]['landmark_face']
            if len(lm_face)==LM_Q_FACE:
                multiVids[-1][-1].extend(array(lm_face, dtype=float64))
            else:
                multiVids[-1][-1].extend(zeros((LM_Q_FACE, 2), dtype=float64))


            lm_pose= vid['landmark'][idx_onVidImg]['landmark_pose']
            if len(lm_pose)==LM_Q_POSE:
                multiVids[-1][-1].extend(array(lm_pose, dtype=float64))
            else:
                multiVids[-1][-1].extend(zeros((LM_Q_POSE, 2), dtype=float64))


            lm_left_hand= vid['landmark'][idx_onVidImg]['landmark_left_hand']
            if len(lm_left_hand)==LM_Q_HAND:
                multiVids[-1][-1].extend(array(lm_left_hand, dtype=float64))
            else:
                multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))
            lm_right_hand= vid['landmark'][idx_onVidImg]['landmark_right_hand']
            if len(lm_right_hand)==LM_Q_HAND:
                multiVids[-1][-1].extend(array(lm_right_hand, dtype=float64))
            else:
                multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float64))

    checkshape= array(multiVids, dtype=float64).shape
    if checkshape[1:]!=(TqFrames, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2):
        raise ValueError(f"problem is on greater than TqFrames generator {checkshape}")
    return tuple(multiVids)


# def getFramesGreaterThanTarget_lessEmpty(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
#     if int(len(vid['images']))<TqFrames:
#         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
#     oqFrames: int= len(vid['images'])
#     o2t_ratio: int= oqFrames//TqFrames
#
#
#     # evenly spaced
#     multiVids: list= [[] for _ in range(o2t_ratio)]
#     for i_mod in range(o2t_ratio):
#         for i in range(TqFrames):
#             img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i*o2t_ratio +i_mod]['file']}.png"))
#             if exists(img_path):
#                 multiVids[i_mod].append(imread(  img_path  ).astype(uint8))
#             else:
#                 raise FileExistsError(f"no file exist on {img_path}")
#     # len(multiVids) is now o2t_ratio, ie. o2t_ratio>=1
#     # append version with missing frames, each be by 2 and 3
#     for i_mod in range(o2t_ratio):
#         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
#         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
#         beEmpty: list= [i for i in range(0, TqFrames, 2)]
#         for i in beEmpty:
#             multiVids[-2][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#             if i+1<TqFrames:
#                 multiVids[-1][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#         beEmpty= [i for i in range(0, TqFrames, 3)]
#         for i in beEmpty:
#             multiVids[-3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
#             if i+1<TqFrames:
#                 multiVids[-2][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
#             if i+2<TqFrames:
#                 multiVids[-1][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
#     del o2t_ratio
#
#
#     # consecutive
#     init_HasHand: int= -1
#     while init_HasHand==-1:
#         i: int= 0
#         while init_HasHand==-1 and i<len(vid['images']):
#             if vid['images'][i]['left_hand'] or vid['images'][i]['right_hand']:
#                 init_HasHand= i
#             i+= 1
#         if init_HasHand==-1:
#             init_HasHand= 0
#     # to check if has atleast enough frames for 1 TqFrames
#     if TqFrames<=(int(len(vid['images'])) -init_HasHand):
#         for init_consecutive in range(init_HasHand, int(len(vid['images']))-TqFrames): # if a>b on range(a,b), then forLoopNotRun
#             multiVids.append([]) # --------------- idx -6
#             multiVids.append([]) # for by 2 -- idx -5
#             multiVids.append([]) # for by 2 ------ idx -4
#             multiVids.append([]) # for by 3 -- idx -3
#             multiVids.append([]) # for by 3 ------ idx -2
#             multiVids.append([]) # for by 3 -- idx -1
#             for i in range(init_consecutive, init_consecutive+TqFrames):
#                 img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
#                 if exists(img_path):
#                     multiVids[-6].append(imread(  img_path  ).astype(uint8))
#                     if i%2==0:
#                         multiVids[-5].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#                         multiVids[-4].append(imread(  img_path  ).astype(uint8))
#                     else:
#                         multiVids[-5].append(imread(  img_path  ).astype(uint8))
#                         multiVids[-4].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#
#
#                     if i%3==0:
#                         multiVids[-3].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#                         multiVids[-2].append(imread(  img_path  ).astype(uint8))
#                         multiVids[-1].append(imread(  img_path  ).astype(uint8))
#                     elif i%3==1:
#                         multiVids[-3].append(imread(  img_path  ).astype(uint8))
#                         multiVids[-2].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#                         multiVids[-1].append(imread(  img_path  ).astype(uint8))
#                     else:
#                         multiVids[-3].append(imread(  img_path  ).astype(uint8))
#                         multiVids[-2].append(imread(  img_path  ).astype(uint8))
#                         multiVids[-1].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#                 else:
#                     raise FileExistsError(f"no file exist on {img_path}")
#
#
#         # use floor in a different way, a different way of evenly splitting to TqFrames
#         multiVids.append([])
#         multiVids.append([])
#         multiVids.append([])
#         o2t_rFloat: float= (len(vid['images'])-init_HasHand)/TqFrames
#         for i in range(TqFrames):
#             idx_onVidImg: int= int(i*o2t_rFloat +init_HasHand)
#             img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][  idx_onVidImg  ]['file']}.png"))
#             if exists(img_path):
#                 multiVids[-3].append(imread(  img_path  ).astype(uint8))
#                 if idx_onVidImg%2==0:
#                     multiVids[-2].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#                     multiVids[-1].append(imread(  img_path  ).astype(uint8))
#                 else:
#                     multiVids[-2].append(imread(  img_path  ).astype(uint8))
#                     multiVids[-1].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
#             else:
#                 raise FileExistsError(f"no file exist on {img_path}")
#
#     return tuple(multiVids)


def create_lmark_npz_data() -> dict:
    landmark_ann_numpy: dict= {
        'train': [],
        'val': [],
        'test': [],
        'label_id2gloss': wlasl_READY_10['label_id2gloss'],
        'label_gloss2id': wlasl_READY_10['label_gloss2id']
    }
    pastFrame: tuple= ()
    for tvt in ['train', 'val', 'test']:
        print(f"processing for {tvt}...")
        for video_ann in wlasl_READY_10[tvt]:
            if len(video_ann['landmark'])<QUANTITY_FRAME:
                pastFrame= getFramesLessThanTarget(video_ann)
            elif len(video_ann['landmark'])==QUANTITY_FRAME:
                pastFrame= getFramesEqualTarget(video_ann)
            else:
                pastFrame= getFramesGreaterThanTarget(video_ann)
            i: int= 0
            while len(pastFrame)!=0:
                sfx_end: str= f"000{i}" if i<10 else (f"00{i}" if i<100 else (f"0{i}" if i<1000 else f"{i}"))
                videodata_lmark: ndarray= array(pastFrame[0], dtype=float64, copy=True)
                videodata_lmark_file_npz: str= f"{video_ann['video_id']}{sfx_end}"
                save(pjoin(LM_NPZ_DIR, videodata_lmark_file_npz), videodata_lmark)
                landmark_ann_numpy[tvt].append({
                    'gloss_id': video_ann['gloss_id'],
                    'file': videodata_lmark_file_npz
                })
                pastFrame= pastFrame[1:]
                i+= 1

    return landmark_ann_numpy

if __name__=="__main__":
    file_npz: str= pjoin(LM_NPZ_DIR, '..', 'wlasl.annotation.landmark.numpy.json')
    if not exists(LM_NPZ_DIR) and not exists(file_npz):
        makedirs(LM_NPZ_DIR)
        print(f"process writing to {LM_NPZ_DIR}...")
        lmark_numpy_dict: dict= create_lmark_npz_data()
        with open(file_npz, 'w') as f:
            dump(lmark_numpy_dict, f)
    else:
        print(f"please move or delete {LM_NPZ_DIR}, due to this is where the data will be written")
        print(f"this is just incase {LM_NPZ_DIR} has some importance to you")
        print(f"and also delete or move {file_npz}, same reason as {LM_NPZ_DIR}")

