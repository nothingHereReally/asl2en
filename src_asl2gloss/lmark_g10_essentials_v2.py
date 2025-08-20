from json import dump
from os import makedirs
from os.path import join as pjoin
from random import shuffle
from typing import Generator
from numpy import float32, load, ndarray, uint16, zeros
from os.path import exists

from .lmark_constant_v2 import LM_NPZ_DIR, LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, PROJ_ROOT, QUANTITY_FRAME, T10_VAL, TRAIN_BATCH, TRAIN_STEPS, wlasl_READY_10




# def getFramesLessThanTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
#     if int(len(vid['landmark']))<1 or TqFrames<=int(len(vid['landmark'])):
#         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
#     oqFrames: int= len(vid['landmark'])
#     t2o_ratio: int= int(ceil(TqFrames/oqFrames))
#     multiVids: list= [[]]
#     # multiVids in end be shape of (1, TqFrames, 518, 2)
#     for i in range(oqFrames):
#         for ii in range(t2o_ratio):
#             if (i*t2o_ratio +ii)<TqFrames:
#                 # have slot for landmarks later ie. face, pose, left_hand, right_hand
#                 multiVids[0].append([])
#                 lm_face= vid['landmark'][i]['landmark_face']
#                 if len(lm_face)==LM_Q_FACE:
#                     multiVids[0][-1].extend(lm_face)
#                 else:
#                     multiVids[0][-1].extend(zeros((LM_Q_FACE, 2), dtype=float32))
#
#
#                 lm_pose= vid['landmark'][i]['landmark_pose']
#                 if len(lm_pose)==LM_Q_POSE:
#                     multiVids[0][-1].extend(lm_pose)
#                 else:
#                     multiVids[0][-1].extend(zeros((LM_Q_POSE, 2), dtype=float32))
#
#
#                 lm_left_hand= vid['landmark'][i]['landmark_left_hand']
#                 if len(lm_left_hand)==LM_Q_HAND:
#                     multiVids[0][-1].extend(lm_left_hand)
#                 else:
#                     multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#                 lm_right_hand= vid['landmark'][i]['landmark_right_hand']
#                 if len(lm_right_hand)==LM_Q_HAND:
#                     multiVids[0][-1].extend(lm_right_hand)
#                 else:
#                     multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#     # # by 2
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
#     # beEmpty: list= [i for i in range(0, TqFrames, 2)]
#     # for i in beEmpty:
#     #     multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+1<TqFrames:
#     #         multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #
#     #
#     # # by 3
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
#     #
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
#     # beEmpty= [i for i in range(0, TqFrames, 3)]
#     # for i in beEmpty:
#     #     multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
#     #     multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
#     #     if i+1<TqFrames:
#     #         multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
#     #         multiVids[6][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
#     #         multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
#     #     if i+2<TqFrames:
#     #         multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
#     #         multiVids[7][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
#     #
#     #
#     # # by 4
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
#     # beEmpty= [i for i in range(0, TqFrames, 4)]
#     # for i in beEmpty:
#     #     multiVids[8][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+1<TqFrames:
#     #         multiVids[9][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+2<TqFrames:
#     #         multiVids[10][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+3<TqFrames:
#     #         multiVids[11][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #
#     #
#     # # by 5
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 15
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 16
#     # beEmpty= [i for i in range(0, TqFrames, 5)]
#     # for i in beEmpty:
#     #     multiVids[12][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+1<TqFrames:
#     #         multiVids[13][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+2<TqFrames:
#     #         multiVids[14][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+3<TqFrames:
#     #         multiVids[15][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+4<TqFrames:
#     #         multiVids[16][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#     # output shape be (1, TqFrames, 518, 2)
#     checkshape= array(multiVids, dtype=float32).shape
#     if checkshape!=(1, TqFrames, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2):
#         raise ValueError("problem is on less than TqFrames generator")
#     return tuple(multiVids)
#
#
# # def getFramesLessThanTarget_lessEmpty(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
# #     if int(len(vid['images']))<1 or TqFrames<=int(len(vid['images'])):
# #         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
# #     oqFrames: int= len(vid['images'])
# #     t2o_ratio: int= int(ceil(TqFrames/oqFrames))
# #     multiVids: list= [[]]
# #     for i in range(oqFrames):
# #         for ii in range(t2o_ratio):
# #             if (i*t2o_ratio +ii)<TqFrames:
# #                 img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
# #                 if exists(img_path):
# #                     multiVids[0].append(imread(  img_path  ).astype(uint8))
# #                 else:
# #                     raise FileExistsError(f"no file exist on {img_path}")
# #     # by 2
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
# #     beEmpty: list= [i for i in range(0, TqFrames, 2)]
# #     for i in beEmpty:
# #         multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+1<TqFrames:
# #             multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #
# #     # by 3
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
# #     beEmpty= [i for i in range(0, TqFrames, 3)]
# #     for i in beEmpty:
# #         multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
# #         if i+1<TqFrames:
# #             multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
# #         if i+2<TqFrames:
# #             multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
# #
# #
# #     # by 4
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
# #     beEmpty= [i for i in range(0, TqFrames, 4)]
# #     for i in beEmpty:
# #         multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+1<TqFrames:
# #             multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+2<TqFrames:
# #             multiVids[8][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+3<TqFrames:
# #             multiVids[9][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #
# #     # by 5
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
# #     beEmpty= [i for i in range(0, TqFrames, 5)]
# #     for i in beEmpty:
# #         multiVids[10][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+1<TqFrames:
# #             multiVids[11][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+2<TqFrames:
# #             multiVids[12][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+3<TqFrames:
# #             multiVids[13][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+4<TqFrames:
# #             multiVids[14][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #     return tuple(multiVids)
#
#
# def getFramesEqualTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
#     if int(len(vid['landmark']))!=TqFrames:
#         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Equal to {TqFrames}( TqFrames )")
#     multiVids: list= [[]]
#     for i in range(TqFrames):
#         # have slot for landmarks later ie. face, pose, left_hand, right_hand
#         multiVids[0].append([])
#         lm_face= vid['landmark'][i]['landmark_face']
#         if len(lm_face)==LM_Q_FACE:
#             multiVids[0][-1].extend(lm_face)
#         else:
#             multiVids[0][-1].extend(zeros((LM_Q_FACE, 2), dtype=float32))
#
#
#         lm_pose= vid['landmark'][i]['landmark_pose']
#         if len(lm_pose)==LM_Q_POSE:
#             multiVids[0][-1].extend(lm_pose)
#         else:
#             multiVids[0][-1].extend(zeros((LM_Q_POSE, 2), dtype=float32))
#
#
#         lm_left_hand= vid['landmark'][i]['landmark_left_hand']
#         if len(lm_left_hand)==LM_Q_HAND:
#             multiVids[0][-1].extend(lm_left_hand)
#         else:
#             multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#         lm_right_hand= vid['landmark'][i]['landmark_right_hand']
#         if len(lm_right_hand)==LM_Q_HAND:
#             multiVids[0][-1].extend(lm_right_hand)
#         else:
#             multiVids[0][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#     # # by 2
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
#     # beEmpty: list= [i for i in range(0, TqFrames, 2)]
#     # for i in beEmpty:
#     #     multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+1<TqFrames:
#     #         multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #
#     #
#     # # by 3
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
#     #
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
#     # beEmpty= [i for i in range(0, TqFrames, 3)]
#     # for i in beEmpty:
#     #     multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
#     #     multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
#     #     if i+1<TqFrames:
#     #         multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
#     #         multiVids[6][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
#     #         multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
#     #     if i+2<TqFrames:
#     #         multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
#     #         multiVids[7][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
#     #
#     #
#     # # by 4
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
#     # beEmpty= [i for i in range(0, TqFrames, 4)]
#     # for i in beEmpty:
#     #     multiVids[8][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+1<TqFrames:
#     #         multiVids[9][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+2<TqFrames:
#     #         multiVids[10][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+3<TqFrames:
#     #         multiVids[11][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #
#     #
#     # # by 5
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 15
#     # multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 16
#     # beEmpty= [i for i in range(0, TqFrames, 5)]
#     # for i in beEmpty:
#     #     multiVids[12][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+1<TqFrames:
#     #         multiVids[13][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+2<TqFrames:
#     #         multiVids[14][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+3<TqFrames:
#     #         multiVids[15][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #     if i+4<TqFrames:
#     #         multiVids[16][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#
#     checkshape= array(multiVids, dtype=float32).shape
#     if checkshape!=(1, TqFrames, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2):
#         raise ValueError("problem is on equal TqFrames generator")
#     return tuple(multiVids)
#
#
# # def getFramesEqualTarget_lessEmpty(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
# #     if int(len(vid['images']))!=TqFrames:
# #         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Equal to {TqFrames}( TqFrames )")
# #     multiVids: list= [[]]
# #     for i in range(TqFrames):
# #         img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
# #         if exists(img_path):
# #             multiVids[0].append(imread(  img_path  ).astype(uint8))
# #         else:
# #             raise FileExistsError(f"no file exist on {img_path}")
# #     # by 2
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
# #     beEmpty: list= [i for i in range(0, TqFrames, 2)]
# #     for i in beEmpty:
# #         multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+1<TqFrames:
# #             multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #
# #     # by 3
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5
# #     beEmpty= [i for i in range(0, TqFrames, 3)]
# #     for i in beEmpty:
# #         multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
# #         if i+1<TqFrames:
# #             multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
# #         if i+2<TqFrames:
# #             multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
# #
# #
# #     # by 4
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
# #     beEmpty= [i for i in range(0, TqFrames, 4)]
# #     for i in beEmpty:
# #         multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+1<TqFrames:
# #             multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+2<TqFrames:
# #             multiVids[8][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+3<TqFrames:
# #             multiVids[9][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #
# #     # by 5
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
# #     multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
# #     beEmpty= [i for i in range(0, TqFrames, 5)]
# #     for i in beEmpty:
# #         multiVids[10][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+1<TqFrames:
# #             multiVids[11][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+2<TqFrames:
# #             multiVids[12][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+3<TqFrames:
# #             multiVids[13][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #         if i+4<TqFrames:
# #             multiVids[14][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #     return tuple(multiVids)
#
#
# def getFramesGreaterThanTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
#     if int(len(vid['landmark']))<TqFrames:
#         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
#     oqFrames: int= len(vid['landmark'])
#     o2t_ratio: int= oqFrames//TqFrames
#
#
#     # evenly spaced
#     multiVids: list= [[] for _ in range(o2t_ratio)]
#     for i_mod in range(o2t_ratio):
#         for i in range(TqFrames):
#             # have slot for landmarks later ie. face, pose, left_hand, right_hand
#             multiVids[i_mod].append([])
#             lm_face= vid['landmark'][i*o2t_ratio +i_mod]['landmark_face']
#             if len(lm_face)==LM_Q_FACE:
#                 multiVids[i_mod][-1].extend(lm_face)
#             else:
#                 multiVids[i_mod][-1].extend(zeros((LM_Q_FACE, 2), dtype=float32))
#
#
#             lm_pose= vid['landmark'][i*o2t_ratio +i_mod]['landmark_pose']
#             if len(lm_pose)==LM_Q_POSE:
#                 multiVids[i_mod][-1].extend(lm_pose)
#             else:
#                 multiVids[i_mod][-1].extend(zeros((LM_Q_POSE, 2), dtype=float32))
#
#
#             lm_left_hand= vid['landmark'][i*o2t_ratio +i_mod]['landmark_left_hand']
#             if len(lm_left_hand)==LM_Q_HAND:
#                 multiVids[i_mod][-1].extend(lm_left_hand)
#             else:
#                 multiVids[i_mod][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#             lm_right_hand= vid['landmark'][i*o2t_ratio +i_mod]['landmark_right_hand']
#             if len(lm_right_hand)==LM_Q_HAND:
#                 multiVids[i_mod][-1].extend(lm_right_hand)
#             else:
#                 multiVids[i_mod][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#     # # len(multiVids) is now o2t_ratio, ie. o2t_ratio>=1
#     # # append version with missing frames, each be by 2 and 3
#     # for i_mod in range(o2t_ratio):
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
#     #     beEmpty: list= [i for i in range(0, TqFrames, 2)]
#     #     for i in beEmpty:
#     #         multiVids[-2][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #         if i+1<TqFrames:
#     #             multiVids[-1][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
#     #
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#     #     multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
#     #     beEmpty= [i for i in range(0, TqFrames, 3)]
#     #     for i in beEmpty:
#     #         multiVids[-5][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
#     #         multiVids[-2][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
#     #         if i+1<TqFrames:
#     #             multiVids[-4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
#     #             multiVids[-2][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
#     #             multiVids[-1][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
#     #         if i+2<TqFrames:
#     #             multiVids[-3][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
#     #             multiVids[-1][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
#     # del o2t_ratio
#
#
#     # consecutive
#     init_HasHand: int= -1
#     while init_HasHand==-1:
#         i: int= 0
#         while init_HasHand==-1 and i<len(vid['landmark']):
#             if 0<len(vid['landmark'][i]['landmark_left_hand']) or 0<len(vid['landmark'][i]['landmark_right_hand']):
#                 init_HasHand= i
#             i+= 1
#         if init_HasHand==-1:
#             init_HasHand= 0
#     # to check if has atleast enough frames for 1 TqFrames
#     if TqFrames<=(oqFrames -init_HasHand):
#         for init_consecutive in range(init_HasHand, oqFrames-TqFrames+1): # if a>b on range(a,b), then forLoopNotRun
#             multiVids.append([]) # append new video for each valid-consecutive
#             for i in range(init_consecutive, init_consecutive+TqFrames):
#                 multiVids[-1].append([]) # append for image(lmark really) of a video
#                 lm_face= vid['landmark'][i]['landmark_face']
#                 if len(lm_face)==LM_Q_FACE:
#                     multiVids[-1][-1].extend(lm_face)
#                 else:
#                     multiVids[-1][-1].extend(zeros((LM_Q_FACE, 2), dtype=float32))
#
#
#                 lm_pose= vid['landmark'][i]['landmark_pose']
#                 if len(lm_pose)==LM_Q_POSE:
#                     multiVids[-1][-1].extend(lm_pose)
#                 else:
#                     multiVids[-1][-1].extend(zeros((LM_Q_POSE, 2), dtype=float32))
#
#
#                 lm_left_hand= vid['landmark'][i]['landmark_left_hand']
#                 if len(lm_left_hand)==LM_Q_HAND:
#                     multiVids[-1][-1].extend(lm_left_hand)
#                 else:
#                     multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#                 lm_right_hand= vid['landmark'][i]['landmark_right_hand']
#                 if len(lm_right_hand)==LM_Q_HAND:
#                     multiVids[-1][-1].extend(lm_right_hand)
#                 else:
#                     multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#                 # img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
#                 #     multiVids[-1].append(imread(  img_path  ).astype(uint8))
#
#
#         # use floor in a different way, a different way of evenly splitting to TqFrames
#         # be included on checking if has atleast enough frames for 1 TqFrames
#         multiVids.append([])
#         o2t_rFloat: float= (oqFrames-init_HasHand)/TqFrames
#         for i in range(TqFrames):
#             idx_onVidImg: int= int(i*o2t_rFloat +init_HasHand)
#             multiVids[-1].append([]) # append for image(lmark really) of a video
#             lm_face= vid['landmark'][idx_onVidImg]['landmark_face']
#             if len(lm_face)==LM_Q_FACE:
#                 multiVids[-1][-1].extend(lm_face)
#             else:
#                 multiVids[-1][-1].extend(zeros((LM_Q_FACE, 2), dtype=float32))
#
#
#             lm_pose= vid['landmark'][idx_onVidImg]['landmark_pose']
#             if len(lm_pose)==LM_Q_POSE:
#                 multiVids[-1][-1].extend(lm_pose)
#             else:
#                 multiVids[-1][-1].extend(zeros((LM_Q_POSE, 2), dtype=float32))
#
#
#             lm_left_hand= vid['landmark'][idx_onVidImg]['landmark_left_hand']
#             if len(lm_left_hand)==LM_Q_HAND:
#                 multiVids[-1][-1].extend(lm_left_hand)
#             else:
#                 multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#             lm_right_hand= vid['landmark'][idx_onVidImg]['landmark_right_hand']
#             if len(lm_right_hand)==LM_Q_HAND:
#                 multiVids[-1][-1].extend(lm_right_hand)
#             else:
#                 multiVids[-1][-1].extend(zeros((LM_Q_HAND, 2), dtype=float32))
#
#     checkshape= array(multiVids, dtype=float32).shape
#     if checkshape[1:]!=(TqFrames, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2):
#         raise ValueError(f"problem is on greater than TqFrames generator {checkshape}")
#     return tuple(multiVids)
#
#
# # def getFramesGreaterThanTarget_lessEmpty(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
# #     if int(len(vid['images']))<TqFrames:
# #         raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
# #     oqFrames: int= len(vid['images'])
# #     o2t_ratio: int= oqFrames//TqFrames
# #
# #
# #     # evenly spaced
# #     multiVids: list= [[] for _ in range(o2t_ratio)]
# #     for i_mod in range(o2t_ratio):
# #         for i in range(TqFrames):
# #             img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i*o2t_ratio +i_mod]['file']}.png"))
# #             if exists(img_path):
# #                 multiVids[i_mod].append(imread(  img_path  ).astype(uint8))
# #             else:
# #                 raise FileExistsError(f"no file exist on {img_path}")
# #     # len(multiVids) is now o2t_ratio, ie. o2t_ratio>=1
# #     # append version with missing frames, each be by 2 and 3
# #     for i_mod in range(o2t_ratio):
# #         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
# #         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 2
# #         beEmpty: list= [i for i in range(0, TqFrames, 2)]
# #         for i in beEmpty:
# #             multiVids[-2][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #             if i+1<TqFrames:
# #                 multiVids[-1][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
# #
# #         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
# #         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
# #         multiVids.append(array(multiVids[i_mod], dtype=uint8, copy=True)) # for by 3
# #         beEmpty= [i for i in range(0, TqFrames, 3)]
# #         for i in beEmpty:
# #             multiVids[-3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
# #             if i+1<TqFrames:
# #                 multiVids[-2][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
# #             if i+2<TqFrames:
# #                 multiVids[-1][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
# #     del o2t_ratio
# #
# #
# #     # consecutive
# #     init_HasHand: int= -1
# #     while init_HasHand==-1:
# #         i: int= 0
# #         while init_HasHand==-1 and i<len(vid['images']):
# #             if vid['images'][i]['left_hand'] or vid['images'][i]['right_hand']:
# #                 init_HasHand= i
# #             i+= 1
# #         if init_HasHand==-1:
# #             init_HasHand= 0
# #     # to check if has atleast enough frames for 1 TqFrames
# #     if TqFrames<=(int(len(vid['images'])) -init_HasHand):
# #         for init_consecutive in range(init_HasHand, int(len(vid['images']))-TqFrames): # if a>b on range(a,b), then forLoopNotRun
# #             multiVids.append([]) # --------------- idx -6
# #             multiVids.append([]) # for by 2 -- idx -5
# #             multiVids.append([]) # for by 2 ------ idx -4
# #             multiVids.append([]) # for by 3 -- idx -3
# #             multiVids.append([]) # for by 3 ------ idx -2
# #             multiVids.append([]) # for by 3 -- idx -1
# #             for i in range(init_consecutive, init_consecutive+TqFrames):
# #                 img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
# #                 if exists(img_path):
# #                     multiVids[-6].append(imread(  img_path  ).astype(uint8))
# #                     if i%2==0:
# #                         multiVids[-5].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #                         multiVids[-4].append(imread(  img_path  ).astype(uint8))
# #                     else:
# #                         multiVids[-5].append(imread(  img_path  ).astype(uint8))
# #                         multiVids[-4].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #
# #
# #                     if i%3==0:
# #                         multiVids[-3].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #                         multiVids[-2].append(imread(  img_path  ).astype(uint8))
# #                         multiVids[-1].append(imread(  img_path  ).astype(uint8))
# #                     elif i%3==1:
# #                         multiVids[-3].append(imread(  img_path  ).astype(uint8))
# #                         multiVids[-2].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #                         multiVids[-1].append(imread(  img_path  ).astype(uint8))
# #                     else:
# #                         multiVids[-3].append(imread(  img_path  ).astype(uint8))
# #                         multiVids[-2].append(imread(  img_path  ).astype(uint8))
# #                         multiVids[-1].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #                 else:
# #                     raise FileExistsError(f"no file exist on {img_path}")
# #
# #
# #         # use floor in a different way, a different way of evenly splitting to TqFrames
# #         multiVids.append([])
# #         multiVids.append([])
# #         multiVids.append([])
# #         o2t_rFloat: float= (len(vid['images'])-init_HasHand)/TqFrames
# #         for i in range(TqFrames):
# #             idx_onVidImg: int= int(i*o2t_rFloat +init_HasHand)
# #             img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][  idx_onVidImg  ]['file']}.png"))
# #             if exists(img_path):
# #                 multiVids[-3].append(imread(  img_path  ).astype(uint8))
# #                 if idx_onVidImg%2==0:
# #                     multiVids[-2].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #                     multiVids[-1].append(imread(  img_path  ).astype(uint8))
# #                 else:
# #                     multiVids[-2].append(imread(  img_path  ).astype(uint8))
# #                     multiVids[-1].append(zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8))
# #             else:
# #                 raise FileExistsError(f"no file exist on {img_path}")
# #
# #     return tuple(multiVids)



def getdata(TrainVal: str= 'train', batch: int=TRAIN_BATCH) -> Generator[tuple, None, None]:
    shuffle(wlasl_READY_10[TrainVal])
    shuffle(wlasl_READY_10[TrainVal])

    glossDist: dict= { i: {
        'gloss_id': i,
        'gloss_name': wlasl_READY_10['label_id2gloss'][i],
        'quantity': 0,
        'video_id': []
    } for i in range(len(wlasl_READY_10['label_id2gloss']))}
    glossDist['split']= TrainVal
    glossDist['split_size']= len(wlasl_READY_10[TrainVal])
    # glossDist= {
    #     0: {
    #         'gloss_id': 0,
    #         'quantity': int(on this gloss id( ie. gloss_id=0 is book ) how many training was done),
    #         'video_id': list(all video_id that training was done, later be processed as unique),
    #         'vid_q_uniq': int(quantity of video_id that are unique, ie. above right after processed unique)
    #       },
    #     1: {...},
    #     2: {...},
    #     ...
    #     9: {...},
    #     'split': str(  'train'|'val'  )
    #     'split_size': int(quantity of video_id on 'train'|'val' ie. len(wlasl_READY_10[TrainVal]))
    # }
    # print(len(wlasl_READY_10['label_id2gloss'])) # correct, it exist
    init_IDXbatch: int= 0
    while True:
        batch_vids: ndarray= zeros(
            (batch, QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
            dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)
        for i in range(batch):
            idx_USE: int= (init_IDXbatch +i) if (init_IDXbatch +i)<len(wlasl_READY_10[TrainVal]) else (init_IDXbatch +i-len(wlasl_READY_10[TrainVal]))
            vid_lmarkNPZ_file: str= str(pjoin(LM_NPZ_DIR, f"{wlasl_READY_10[TrainVal][idx_USE]['file']}.npy"))
            if exists(vid_lmarkNPZ_file):
                with open(vid_lmarkNPZ_file, 'rb') as f:
                    gloss_id: int= int(wlasl_READY_10[TrainVal][idx_USE]['gloss_id'])
                    batch_vids[i]= load(f)
                    batch_class[i]= gloss_id
                    glossDist[  gloss_id  ]['video_id'].append(
                        str(wlasl_READY_10[TrainVal][idx_USE]['file'])[:-4]
                    )
                    glossDist[  gloss_id  ]['quantity']+= 1
            else:
                raise FileNotFoundError(f"file {vid_lmarkNPZ_file} does not exist")
        init_IDXbatch+= batch


        if TrainVal=='train' and init_IDXbatch==batch*TRAIN_STEPS:
            # overwrites for every epoch
            for i in range(len(wlasl_READY_10['label_id2gloss'])):
                glossDist[ i ]['video_id']= list(set(
                    glossDist[ i ]['video_id']
                ))
                glossDist[ i ]['vid_q_uniq']= int(len(glossDist[ i ]['video_id']))
            if not exists(str(pjoin(PROJ_ROOT, f"training_{TrainVal}"))):
                makedirs(str(pjoin(PROJ_ROOT, f"training_{TrainVal}")))
            # below overwrite prev epoch written json
            with open(str(pjoin(PROJ_ROOT, f"training_{TrainVal}", f"{TrainVal}_{init_IDXbatch}.json")), 'w') as f:
                dump(glossDist, f)
        elif TrainVal=='val' and init_IDXbatch==batch*T10_VAL:
            # overwrites for every epoch
            for i in range(len(wlasl_READY_10['label_id2gloss'])):
                glossDist[ i ]['video_id']= list(set(
                    glossDist[ i ]['video_id']
                ))
                glossDist[ i ]['vid_q_uniq']= int(len(glossDist[ i ]['video_id']))
            if not exists(str(pjoin(PROJ_ROOT, f"training_{TrainVal}"))):
                makedirs(str(pjoin(PROJ_ROOT, f"training_{TrainVal}")))
            # below overwrite prev epoch written json
            with open(str(pjoin(PROJ_ROOT, f"training_{TrainVal}", f"{TrainVal}_{init_IDXbatch}.json")), 'w') as f:
                dump(glossDist, f)


        if len(wlasl_READY_10[TrainVal])<=init_IDXbatch:
            init_IDXbatch-= int(len(wlasl_READY_10[TrainVal]))
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))


# if __name__=="__main__":
#     g= getdata()
#     print(f"blah {array(next(g)[0][0], dtype=float32).shape}")
#     print(f"blah {array(next(g)[0][0][0][0], dtype=float32)}")

