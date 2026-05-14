from cv2 import circle, imwrite, line
from json import load as jsonload, dump as jsonsave
from keras.src.saving import load_model
from math import ceil
from numpy import array, float32, load as npload, ndarray, save as numpysave, sum as npsum, uint8, zeros
from os import makedirs
from os.path import exists
from pathlib import Path
from sys import stderr


PROJ_ROOT: str= str(Path(__file__).parent.parent.parent)

MODEL_DIR: str= str(Path(PROJ_ROOT)/"model")
MODEL_FILE_LIST: tuple= ("aslvid2gloss_v30.keras",)
ASL2GLOSS_MODEL_LIST: tuple= tuple(load_model(Path(MODEL_DIR)/el) for el in MODEL_FILE_LIST)

GLASL_DIR: str= str(Path(PROJ_ROOT)/"dataset"/"glasl")
LANDMARK_origin: str= str(Path(GLASL_DIR)/"landmark")
LANDMARK_dir: str= str(Path(GLASL_DIR)/"image_landmark")
SKELETON_dir: str= str(Path(GLASL_DIR)/"image_skeleton")
KEY_TRAIN: str= "train"
KEY_VAL: str= "val"
KEY_TEST: str= "test"
KEY_ID2G: str= "id2gloss"
KEY_G2ID: str= "gloss2id"
QUANTITY_FRAME: int= 22
IMG_SIZE: int= 240
KEY_G_ID: str= 'gloss_id'
KEY_V_ID: str= 'video_id'
KEY_MODEL_ACC: str= 'model'
KEY_V_IMGs_ID_origin: str= 'landmark'
KEY_V_IMGs_ID: str= 'image'
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




def getGreaterThan_landmark_allHasHand(lmark_: dict, landmark_directory: str) -> list:
    '''
    to be used for when len(lmark_['landmark']) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 86, 2 ____)
    '''
    lmark_numpy_MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

    lmark_all: list= []
    idx_init_has_hand: int= -1
    for i in range(len(lmark_[KEY_V_IMGs_ID_origin])):
        with open(f"{Path(landmark_directory)/lmark_[KEY_V_ID]/lmark_[KEY_V_IMGs_ID_origin][i]['file']}", 'rb') as f:
            lmark_all.append(npload(f))
        if idx_init_has_hand==-1:
            if lmark_[KEY_V_IMGs_ID_origin][i]['left_hand'] or lmark_[KEY_V_IMGs_ID_origin][i]['right_hand']:
                idx_init_has_hand= i
    if idx_init_has_hand==-1:
        return []

    # part 1, floor at index level, still idx_init_has_hand
    o2t_ratio: float= (len(lmark_[KEY_V_IMGs_ID_origin])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        append_if_valid: int= idx_init_has_hand+int(i*o2t_ratio)
        if lmark_[KEY_V_IMGs_ID_origin][append_if_valid]['left_hand'] or \
            lmark_[KEY_V_IMGs_ID_origin][append_if_valid]['right_hand']:
            lmark_numpy_MANY_VIDS[0].append(lmark_all[append_if_valid]) # floor
        else:
            lmark_numpy_MANY_VIDS[0].append(lmark_numpy_MANY_VIDS[0][-1])
    if len(lmark_numpy_MANY_VIDS[0])!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_landmark_allHasHand on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del o2t_ratio
    # lmark_numpy_MANY_VIDS[0] is of shape (QUANTITY_FRAME, 518, 2), but
    # here lmark_numpy_MANY_VIDS is of shape (1, QUANTITY_FRAME, 518, 2)

    len_available_images: int= len(lmark_[KEY_V_IMGs_ID_origin])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
        notIncludedOn_mod: int= len(lmark_[KEY_V_IMGs_ID_origin])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
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
                    if lmark_[KEY_V_IMGs_ID_origin][append_if_valid]['left_hand'] or \
                        lmark_[KEY_V_IMGs_ID_origin][append_if_valid]['right_hand']:
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
                if lmark_[KEY_V_IMGs_ID_origin][append_if_valid]['left_hand'] or \
                    lmark_[KEY_V_IMGs_ID_origin][append_if_valid]['right_hand']:
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
                    if lmark_[KEY_V_IMGs_ID_origin][i]['left_hand'] or lmark_[KEY_V_IMGs_ID_origin][i]['right_hand']:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[  i  ])
                    else:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
        if len(lmark_numpy_MANY_VIDS[-1])!=QUANTITY_FRAME:
            raise ValueError("incorrect implementation on idx idx_init_has_hand!=-1 and len_available_images<QUANTITY_FRAME, getGreaterThan_landmark_allHasHand")
    del len_available_images

    return lmark_numpy_MANY_VIDS


def getLessThanOrEqual_landmark_allHasHand(lmark_: dict, landmark_directory: str) -> list:
    '''
    to be used for when len(lmark_['landmark']) <= QUANTITY_FRAME
    output be of shape(____ QUANTITY_FRAME, 86, 2 ____)
    '''
    def getIdxStartHand(image_list: list) -> int:
        for i in range(len(image_list)):
            if image_list[i]['left_hand'] or image_list[i]['right_hand']:
                return i
        return -1
    idx_init_has_hand: int= getIdxStartHand(image_list=lmark_[KEY_V_IMGs_ID_origin])
    if idx_init_has_hand==-1:
        return []
    lmark_numpy: list= []
    t2o_ratio: int= int(ceil(QUANTITY_FRAME/(len(lmark_[KEY_V_IMGs_ID_origin])-idx_init_has_hand)))
    for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, len(lmark_[KEY_V_IMGs_ID_origin])), range(len(lmark_[KEY_V_IMGs_ID_origin])-idx_init_has_hand)):
        landmark_data_numpy= None
        with open(f"{Path(landmark_directory)/lmark_[KEY_V_ID]/lmark_[KEY_V_IMGs_ID_origin][  i  ]['file']}", 'rb') as f:
            landmark_data_numpy= npload(f)
        for ii in range(t2o_ratio):
            if (i_0to_t2o_multiplier*t2o_ratio+ii)<QUANTITY_FRAME:
                # i_0to_t2o_multiplier*t2o_ratio, due to since: getLessThanOrEqual_landmark,
                # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                # multiple/( or 1 time if equal and idx 0 has hand ) times ie. int(t2o_ratio) times
                # then +ii, due to current be added mod of from int(t2o_ratio),
                # thus i_0to_t2o_multiplier*t2o_ratio+ii
                if lmark_[KEY_V_IMGs_ID_origin][i]['left_hand'] or lmark_[KEY_V_IMGs_ID_origin][i]['right_hand']:
                    lmark_numpy.append( landmark_data_numpy )
                else:
                    lmark_numpy.append( lmark_numpy[-1] )
    if len(lmark_numpy)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getLessThanOrEqual_landmark_allHasHand, due to len(lmark_numpy)!=QUANTITY_FRAME")
    return lmark_numpy


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
def drawFacePoseHand(
        img_write_to: ndarray,
        landmark_numpy: ndarray,
        has_face: bool,
        has_pose: bool,
        has_left_hand: bool,
        has_right_hand: bool
    ) -> ndarray:
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
    quantity_lm_face: int= len(WORTHY_FACE_IDX)
    quantity_lm_pose: int= len(WORTHY_POSE_IDX)
    if has_face:
        # print(f"{array(landmark_numpy[:len(WORTHY_FACE_IDX)]).shape}")
        img_write_to= recalcDrawFace(img_write_to, tuple(landmark_numpy.tolist()[:quantity_lm_face]))
    if has_pose:
        # print(f"{array(landmark_numpy[len(WORTHY_FACE_IDX):len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)]).shape}")
        img_write_to= recalcDrawPose(img_write_to, tuple(landmark_numpy.tolist()[quantity_lm_face:quantity_lm_face+quantity_lm_pose]))
    if has_left_hand:
        # print(f"{array(landmark_numpy[len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX):len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+21]).shape}")
        img_write_to= recalcDrawLeftHands(img_write_to, tuple(landmark_numpy.tolist()[quantity_lm_face+quantity_lm_pose:quantity_lm_face+quantity_lm_pose+21]))
    if has_right_hand:
        # print(f"{array(landmark_numpy[-21:]).shape}")
        img_write_to= recalcDrawRightHands(img_write_to, tuple(landmark_numpy.tolist()[-21:]))

    return img_write_to


def get_video_skeletons(gloss_video_lmark: dict) -> ndarray:
    allImg_skeleton: list= []
    for lmark_np in gloss_video_lmark[KEY_V_IMGs_ID_origin]:
        skeleton__image= drawFacePoseHand(
            img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
            landmark_numpy=npload(f"{Path(LANDMARK_origin) / gloss_video_lmark[KEY_V_ID] / lmark_np['file']}"),
            has_face=lmark_np['face'],
            has_pose=lmark_np['pose'],
            has_left_hand=lmark_np['left_hand'],
            has_right_hand=lmark_np['right_hand'],
        )
        allImg_skeleton.append(skeleton__image)

    if len(gloss_video_lmark[KEY_V_IMGs_ID_origin])!=len(allImg_skeleton):
        print(f"len gloss_video_lmark[{KEY_V_IMGs_ID_origin}] --> {len(gloss_video_lmark[KEY_V_IMGs_ID_origin])}", file=stderr)
        print(f"len allImg_skeleton --> {len(allImg_skeleton)}", file=stderr)
        raise NotImplementedError("Incorrect implementation due to mandatory all 4 be having same quantity of elements")
    return array(allImg_skeleton, dtype=uint8)


def mandatory_all_2_notExist() -> None:
    if exists(LANDMARK_dir):
        raise FileExistsError(f"please delete this folder {LANDMARK_dir}, will be the one to create it for you.")
    if exists(SKELETON_dir):
        raise FileExistsError(f"please delete this folder {SKELETON_dir}, will be the one to create it for you.")

    makedirs(LANDMARK_dir)
    makedirs(SKELETON_dir)


def init_vars() -> tuple:
    glasl_clean_landmark: dict= {}
    with open(f"{Path(GLASL_DIR)/"glasl.annotation.landmark.json"}", 'r') as f:
        glasl_clean_landmark= jsonload(f)
    glasl_LANDMARK: dict= {
        KEY_TRAIN: [],
        KEY_VAL: [],
        KEY_TEST: [],
        KEY_ID2G: [ins for ins in glasl_clean_landmark[KEY_ID2G]],
        KEY_G2ID: {glasl_clean_landmark[KEY_ID2G][i]: i for i in range(len(glasl_clean_landmark[KEY_ID2G]))},
        KEY_MODEL_ACC: MODEL_FILE_LIST,
    }
    return (glasl_clean_landmark, glasl_LANDMARK)


if __name__=='__main__':
    mandatory_all_2_notExist()
    glasl_c_landmark, glasl_LANDMARK= init_vars()


    for data_split in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        for a_gloss_video in glasl_c_landmark[data_split]: # on each split has many videos, now for each videos
            # gloss_video['gloss_id']
            # gloss_video['video_id']
            # gloss_video['landmark']
            # ---- gloss_video['landmark'][...] is has key
            # -------- gloss_video['landmark'][...]['file']
            # -------- gloss_video['landmark'][...]['face']
            # -------- gloss_video['landmark'][...]['pose']
            # -------- gloss_video['landmark'][...]['left_hand']
            # -------- gloss_video['landmark'][...]['right_hand']
            imgs_skeleton= get_video_skeletons(a_gloss_video)
            glasl_LANDMARK[ data_split ].append({
                KEY_G_ID: a_gloss_video[KEY_G_ID],
                KEY_V_ID: a_gloss_video[KEY_V_ID],
                KEY_MODEL_ACC: [],
                KEY_V_IMGs_ID: [],
            })
            for model_idx in range(len(MODEL_FILE_LIST)):
                landmark_dataset_elements: list= []
                if len(a_gloss_video[KEY_V_IMGs_ID_origin])<=QUANTITY_FRAME:
                    landmark_dataset_elements.append(getLessThanOrEqual_landmark_allHasHand(
                        lmark_=a_gloss_video,
                        landmark_directory=LANDMARK_origin,
                    ))
                else:
                    landmark_dataset_elements= getGreaterThan_landmark_allHasHand(
                        lmark_=a_gloss_video,
                        landmark_directory=LANDMARK_origin,
                    )
                quantity_of_elements: int= array(landmark_dataset_elements).shape[0]
                modelPredict= ASL2GLOSS_MODEL_LIST[model_idx].predict(
                    x=array(landmark_dataset_elements, dtype=float32),
                    batch_size=quantity_of_elements,
                )
                modelPredict= npsum(modelPredict, axis=0) / quantity_of_elements
                glasl_LANDMARK[ data_split ][-1][KEY_MODEL_ACC].append({
                    'model_file': MODEL_FILE_LIST[model_idx],
                    'accuracy': tuple(modelPredict.tolist())
                })
            for i in range(imgs_skeleton.shape[0]): # each video has many images, now for each images
                file2create: str= f"{a_gloss_video[KEY_V_ID]}_{a_gloss_video[KEY_V_IMGs_ID_origin][i]['file'][:-4]}"
                filename_abs_landmark_w: str= f"{Path(LANDMARK_dir) / file2create}.npy"
                filename_abs_skeleton_w: str= f"{Path(SKELETON_dir) / file2create}.png"
                with open(filename_abs_landmark_w, "wb") as f:
                    numpysave(file=f, arr=npload(f"{Path(LANDMARK_origin) / a_gloss_video[KEY_V_ID] / a_gloss_video[KEY_V_IMGs_ID_origin][i]['file']}"))
                imwrite(filename=filename_abs_skeleton_w, img=imgs_skeleton[i])
                glasl_LANDMARK[ data_split ][-1][KEY_V_IMGs_ID].append({
                    "numpy_file": f"{file2create}.npy",
                    "skeleton_file": f"{file2create}.png",
                    "face": a_gloss_video[KEY_V_IMGs_ID_origin][i]['face'],
                    "pose": a_gloss_video[KEY_V_IMGs_ID_origin][i]['pose'],
                    "left_hand": a_gloss_video[KEY_V_IMGs_ID_origin][i]['left_hand'],
                    "right_hand": a_gloss_video[KEY_V_IMGs_ID_origin][i]['right_hand'],
                })
    with open(f"{Path(GLASL_DIR)/"glasl.annotation.image_landmark.json"}", 'w') as f:
        jsonsave(glasl_LANDMARK, f, indent=4)
