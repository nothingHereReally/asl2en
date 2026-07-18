from cv2 import circle, imwrite, line
from json import load as jsonload, dump as jsonsave
from keras.src.saving import load_model
from math import ceil
from numpy import array, float32, load as loadnp, ndarray, save as numpysave, sum as npsum, uint8, zeros
from os import makedirs
from os.path import exists
from pathlib import Path
from sys import stderr

from src_asl2gloss.lmark_constant import (
    GLASL_LANDMARK_DIR,
    KEY_FILE,
    KEY_GLOSS,
    KEY_LHAND,
    KEY_LMARK,
    KEY_RHAND,
    KEY_VIDEO,
    LANDMARK_SHAPE
)


PROJ_ROOT: Path= Path(__file__).parent.parent.parent

MODEL_DIR: Path= PROJ_ROOT /"model"
MODEL_FILE_LIST: tuple= (
    "aslvid2gloss_v25.keras",
    "aslvid2gloss_v30.keras",
)
ASL2GLOSS_MODEL_LIST: tuple= tuple(load_model(str(MODEL_DIR /el)) for el in MODEL_FILE_LIST)

GLASL_DIR: Path= Path(PROJ_ROOT) /"dataset" /"glasl"
LANDMARK_origin: Path= Path(GLASL_DIR) /"landmark"
LANDMARK_dir: Path= Path(GLASL_DIR) /"image_landmark"
SKELETON_dir: Path= Path(GLASL_DIR) /"image_skeleton"
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




def get_idx_start_hand(annotated_images: list) -> int:
    for idx in range(len(annotated_images)):
        if annotated_images[idx][KEY_LHAND] or annotated_images[idx][KEY_RHAND]:
            return idx
    return -1


def get_landmark4less_or_equal(a_raw_video: dict, idx_init_has_hand: int|None=None) -> list:
    '''
    output be of shape(____ QUANTITY_FRAME, 86, 2 ____)
    '''
    if idx_init_has_hand is None:
        idx_init_has_hand= get_idx_start_hand(a_raw_video[KEY_LMARK])
    if idx_init_has_hand==-1:
        return list()

    lmark_numpy_out: list= []
    ratio_what: int= int(ceil(
        QUANTITY_FRAME  /  (len(a_raw_video[KEY_LMARK])-idx_init_has_hand)
    ))
    for idx in range(idx_init_has_hand, len(a_raw_video[KEY_LMARK])):
        with open(f"{GLASL_LANDMARK_DIR /a_raw_video[KEY_VIDEO] /a_raw_video[KEY_LMARK][idx][KEY_FILE]}", 'rb') as f:
            load_an_image_landmarks= loadnp(f)
        for _ in range(min(ratio_what, QUANTITY_FRAME-len(lmark_numpy_out))):
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                lmark_numpy_out.append(load_an_image_landmarks)
            else:
                lmark_numpy_out.append(lmark_numpy_out[-1])
    check_shape: ndarray= array(lmark_numpy_out, dtype=float32)
    if check_shape.shape!=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]):
        raise NotImplementedError(
            f"incorrect implementation on get_landmark4less_or_equal(), due to lmark_numpy_out should be of shape {
            tuple((QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]))
            }, but got {check_shape.shape}"
        )
    return lmark_numpy_out


def get_landmark4greater(a_raw_video: dict) -> list:
    '''
    output be of shape(____ QUANTITY_FRAME, 86, 2 ____)
    '''
    lmark_load_ALL: list= []
    idx_init_has_hand: int= -1
    for idx in range(len(a_raw_video[KEY_LMARK])):
        with open(f"{GLASL_LANDMARK_DIR /a_raw_video[KEY_VIDEO] /a_raw_video[KEY_LMARK][idx][KEY_FILE]}", 'rb') as f:
            lmark_load_ALL.append(loadnp(f))
        if idx_init_has_hand==-1:
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                idx_init_has_hand= idx
    if idx_init_has_hand==-1:
        raise ValueError(f"video {a_raw_video[KEY_VIDEO]}.mp4 has no hands")
    len_available_images: int= len(a_raw_video[KEY_LMARK])-idx_init_has_hand

    if QUANTITY_FRAME<len_available_images:
        init_hand_idxs: tuple= tuple(range(idx_init_has_hand, len(a_raw_video[KEY_LMARK])))
        past_img_has_hand: ndarray= lmark_load_ALL[idx_init_has_hand]
        tmp_part4: list= []
        for check_mod_init_0, idx in zip(range(len_available_images), init_hand_idxs):
            if check_mod_init_0%5 == 4:
                if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or \
                    a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                    tmp_part4.append(lmark_load_ALL[idx])
                else:
                    tmp_part4.append(past_img_has_hand)
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or \
                a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                past_img_has_hand= lmark_load_ALL[idx]
        if QUANTITY_FRAME<len(tmp_part4):
            start_where: int= int((len(tmp_part4)-QUANTITY_FRAME+1)//2)
            return tmp_part4[start_where:start_where+QUANTITY_FRAME]
        else:
            tmp_for_less_or_qual: list= []
            ratio4mod_part4: int= int(ceil(QUANTITY_FRAME/len(tmp_part4)))
            for an_image_landmarks in tmp_part4:
                tmp_for_less_or_qual.extend(list([an_image_landmarks]) *ratio4mod_part4)
            tmp_for_less_or_qual= tmp_for_less_or_qual[:QUANTITY_FRAME]
            return tmp_for_less_or_qual
    # len_available_images <= QUANTITY_FRAME
    copy_a_raw_video: dict= {
        KEY_GLOSS: a_raw_video[KEY_GLOSS],
        KEY_VIDEO: a_raw_video[KEY_VIDEO],
        KEY_LMARK: a_raw_video[KEY_LMARK][idx_init_has_hand:]
    }
    return get_landmark4less_or_equal(copy_a_raw_video, 0)


def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: ndarray, \
                    lmark_coordinates: tuple, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> ndarray:
    img_wh: dict= {"wx": image.shape[1], "hy": image.shape[0]}


    # drawing the lines between 2 landmark connections
    if color_line!=None or color_dot!=None:
        for lmark_idx_pair in connections_idxs:
            pA: tuple= (
                lmark_coordinates[  lmark_idx_pair[0]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[0]  ][1]  # y
            )
            pB: tuple= (
                lmark_coordinates[  lmark_idx_pair[1]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[1]  ][1]  # y
            )
            if isOkPlot(pA) and isOkPlot(pB):
                if color_dot!=None:
                    circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    circle(
                        img=image,
                        center=(
                            int(pB[0]*img_wh['wx']),
                            int(pB[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                if color_line!=None:
                    line(
                        img=image,
                        pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                        pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                        color=color_line,
                        thickness=thick
                    )
            else:
                raise NotImplementedError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0, on both x and y coordinates")
            del pA
            del pB
    return image
def drawFacePoseHand(
        img_write_to: ndarray,
        landmark_numpy: ndarray,
        has_face: bool,
        has_pose: bool,
        has_left_hand: bool,
        has_right_hand: bool
    ) -> ndarray:
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
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=tuple(landmark_numpy.tolist()[:quantity_lm_face]),
            connections_idxs=FACE_CONNECTIONS,
            thick=1,
            color_line=(0, 153, 0), # 153/255= 0.6
            color_dot=None,
        )
    if has_pose:
        # print(f"{array(landmark_numpy[len(WORTHY_FACE_IDX):len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)]).shape}")
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=tuple(landmark_numpy.tolist()[quantity_lm_face:quantity_lm_face+quantity_lm_pose]),
            connections_idxs=POSE_CONNECTIONS,
            thick=1,
            color_line=(0, 0, 153), # 153/255= 0.6
            color_dot=None,
        )
    if has_left_hand:
        # print(f"{array(landmark_numpy[len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX):len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+21]).shape}")
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=tuple(landmark_numpy.tolist()[quantity_lm_face+quantity_lm_pose:quantity_lm_face+quantity_lm_pose+21]),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_line=(255, 255, 255),
            color_dot=None,
        )
    if has_right_hand:
        # print(f"{array(landmark_numpy[-21:]).shape}")
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=tuple(landmark_numpy.tolist()[-21:]),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_line=(204, 204, 204), # 204/255= 0.8
            color_dot=None,
        )

    return img_write_to


def get_video_skeletons(gloss_video_lmark: dict) -> ndarray:
    allImg_skeleton: list= []
    for lmark_np in gloss_video_lmark[KEY_V_IMGs_ID_origin]:
        skeleton__image= drawFacePoseHand(
            img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
            landmark_numpy=loadnp(f"{Path(LANDMARK_origin) /gloss_video_lmark[KEY_V_ID] /lmark_np['file']}"),
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
        raise FileExistsError(f"please delete this folder {LANDMARK_dir}, it will be the one to create it for you.")
    if exists(SKELETON_dir):
        raise FileExistsError(f"please delete this folder {SKELETON_dir}, it will be the one to create it for you.")

    makedirs(LANDMARK_dir)
    makedirs(SKELETON_dir)


def init_vars() -> tuple:
    glasl_clean_landmark: dict= {}
    with open(f"{GLASL_DIR /"glasl.annotation.landmark.json"}", 'r') as f:
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


def main():
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
                if a_gloss_video[KEY_G_ID]<ASL2GLOSS_MODEL_LIST[model_idx].output_shape[-1]:
                    landmark_dataset_elements: list= []
                    if len(a_gloss_video[KEY_V_IMGs_ID_origin])<=QUANTITY_FRAME:
                        landmark_dataset_elements.append(get_landmark4less_or_equal(
                            a_raw_video=a_gloss_video
                        ))
                    else:
                        landmark_dataset_elements.append(get_landmark4greater(
                            a_raw_video=a_gloss_video
                        ))
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
                filename_abs_landmark_w: str= f"{LANDMARK_dir / file2create}.npy"
                filename_abs_skeleton_w: str= f"{SKELETON_dir / file2create}.png"
                with open(filename_abs_landmark_w, "wb") as f:
                    numpysave(file=f, arr=loadnp(f"{Path(LANDMARK_origin) /a_gloss_video[KEY_V_ID] /a_gloss_video[KEY_V_IMGs_ID_origin][i]['file']}"))
                imwrite(filename=filename_abs_skeleton_w, img=imgs_skeleton[i])
                glasl_LANDMARK[ data_split ][-1][KEY_V_IMGs_ID].append({
                    "numpy_file": f"{file2create}.npy",
                    "skeleton_file": f"{file2create}.png",
                    "face": a_gloss_video[KEY_V_IMGs_ID_origin][i]['face'],
                    "pose": a_gloss_video[KEY_V_IMGs_ID_origin][i]['pose'],
                    "left_hand": a_gloss_video[KEY_V_IMGs_ID_origin][i]['left_hand'],
                    "right_hand": a_gloss_video[KEY_V_IMGs_ID_origin][i]['right_hand'],
                })
    with open(f"{GLASL_DIR /"glasl.annotation.image_landmark.json"}", 'w') as f:
        jsonsave(glasl_LANDMARK, f, indent=4)


if __name__=='__main__':
    main()
