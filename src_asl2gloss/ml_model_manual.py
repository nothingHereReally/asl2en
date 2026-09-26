import cv2
from pathlib import Path
from json import load as loadJson, dump as writeJson
import math
import numpy as np
# import jax.numpy as jnp


PROJ_ROOT: Path= Path(__file__).parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
LANDMARK_dir: Path= DS_DIR /"ds_landmark"
MODEL_LANDMARK_DIR: Path= DS_DIR /"model_landmark"
MODEL_SKELETON_DIR: Path= DS_DIR /"model_skeleton"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
QUANTITY_FRAME: int= 8
MOD_PART_3: tuple= (3, 4, 5, 6, 7)
LM_SHAPE_NORMALIZED: tuple= (86, 2)
IMG_SIZE: int= 158
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
QUANTITY_HAND_LMARK: int= 21




def has_atleast_1hand(annotation: dict) -> bool:
    return (annotation[KEY_LHAND] or annotation[KEY_RHAND]) and \
            annotation[KEY_FACE] and annotation[KEY_POSE]
def idx_init_has_hand(landmarks: list) -> int:
    for idx in range(len(landmarks)):
        if has_atleast_1hand(landmarks[idx]):
            return idx
    return -1
def q_imgs_less_than_or_equal(
    landmarks: list,
    parent_folder: str,
) -> tuple:
    lm_data_npy: list= []
    annotations: list= []

    idx_init: int= idx_init_has_hand(landmarks)
    assert idx_init!=-1
    ratio: int= math.ceil(QUANTITY_FRAME /(len(landmarks) -idx_init))
    past_npy: np.ndarray= np.zeros(LM_SHAPE_NORMALIZED)
    for a_landmark in landmarks[idx_init:]:
        if has_atleast_1hand(a_landmark):
            with open(f"{LANDMARK_dir /parent_folder /a_landmark[KEY_FILE]}", 'rb') as f:
                tmp_npy_data: np.ndarray= np.load(f)
                for _ in range(min(ratio, QUANTITY_FRAME-len(lm_data_npy))):
                    lm_data_npy.append(tmp_npy_data.copy())
                    annotations.append({
                        KEY_FACE: a_landmark[KEY_FACE],
                        KEY_POSE: a_landmark[KEY_POSE],
                        KEY_LHAND: a_landmark[KEY_LHAND],
                        KEY_RHAND: a_landmark[KEY_RHAND],
                    })
                    past_npy= np.array(lm_data_npy[-1]).copy()
        else:
            for _ in range(min(ratio, QUANTITY_FRAME-len(lm_data_npy))):
                lm_data_npy.append(past_npy)
                annotations.append(annotations[-1])
    assert np.array(lm_data_npy).shape==(QUANTITY_FRAME, LM_SHAPE_NORMALIZED[0], LM_SHAPE_NORMALIZED[1])
    return (lm_data_npy, annotations)
def greater_than_qf_p1(
    landmarks: list,
    parent_folder: str,
) -> tuple:
    lm_data_npy: list= []
    annotations: list= []

    ratio: int= math.floor(len(landmarks)/QUANTITY_FRAME)
    past_npy: np.ndarray= np.zeros(LM_SHAPE_NORMALIZED)
    for idx in range(QUANTITY_FRAME):
        a_landmark= landmarks[idx*ratio]
        if has_atleast_1hand(a_landmark):
            with open(f"{LANDMARK_dir /parent_folder /a_landmark[KEY_FILE]}", 'rb') as f:
                lm_data_npy.append(np.load(f))
                annotations.append({
                    KEY_FACE: a_landmark[KEY_FACE],
                    KEY_POSE: a_landmark[KEY_POSE],
                    KEY_LHAND: a_landmark[KEY_LHAND],
                    KEY_RHAND: a_landmark[KEY_RHAND],
                })
                past_npy= np.array(lm_data_npy[-1]).copy()
        else:
            lm_data_npy.append(past_npy)
            annotations.append(annotations[-1])
    assert np.array(lm_data_npy).shape==(QUANTITY_FRAME, LM_SHAPE_NORMALIZED[0], LM_SHAPE_NORMALIZED[1])
    return (lm_data_npy, annotations)
def greater_than_qf_p2(
    landmarks: list,
    parent_folder: str,
) -> tuple:
    lm_data_npy_many: list= [] # each element inside is of shape (QUANTITY_FRAME, 86, 2)
    annotations_many: list= []

    how_many_qfs: int= math.floor(len(landmarks)/QUANTITY_FRAME)
    remains: int= len(landmarks) -how_many_qfs*QUANTITY_FRAME
    mod_list: tuple= tuple(range(how_many_qfs))
    load_lm_data: list= []
    for a_landmark in landmarks:
        with open(f"{LANDMARK_dir /parent_folder /a_landmark[KEY_FILE]}", 'rb') as f:
            load_lm_data.append(np.load(f))
    past_npy: np.ndarray= load_lm_data[0].copy()
    past_notation: dict= {
        KEY_FACE: landmarks[0][KEY_FACE],
        KEY_POSE: landmarks[0][KEY_POSE],
        KEY_LHAND: landmarks[0][KEY_LHAND],
        KEY_RHAND: landmarks[0][KEY_RHAND],
    }
    for idx_init_only in range(remains+1):
        tmp_lm_data: list= [[] for _ in range(how_many_qfs)]
        tmp_annotations: list= [[] for _ in range(how_many_qfs)]
        for idx in range(
            idx_init_only,
            idx_init_only +(how_many_qfs*QUANTITY_FRAME)
        ):
            for mod_what in mod_list:
                if idx%how_many_qfs==mod_what:
                    if has_atleast_1hand(landmarks[idx]):
                        tmp_lm_data[mod_what].append(load_lm_data[idx])
                        tmp_annotations[mod_what].append({
                            KEY_FACE: landmarks[idx][KEY_FACE],
                            KEY_POSE: landmarks[idx][KEY_POSE],
                            KEY_LHAND: landmarks[idx][KEY_LHAND],
                            KEY_RHAND: landmarks[idx][KEY_RHAND],
                        })
                        past_npy= load_lm_data[idx].copy()
                        past_notation= {
                            KEY_FACE: landmarks[idx][KEY_FACE],
                            KEY_POSE: landmarks[idx][KEY_POSE],
                            KEY_LHAND: landmarks[idx][KEY_LHAND],
                            KEY_RHAND: landmarks[idx][KEY_RHAND],
                        }
                    else:
                        tmp_lm_data[mod_what].append(past_npy)
                        tmp_annotations[mod_what].append(past_notation)
        lm_data_npy_many.extend(tmp_lm_data)
        annotations_many.extend(tmp_annotations)
        assert np.array(lm_data_npy_many).shape[1:]==(QUANTITY_FRAME, *LM_SHAPE_NORMALIZED)
    return (lm_data_npy_many, annotations_many)
def greater_than_qf_p3(
    landmarks: list,
    parent_folder: str,
) -> tuple:
    lm_data_npy_many: list= [] # each element inside is of shape (QUANTITY_FRAME, 86, 2)
    annotations_many: list= [] # eachElementInsideIsAnArrayOf len QUANTITY_FRAME, thenEachIs dict

    groups_lm: list= [[] for _ in MOD_PART_3]
    groups_notation: list= [[] for _ in MOD_PART_3]
    load_lm_data: list= []
    for a_landmark in landmarks:
        with open(f"{LANDMARK_dir /parent_folder /a_landmark[KEY_FILE]}", 'rb') as f:
            load_lm_data.append(np.load(f))
    past_npy: np.ndarray= load_lm_data[0].copy()
    past_notation: dict= {
        KEY_FACE: landmarks[0][KEY_FACE],
        KEY_POSE: landmarks[0][KEY_POSE],
        KEY_LHAND: landmarks[0][KEY_LHAND],
        KEY_RHAND: landmarks[0][KEY_RHAND],
    }
    for idx in range(len(landmarks)):
        for idx_mod, what_mod in enumerate(MOD_PART_3):
            if (idx+1)%what_mod==0:
                if has_atleast_1hand(landmarks[idx]):
                    groups_lm[idx_mod].append(load_lm_data[idx])
                    groups_notation[idx_mod].append({
                        KEY_FACE: landmarks[idx][KEY_FACE],
                        KEY_POSE: landmarks[idx][KEY_POSE],
                        KEY_LHAND: landmarks[idx][KEY_LHAND],
                        KEY_RHAND: landmarks[idx][KEY_RHAND],
                    })
                    past_npy= load_lm_data[idx].copy()
                    past_notation= {
                        KEY_FACE: landmarks[idx][KEY_FACE],
                        KEY_POSE: landmarks[idx][KEY_POSE],
                        KEY_LHAND: landmarks[idx][KEY_LHAND],
                        KEY_RHAND: landmarks[idx][KEY_RHAND],
                    }
                else:
                    groups_lm[idx_mod].append(past_npy)
                    groups_notation[idx_mod].append(past_notation)
    for idx_mod in range(len(MOD_PART_3)):
        if len(groups_lm[idx_mod])<=QUANTITY_FRAME:
            ratio: int= math.ceil(QUANTITY_FRAME/len(groups_lm[idx_mod]))
            lm_data_npy_many.append([])
            annotations_many.append([])
            for idx_g_lm in range(len(groups_lm[idx_mod])):
                for _ in range(min(ratio, QUANTITY_FRAME -len(lm_data_npy_many[-1]))):
                    lm_data_npy_many[-1].append(groups_lm[idx_mod][idx_g_lm])
                    annotations_many[-1].append(groups_notation[idx_mod][idx_g_lm])
        else:
            for idx_init in range(len(groups_lm[idx_mod]) -QUANTITY_FRAME +1):
                lm_data_npy_many.append(groups_lm[idx_mod][
                    idx_init:
                    idx_init +QUANTITY_FRAME
                ])
                annotations_many.append(groups_notation[idx_mod][
                    idx_init:
                    idx_init +QUANTITY_FRAME
                ])
    return (lm_data_npy_many, annotations_many)
def q_imgs_greater_than(
    landmarks: list,
    parent_folder: str,
) -> tuple:
    lm_data_npy_many: list= []
    annotations_many: list= []

    idx_init: int= idx_init_has_hand(landmarks)
    assert idx_init!=-1
    q_imgs_available: int= len(landmarks)-idx_init
    if q_imgs_available<=QUANTITY_FRAME:
        lm_data, annotation= q_imgs_less_than_or_equal(
            landmarks=landmarks[idx_init:],
            parent_folder=parent_folder
        )
        lm_data_npy_many.append(lm_data)
        annotations_many.append(annotation)
    else:
        # ---- part 1 ----
        lm_data, annotation= greater_than_qf_p1(
            landmarks=landmarks[idx_init:],
            parent_folder=parent_folder,
        )
        lm_data_npy_many.append(lm_data)
        annotations_many.append(annotation)

        # ---- part 2 ----
        lm_data, annotation= greater_than_qf_p2(
            landmarks=landmarks[idx_init:],
            parent_folder=parent_folder,
        )
        lm_data_npy_many.extend(lm_data)
        annotations_many.extend(annotation)

        # ---- part 3 ----
        lm_data, annotation= greater_than_qf_p3(
            landmarks=landmarks[idx_init:],
            parent_folder=parent_folder,
        )
        assert np.array(lm_data).shape[1:]==(QUANTITY_FRAME, *LM_SHAPE_NORMALIZED)
        lm_data_npy_many.extend(lm_data)
        annotations_many.extend(annotation)
    return (lm_data_npy_many, annotations_many)
def landmarks_of_gloss(a_gloss: dict) -> tuple:
    gloss_lm_presented: list= []
    gloss_annotations: list= []
    for a_video in a_gloss[KEY_VIDS]:
        for a_start_end in a_video[KEY_LANDMARK]:
            if len(a_start_end[KEY_LANDMARK])<=QUANTITY_FRAME:
                tmp_lm, tmp_notation= q_imgs_less_than_or_equal(
                    landmarks=a_start_end[KEY_LANDMARK],
                    parent_folder=a_start_end[KEY_PFOLDER],
                )
                gloss_lm_presented.append(tmp_lm)
                gloss_annotations.append({
                    KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                    KEY_LANDMARK: tmp_notation,
                })
            else:
                tmp_lm, tmp_notation= q_imgs_greater_than(
                    landmarks=a_start_end[KEY_LANDMARK],
                    parent_folder=a_start_end[KEY_PFOLDER],
                )
                gloss_lm_presented.extend(tmp_lm)
                gloss_annotations.extend([{
                    KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                    KEY_LANDMARK: el,
                } for el in tmp_notation])
    '''
    gloss_lm_presented of shape (INT, QUANTITY_FRAME, 86, 2)
    gloss_annotations is a list(
        each element is dict_keys(
            KEY_PFOLDER: str,
            KEY_LANDMARK: list_of_len --> QUANTITY_FRAME --> each element is dict_keys(
                KEY_FACE: bool,
                KEY_POSE: bool,
                KEY_LHAND: bool,
                KEY_RHAND: bool,
            )
        )
    )
    '''
    return (gloss_lm_presented, gloss_annotations)
def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: np.ndarray, \
                    lmark_coordinates: list, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> np.ndarray:
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
                    cv2.circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    cv2.circle(
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
                    cv2.line(
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
def get_lmark_face(landmark):
    return landmark[:len(WORTHY_FACE_IDX)]
def get_lmark_pose(landmark):
    return landmark[
        len(WORTHY_FACE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
    ]
def get_lmark_lhand(landmark):
    return landmark[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ]
def get_lmark_rhand(landmark):
    return landmark[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ]
def drawFacePoseHand(img_write_to: np.ndarray, landmarks: np.ndarray, hasLandmarks: dict) -> np.ndarray:
    if hasLandmarks[KEY_FACE] \
        or hasLandmarks[KEY_POSE] \
        or hasLandmarks[KEY_LHAND] \
        or hasLandmarks[KEY_RHAND]:

        # ---- face landmarks ----
        if hasLandmarks[KEY_FACE]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_face(landmarks).tolist(),
                connections_idxs=FACE_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(0, 153, 0), # 153/255= 0.6
            )

        # ---- pose landmarks ----
        if hasLandmarks[KEY_POSE]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_pose(landmarks).tolist(),
                connections_idxs=POSE_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(0, 0, 153), # 153/255= 0.6
            )

        # ---- left hand landmarks ----
        if hasLandmarks[KEY_LHAND]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_lhand(landmarks).tolist(),
                connections_idxs=HAND_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(255, 255, 255)
            )

        # ---- right hand landmarks ----
        if hasLandmarks[KEY_RHAND]:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=get_lmark_rhand(landmarks).tolist(),
                connections_idxs=HAND_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(153, 204, 204), # 204/255= 0.8
            )

        # HERE ORDER OF LANDMARKS
        # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to
def save_skeleton_landmark(
    landmarks: list,
    annotations: list,
) -> tuple:
    '''
    landmarks of shape (INT, QUANTITY_FRAME, 86, 2)
    annotations is a list(
        each element is dict_keys(
            KEY_PFOLDER: str,
            KEY_LANDMARK: list_of_len --> QUANTITY_FRAME --> each element is dict_keys(
                KEY_FACE: bool,
                KEY_POSE: bool,
                KEY_LHAND: bool,
                KEY_RHAND: bool,
            )
        )
    )
    '''
    out_landmarks: list= []
    out_skeletons: list= []
    for idx_video_qf in range(len(landmarks)):
        new_pfolder: str= f"{annotations[idx_video_qf][KEY_PFOLDER][:-8]}_{str(idx_video_qf+1).zfill(5)}"
        new_pfolder= f"{new_pfolder}_{annotations[idx_video_qf][KEY_PFOLDER][-7:]}"
        abs_pfolder_landmark: Path= MODEL_LANDMARK_DIR /new_pfolder
        abs_pfolder_landmark.mkdir()
        abs_pfolder_skeleton: Path= MODEL_SKELETON_DIR /new_pfolder
        abs_pfolder_skeleton.mkdir()
        out_landmarks.append({
            KEY_PFOLDER: new_pfolder,
            KEY_LANDMARK: []
        })
        out_skeletons.append({
            KEY_PFOLDER: new_pfolder,
            KEY_SKELETON: []
        })
        assert len(annotations[idx_video_qf][KEY_LANDMARK])==QUANTITY_FRAME
        for idx_img, an_img_detail in enumerate(annotations[idx_video_qf][KEY_LANDMARK]):
            a_lm_skeleton_filename: str= str(idx_img+1).zfill(8)
            with open(f"{abs_pfolder_landmark /a_lm_skeleton_filename}.npy", 'wb') as f:
                np.save(f, landmarks[idx_video_qf][idx_img])
            cv2.imwrite(
                f"{abs_pfolder_skeleton /a_lm_skeleton_filename}.jpg",
                drawFacePoseHand(
                    np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                    landmarks=landmarks[idx_video_qf][idx_img],
                    hasLandmarks=an_img_detail,
                )
            )
            out_landmarks[-1][KEY_LANDMARK].append({
                KEY_FILE: f'{a_lm_skeleton_filename}.npy',
                KEY_FACE: an_img_detail[KEY_FACE],
                KEY_POSE: an_img_detail[KEY_POSE],
                KEY_LHAND: an_img_detail[KEY_LHAND],
                KEY_RHAND: an_img_detail[KEY_RHAND],
            })
            out_skeletons[-1][KEY_SKELETON].append({
                KEY_FILE: f'{a_lm_skeleton_filename}.jpg',
                KEY_FACE: an_img_detail[KEY_FACE],
                KEY_POSE: an_img_detail[KEY_POSE],
                KEY_LHAND: an_img_detail[KEY_LHAND],
                KEY_RHAND: an_img_detail[KEY_RHAND],
            })
    return (out_landmarks, out_skeletons)
def init_directories() -> None:
    err_msg: list= []
    if not LANDMARK_dir.exists():
        err_msg.append(f"Folder Not Found: {LANDMARK_dir}")
    if MODEL_LANDMARK_DIR.exists():
        err_msg.append(f"Please delete folder: {MODEL_LANDMARK_DIR}")
    if MODEL_SKELETON_DIR.exists():
        err_msg.append(f"Please delete folder: {MODEL_SKELETON_DIR}")
    if 0<len(err_msg):
        print("---------------------- error message ----------------------")
        for msg in err_msg:
            print(msg)
        raise FileNotFoundError("Please see message above.")
    else:
        MODEL_LANDMARK_DIR.mkdir()
        MODEL_SKELETON_DIR.mkdir()
def process_dataset(ds_landmark: list) -> tuple:
    landmarks_data: list= []
    skeletons_data: list= []
    for a_gloss in ds_landmark:
        tmp_gloss_lm_presented, tmp_gloss_annotations= landmarks_of_gloss(a_gloss=a_gloss)
        notation_landmarks, notation_skeletons= save_skeleton_landmark(
            landmarks=tmp_gloss_lm_presented,
            annotations=tmp_gloss_annotations,
        )
        landmarks_data.append({
            KEY_G: a_gloss[KEY_G],
            KEY_LANDMARK: notation_landmarks,
        })
        skeletons_data.append({
            KEY_G: a_gloss[KEY_G],
            KEY_SKELETON: notation_skeletons,
        })
    return landmarks_data, skeletons_data
def main() -> None:
    init_directories()
    ds_landmark: list
    with open(DS_DIR /"ds_landmark.json", 'r') as f:
        ds_landmark= loadJson(f)
    landmarks_data, skeletons_data= process_dataset(ds_landmark)
    with open(f"{DS_DIR /'model_landmark'}.json", 'w') as f:
        writeJson(landmarks_data, f, indent=4)
    with open(f"{DS_DIR /'model_skeleton'}.json", 'w') as f:
        writeJson(skeletons_data, f, indent=4)
if __name__=="__main__":
    main()
