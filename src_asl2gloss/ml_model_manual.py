from pathlib import Path
from json import load as loadJson
import math
import numpy as np
# import jax.numpy as jnp


PROJ_ROOT: Path= Path(__file__).parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
LANDMARK_dir: Path= DS_DIR /"ds_landmark"
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
QUANTITY_FRAME: int= 8
MOD_PART_3: tuple= (3, 4, 5, 6, 7)
LM_SHAPE_NORMALIZED: tuple= (86, 2)
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
                lm_data_npy_many.append(groups_lm[
                    idx_init:
                    idx_init +QUANTITY_FRAME
                ])
                annotations_many.append(groups_notation[
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
        lm_data_npy_many.extend(lm_data)
        annotations_many.extend(annotation)
    return (lm_data_npy_many, annotations_many)
def main() -> None:
    ds_landmark: list
    with open(DS_DIR /"ds_landmark.json", 'r') as f:
        ds_landmark= loadJson(f)
    # can RAM handle if we load all numpy data
    lm_data_npy: list= []
    for a_gloss in ds_landmark:
        # a_gloss[KEY_G]
        for a_video in a_gloss[KEY_VIDS]:
            gloss_lm_presented: list= []
            gloss_annotations: list= []
            for a_start_end in a_video[KEY_LANDMARK]:
                if len(a_start_end[KEY_LANDMARK])<=QUANTITY_FRAME:
                    tmp_lm, tmp_notation= q_imgs_less_than_or_equal(
                        landmarks=a_start_end[KEY_LANDMARK],
                        parent_folder=a_start_end[KEY_PFOLDER],
                    )
                    gloss_lm_presented.append(tmp_lm)
                    gloss_annotations.append(tmp_notation)
                else:
                    tmp_lm, tmp_notation= q_imgs_greater_than(
                        landmarks=a_start_end[KEY_LANDMARK],
                        parent_folder=a_start_end[KEY_PFOLDER],
                    )
                    gloss_lm_presented.extend(tmp_lm)
                    gloss_annotations.extend(tmp_notation)
    print(len(ds_landmark))
if __name__=="__main__":
    main()
