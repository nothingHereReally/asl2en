from math import ceil, floor
# from numpy import array, load as loadnp, ndarray, uint16
import numpy as np
from random import sample
from typing import Generator
from pathlib import Path


from .constant import (
    KEY_PFOLDER_FROM,
    KEY_VIDS,
    LANDMARK_DIR_WLASL,
    LANDMARK_DIR_GLASL,
    BATCH,
    MOD_PART_3,
    QUANTITY_FRAME,
    QUANTITY_HAND_LMARK,
    VAL_TEST,
    WORTHY_FACE_IDX,
    WORTHY_POSE_IDX,
    wlasl_ds,
    glasl_ds,

    KEY_GID,
    KEY_LANDMARK,

    KEY_PFOLDER,
    KEY_SPLIT,
    VAL_TRAIN,

    KEY_FILE,
    KEY_FACE,
    KEY_POSE,
    KEY_LHAND,
    KEY_RHAND,
)


def flip_vertical_x_axis(landmarks: np.ndarray) -> np.ndarray:
    '''
    landmarks be of shape (INT, ..., 86, 2)
    '''
    new_landmark= landmarks.copy()
    new_landmark[..., 0]= 1 -new_landmark[..., 0]
    lhand2rhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ,:].copy()
    rhand2lhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ,:].copy()
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ,:]= rhand2lhand # ie. past was right hand, now be left hand
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ,:]= lhand2rhand # ie. past was left hand, now be right hand

    # -- flip index pair face
    # -- ( 12, 28 )
    tmp= new_landmark[..., 12:13, :].copy()
    new_landmark[..., 12:13, :]= new_landmark[..., 28:29, :].copy()
    new_landmark[..., 28:29, :]= tmp

    # -- ( 19, 34 )
    tmp = new_landmark[..., 19:20, :].copy()
    new_landmark[..., 19:20, :] = new_landmark[..., 34:35, :].copy()
    new_landmark[..., 34:35, :] = tmp

    # -- ( 11, 27 )
    tmp = new_landmark[..., 11:12, :].copy()
    new_landmark[..., 11:12, :] = new_landmark[..., 27:28, :].copy()
    new_landmark[..., 27:28, :] = tmp

    # -- ( 21, 35 )
    tmp = new_landmark[..., 21:22, :].copy()
    new_landmark[..., 21:22, :] = new_landmark[..., 35:36, :].copy()
    new_landmark[..., 35:36, :] = tmp

    # -- ( 14, 30 )
    tmp = new_landmark[..., 14:15, :].copy()
    new_landmark[..., 14:15, :] = new_landmark[..., 30:31, :].copy()
    new_landmark[..., 30:31, :] = tmp

    # -- ( 13, 29 )
    tmp = new_landmark[..., 13:14, :].copy()
    new_landmark[..., 13:14, :] = new_landmark[..., 29:30, :].copy()
    new_landmark[..., 29:30, :] = tmp

    # -- ( 10, 26 )
    tmp = new_landmark[..., 10:11, :].copy()
    new_landmark[..., 10:11, :] = new_landmark[..., 26:27, :].copy()
    new_landmark[..., 26:27, :] = tmp

    # -- ( 15, 31 )
    tmp = new_landmark[..., 15:16, :].copy()
    new_landmark[..., 15:16, :] = new_landmark[..., 31:32, :].copy()
    new_landmark[..., 31:32, :] = tmp

    # -- ( 18, 33 )
    tmp = new_landmark[..., 18:19, :].copy()
    new_landmark[..., 18:19, :] = new_landmark[..., 33:34, :].copy()
    new_landmark[..., 33:34, :] = tmp

    # -- (  7, 23 )
    tmp = new_landmark[..., 7:8, :].copy()
    new_landmark[..., 7:8, :] = new_landmark[..., 23:24, :].copy()
    new_landmark[..., 23:24, :] = tmp

    # -- ( 16, 32 )
    tmp = new_landmark[..., 16:17, :].copy()
    new_landmark[..., 16:17, :] = new_landmark[..., 32:33, :].copy()
    new_landmark[..., 32:33, :] = tmp

    # -- (  9, 25 )
    tmp = new_landmark[..., 9:10, :].copy()
    new_landmark[..., 9:10, :] = new_landmark[..., 25:26, :].copy()
    new_landmark[..., 25:26, :] = tmp

    # -- (  8, 24 )
    tmp = new_landmark[..., 8:9, :].copy()
    new_landmark[..., 8:9, :] = new_landmark[..., 24:25, :].copy()
    new_landmark[..., 24:25, :] = tmp




    # -- flip index pair pose
    # -- ( 37, 36 )
    tmp = new_landmark[..., 37:38, :].copy()
    new_landmark[..., 37:38, :] = new_landmark[..., 36:37, :].copy()
    new_landmark[..., 36:37, :] = tmp

    # -- ( 43, 42 )
    tmp = new_landmark[..., 43:44, :].copy()
    new_landmark[..., 43:44, :] = new_landmark[..., 42:43, :].copy()
    new_landmark[..., 42:43, :] = tmp

    # -- ( 39, 38 )
    tmp = new_landmark[..., 39:40, :].copy()
    new_landmark[..., 39:40, :] = new_landmark[..., 38:39, :].copy()
    new_landmark[..., 38:39, :] = tmp

    # -- ( 41, 40 )
    tmp = new_landmark[..., 41:42, :].copy()
    new_landmark[..., 41:42, :] = new_landmark[..., 40:41, :].copy()
    new_landmark[..., 40:41, :] = tmp


    return new_landmark


def extract_ds_annotation_split(wlasl_or_glasl: str, split: str) -> list:
    out: list= []
    assert split==VAL_TRAIN or split==VAL_TEST
    if wlasl_or_glasl == 'glasl':
        for idx_gloss, a_gloss in enumerate(glasl_ds):
            approved_by_split: list= list(filter(
                lambda el: el[KEY_SPLIT]==split,
                a_gloss[KEY_VIDS],
            ))
            out.extend([{
                KEY_GID: idx_gloss,
                KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                KEY_LANDMARK: a_start_end[KEY_LANDMARK],
            })
    return out


def load_npfiles(landmark: dict) -> list:
    out: list= []
    for an_img_detail in landmark[KEY_LANDMARK]:
        with open(f"{DS_LANDMARK_DIR /landmark[KEY_PFOLDER] /an_img_detail[KEY_FILE]}", 'rb') as f:
            out.append(loadnp(f))
    return out
def load_elements_npfiles(elements: list) -> list:
    out: list= []
    for el in elements:
        out.append(load_npfiles(el))
    return out


def get_data_landmark(
    train_val: str=VAL_TRAIN,
    batch_size: int=ON_TRAINING_BATCH
) -> Generator:
    dataset: list= extract_ds_annotation_split(train_val)
    dataset_idxs: tuple= tuple(sample(
        range(len(dataset)),
        len(dataset)
    ))
    batch_videos: list= []
    batch_class: list= []
    while True:
        for idx_ds in dataset_idxs:
            batch_class.append(dataset[idx_ds][KEY_GID])
            batch_videos.append(dataset[idx_ds])
            # ---- now ready ----
            while batch_size<=len(batch_videos):
                out_inputs: ndarray= array(
                    load_elements_npfiles(batch_videos[:batch_size]),
                )
                assert out_inputs.shape==(batch_size, QUANTITY_FRAME, 86, 2)
                out_expected_outputs: ndarray= array(batch_class[:batch_size], dtype=uint16)
                batch_videos= batch_videos[batch_size:]
                batch_class= batch_class[batch_size:]
                yield (out_inputs, out_expected_outputs)
