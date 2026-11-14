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
    LEFT_HAND_DIR_FILE,
    LEN_W_FACE,
    LEN_W_POSE,
    LM_IDX_LEFT_ARM_TIP,
    LM_IDX_LEFT_ELBOW,
    LM_IDX_LEFT_HAND_WRIST,
    LM_IDX_LEFT_HIP,
    LM_IDX_LEFT_SHOULDER,
    LM_IDX_RIGHT_ARM_TIP,
    LM_IDX_RIGHT_ELBOW,
    LM_IDX_RIGHT_HAND_WRIST,
    LM_IDX_RIGHT_HIP,
    LM_IDX_RIGHT_SHOULDER,
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


def flip_vertical_y_axis(landmarks) -> np.ndarray:
    '''
    landmarks be of shape (INT, ..., 86, 2)
    '''
    new_landmark= np.array(landmarks).copy()
    new_landmark[..., 0]= 1 -new_landmark[..., 0]
    lhand2rhand: np.ndarray= new_landmark[...,
        Q_W_FACE +Q_W_POSE:
        Q_W_FACE +Q_W_POSE +Q_W_HAND
    ,:].copy()
    rhand2lhand: np.ndarray= new_landmark[...,
        Q_W_FACE +Q_W_POSE +Q_W_HAND:
    ,:].copy()
    new_landmark[...,
        Q_W_FACE +Q_W_POSE:
        Q_W_FACE +Q_W_POSE +Q_W_HAND
    ,:]= rhand2lhand # ie. past was right hand, now be left hand
    new_landmark[...,
        Q_W_FACE +Q_W_POSE +Q_W_HAND:
    ,:]= lhand2rhand # ie. past was left hand, now be right hand

    indices2flip: tuple[tuple, ...]= (
        # -- face --
        ( 12, 28 ),
        ( 19, 34 ),
        ( 11, 27 ),
        ( 21, 35 ),
        ( 14, 30 ),
        ( 13, 29 ),
        ( 10, 26 ),
        ( 15, 31 ),
        ( 18, 33 ),
        (  7, 23 ),
        ( 16, 32 ),
        (  9, 25 ),
        (  8, 24 ),

        # -- pose --
        ( 37, 36 ),
        ( 43, 42 ),
        ( 39, 38 ),
        ( 41, 40 ),
    )
    # -- flip index pair face
    # -- ( 12, 28 )
    for p_a, p_b in indices2flip:
        tmp= new_landmark[..., p_a:p_a+1, :].copy()
        new_landmark[..., p_a:p_a+1, :]= new_landmark[..., p_b:p_b+1, :].copy()
        new_landmark[..., p_b:p_b+1, :]= tmp

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
                KEY_PFOLDER_FROM: LANDMARK_DIR_GLASL,
                KEY_PFOLDER: el[KEY_PFOLDER],
                KEY_LANDMARK: el[KEY_LANDMARK],
            } for el in approved_by_split])
    elif wlasl_or_glasl == 'wlasl':
        for idx_gloss, a_gloss in enumerate(wlasl_ds):
            approved_by_split: list= list(filter(
                lambda el: el[KEY_SPLIT]==split,
                a_gloss[KEY_VIDS],
            )) # -- element is dict 3keys( video_file, split, landmark:list[dict..] )
            for a_video in approved_by_split:
                out.extend([{
                    KEY_GID: idx_gloss,
                    KEY_PFOLDER_FROM: LANDMARK_DIR_WLASL,
                    KEY_PFOLDER: el[KEY_PFOLDER],
                    KEY_LANDMARK: el[KEY_LANDMARK],
                } for el in a_video[KEY_LANDMARK]])
    else:
        raise ValueError(
            f'wlasl_or_glasl can only be "wlasl" or' +
            f'"glasl", given value: {wlasl_or_glasl}'
        )
    return out


def calculate_steps_needed(train_test: str=VAL_TRAIN, batch_size: int=BATCH) -> int:
    total_DS: int= 0
    dataset: list= extract_ds_annotation_split(
        wlasl_or_glasl='glasl',
        split=train_test,
    )
    for a_video in dataset:
        idx_init: int= idx_init_has_hand(a_video[KEY_LANDMARK])
        if idx_init!=-1:
            if len(a_video[KEY_LANDMARK])<=QUANTITY_FRAME:
                total_DS+= 1
            else:
                len_available_images: int= len(a_video[KEY_LANDMARK])-idx_init
                o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
                total_DS+= 1 # for part 1
                if QUANTITY_FRAME<len_available_images:
                    total_DS+= int(
                        o2t_mod *(len(a_video[KEY_LANDMARK]) -(idx_init+ QUANTITY_FRAME*o2t_mod))
                    ) # for part 2
                    for work4mod_what in MOD_PART_3: # for part 3
                        if 0<int(len_available_images//work4mod_what):
                            total_DS+= (len_available_images//work4mod_what) -QUANTITY_FRAME +1
                        else:
                            total_DS+= 1
                else:
                    total_DS+= 1
    dataset= extract_ds_annotation_split(
        wlasl_or_glasl='wlasl',
        split=train_test,
    )
    for a_video in dataset:
        idx_init: int= idx_init_has_hand(a_video[KEY_LANDMARK])
        if idx_init!=-1:
            if len(a_video[KEY_LANDMARK])<=QUANTITY_FRAME:
                total_DS+= 1
            else:
                len_available_images: int= len(a_video[KEY_LANDMARK])-idx_init
                o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
                total_DS+= 1 # for part 1
                if QUANTITY_FRAME<len_available_images:
                    total_DS+= int(
                        o2t_mod *(len(a_video[KEY_LANDMARK]) -(idx_init+ QUANTITY_FRAME*o2t_mod))
                    ) # for part 2
                    for work4mod_what in MOD_PART_3: # for part 3
                        if 0<int(len_available_images//work4mod_what):
                            total_DS+= (len_available_images//work4mod_what) -QUANTITY_FRAME +1
                        else:
                            total_DS+= 1
                else:
                    total_DS+= 1
    return int(ceil(total_DS/float(batch_size)))


def correct_hand_position(landmarks: np.ndarray, left_or_right: str) -> np.ndarray:
    move_howmuch: float= 0.02
    assert landmarks.shape == (86, 2)
    if left_or_right == KEY_LHAND:
        target= landmarks[LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] + move_howmuch
        move= target - landmarks[LM_IDX_LEFT_HAND_WRIST:LM_IDX_LEFT_HAND_WRIST+1, :]
        landmarks[
            LEN_W_FACE+LEN_W_POSE :LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK,
            :
        ]= landmarks[
            LEN_W_FACE+LEN_W_POSE :LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK,
            :
        ] + move
        target[:, 1]= target[:, 1] - move_howmuch
        landmarks[LM_IDX_LEFT_ARM_TIP]= target.copy()
        landmarks[LM_IDX_LEFT_ELBOW, :]= (
            landmarks[LM_IDX_LEFT_SHOULDER, :] +
            target
        )/2.0
    elif left_or_right == KEY_RHAND:
        target= landmarks[LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :] - move_howmuch
        move= target - landmarks[LM_IDX_RIGHT_HAND_WRIST:LM_IDX_RIGHT_HAND_WRIST+1, :]
        landmarks[
            LEN_W_FACE +LEN_W_POSE +QUANTITY_HAND_LMARK:,
            :
        ]= landmarks[
            LEN_W_FACE +LEN_W_POSE +QUANTITY_HAND_LMARK:,
            :
        ] + move
        target[:, 1]= target[:, 1] - move_howmuch
        landmarks[LM_IDX_RIGHT_ARM_TIP]= target.copy()
        landmarks[LM_IDX_RIGHT_ELBOW, :]= (
            landmarks[LM_IDX_RIGHT_SHOULDER, :] +
            target
        )/2.0
    else:
        raise ValueError(f'left_or_right can only be: {KEY_LHAND} or {KEY_RHAND}')
    return landmarks
def load_npfiles(landmark: dict) -> list[np.ndarray]:
    out: list= []
    for an_img_detail in landmark[KEY_LANDMARK]:
        assert an_img_detail[KEY_FACE] and an_img_detail[KEY_POSE] and \
            ( an_img_detail[KEY_LHAND] or an_img_detail[KEY_RHAND] )
        with open(f"{landmark[KEY_PFOLDER_FROM] /landmark[KEY_PFOLDER] /an_img_detail[KEY_FILE]}", 'rb') as f:
            if not an_img_detail[KEY_LHAND] or not an_img_detail[KEY_RHAND]:
                with open(LEFT_HAND_DIR_FILE, 'rb') as lhf:
                    left_hand_data= np.load(lhf)
                tmp: np.ndarray= np.load(f)
                if an_img_detail[KEY_LHAND]:
                    tmp[LEN_W_FACE+LEN_W_POSE:LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK, :]= left_hand_data
                else:
                    left_hand_data[:, 0]= 1- left_hand_data[:, 0]
                    tmp[LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK:, :]= left_hand_data
                out.append(correct_hand_position(
                    landmarks=tmp,
                    left_or_right=KEY_LHAND if an_img_detail[KEY_LHAND] else KEY_RHAND,
                ))
            else:
                out.append(np.load(f))
    return out
def load_elements_npfiles(elements: list) -> list:
    out: list= []
    for el in elements:
        out.append(load_npfiles(el))
    return out


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
    parent_folder_from: Path,
    parent_folder: str,
) -> dict:
    annotations: dict= {
        KEY_PFOLDER_FROM: parent_folder_from,
        KEY_PFOLDER: parent_folder,
        KEY_LANDMARK: [],
    }

    idx_init: int= idx_init_has_hand(landmarks)
    ratio: int= int(ceil(QUANTITY_FRAME / (len(landmarks)-idx_init)))
    for landmark in landmarks[idx_init:]:
        if has_atleast_1hand(landmark) and len(annotations[KEY_LANDMARK])<QUANTITY_FRAME:
            annotations[KEY_LANDMARK].extend([
                {
                    KEY_FILE:  landmark[KEY_FILE],
                    KEY_FACE:  landmark[KEY_FACE],
                    KEY_POSE:  landmark[KEY_POSE],
                    KEY_LHAND: landmark[KEY_LHAND],
                    KEY_RHAND: landmark[KEY_RHAND],
                }
            ] * min(ratio, QUANTITY_FRAME -len(annotations[KEY_LANDMARK])))
        elif len(annotations[KEY_LANDMARK])<QUANTITY_FRAME:
            annotations[KEY_LANDMARK].extend([
                {
                    KEY_FILE:  annotations[KEY_LANDMARK][-1][KEY_FILE],
                    KEY_FACE:  annotations[KEY_LANDMARK][-1][KEY_FACE],
                    KEY_POSE:  annotations[KEY_LANDMARK][-1][KEY_POSE],
                    KEY_LHAND: annotations[KEY_LANDMARK][-1][KEY_LHAND],
                    KEY_RHAND: annotations[KEY_LANDMARK][-1][KEY_RHAND],
                }
            ] * min(ratio, QUANTITY_FRAME -len(annotations[KEY_LANDMARK])))

    return annotations
def greater_than_qf_p1(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> dict:
    assert has_atleast_1hand(landmarks[0])
    annotations: dict= {
        KEY_PFOLDER_FROM: parent_folder_from,
        KEY_PFOLDER: parent_folder,
        KEY_LANDMARK: [],
    }

    ratio: int= floor(len(landmarks) / QUANTITY_FRAME)
    for idx in range(QUANTITY_FRAME):
        a_landmark = landmarks[idx * ratio]
        if has_atleast_1hand(a_landmark):
            annotations[KEY_LANDMARK].append({
                KEY_FILE: a_landmark[KEY_FILE],
                KEY_FACE: a_landmark[KEY_FACE],
                KEY_POSE: a_landmark[KEY_POSE],
                KEY_LHAND: a_landmark[KEY_LHAND],
                KEY_RHAND: a_landmark[KEY_RHAND],
            })
        else:
            annotations[KEY_LANDMARK].append({
                KEY_FILE:  annotations[KEY_LANDMARK][-1][KEY_FILE],
                KEY_FACE:  annotations[KEY_LANDMARK][-1][KEY_FACE],
                KEY_POSE:  annotations[KEY_LANDMARK][-1][KEY_POSE],
                KEY_LHAND: annotations[KEY_LANDMARK][-1][KEY_LHAND],
                KEY_RHAND: annotations[KEY_LANDMARK][-1][KEY_RHAND],
            })

    return annotations
def greater_than_qf_p2(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> list:
    assert has_atleast_1hand(landmarks[0])
    annotations_many: list= []

    how_many_qfs: int= floor(len(landmarks)/QUANTITY_FRAME)
    remains: int=      len(landmarks) - how_many_qfs*QUANTITY_FRAME
    mod_list: tuple=   tuple(range(how_many_qfs))
    past_notation: dict= {
        KEY_FILE: landmarks[0][KEY_FILE],
        KEY_FACE: landmarks[0][KEY_FACE],
        KEY_POSE: landmarks[0][KEY_POSE],
        KEY_LHAND: landmarks[0][KEY_LHAND],
        KEY_RHAND: landmarks[0][KEY_RHAND],
    }
    for idx_init_only in range(remains+1):
        tmp_annotations: list= [{
            KEY_PFOLDER_FROM: parent_folder_from,
            KEY_PFOLDER: parent_folder,
            KEY_LANDMARK: [],
        } for _ in range(how_many_qfs)]
        for idx in range(
            idx_init_only,
            idx_init_only +(how_many_qfs*QUANTITY_FRAME)
        ):
            for mod_what in mod_list:
                if (idx%how_many_qfs)==mod_what:
                    if has_atleast_1hand(landmarks[idx]):
                        tmp_annotations[mod_what][KEY_LANDMARK].append({
                            KEY_FILE:  landmarks[idx][KEY_FILE],
                            KEY_FACE:  landmarks[idx][KEY_FACE],
                            KEY_POSE:  landmarks[idx][KEY_POSE],
                            KEY_LHAND: landmarks[idx][KEY_LHAND],
                            KEY_RHAND: landmarks[idx][KEY_RHAND],
                        })
                        past_notation= {
                            KEY_FILE:  landmarks[idx][KEY_FILE],
                            KEY_FACE:  landmarks[idx][KEY_FACE],
                            KEY_POSE:  landmarks[idx][KEY_POSE],
                            KEY_LHAND: landmarks[idx][KEY_LHAND],
                            KEY_RHAND: landmarks[idx][KEY_RHAND],
                        }
                    else:
                        tmp_annotations[mod_what][KEY_LANDMARK].append(past_notation.copy())
        annotations_many.extend(tmp_annotations)
    return annotations_many
def greater_than_qf_p3(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> list:
    assert has_atleast_1hand(landmarks[0])

    annotations_many: list= []
    groups_notation: list= [[] for _ in MOD_PART_3]
    past_annotation: dict= {
        KEY_FILE:  landmarks[0][KEY_FILE],
        KEY_FACE:  landmarks[0][KEY_FACE],
        KEY_POSE:  landmarks[0][KEY_POSE],
        KEY_LHAND: landmarks[0][KEY_LHAND],
        KEY_RHAND: landmarks[0][KEY_RHAND],
    }

    for idx, a_landmark in enumerate(landmarks):
        for idx_mod, what_mod in enumerate(MOD_PART_3):
            if ((idx + 1) % what_mod) == 0:
                if has_atleast_1hand(a_landmark):
                    past_annotation= {
                        KEY_FILE: a_landmark[KEY_FILE],
                        KEY_FACE: a_landmark[KEY_FACE],
                        KEY_POSE: a_landmark[KEY_POSE],
                        KEY_LHAND: a_landmark[KEY_LHAND],
                        KEY_RHAND: a_landmark[KEY_RHAND],
                    }
                groups_notation[idx_mod].append(past_annotation.copy())

    for group in groups_notation:
        if len(group) <= QUANTITY_FRAME:
            annotations: list= []
            ratio: int= ceil(QUANTITY_FRAME / len(group))

            for annotation in group:
                if len(annotations)<QUANTITY_FRAME:
                    annotations.extend(
                        [annotation.copy()] * min(
                            ratio, QUANTITY_FRAME - len(annotations)
                        )
                    )

            annotations_many.append({
                KEY_PFOLDER_FROM: parent_folder_from,
                KEY_PFOLDER: parent_folder,
                KEY_LANDMARK: annotations,
            })
        else:
            for idx_init in range(len(group) - QUANTITY_FRAME + 1):
                annotations_many.append({
                    KEY_PFOLDER_FROM: parent_folder_from,
                    KEY_PFOLDER: parent_folder,
                    KEY_LANDMARK: group[
                        idx_init:idx_init + QUANTITY_FRAME
                    ],
                })

    return annotations_many
def q_imgs_greater_than(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> list:
    annotations_many: list= []

    idx_init: int= idx_init_has_hand(landmarks)
    q_imgs_available: int= len(landmarks)-idx_init
    if q_imgs_available<=QUANTITY_FRAME:
        annotation: dict= q_imgs_less_than_or_equal(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.append(annotation)
    else:
        # ---- part 1 ----
        annotation: dict= greater_than_qf_p1(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.append(annotation)

        # ---- part 2 ----
        annotations: list= greater_than_qf_p2(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.extend(annotations)

        # ---- part 3 ----
        annotations= greater_than_qf_p3(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.extend(annotations)
    return annotations_many
def get_data_landmark(
    train_test: str=VAL_TRAIN,
    batch_size: int=BATCH
) -> Generator:
    dataset: list= extract_ds_annotation_split(
        wlasl_or_glasl='wlasl',
        split=train_test,
    )
    # dataset.extend(extract_ds_annotation_split(
    #     wlasl_or_glasl='glasl',
    #     split=train_test,
    # ))
    dataset_idxs: tuple= tuple(sample(
        range(len(dataset)),
        len(dataset)
    ))

    batch_ds_elements: list= []
    batch_ds_class: list= []
    while True:
        for idx_ds in dataset_idxs:
            an_element_ds: dict= dataset[idx_ds]
            if len(an_element_ds[KEY_LANDMARK])<=QUANTITY_FRAME:
                batch_ds_class.append(an_element_ds[KEY_GID])
                batch_ds_elements.append(q_imgs_less_than_or_equal(
                    landmarks=an_element_ds[KEY_LANDMARK],
                    parent_folder_from=an_element_ds[KEY_PFOLDER_FROM],
                    parent_folder=an_element_ds[KEY_PFOLDER],
                ))
            else:
                result_from_many: list= q_imgs_greater_than(
                    landmarks=an_element_ds[KEY_LANDMARK],
                    parent_folder_from=an_element_ds[KEY_PFOLDER_FROM],
                    parent_folder=an_element_ds[KEY_PFOLDER],
                )
                batch_ds_class.extend( [an_element_ds[KEY_GID]] * len(result_from_many) )
                batch_ds_elements.extend(result_from_many)
            # ---- now ready ----
            while batch_size<=len(batch_ds_class):
                out_inputs: np.ndarray= np.array(
                    load_elements_npfiles(batch_ds_elements[:batch_size]),
                )
                out_inputs= np.concatenate([
                    out_inputs,
                    flip_vertical_x_axis(out_inputs)
                ], axis=0)
                out_expected_outputs: np.ndarray= np.array(batch_ds_class[:batch_size])
                out_expected_outputs= np.concatenate([
                    out_expected_outputs,
                    out_expected_outputs,
                ], axis=0)
                assert out_inputs.shape==(batch_size*2, QUANTITY_FRAME, 86, 2)
                assert out_expected_outputs.shape==(batch_size*2,)
                assert out_inputs.shape[0]==out_expected_outputs.shape[0]

                batch_ds_elements= batch_ds_elements[batch_size:]
                batch_ds_class= batch_ds_class[batch_size:]

                yield (out_inputs, out_expected_outputs)
