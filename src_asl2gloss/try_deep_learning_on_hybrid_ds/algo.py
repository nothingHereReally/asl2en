from math import ceil
from numpy import array, load as loadnp, ndarray, uint16
from random import sample
from typing import Generator


from .constant import (
    DS_LANDMARK_DIR,
    ON_TRAINING_BATCH,
    QUANTITY_FRAME,
    ds_landmark as DATASET,

    KEY_GID,
    KEY_LANDMARK,

    KEY_PFOLDER,
    KEY_SPLIT,
    VAL_TRAIN,

    KEY_FILE,
)


def calculate_steps_needed(train_val: str=VAL_TRAIN, batch_size: int=ON_TRAINING_BATCH) -> int:
    total_DS: int= 0
    for a_gloss in DATASET:
        total_DS+= list(filter(
            lambda el: el[KEY_SPLIT]==train_val,
            a_gloss[KEY_LANDMARK],
        )).__len__()
    return int(ceil(total_DS/float(batch_size)))


def extract_ds_annotation_split(split: str) -> list:
    out: list= []
    for idx_gloss, a_gloss in enumerate(DATASET):
        approved_by_split: list= list(filter(
            lambda el: el[KEY_SPLIT]==split,
            a_gloss[KEY_LANDMARK]
        ))
        for a_start_end in approved_by_split:
            out.append({
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
