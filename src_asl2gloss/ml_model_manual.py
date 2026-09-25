from pathlib import Path
from json import load as loadJson
import numpy as np
from time import sleep


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
                for _ in range(min(ratio, QUANTITY_FRAME-len(lm_data_npy))):
                    lm_data_npy.append(np.load(f))
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
    assert np.array(lm_data_npy)==(QUANTITY_FRAME, LM_SHAPE_NORMALIZED[0], LM_SHAPE_NORMALIZED[1])
    return (lm_data_npy, annotations)
def main() -> None:
    ds_landmark: list
    with open(DS_DIR /"ds_landmark.json", 'r') as f:
        ds_landmark= loadJson(f)
    # can RAM handle if we load all numpy data
    lm_data_npy: list= []
    for a_gloss in ds_landmark:
        # a_gloss[KEY_G]
        for a_video in a_gloss[KEY_VIDS]:
            for a_start_end in a_video[KEY_LANDMARK]:
                for an_img_details in a_start_end[KEY_LANDMARK]:
                    with open(f"{LANDMARK_dir /a_start_end[KEY_PFOLDER] /an_img_details[KEY_FILE]}", 'rb') as f:
                        lm_data_npy.append(
                            np.load(f)
                        )
    for a_landmark in lm_data_npy:
        print(f"--> {a_landmark.shape} -- {a_landmark[-1]}")
        blah= a_landmark*2.1
        del blah
    print(f"loaded numpy files {len(lm_data_npy)} --> now be sleeping for 10.5 seconds")
    sleep(10.5)
    print(len(ds_landmark))
if __name__=="__main__":
    main()
