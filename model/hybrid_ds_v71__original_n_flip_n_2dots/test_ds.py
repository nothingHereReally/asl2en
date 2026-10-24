from json import load as loadJson
from pathlib import Path
from keras.src.layers import Layer
from keras.src.saving import register_keras_serializable, load_model
from keras.src import ops
from typing import Any
import numpy as np


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
LANDMARK_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset" /"ds_landmark"
KEY_G: str= 'gloss'
KEY_G_IDX: str= 'gloss_idx'
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
# ---------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
# ---------------------
LM_IDX_TOP_LIP: int        =  0
LM_IDX_TIP_OF_FOREHEAD: int=  3
LM_IDX_RIGHT_MOST_EYE: int =  7
LM_IDX_RIGHT_MOST_LIP: int =  8
LM_IDX_RIGHT_EAR: int      = 19
LM_IDX_UP_TIP_NOSE: int    = 20
LM_IDX_LEFT_MOST_EYE: int  = 23
LM_IDX_LEFT_MOST_LIP: int  = 24
LM_IDX_LEFT_EAR: int       = 34
LM_IDX_LEFT_ELBOW: int     = 38
LM_IDX_RIGHT_ELBOW: int    = 39
LM_IDX_LEFT_ARM_TIP: int   = 40
LM_IDX_RIGHT_ARM_TIP: int  = 41
# -----------------------------
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
LM_IDX_LEFT_HAND_F_THUMB: int= 48
LM_IDX_LEFT_HAND_F_INDEX: int= 52
LM_IDX_LEFT_HAND_F_MIDLE: int= 56
LM_IDX_LEFT_HAND_F_RING: int = 60
LM_IDX_LEFT_HAND_F_PINKY: int= 64
# -----------------------------
LM_IDX_RIGHT_HAND_F_THUMB: int= 48
LM_IDX_RIGHT_HAND_F_INDEX: int= 52
LM_IDX_RIGHT_HAND_F_MIDLE: int= 56
LM_IDX_RIGHT_HAND_F_RING: int = 60
LM_IDX_RIGHT_HAND_F_PINKY: int= 64
# -----------------------------
LM_IMPORTANT_FACE_IDX: tuple= (
    LM_IDX_TIP_OF_FOREHEAD, # ok

    LM_IDX_LEFT_MOST_EYE,   # ok
    LM_IDX_RIGHT_MOST_EYE,  # ok

    LM_IDX_LEFT_EAR,        # ok
    LM_IDX_RIGHT_EAR,       # ok

    LM_IDX_UP_TIP_NOSE,     # ok

    LM_IDX_LEFT_MOST_LIP,   # ok
    LM_IDX_RIGHT_MOST_LIP,  # ok
    LM_IDX_TOP_LIP,         # ok
)
# -----------------------------
LM_LEFT_HAND_TIP: tuple= (
    LM_IDX_LEFT_HAND_F_THUMB,
    LM_IDX_LEFT_HAND_F_INDEX,
    LM_IDX_LEFT_HAND_F_MIDLE,
    LM_IDX_LEFT_HAND_F_RING,
    LM_IDX_LEFT_HAND_F_PINKY,
)
LM_LEFT_HAND_NOT_TIP: tuple= (
    44, 45, 46, 47,
    49, 50, 51,
    53, 54, 55,
    57, 58, 59,
    61, 62, 63,
)
LM_RIGHT_HAND_TIP: tuple= (
    LM_IDX_RIGHT_HAND_F_THUMB,
    LM_IDX_RIGHT_HAND_F_INDEX,
    LM_IDX_RIGHT_HAND_F_MIDLE,
    LM_IDX_RIGHT_HAND_F_RING,
    LM_IDX_RIGHT_HAND_F_PINKY,
)
LM_RIGHT_HAND_NOT_TIP: tuple= (
    65, 66, 67, 68,
    70, 71, 72,
    74, 75, 76,
    78, 79, 80,
    82, 83, 84,
)
# ---------------------------------------------------------------------------
@register_keras_serializable()
class SelectLM(Layer):
    def __init__(self, name: str='select49', **kwargs):
        super().__init__(**kwargs)
        self.name= name
    def build(self, input_shape):
        del input_shape
    def call(self, input):
        # left_hand_tip: Any= input[..., LM_LEFT_HAND_TIP, :]
        # right_hand_tip: Any= input[..., LM_RIGHT_HAND_TIP, :]
        left_hand_tip: Any= ops.take(
            input,
            indices=LM_LEFT_HAND_TIP,
            axis=-2,
        )
        right_hand_tip: Any= ops.take(
            input,
            indices=LM_RIGHT_HAND_TIP,
            axis=-2,
        )
        left_hand_tip= ops.expand_dims(left_hand_tip, axis=-3)
        right_hand_tip= ops.expand_dims(right_hand_tip, axis=-3)
        center_shoulder= (
            input[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
            input[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
        )/2.0
        center_hip= (
            input[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
            input[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
        )/2.0
        center_shoulder_hip= (center_shoulder + center_hip)/2.0
        the_rest= ops.concatenate([
            ops.take(
                input,
                indices=LM_IMPORTANT_FACE_IDX,
                axis=-2,
            ),
            input[..., 36:36+8, :], # -- pose landmarks
            (center_shoulder + center_shoulder_hip)/2.0, # -- upper dot on 2 dots
            (center_hip + center_shoulder_hip)/2.0,      # -- lower dot on 2 dots
            ops.take(
                input,
                indices=LM_LEFT_HAND_NOT_TIP,
                axis=-2,
            ),
            ops.take(
                input,
                indices=LM_RIGHT_HAND_NOT_TIP,
                axis=-2,
            ),
        ], axis=-2)
        the_rest= ops.expand_dims(the_rest, axis=-2)

        left_hand_tip= (left_hand_tip - the_rest)**2
        left_hand_tip= ops.sum(left_hand_tip, axis=-1, keepdims=False)
        left_hand_tip= left_hand_tip**0.5
        shape: Any= ops.shape(left_hand_tip)
        left_hand_tip= ops.reshape(left_hand_tip, (-1, shape[1], shape[2]*shape[3]))

        right_hand_tip= (right_hand_tip - the_rest)**2
        right_hand_tip= ops.sum(right_hand_tip, axis=-1, keepdims=False)
        right_hand_tip= right_hand_tip**0.5
        shape: Any= ops.shape(right_hand_tip)
        right_hand_tip= ops.reshape(right_hand_tip, (-1, shape[1], shape[2]*shape[3]))

        out= ops.concatenate([
            left_hand_tip,
            right_hand_tip,
        ], axis=-1)
        return out
        # out= ops.concatenate([
        #     input[..., 44:, :], # -- left and right hand
        #     input[..., 36:36+8, :], # -- pose landmarks
        #     ops.take(
        #         input,
        #         indices=LM_IMPORTANT_FACE_IDX,
        #         axis=-2,
        #     ),
        # ], axis=-2)
        # out= ops.concatenate([
        #     out,
        # ], axis=-2)
        # shape: Any= ops.shape(out)
        # out= ops.reshape(out, (-1, shape[1], shape[2]*shape[3]))
        # return out
        # out= input[..., 44:, :]
        # shape: Any= ops.shape(out)
        # out= ops.reshape(out, (-1, shape[1], shape[2]*shape[3]))
        # return out
    def get_config(self):
        return {
            **super().get_config(),
            "name":      self.name,
        }
# ---------------------------------------------------------------------------
MODEL_ASL: Any= load_model(PROJ_ROOT /"model" /"hybrid_ds_v71__original_n_flip_n_2dots" /"aslvid2gloss_v71.keras")




def loadnpfiles(parent_folder: str, landmarks: list) -> np.ndarray:
    out: list= []
    for a_lm in landmarks:
        with open(f"{LANDMARK_DIR /parent_folder /a_lm[KEY_FILE]}", 'rb') as f:
            out.append(np.load(f))
    return np.array(out)
def split_ds(annotations: list) -> tuple:
    ds_train: list= []
    ds_test: list= []
    labels: list= []
    for idx_gloss, a_gloss in enumerate(annotations):
        labels.append(a_gloss[KEY_G])
        train_count: int= 0
        test_count: int= 0
        for a_start_end in a_gloss[KEY_LANDMARK]:
            if a_start_end[KEY_SPLIT] == VAL_TRAIN and train_count<160:
                ds_train.append({
                    KEY_G_IDX: idx_gloss,
                    KEY_G: a_gloss[KEY_G],
                    KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                    KEY_LANDMARK: a_start_end[KEY_LANDMARK],
                })
                train_count+= 1
            elif a_start_end[KEY_SPLIT] == VAL_TEST and test_count<40:
                ds_test.append({
                    KEY_G_IDX: idx_gloss,
                    KEY_G: a_gloss[KEY_G],
                    KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                    KEY_LANDMARK: a_start_end[KEY_LANDMARK],
                })
                test_count+= 1
    return (ds_train, ds_test, labels)
def get_results(ds_train: list, ds_test: list, labels_str: list[str]) -> tuple:
    train_results: list= [{
        'correct': 0,
        'total': 0,
        KEY_PFOLDER: []
    } for _ in labels_str]
    test_results: list= [{
        'correct': 0,
        'total': 0,
        KEY_PFOLDER: []
    } for _ in labels_str]

    batch_input: list= []
    batch_cat: list= []
    batch_pfolder: list= []
    batch_size: int= 32
    for an_element in ds_train:
        batch_input.append(loadnpfiles(
            an_element[KEY_PFOLDER],
            an_element[KEY_LANDMARK]
        ))
        batch_cat.append(an_element[KEY_G_IDX])
        batch_pfolder.append(an_element[KEY_PFOLDER])
        while batch_size <= len(batch_cat):
            pred_result= MODEL_ASL.predict(np.array(batch_input[:batch_size]), batch_size)
            for a_pred, a_cat, a_pfolder in zip(pred_result, batch_cat[:batch_size], batch_pfolder[:batch_size]):
                if np.argmax(a_pred, axis=-1) == a_cat:
                    train_results[a_cat]['correct']+= 1
                else:
                    train_results[a_cat][KEY_PFOLDER].append(a_pfolder)
                train_results[a_cat]['total']+= 1
            batch_input= batch_input[batch_size:]
            batch_cat= batch_cat[batch_size:]
            batch_pfolder= batch_pfolder[batch_size:]
    if len(batch_cat)!=0:
        pred_result= MODEL_ASL.predict(np.array(batch_input[:batch_size]), batch_size)
        for a_pred, a_cat, a_pfolder in zip(pred_result, batch_cat[:batch_size], batch_pfolder[:batch_size]):
            if np.argmax(a_pred, axis=-1) == a_cat:
                train_results[a_cat]['correct']+= 1
            else:
                train_results[a_cat][KEY_PFOLDER].append(a_pfolder)
            train_results[a_cat]['total']+= 1
        batch_input= batch_input[batch_size:]
        batch_cat= batch_cat[batch_size:]
        batch_pfolder= batch_pfolder[batch_size:]

    for an_element in ds_test:
        batch_input.append(loadnpfiles(
            an_element[KEY_PFOLDER],
            an_element[KEY_LANDMARK]
        ))
        batch_cat.append(an_element[KEY_G_IDX])
        batch_pfolder.append(an_element[KEY_PFOLDER])
        while batch_size <= len(batch_cat):
            pred_result= MODEL_ASL.predict(np.array(batch_input[:batch_size]), batch_size)
            for a_pred, a_cat, a_pfolder in zip(pred_result, batch_cat[:batch_size], batch_pfolder[:batch_size]):
                if np.argmax(a_pred, axis=-1) == a_cat:
                    test_results[a_cat]['correct']+= 1
                else:
                    test_results[a_cat][KEY_PFOLDER].append(a_pfolder)
                test_results[a_cat]['total']+= 1
            batch_input= batch_input[batch_size:]
            batch_cat= batch_cat[batch_size:]
            batch_pfolder= batch_pfolder[batch_size:]
    if len(batch_cat)!=0:
        pred_result= MODEL_ASL.predict(np.array(batch_input[:batch_size]), batch_size)
        for a_pred, a_cat, a_pfolder in zip(pred_result, batch_cat[:batch_size], batch_pfolder[:batch_size]):
            if np.argmax(a_pred, axis=-1) == a_cat:
                test_results[a_cat]['correct']+= 1
            else:
                test_results[a_cat][KEY_PFOLDER].append(a_pfolder)
            test_results[a_cat]['total']+= 1
        batch_input= batch_input[batch_size:]
        batch_cat= batch_cat[batch_size:]
        batch_pfolder= batch_pfolder[batch_size:]
    return (train_results, test_results)
def show_results(train_results, labels_str) -> None:
    correct: int= 0
    total: int= 0
    for idx, element in enumerate(train_results):
        print(f"{labels_str[idx]}")
        print(f"    correct: {element['correct']} / {element['total']} -- {element['correct']/element['total']*100:.4f}%")
        print(f"    INcorrect: {element['total'] - element['correct']} / {element['total']} -- {(element['total'] - element['correct'])/element['total']*100:.4f}%")
        correct+= element['correct']
        total+= element['total']
        if len(element[KEY_PFOLDER])>0:
            print("    parent folder incorrect")
        for a_pfolder in element[KEY_PFOLDER]:
            print(f"        {a_pfolder}")
        print()
    print("----------------------------------------")
    print("----------------------------------------")
    for idx, element in enumerate(train_results):
        print(f"{labels_str[idx]}")
        print(f"    correct: {element['correct']} / {element['total']} -- {element['correct']/element['total']*100:.4f}%")
        print(f"    INcorrect: {element['total'] - element['correct']} / {element['total']} -- {(element['total'] - element['correct'])/element['total']*100:.4f}%")
        print()
    print("-------------------------------------")
    print(f"correct: {correct} / {total} -- {correct/total*100:.4f}%")
    print("-------------------------------------")
    print("-------------------------------------")
def main() -> None:
    with open(f"{PROJ_ROOT /"dataset" /"clean_dataset" /"ds_landmark.json"}", 'r') as f:
        annotations: list= loadJson(f)
    ds_train, ds_test, labels_str= split_ds(annotations)
    train_results, test_results= get_results(ds_train, ds_test, labels_str)
    show_results(train_results, labels_str)
    show_results(test_results, labels_str)
if __name__=="__main__":
    main()
