from json import load as loadJson
from pathlib import Path
from keras.src.activations.activations import ReLU
from keras.src.layers import Attention, Dense, Layer
from keras.src.saving import register_keras_serializable, load_model
from keras.src import ops
from typing import Any
import numpy as np
from numpy import float64


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
WORTHY_LM_IMPORTANT_IDX: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_UP_TIP_NOSE,
    # -- on later process here to insert up and left body dots
    LM_IDX_LEFT_ARM_TIP,
    LM_IDX_RIGHT_ARM_TIP,
)
# ---------------------------------------------------------------------------
@register_keras_serializable()
class SelectLM(Layer):
    def __init__(self, name: str='select49', **kwargs):
        super().__init__(**kwargs)
        self.name= name
    def build(self, input_shape):
        pass
    def call(self, input):
        out= ops.concatenate([
            input[..., 44:, :],
            ops.take(
                input,
                WORTHY_LM_IMPORTANT_IDX,
                axis=-2
            ),
        ], axis=-2)
        center_shoulder= (
            input[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
            input[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
        )/2.0
        center_hip= (
            input[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
            input[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
        )/2.0
        center_shoulder_hip= (center_shoulder + center_hip)/2.0
        out= ops.concatenate([
            out,
            (center_shoulder + center_shoulder_hip)/2.0,
            (center_hip + center_shoulder_hip)/2.0,
        ], axis=-2)
        shape: Any= ops.shape(out)
        out= ops.reshape(out, (-1, shape[1], shape[-2]*shape[3]))
        return out
    def get_config(self):
        return {
            **super().get_config(),
            "name":      self.name,
        }
@register_keras_serializable()
class ShapeReduceViaAddAndAttention(Layer):
    def __init__(self, target: int, num_heads: int, dropout: float, name: str='reduce_shape21', **kwargs):
        super().__init__(**kwargs)
        self.target=    target
        self.num_heads= num_heads
        self.dropout=   dropout
        self.name=      name
    def build(self, input_shape):
        assert input_shape[-2]%2==0
        self.target_query= [Dense(
                units=self.target,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.target_value= [Dense(
                units=self.target,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.target_key= [Dense(
                units=self.target,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.target_attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float64,
            ) for _ in range(self.num_heads)]
        # -----------------------------------
        self.query= [Dense(
                units=self.target*input_shape[-2]//2,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.value= [Dense(
                units=self.target*input_shape[-2]//2,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.key= [Dense(
                units=self.target*input_shape[-2]//2,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float64,
            ) for _ in range(self.num_heads)]
    def call(self, data_self_att):
        origin_shape: Any= ops.shape(data_self_att) # shape(BATCH_SIZE, QUANTITY_FRAME, 43)
        data= ops.reshape(data_self_att, (-1, origin_shape[-2]//2, 2, origin_shape[-1]))
        data= ops.sum(data, axis=-2) # shape(BATCH_SIZE, QUANTITY_FRAME/2, 43)
        shape: Any= ops.shape(data)
        data=  ops.reshape(data, (-1, shape[-1])) # shape(BATCH_SIZE*QUANTITY_FRAME/2, 43)
        # ----------------------------------------------------------------------
        query= [query_ann(data) for query_ann in self.target_query]
        value= [value_ann(data) for value_ann in self.target_value]
        key=   [key_ann(data)   for key_ann   in self.target_key]
        query= [ops.reshape(query_ann, (-1, shape[-2], self.target)) for query_ann in query]
        value= [ops.reshape(value_ann, (-1, shape[-2], self.target)) for value_ann in value]
        key=   [ops.reshape(key_ann,   (-1, shape[-2], self.target)) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.target_attention)]
        output= ops.sum(output, axis=0)
        shape= ops.shape(output) # (BATCH_SIZE, QUANTITY_FRAME/2, self.target)
        # ----------------------------------------------------------------------
        output= ops.reshape(output, (-1, shape[-2]*shape[-1])) # shape(BATCH_SIZE, QUANTITY_FRAME/2*self.target)
        query= [query_ann(output) for query_ann in self.query]
        value= [value_ann(output) for value_ann in self.value]
        key=   [key_ann(output)   for key_ann   in self.key]
        query= [ops.reshape(query_ann, (-1, shape[-2], self.target)) for query_ann in query]
        value= [ops.reshape(value_ann, (-1, shape[-2], self.target)) for value_ann in value]
        key=   [ops.reshape(key_ann,   (-1, shape[-2], self.target)) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.attention)]
        # input_shape (BATCH_SIZE, QUANTITY_FRAME, 43)
        # output_shape (BATCH_SIZE, QUANTITY_FRAME/2, self.target)
        return ops.sum(output, axis=0)
    def get_config(self):
        return {
            **super().get_config(),
            "target":    self.target,
            "num_heads": self.num_heads,
            "dropout":   self.dropout,
            "name":      self.name,
        }
# ---------------------------------------------------------------------------
MODEL_ASL: Any= load_model(PROJ_ROOT /"model" /"hybrid_ds_v69_select_lm" /"aslvid2gloss_v69.keras")




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
