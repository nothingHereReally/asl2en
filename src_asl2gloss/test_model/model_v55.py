from typing import Any
from keras.src.layers import Layer, Dense, Attention
from keras.src.activations.activations import ReLU
from keras.src.saving import load_model, register_keras_serializable
from numpy import argmax, array, float32, load as loadnp, ndarray
from json import load as loadjson
from math import ceil
import tensorflow as tf


from ..lmark_constant import KEY_FILE, KEY_GLOSS, KEY_ID2G, KEY_LHAND, KEY_LMARK, KEY_RHAND, KEY_TEST, KEY_TRAIN, KEY_VAL, KEY_VIDEO, PROJ_ROOT, GLASL_LANDMARK_DIR


LANDMARK_SHAPE: tuple= (36 +8 +21*2, 2)
@register_keras_serializable()
class ShapeReduce43ViaAttention(Layer):
    def __init__(self, num_heads: int, dropout: float, name: str='reduce_shape', **kwargs):
        super().__init__(**kwargs)
        self.num_heads=   num_heads
        self.dropout=     dropout
        self.name=        name
    def build(self, input_shape):
        self.in_query= [Dense(
                units=input_shape[-2],
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.in_value= [Dense(
                units=input_shape[-2],
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.in_key= [Dense(
                units=input_shape[-2],
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.in_attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float32,
            ) for _ in range(self.num_heads)]
        # -----------------------------------
        self.query= [Dense( # ---------------
                units=input_shape[-2]//2,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.value= [Dense(
                units=input_shape[-2]//2,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.key= [Dense(
                units=input_shape[-2]//2,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float32,
            ) for _ in range(self.num_heads)]
    def call(self, query, value, key):
        shape: Any= tf.shape(query) # shape(BATCH_SIZE, QUANTITY_FRAME, 86, 2)
        query= tf.reshape(query, (-1, shape[-2]*shape[-1]))
        value= tf.reshape(value, (-1, shape[-2]*shape[-1]))
        key=   tf.reshape(key,   (-1, shape[-2]*shape[-1]))
        # -------------------------------------------------
        query= [query_ann(query) for query_ann in self.in_query] # (-1, 172) -> shape(-1, 86)
        value= [value_ann(value) for value_ann in self.in_value] # (-1, 172) -> shape(-1, 86)
        key=   [key_ann(key)     for key_ann   in self.in_key]   # (-1, 172) -> shape(-1, 86)
        query= [tf.reshape(query_ann, (-1, shape[-3], shape[-2])) for query_ann in query]
        value= [tf.reshape(value_ann, (-1, shape[-3], shape[-2])) for value_ann in value]
        key=   [tf.reshape(key_ann,   (-1, shape[-3], shape[-2])) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.in_attention)]
        output= tf.reduce_sum(output, axis=0)
        # ----------------------------------------------------------------------
        query= [query_ann(output) for query_ann in self.query]
        value= [value_ann(output) for value_ann in self.value]
        key=   [key_ann(output)   for key_ann   in self.key]
        query= [tf.reshape(query_ann, (-1, shape[-3], shape[-2]//2)) for query_ann in query]
        value= [tf.reshape(value_ann, (-1, shape[-3], shape[-2]//2)) for value_ann in value]
        key=   [tf.reshape(key_ann,   (-1, shape[-3], shape[-2]//2)) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.attention)]
        # input_shape (BATCH_SIZE, QUANTITY_FRAME, 86, 2)
        # output_shape (BATCH_SIZE, QUANTITY_FRAME, 43=86//2)
        return tf.reduce_sum(output, axis=0)
    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "dropout":   self.dropout,
            "name":      self.name,
        }
@register_keras_serializable()
class ShapeReduce21ViaAttention(Layer):
    def __init__(self, target: int, num_heads: int, dropout: float, name: str='reduce_shape21', **kwargs):
        super().__init__(**kwargs)
        self.target=    target
        self.num_heads= num_heads
        self.dropout=   dropout
        self.name=      name
    def build(self, input_shape):
        quantity_frame_here= input_shape[-2]//2
        self.query= [Dense(
                units=self.target*quantity_frame_here,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.value= [Dense(
                units=self.target*quantity_frame_here,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.key= [Dense(
                units=self.target*quantity_frame_here,
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float32,
            ) for _ in range(self.num_heads)]
        self.attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float32,
            ) for _ in range(self.num_heads)]
    def call(self, date_self_att):
        origin_shape: Any= tf.shape(date_self_att) # shape(BATCH_SIZE, QUANTITY_FRAME, 43)
        data= tf.reshape(date_self_att, (-1, origin_shape[-2]//2, 2, origin_shape[-1]))
        data= tf.reduce_sum(data,   axis=-2) # shape(BATCH_SIZE, QUANTITY_FRAME/2, 43)
        data= tf.reshape(data, (-1, origin_shape[-2]//2, origin_shape[-1]))
        shape: Any= tf.shape(data)
        data=  tf.reshape(data, (-1, shape[-2]*shape[-1])) # shape(BATCH_SIZE, QUANTITY_FRAME/2*43)
        # ----------------------------------------------------------------------
        query= [query_ann(data) for query_ann in self.query]
        value= [value_ann(data) for value_ann in self.value]
        key=   [key_ann(data)   for key_ann   in self.key]
        query= [tf.reshape(query_ann, (-1, shape[-2], self.target)) for query_ann in query]
        value= [tf.reshape(value_ann, (-1, shape[-2], self.target)) for value_ann in value]
        key=   [tf.reshape(key_ann,   (-1, shape[-2], self.target)) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.attention)]
        # input_shape (BATCH_SIZE, QUANTITY_FRAME, 43)
        # output_shape (BATCH_SIZE, QUANTITY_FRAME/2, self.target)
        return tf.reduce_sum(output, axis=0)
    def get_config(self):
        return {
            **super().get_config(),
            "target": self.target,
            "num_heads": self.num_heads,
            "dropout":   self.dropout,
            "name":      self.name,
        }
model: Any= load_model(f"{PROJ_ROOT /"model" /"aslvid2gloss_v55.keras"}")
TRAIN_GLOSS: int= model.output_shape[-1]   # 22 categories
QUANTITY_FRAME: int= model.input_shape[-3] # 22 frames/images
TRAIN_BATCH: int= 2


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








if __name__=="__main__":
    glasl_landmark: dict= {}
    with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl.annotation.landmark.45videos.json"}", 'r') as f:
        glasl_landmark= loadjson(f)


    batch: int= 8
    details: dict= {
        KEY_TRAIN: {
            'correct':          [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
            'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
            'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_VAL: {
            'correct':          [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
            'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
            'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_TEST: {
            'correct':          [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
            'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
            'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
        },
        "summary": {
            KEY_TRAIN: {
                'correct':          [0 for _ in range(TRAIN_GLOSS)],
                'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
                'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
                'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
            },
            KEY_VAL: {
                'correct':          [0 for _ in range(TRAIN_GLOSS)],
                'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
                'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
                'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
            },
            KEY_TEST: {
                'correct':          [0 for _ in range(TRAIN_GLOSS)],
                'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
                'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
                'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
            },
        }
    }
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        batch_vid_lm: list= []
        batch_gloss: list= []
        i: int= 0
        while i<len(glasl_landmark[tvt_idv]) and glasl_landmark[tvt_idv][i]['gloss_id']<TRAIN_GLOSS:
            if len(batch_vid_lm)<batch:
                tmp= []
                if len(glasl_landmark[tvt_idv][i]['landmark'])<=QUANTITY_FRAME:
                    batch_vid_lm.append(array(get_landmark4less_or_equal(glasl_landmark[tvt_idv][i]), dtype=float32))
                else:
                    batch_vid_lm.append(array(get_landmark4greater(glasl_landmark[tvt_idv][i]), dtype=float32))
                batch_gloss.append(int(glasl_landmark[tvt_idv][i]['gloss_id']))
                i+= 1
            else:
                y_pred= model.predict(
                    x=array(batch_vid_lm, dtype=float32),
                    batch_size=len(batch_vid_lm)
                )
                for y_out, y_shouldbe in zip(y_pred, batch_gloss):
                    y_out_g_id= int(argmax(y_out, axis=-1))
                    if y_out_g_id==y_shouldbe:
                        details[tvt_idv]['correct'][y_shouldbe]+= 1
                        details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                    details[tvt_idv]['overall_accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                    details[tvt_idv]['total_vid'][y_shouldbe]+= 1
                batch_vid_lm= []
                batch_gloss= []
        if len(batch_vid_lm)>0:
            y_pred= model.predict(
                x=array(batch_vid_lm, dtype=float32),
                batch_size=len(batch_vid_lm)
            )
            for y_out, y_shouldbe in zip(y_pred, batch_gloss):
                y_out_g_id= int(argmax(y_out, axis=-1))
                if y_out_g_id==y_shouldbe:
                    details[tvt_idv]['correct'][y_shouldbe]+= 1
                    details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                details[tvt_idv]['overall_accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                details[tvt_idv]['total_vid'][y_shouldbe]+= 1
            batch_vid_lm= []
            batch_gloss= []
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"__________________________ {tvt_idv} ____")
        q_correct: int= 0
        q_vid: int= 0
        overall_accuracy: float= 0.0
        quantity_videos: int= 0
        for g_id in range(TRAIN_GLOSS):
            q_correct+= details[tvt_idv]['correct'][g_id]
            q_vid+= details[tvt_idv]['total_vid'][g_id]
            print(f"{g_id}: {glasl_landmark[KEY_ID2G][g_id]} --> {details[tvt_idv]['correct'][g_id]}/{details[tvt_idv]['total_vid'][g_id]}")
            print(f"precentage correct: {details[tvt_idv]['correct'][g_id] / details[tvt_idv]['total_vid'][g_id] *100}%")
            print(f"accuracy: {details[tvt_idv]['accuracy'][g_id] / (details[tvt_idv]['correct'][g_id] if details[tvt_idv]['correct'][g_id]>0 else 1) *100}%")
            print(f"overall accuracy: {details[tvt_idv]['overall_accuracy'][g_id] / details[tvt_idv]['total_vid'][g_id] *100}%")
            overall_accuracy+= details[tvt_idv]['overall_accuracy'][g_id]
            quantity_videos+= details[tvt_idv]['total_vid'][g_id]
            print()
        print("<< ----------------------------------------------------------------- >>")
        print(f"____ {tvt_idv} --> {q_correct}/{q_vid} --> {q_correct/q_vid*100}%")
        print(f"____ overall accuracy {tvt_idv} --> {overall_accuracy/quantity_videos*100}%")
        print("<< ----------------------------------------------------------------- >>")
    print("\n\nsummary")
    summary_details: dict= {
        'correct': 0,
        'total_vid': 0,
        'accuracy': 0,
        'overall_accuracy': 0
    }
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"-----------------------------------------------------------------")
        print(f"____ {tvt_idv} --> {  sum(details[tvt_idv]['correct'])  }/{  sum(details[tvt_idv]['total_vid'])  } --> {sum(details[tvt_idv]['correct'])  /  sum(details[tvt_idv]['total_vid'])*100}%")
        print(f"____ overall accuracy {tvt_idv} --> {sum(details[tvt_idv]['overall_accuracy'])/sum(details[tvt_idv]['total_vid'])*100}%")

        summary_details['correct']+= sum(details[tvt_idv]['correct'])
        summary_details['total_vid']+= sum(details[tvt_idv]['total_vid'])
        summary_details['accuracy']+= sum(details[tvt_idv]['correct'])  /  sum(details[tvt_idv]['total_vid'])
        summary_details['overall_accuracy']+= sum(details[tvt_idv]['overall_accuracy'])/sum(details[tvt_idv]['total_vid'])
    print(f"-----------------------------------------------------------------")
    print(f"____ percentage correct --> {summary_details['correct']} / {summary_details['total_vid']} --> {summary_details['correct'] /summary_details['total_vid']  *100}%")
    print(f"____ hardmax accuracy --> {summary_details['accuracy']  /3  *100}%")
    print(f"____ softmax accuracy --> {summary_details['overall_accuracy']  /3  *100}%")
