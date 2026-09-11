from typing import Any
from keras.src.layers import Layer, Dense, Attention
from keras.src.activations.activations import ReLU
from keras.src.saving import load_model, register_keras_serializable
from numpy import argmax, array, float32, float64, load as loadnp, ndarray
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
                units=input_shape[-2], # 86
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.in_value= [Dense(
                units=input_shape[-2], # 86
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.in_key= [Dense(
                units=input_shape[-2], # 86
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.in_attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float64,
            ) for _ in range(self.num_heads)]
        # -----------------------------------
        self.query= [Dense( # ---------------
                units=input_shape[-2]//2, # 43
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.value= [Dense(
                units=input_shape[-2]//2, # 43
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.key= [Dense(
                units=input_shape[-2]//2, # 43
                activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
                dtype=float64,
            ) for _ in range(self.num_heads)]
        self.attention= [Attention(
                use_scale=True,
                dropout=self.dropout,
                dtype=float64,
            ) for _ in range(self.num_heads)]
    def call(self, data_self_att):
        shape: Any= tf.shape(data_self_att) # shape(BATCH_SIZE, QUANTITY_FRAME, 86, 2)
        data= tf.reshape(data_self_att, (-1, shape[-2]*shape[-1])) # shape(BATCH_SIZE*QUANTITY_FRAME, 86*2)
        query= [query_ann(data) for query_ann in self.in_query] # (-1, 172) -> shape(-1, 86)
        value= [value_ann(data) for value_ann in self.in_value] # (-1, 172) -> shape(-1, 86)
        key=   [key_ann(data)   for key_ann   in self.in_key]   # (-1, 172) -> shape(-1, 86)
        query= [tf.reshape(query_ann, (-1, shape[-3], shape[-2])) for query_ann in query]
        value= [tf.reshape(value_ann, (-1, shape[-3], shape[-2])) for value_ann in value]
        key=   [tf.reshape(key_ann,   (-1, shape[-3], shape[-2])) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.in_attention)]
        output= tf.reduce_sum(output, axis=0) # shape(BATCH_SIZE, QUANTITY_FRAME, 86)
        # ----------------------------------------------------------------------
        output= tf.reshape(output, (-1, shape[-2])) # shape(BATCH_SIZE*QUANTITY_FRAME, 86)
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
        origin_shape: Any= tf.shape(data_self_att) # shape(BATCH_SIZE, QUANTITY_FRAME, 43)
        data= tf.reshape(data_self_att, (-1, origin_shape[-2]//2, 2, origin_shape[-1]))
        data= tf.reduce_sum(data, axis=-2) # shape(BATCH_SIZE, QUANTITY_FRAME/2, 43)
        shape: Any= tf.shape(data)
        data=  tf.reshape(data, (-1, shape[-1])) # shape(BATCH_SIZE*QUANTITY_FRAME/2, 43)
        # ----------------------------------------------------------------------
        query= [query_ann(data) for query_ann in self.target_query]
        value= [value_ann(data) for value_ann in self.target_value]
        key=   [key_ann(data)   for key_ann   in self.target_key]
        query= [tf.reshape(query_ann, (-1, shape[-2], self.target)) for query_ann in query]
        value= [tf.reshape(value_ann, (-1, shape[-2], self.target)) for value_ann in value]
        key=   [tf.reshape(key_ann,   (-1, shape[-2], self.target)) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.target_attention)]
        output= tf.reduce_sum(output, axis=0)
        shape= tf.shape(output) # (BATCH_SIZE, QUANTITY_FRAME/2, self.target)
        # ----------------------------------------------------------------------
        output= tf.reshape(output, (-1, shape[-2]*shape[-1])) # shape(BATCH_SIZE, QUANTITY_FRAME/2*self.target)
        query= [query_ann(output) for query_ann in self.query]
        value= [value_ann(output) for value_ann in self.value]
        key=   [key_ann(output)   for key_ann   in self.key]
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
            "target":    self.target,
            "num_heads": self.num_heads,
            "dropout":   self.dropout,
            "name":      self.name,
        }
model: Any= load_model(f"{PROJ_ROOT /"model" /"aslvid2gloss_v60.keras"}")
TRAIN_GLOSS: int= model.output_shape[-1]   # 22 categories
QUANTITY_FRAME: int= model.input_shape[-3] # 4 frames/images
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
            'accuracy_correct':    [0 for _ in range(TRAIN_GLOSS)],
            'accuracy_overall':    [0 for _ in range(TRAIN_GLOSS)],
            'count_correct':       [0 for _ in range(TRAIN_GLOSS)],
            'count_videos':        [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_VAL: {
            'accuracy_correct':    [0 for _ in range(TRAIN_GLOSS)],
            'accuracy_overall':    [0 for _ in range(TRAIN_GLOSS)],
            'count_correct':       [0 for _ in range(TRAIN_GLOSS)],
            'count_videos':        [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_TEST: {
            'accuracy_correct':    [0 for _ in range(TRAIN_GLOSS)],
            'accuracy_overall':    [0 for _ in range(TRAIN_GLOSS)],
            'count_correct':       [0 for _ in range(TRAIN_GLOSS)],
            'count_videos':        [0 for _ in range(TRAIN_GLOSS)]
        },
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
                        details[tvt_idv]['accuracy_correct'][y_shouldbe]+= y_out[y_out_g_id]
                        details[tvt_idv]['count_correct'][y_shouldbe]+= 1
                    details[tvt_idv]['accuracy_overall'][y_shouldbe]+= y_out[y_out_g_id]
                    details[tvt_idv]['count_videos'][y_shouldbe]+= 1
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
                    details[tvt_idv]['accuracy_correct'][y_shouldbe]+= y_out[y_out_g_id]
                    details[tvt_idv]['count_correct'][y_shouldbe]+= 1
                details[tvt_idv]['accuracy_overall'][y_shouldbe]+= y_out[y_out_g_id]
                details[tvt_idv]['count_videos'][y_shouldbe]+= 1
            batch_vid_lm= []
            batch_gloss= []
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"__________________________ {tvt_idv} ____")
        for g_id in range(TRAIN_GLOSS):
            print(f"{g_id}: {glasl_landmark[KEY_ID2G][g_id]} --> {details[tvt_idv]['count_correct'][g_id]}/{details[tvt_idv]['count_videos'][g_id]} --> {
                (details[tvt_idv]['count_correct'][g_id] / details[tvt_idv]['count_videos'][g_id] * 100):.2f
            }%")
            print(f"accuracy on correct: {
                details[tvt_idv]['accuracy_correct'][g_id] /
                (details[tvt_idv]['count_correct'][g_id] if details[tvt_idv]['count_correct'][g_id]>0 else 1) *
                100
            }%")
            print(f"accuracy( overall ): {
                details[tvt_idv]['accuracy_overall'][g_id] /
                (details[tvt_idv]['count_videos'][g_id] if details[tvt_idv]['count_videos'][g_id]>0 else 1) *
                100
            }%")
            print()
        print("<< ----------------------------------------------------------------- >>")
        print(f"____ {tvt_idv} --> {sum(details[tvt_idv]['count_correct'])}/{sum(details[tvt_idv]['count_videos'])} --> {
            sum(details[tvt_idv]['count_correct']) / sum(details[tvt_idv]['count_videos']) *100
        }%")
        print(f"____ accuracy on correct {tvt_idv} --> {
            sum(details[tvt_idv]['accuracy_correct']) / sum(details[tvt_idv]['count_correct']) *100
        }%")
        print(f"____ overall accuracy {tvt_idv} --> {
            sum(details[tvt_idv]['accuracy_overall']) / sum(details[tvt_idv]['count_videos']) *100
        }%")
        print("<< ----------------------------------------------------------------- >>")
    print("\n\nsummary")
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"-----------------------------------------------------------------")
        print(f"____ {tvt_idv} --> {  sum(details[tvt_idv]['count_correct'])  }/{  sum(details[tvt_idv]['count_videos'])  } --> {
            sum(details[tvt_idv]['count_correct']) / sum(details[tvt_idv]['count_videos']) *100
        }%")
        print(f"____ accuracy on correct {tvt_idv} --> {sum(details[tvt_idv]['accuracy_correct'])/sum(details[tvt_idv]['count_correct'])*100}%")
        print(f"____ overall accuracy {tvt_idv} --> {sum(details[tvt_idv]['accuracy_overall'])/sum(details[tvt_idv]['count_videos'])*100}%")
    print(f"-----------------------------------------------------------------")
    print(f"____ percentage correct --> {
        sum(details[KEY_TRAIN]['count_correct'])+
        sum(details[KEY_VAL  ]['count_correct'])+
        sum(details[KEY_TEST ]['count_correct'])
    } / {
        sum(details[KEY_TRAIN]['count_videos'])+
        sum(details[KEY_VAL  ]['count_videos'])+
        sum(details[KEY_TEST ]['count_videos'])
    } --> {
        (
            sum(details[KEY_TRAIN]['count_correct'])+
            sum(details[KEY_VAL  ]['count_correct'])+
            sum(details[KEY_TEST ]['count_correct'])
        ) / (
            sum(details[KEY_TRAIN]['count_videos'])+
            sum(details[KEY_VAL  ]['count_videos'])+
            sum(details[KEY_TEST ]['count_videos'])
        )*100
    }%")
    print(f"____ accuracy on correct --> {
        (
            sum(details[KEY_TRAIN]['accuracy_correct'])+
            sum(details[KEY_VAL  ]['accuracy_correct'])+
            sum(details[KEY_TEST ]['accuracy_correct'])
        ) / (
            sum(details[KEY_TRAIN]['count_correct'])+
            sum(details[KEY_VAL  ]['count_correct'])+
            sum(details[KEY_TEST ]['count_correct'])
        )*100
    }%")
    print(f"____ accuracy( overall ) --> {
        (
            sum(details[KEY_TRAIN]['accuracy_overall'])+
            sum(details[KEY_VAL  ]['accuracy_overall'])+
            sum(details[KEY_TEST ]['accuracy_overall'])
        ) / (
            sum(details[KEY_TRAIN]['count_videos'])+
            sum(details[KEY_VAL  ]['count_videos'])+
            sum(details[KEY_TEST ]['count_videos'])
        )*100
    }%")
    print(f"____ accuracy on correct( avg.({KEY_TRAIN}, {KEY_VAL}, {KEY_TEST}) --> {
        (
            sum(details[KEY_TRAIN]['accuracy_correct']) / sum(details[KEY_TRAIN]['count_correct']) +
            sum(details[KEY_VAL  ]['accuracy_correct']) / sum(details[KEY_VAL  ]['count_correct']) +
            sum(details[KEY_TEST ]['accuracy_correct']) / sum(details[KEY_TEST ]['count_correct'])
        )/3.0*100
    }%")
    print(f"____ accuracy( overall )( avg.({KEY_TRAIN}, {KEY_VAL}, {KEY_TEST}) --> {
        (
            sum(details[KEY_TRAIN]['accuracy_overall']) / sum(details[KEY_TRAIN]['count_videos']) +
            sum(details[KEY_VAL  ]['accuracy_overall']) / sum(details[KEY_VAL  ]['count_videos']) +
            sum(details[KEY_TEST ]['accuracy_overall']) / sum(details[KEY_TEST ]['count_videos'])
        )/3.0*100
    }%")
