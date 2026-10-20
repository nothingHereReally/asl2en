from keras.src.activations.activations import ReLU
from keras.src.layers import Attention, Flatten, Dense, Input, MultiHeadAttention, Layer
from keras.src.activations import softmax
from keras.src.saving import register_keras_serializable
from keras.src import ops
from typing import Any
from numpy import float32, float64


from .constant import LANDMARK_SHAPE, LM_IDX_LEFT_HIP, LM_IDX_LEFT_SHOULDER, LM_IDX_RIGHT_HIP, LM_IDX_RIGHT_SHOULDER, LM_LEFT_HAND_NOT_TIP, LM_LEFT_HAND_TIP, LM_RIGHT_HAND_NOT_TIP, LM_RIGHT_HAND_TIP, QUANTITY_FRAME, LEN_GLOSS, LM_IMPORTANT_FACE_IDX


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
        shape: Any= ops.shape(data_self_att) # shape(BATCH_SIZE, QUANTITY_FRAME, 86, 2)
        data= ops.reshape(data_self_att, (-1, shape[-2]*shape[-1])) # shape(BATCH_SIZE*QUANTITY_FRAME, 86*2)
        query= [query_ann(data) for query_ann in self.in_query] # (-1, 172) -> shape(-1, 86)
        value= [value_ann(data) for value_ann in self.in_value] # (-1, 172) -> shape(-1, 86)
        key=   [key_ann(data)   for key_ann   in self.in_key]   # (-1, 172) -> shape(-1, 86)
        query= [ops.reshape(query_ann, (-1, shape[-3], shape[-2])) for query_ann in query]
        value= [ops.reshape(value_ann, (-1, shape[-3], shape[-2])) for value_ann in value]
        key=   [ops.reshape(key_ann,   (-1, shape[-3], shape[-2])) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.in_attention)]
        output= ops.sum(output, axis=0) # shape(BATCH_SIZE, QUANTITY_FRAME, 86)
        # ----------------------------------------------------------------------
        output= ops.reshape(output, (-1, shape[-2])) # shape(BATCH_SIZE*QUANTITY_FRAME, 86)
        query= [query_ann(output) for query_ann in self.query]
        value= [value_ann(output) for value_ann in self.value]
        key=   [key_ann(output)   for key_ann   in self.key]
        query= [ops.reshape(query_ann, (-1, shape[-3], shape[-2]//2)) for query_ann in query]
        value= [ops.reshape(value_ann, (-1, shape[-3], shape[-2]//2)) for value_ann in value]
        key=   [ops.reshape(key_ann,   (-1, shape[-3], shape[-2]//2)) for key_ann   in key]
        output= [attention(
            [query[idx], value[idx], key[idx]]
        ) for idx, attention in enumerate(self.attention)]
        # input_shape (BATCH_SIZE, QUANTITY_FRAME, 86, 2)
        # output_shape (BATCH_SIZE, QUANTITY_FRAME, 43=86//2)
        return ops.sum(output, axis=0)
    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "dropout":   self.dropout,
            "name":      self.name,
        }
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
layer_in= Input(
    shape=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]),
    dtype=float32,
    name='batch_vid',
)
x= SelectLM(name='select49_from86')(layer_in)
# ---------------------------------------------------------------------------
# x= ShapeReduce43ViaAttention(
#     num_heads=8,
#     dropout=0.1, # (QUANTITY_FRAME, LANDMARK_SHAPE[0]//2==43)
# )(x)
x= MultiHeadAttention(
    num_heads=8,
    key_dim=43//8,
    dropout=0.1,
    use_bias=True,
    name='mha_86',
)(
    query=x,
    value=x,
    key=x,
) # (QUANTITY_FRAME, 61)
# x= ShapeReduceViaAddAndAttention(
#     target=21,
#     num_heads=8,
#     dropout=0.1
# )(x)
ann= Flatten()(x)
ann= Dense(
    units=LANDMARK_SHAPE[0]*2,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
)(ann)
# ---------------------------------------------------------------------------
layer_out = Dense(LEN_GLOSS, activation=softmax, name='batch_class')(ann)
