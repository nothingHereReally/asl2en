from keras.src.activations.activations import ReLU
from keras.src.layers import Dense, Input, LayerNormalization, MultiHeadAttention, Layer
from keras.src.activations import softmax
from keras.src import ops
from keras.src.saving import register_keras_serializable
from typing import Any


from .constant import (
    LANDMARK_SHAPE,
    LM_IDX_LEFT_INDEXF_TIP,

    LM_IDX_LEFT_SHOULDER,
    LM_IDX_RIGHT_INDEXF_TIP,
    LM_IDX_RIGHT_SHOULDER,
    LM_IDX_LEFT_HIP,
    LM_IDX_RIGHT_HIP,
    Q_W_FACE,
    Q_W_POSE,

    QUANTITY_FRAME,
    LEN_GLOSS,
    LMs_IDXs_FACE_GOOD,
    LMs_IDXs_LEFT_H_NOT_INDEXF,
    LMs_IDXs_RIGHT_H_NOT_INDEXF,
)


@register_keras_serializable()
class EncoderTransformer( Layer ):
    def __init__(
        self,
        num_heads: int=4,  # 8 on 'attention is all u need' (2017)
        key_dim: int=43,   # 64 on 'attention is all u need' (2017)
        value_dim: int=43, # --> so that 43*4= 172= 86*2
        dropout: float=0.1,
        ann_units: int=2048,
        name: str='encoder_transformer',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads= num_heads
        self.key_dim=   key_dim
        self.value_dim= value_dim
        self.dropout=   dropout
        self.ann_units= ann_units
        self.name=      name
    def build(self, input_shape):
        # assert len(input_shape)==3
        d_model= input_shape[-1]
        self.multiHeadAtt= MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            dropout=self.dropout,
            use_bias=True,
        )
        self.normAfterMHA= LayerNormalization()
        self.ann= Dense(
            units=self.ann_units,
            activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
        )
        self.annBringBackShape= Dense(
            units=d_model,
            activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
        )
        self.normAfterAnn= LayerNormalization()
    def call(self, data_input):
        data= self.multiHeadAtt(
            query=data_input,
            value=data_input,
            key=data_input,
        )
        data= ops.add(data, data_input)
        data_after_mha= self.normAfterMHA(data)

        data= self.ann(data_after_mha)
        data= self.annBringBackShape(data)
        data= ops.add(data, data_after_mha)
        data= self.normAfterAnn(data)

        return data
    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "value_dim": self.value_dim,
            "dropout":   self.dropout,
            "ann_units": self.ann_units,
            "name":      self.name,
        }
@register_keras_serializable()
class NTimesEncoderTransformer(Layer):
    def __init__(
        self,
        n_times: int=6,
        num_heads: int=4,  # 8 on 'attention is all u need' (2017)
        key_dim: int=43,   # 64 on 'attention is all u need' (2017)
        value_dim: int=43, # --> so that 43*4= 172= 86*2
        dropout: float=0.1,
        ann_units: int=2048,
        name: str='multi_encoder_in_sequence',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_times=   n_times
        self.num_heads= num_heads
        self.key_dim=   key_dim
        self.value_dim= value_dim
        self.dropout=   dropout
        self.ann_units= ann_units
        self.name=      name
    def build(self, input_shape):
        # assert len(input_shape)==3
        del input_shape
        self.encoder_seq= [EncoderTransformer(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            dropout=self.dropout,
            ann_units=self.ann_units,
            name=f"enc_{idx}",
        ) for idx in range(self.n_times)]
    def call(self, data_input):
        data= data_input
        for encoder in self.encoder_seq:
            data= encoder(data)
        return data
    def get_config(self):
        return {
            **super().get_config(),
            "n_times":   self.n_times,
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "value_dim": self.value_dim,
            "dropout":   self.dropout,
            "ann_units": self.ann_units,
            "name":      self.name,
        }
@register_keras_serializable()
class AttentionPooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scoring_layer= Dense(1, activation="tanh")
        del input_shape
    def call(self, inputs):
        # -- inputs shape (B, T, D)
        scores= self.scoring_layer(inputs) # be shape (B, T, 1)
        weights= ops.softmax(scores, axis=1) # norm via softmax still shape (B, T, 1)
        # weights, accross what time is it most important
        return ops.sum(inputs * weights, axis=1) # shape (B, D)
@register_keras_serializable()
class ExtractFeatures(Layer):
    def __init__(self, name: str='extract_features', **kwargs):
        super().__init__(**kwargs)
        self.name= name
    def build(self, input_shape):
        del input_shape
    def _get2dots_frombody(self, landmarks):
        '''
        input -- shape of (..., be >= 86, 2)
        -------------------------------
        returned shape is (...,  2, 2)
        '''
        shoulder_center= (
            landmarks[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
            landmarks[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
        )/2.0
        hip_center= (
            landmarks[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
            landmarks[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
        )/2.0
        shoulder_hip_center= (shoulder_center +hip_center)/2.0
        '''
        shoulder_center and hip_center, both has shape (..., 1, 2)
        so as shoulder_hip_center has shape (..., 1, 2)
        '''
        return ops.concatenate([
            (shoulder_center+shoulder_hip_center)/2.0,
            (shoulder_hip_center+hip_center)/2.0,
        ], axis=-2)
    def call(self, input):
        left_indexf_tip: Any= input[..., LM_IDX_LEFT_INDEXF_TIP:LM_IDX_LEFT_INDEXF_TIP+1, :]
        right_indexf_tip: Any= input[..., LM_IDX_RIGHT_INDEXF_TIP:LM_IDX_RIGHT_INDEXF_TIP+1, :]
        the_rest: Any= ops.concatenate(
            [
                ops.take(input, indices=LMs_IDXs_FACE_GOOD, axis=-2),
                self._get2dots_frombody(input),
                input[..., Q_W_FACE:Q_W_FACE+Q_W_POSE, :],
                ops.take(input, indices=LMs_IDXs_LEFT_H_NOT_INDEXF, axis=-2),
                ops.take(input, indices=LMs_IDXs_RIGHT_H_NOT_INDEXF, axis=-2),
            ],
            axis=-2,
        )
        '''
        left_indexf_tip  shape (B, QUANTITY_FRAME, 1, 2)
        right_indexf_tip shape (B, QUANTITY_FRAME, 1, 2)
        the_rest of shape (B, QUANTITY_FRAME, 10 +2 +8 +20 +20, 2) --> (B, QF, 60, 2)
        '''
        left_indexf_tip= (the_rest - left_indexf_tip)**2
        right_indexf_tip= (the_rest - right_indexf_tip)**2
        left_indexf_tip= ops.sqrt(ops.sum(left_indexf_tip, axis=-1, keepdims=False))
        right_indexf_tip= ops.sqrt(ops.sum(right_indexf_tip, axis=-1, keepdims=False))
        '''
        left_indexf_tip  shape (B, QUANTITY_FRAME, 60)
        right_indexf_tip shape (B, QUANTITY_FRAME, 60)
        '''
        left_indexf_tip=  left_indexf_tip  * input[..., -1:, 0]
        right_indexf_tip= right_indexf_tip * input[..., -1:, 1]

        return ops.concatenate([
            left_indexf_tip,
            right_indexf_tip,
        ], axis=-1)
    def get_config(self):
        return {
            **super().get_config(),
            "name":      self.name,
        }
# ---------------------------------------------------------------------------
layer_in= Input(
    shape=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]),
    name='batch_vid',
)
# ---------------------------------------------------------------------------
x= ExtractFeatures()(layer_in)
x= NTimesEncoderTransformer(
    n_times=4,
    num_heads=8,
    key_dim=8,
    value_dim=8,
    dropout=0.1,
    ann_units=16,
    name='multi_encoder_in_sequence',
)(x)
ann= AttentionPooling()(x)
layer_out= Dense(LEN_GLOSS, activation=softmax, name='batch_class')(ann)
