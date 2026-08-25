from keras.src import ops
from keras.src.activations.activations import ReLU
from keras.src.layers import (
    Dense,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Layer,
    Reshape,
)
from keras.src.activations import softmax
from keras.src.saving import register_keras_serializable
from keras_hub.layers import SinePositionEncoding
from numpy import float32


from .lmark_constant import LANDMARK_SHAPE, QUANTITY_FRAME, LEN_GLOSS


@register_keras_serializable()
class SinCosPostionalEncoding( Layer ):
    def __init__(
        self,
        name: str='positional_encoding',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name= name
    def build(self, input_shape):
        assert len(input_shape)==3
        self.positional_encoding= SinePositionEncoding(
            max_wavelength=10000,
        )(x)
    def call(self, data_input):
        data= ops.add(
            data_input,
            self.positional_encoding(data_input)
        )

        return data
    def get_config(self):
        return {
            **super().get_config(),
            "name":      self.name,
        }
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
        data= self.multiHeadAtt(data_input)
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
# ---------------------------------------------------------------------------
data_in= Input(
    shape=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]),
    dtype=float32,
    name='batch_vid',
)
# ---------------------------------------------------------------------------
x= Reshape(
    target_shape=(
        QUANTITY_FRAME,
        LANDMARK_SHAPE[0]*LANDMARK_SHAPE[1]
    ),
    name="reshape_QF_172"
)(data_in)
x= SinCosPostionalEncoding(
    name="positional_encoding"
)(x)
x= NTimesEncoderTransformer(
    n_times=6,
    name='encoder_transformer_6ns',
)(x) # (QUANTITY_FRAME, LANDMARK_SHAPE[0]*LANDMARK_SHAPE[1])
ann= GlobalAveragePooling1D(
    data_format='channels_last',
    keepdims=False,
)(x)
# ---------------------------------------------------------------------------
data_out = Dense(LEN_GLOSS, activation=softmax, name='batch_class')(ann)
