from keras.src.activations.activations import ReLU
from keras.src.layers import Flatten, MultiHeadAttention, Reshape, Dense, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32, float64


from .lmark_constant_v2 import LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
    dtype=float32,
    name='batch_vid',
)
x= Reshape(
    target_shape=(QUANTITY_FRAME, -1),
    dtype=float64,
    name='reshape'
)(data_in)


x= MultiHeadAttention(
    num_heads=8,
    key_dim=(LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2)* 2)*1,
    value_dim=(LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2)* 2)*1,
    name="mutli_head_1",
    dtype=float64,
)(query=x, value=x, key=x)
query= TimeDistributed(Dense(
    units=256,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
value= TimeDistributed(Dense(
    units=256,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
key= TimeDistributed(Dense(
    units=256,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
x= MultiHeadAttention(
    num_heads=8,
    key_dim=256,
    value_dim=256,
    name="mutli_head_2",
    dtype=float64,
)(query=query, value=value, key=key)
query= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
value= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
key= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
x= MultiHeadAttention(
    num_heads=8,
    key_dim=128,
    value_dim=128,
    name="mutli_head_3",
    dtype=float64,
)(query=query, value=value, key=key)
query= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
value= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
key= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
x= MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    value_dim=64,
    name="mutli_head_4",
    dtype=float64,
)(query=query, value=value, key=key)
query= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
value= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
key= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
))(x)
x= MultiHeadAttention(
    num_heads=8,
    key_dim=32,
    value_dim=32,
    name="mutli_head_5",
    dtype=float64,
)(query=query, value=value, key=key)
ann= Flatten()(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(ann)
