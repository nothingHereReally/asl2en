# from types import LambdaType
# from typing import Any
from keras.src.activations.activations import ReLU
from keras.src.layers import LSTM, Attention, Dense, Flatten, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32


from .lmark_constant_v2 import LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
    dtype=float32,
    name='batch_vid',
)
qatt_x= TimeDistributed(LSTM(
    units=64,
    return_sequences=True
))(data_in)
qatt_x= TimeDistributed(LSTM(
    units=64,
    return_sequences=True
))(qatt_x)
qatt_x= TimeDistributed(LSTM(
    units=64
))(qatt_x)








vatt_x= TimeDistributed(LSTM(
    units=64,
    return_sequences=True
))(data_in)
vatt_x= TimeDistributed(LSTM(
    units=64,
    return_sequences=True
))(vatt_x)
vatt_x= TimeDistributed(LSTM(
    units=64
))(vatt_x)








katt_x= TimeDistributed(LSTM(
    units=64,
    return_sequences=True
))(data_in)
katt_x= TimeDistributed(LSTM(
    units=64,
    return_sequences=True
))(katt_x)
katt_x= TimeDistributed(LSTM(
    units=64
))(katt_x)








x= Attention()([qatt_x, vatt_x, katt_x])
x= Flatten()(x)
x= Dense(
    units=256,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
)(x)
x= Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
)(x)
x= Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
)(x)
x= Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
)(x)
# x= Reshape(
#     target_shape=(QUANTITY_FRAME, -1)
# )(x)
# x= LSTM(
#     units=64,
#     return_sequences=True,
#     activation='relu'
# )(x)
# x= LSTM(
#     units=128,
#     return_sequences=True,
#     activation='relu'
# )(x)
# x= LSTM(
#     units=64,
#     return_sequences=False,
#     activation='relu'
# )(x)
# x= Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
# )(x)
# x= Dense(
#     units=32,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
# )(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
