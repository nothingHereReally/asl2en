from keras.src.activations.activations import ReLU
from keras.src.layers import LSTM, Reshape, Dense, Input
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
x= LSTM(
    units=LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2),
    return_sequences=True,
    dtype=float64,
    name='lstm_1a'
)(x)
x= LSTM(
    units=LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2),
    return_sequences=True,
    dtype=float64,
    name='lstm_1b'
)(x)
# x= LSTM(
#     units=256,
#     return_sequences=True,
#     dtype=float64,
#     name='lstm2a'
# )(x)
# x= LSTM(
#     units=256,
#     return_sequences=True,
#     dtype=float64,
#     name='lstm2b'
# )(x)
# x= Dropout(
#     rate=0.1,
#     dtype=float64,
# )(x)
# x= LSTM(
#     units=128,
#     return_sequences=True,
#     dtype=float64,
#     name='lstm3a'
# )(x)
# x= LSTM(
#     units=128,
#     return_sequences=True,
#     dtype=float64,
#     name='lstm3b'
# )(x)
x= LSTM(
    units=64,
    return_sequences=True,
    dtype=float64,
    name='lstm4a'
)(x)
x= LSTM(
    units=64,
    return_sequences=True,
    dtype=float64,
    name='lstm4b'
)(x)
x= LSTM(
    units=64,
    dtype=float64,
    name='lstm_out'
)(x)


x= Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
    name='ann'
)(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
