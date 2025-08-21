from keras.src.activations.activations import ReLU
from keras.src.layers import LSTM, Conv1D, Dropout, MaxPooling2D, Reshape, Dense, Input, TimeDistributed
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
    units=512,
    return_sequences=True,
    dtype=float64,
    name='lstm'
)(x)
x= LSTM(
    units=256,
    return_sequences=True,
    dtype=float64,
    name='lstm1'
)(x)
x= LSTM(
    units=256,
    return_sequences=True,
    dtype=float64,
    name='lstm2'
)(x)
x= Dropout(
    rate=0.1,
    dtype=float64,
)(x)
x= LSTM(
    units=128,
    dtype=float64,
    name='lstm_out'
)(x)


x= Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
    name='ann'
)(x)
x= Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
    name='ann1'
)(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
