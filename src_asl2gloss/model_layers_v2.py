from keras.src.activations.activations import ReLU
from keras.src.layers import LSTM, Conv1D, MaxPooling2D, Reshape, Dense, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32


from .lmark_constant_v2 import LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
    dtype=float32,
    name='batch_vid',
)
x= TimeDistributed(Conv1D(
    filters=8,
    kernel_size=3,
    strides=1,
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='cnn'
)(data_in)
x= MaxPooling2D(
    pool_size=(1,2),
    strides=(1,2),
    padding='valid',
    data_format='channels_last',
    name='mp'
)(x)
x= TimeDistributed(Conv1D(
    filters=16,
    kernel_size=3,
    strides=1,
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='cnn1'
)(data_in)
x= MaxPooling2D(
    pool_size=(1,2),
    strides=(1,2),
    padding='valid',
    data_format='channels_last',
    name='mp1'
)(x)
x= TimeDistributed(LSTM(
    units=128,
    return_sequences=True
),
    name='lstm'
)(x)
x= TimeDistributed(LSTM(
    units=128
),
    name='lstm1'
)(x)
x= LSTM(
    units=64,
    name='lstm2'
)(x)


x= Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='ann'
)(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
