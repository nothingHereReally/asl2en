from keras.src.activations.activations import ReLU
from keras.src.layers import LSTM, Reshape, Attention, Dense, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32


from .lmark_constant_v2 import LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
    dtype=float32,
    name='batch_vid',
)
qatt_x= TimeDistributed(TimeDistributed(Dense(
    units=1
)),
    name='q_ann'
)(data_in)
qatt_x= Reshape(
    target_shape=(QUANTITY_FRAME, -1)
)(qatt_x)
qatt_x= LSTM(
    units=32,
    return_sequences=True,
    name='q_p1_lstm'
)(qatt_x)
qatt_x= LSTM(
    units=64,
    return_sequences=True,
    name='q_p2_lstm'
)(qatt_x)








vatt_x= TimeDistributed(TimeDistributed(Dense(
    units=1
)),
    name='v_ann'
)(data_in)
vatt_x= Reshape(
    target_shape=(QUANTITY_FRAME, -1)
)(vatt_x)
vatt_x= LSTM(
    units=32,
    return_sequences=True,
    name='v_p1_lstm'
)(vatt_x)
vatt_x= LSTM(
    units=64,
    return_sequences=True,
    name='v_p2_lstm'
)(vatt_x)








katt_x= TimeDistributed(TimeDistributed(Dense(
    units=1
)),
    name='k_ann'
)(data_in)
katt_x= Reshape(
    target_shape=(QUANTITY_FRAME, -1)
)(katt_x)
katt_x= LSTM(
    units=32,
    return_sequences=True,
    name='k_p1_lstm'
)(katt_x)
katt_x= LSTM(
    units=64,
    return_sequences=True,
    name='k_p2_lstm'
)(katt_x)








x= Attention()([qatt_x, vatt_x, katt_x])
_, _, x= LSTM(
    units=64,
    return_state=True
)(x)
x= Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
)(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
