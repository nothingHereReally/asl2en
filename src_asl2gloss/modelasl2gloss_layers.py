from keras.src.layers import Conv2D, Conv3D, ConvLSTM2D, Dense, Flatten, Input, MaxPooling2D, MaxPooling3D, Rescaling, TimeDistributed
from keras.src.activations import relu, sigmoid, softmax
from numpy import float32


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, TOTAL_GLOSS_UNIQ


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)



x= TimeDistributed(Conv2D(
    filters=8,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
),
    name='conv2d_1'
)(data_in)
x= MaxPooling3D(
    pool_size=(1,3,3),
    strides=(1,3,3),
    padding='valid',
    dtype=float32,
    name='mp_1'
)(x)


x= TimeDistributed(Conv2D(
    filters=16,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
),
    name='conv2d_2'
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    strides=(1,2,2),
    padding='valid',
    dtype=float32,
    name='mp_2'
)(x)


x= TimeDistributed(Conv2D(
    filters=24,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
),
    name='conv2d_3'
)(x)
x= MaxPooling3D(
    pool_size=(1,4,4),
    strides=(1,4,4),
    padding='valid',
    dtype=float32,
    name='mp_3'
)(x)


x= ConvLSTM2D(
    filters=32,
    kernel_size=(3,3),
    strides=1,
    padding='valid',
    return_sequences=False,
    dtype=float32,
    name='convLstm2d_5'
)(x)
# x= ConvLSTM2D(
#     filters=32,
#     kernel_size=(3,3),
#     strides=1,
#     padding='valid',
#     return_sequences=False,
#     dtype=float32,
#     name='convLstm2d_5'
# )(x)


x= Flatten(
    name='flat_1'
)(x)
data_out= Dense(
    units=TOTAL_GLOSS_UNIQ,
    activation=softmax,
    name='batch_class'
)(x)
