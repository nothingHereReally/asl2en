from keras.src.layers import Conv2D, Conv3D, ConvLSTM2D, Dense, Dropout, Flatten, Input, MaxPooling2D, MaxPooling3D, Rescaling, TimeDistributed
from keras.src.activations import relu, softmax
from numpy import float32


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, TOTAL_GLOSS_UNIQ


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)



x= ConvLSTM2D(
    filters=4,
    kernel_size=(3,3),
    activation='tanh',
    data_format='channels_last',
    recurrent_dropout=0.2,
    return_sequences=True
)(data_in)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last'
)(x)
x= TimeDistributed(Dropout(0.2))(x)
x= ConvLSTM2D(
    filters=8,
    kernel_size=(3,3),
    activation='tanh',
    data_format='channels_last',
    recurrent_dropout=0.2,
    return_sequences=True
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last'
)(x)
x= TimeDistributed(Dropout(0.2))(x)
x= ConvLSTM2D(
    filters=14,
    kernel_size=(3,3),
    activation='tanh',
    data_format='channels_last',
    recurrent_dropout=0.2,
    return_sequences=True
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last'
)(x)
x= TimeDistributed(Dropout(0.2))(x)
x= ConvLSTM2D(
    filters=16,
    kernel_size=(3,3),
    activation='tanh',
    data_format='channels_last',
    recurrent_dropout=0.2,
    return_sequences=True
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last'
)(x)


x= Flatten(
    name='flat_1'
)(x)
data_out= Dense(
    units=TOTAL_GLOSS_UNIQ,
    activation=softmax,
    name='batch_class'
)(x)
