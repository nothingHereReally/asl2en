from keras.src.layers import Conv3D, Dense, Flatten, Input, MaxPooling3D, Rescaling
from keras.src.activations import relu, sigmoid, softmax
from numpy import float32


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, TOTAL_GLOSS_UNIQ


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)
# x= Rescaling(
#     scale=1./255,
#     offset=0.0,
#     dtype=float32,
#     name='scale_0.0_1.0'
# )(data_in)



x= Conv3D(
    filters=16,
    # kernel_size=(13,3,3),
    # strides=(1,1,1),
    kernel_size=(5,5,5),
    strides=(3,3,3),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_1'
# )(x)
)(data_in)
x= MaxPooling3D(
    # pool_size=(5,5,5),
    # strides=(3,3,3),
    pool_size=(3,3,3),
    strides=(2,2,2),
    padding='valid',
    dtype=float32,
    name='maxpool_1'
)(x)
x= Conv3D(
    filters=64,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_2'
)(x)
x= MaxPooling3D(
    # pool_size=(7,7,7),
    # strides=(5,5,5),
    pool_size=(3,3,3),
    strides=(2,2,2),
    padding='valid',
    dtype=float32,
    name='maxpool_2'
)(x)
x= Conv3D(
    filters=16,
    # kernel_size=(1,3,3),
    kernel_size=(2,5,5),
    strides=(1,1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_3'
)(x)


x= Flatten()(x)
x= Dense(
    # units=16,
    units=TOTAL_GLOSS_UNIQ//100,
    activation=sigmoid,
    dtype=float32,
    name='dense_1'
)(x)
data_out= Dense(
    units=TOTAL_GLOSS_UNIQ,
    activation=softmax,
    dtype=float32,
    name='batch_class'
)(x)








data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)



x= Conv3D(
    filters=8,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_1'
)(data_in)
x= MaxPooling3D(
    pool_size=(2,2,2),
    strides=(1,1,1),
    padding='valid',
    dtype=float32,
    name='maxpool_1'
)(x)


x= Conv3D(
    filters=16,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=sigmoid,
    dtype=float32,
    name='conv_2'
)(x)
x= MaxPooling3D(
    pool_size=(2,2,2),
    strides=(1,1,1),
    padding='valid',
    dtype=float32,
    name='maxpool_2'
)(x)


x= Conv3D(
    filters=32,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_3'
)(x)
x= MaxPooling3D(
    pool_size=(5,5,5),
    strides=(2,2,2),
    padding='valid',
    dtype=float32,
    name='maxpool_3'
)(x)


x= Conv3D(
    filters=64,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=sigmoid,
    dtype=float32,
    name='conv_4'
)(x)
x= MaxPooling3D(
    pool_size=(3,5,5),
    strides=(2,2,2),
    padding='valid',
    dtype=float32,
    name='maxpool_4'
)(x)


x= Conv3D(
    filters=128,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_5'
)(x)
x= MaxPooling3D(
    pool_size=(2,2,2),
    strides=(1,1,1),
    padding='valid',
    dtype=float32,
    name='maxpool_5'
)(x)


x= Conv3D(
    filters=256,
    kernel_size=(3,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=sigmoid,
    dtype=float32,
    name='conv_6'
)(x)
x= MaxPooling3D(
    pool_size=(2,2,2),
    strides=(1,1,1),
    padding='valid',
    dtype=float32,
    name='maxpool_6'
)(x)


x= Conv3D(
    filters=512,
    kernel_size=(2,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_7'
)(x)
x= MaxPooling3D(
    pool_size=(1,5,5),
    strides=(1,2,2),
    padding='valid',
    dtype=float32,
    name='maxpool_7'
)(x)


x= Conv3D(
    filters=1024,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    activation=sigmoid,
    dtype=float32,
    name='conv_8'
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    strides=(1,1,1),
    padding='valid',
    dtype=float32,
    name='maxpool_8'
)(x)


x= Conv3D(
    filters=2048,
    kernel_size=(1,3,3),
    strides=(1,2,2),
    padding='valid',
    activation=relu,
    dtype=float32,
    name='conv_9'
)(x)
x= MaxPooling3D(
    pool_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    dtype=float32,
    name='maxpool_9'
)(x)


# # x= Conv3D(
# #     filters=2048,
# #     kernel_size=(1,3,3),
# #     strides=(1,1,1),
# #     padding='valid',
# #     activation=relu,
# #     dtype=float32,
# #     name='conv_9'
# # )(x)


x= Flatten()(x)
x= Dense(
    units=TOTAL_GLOSS_UNIQ//100,
    activation=sigmoid,
    dtype=float32,
    name='dense_1'
)(x)
data_out= Dense(
    units=TOTAL_GLOSS_UNIQ,
    activation=softmax,
    dtype=float32,
    name='batch_class'
)(x)
