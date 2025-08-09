from keras.src.activations.activations import ReLU
from keras.src.layers import Conv3D, Dense, Dropout, Flatten, Input, MaxPooling3D
from keras.src.activations import softmax
from numpy import float32


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, TOTAL_GLOSS_UNIQ


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)


x= Conv3D(
    filters=8,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p1_cnn_2d'
)(data_in)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='p1_mp_2d'
)(x)
x= Conv3D(
    filters=8,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p1_cnn_3d'
)(x)
x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='p1_mp_3d'
)(x)
# x= Dropout(
#     rate=0.1,
#     name='p1_do'
# )(x)




x= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p2_cnn_2d'
)(x)
x= MaxPooling3D(
    pool_size=(1,3,3),
    padding='valid',
    data_format='channels_last',
    name='p2_mp_2d'
)(x)
x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p2_cnn_3d'
)(x)
# x= MaxPooling3D(
#     pool_size=(2,1,1),
#     padding='valid',
#     data_format='channels_last',
#     name='p2_mp_3d'
# )(x)
# x= Dropout(
#     rate=0.1,
#     name='p2_do'
# )(x)




x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p3_cnn_2d'
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='p3_mp_2d'
)(x)
x= Conv3D(
    filters=32,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p3_cnn_3d'
)(x)
# x= Dropout(
#     rate=0.1,
#     name='p3_do'
# )(x)
x= MaxPooling3D(
    pool_size=(1,3,3),
    strides=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='p3_mp_3d'
)(x)
x= Conv3D(
    filters=32,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p3_cnn_3d_1'
)(x)
x= Flatten(name='flat')(x)


data_out = Dense(TOTAL_GLOSS_UNIQ, activation=softmax, name='batch_class')(x)
