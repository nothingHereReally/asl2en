# from types import LambdaType
# from typing import Any
from keras.src.activations.activations import ReLU
from keras.src.layers import Attention, Conv2D, Conv3D, Dense, Dropout, Flatten, Input, MaxPooling2D, MaxPooling3D, Permute, Reshape
from keras.src.activations import softmax
# from keras.ops import expand_dims
# from tensorflow import reshape, convert_to_tensor
# from keras.saving import register_keras_serializable
from numpy import float32


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid',
)
cx= Conv3D(
    filters=8,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p1_cnn_2d'
)(data_in)
cx= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c0p1_mp_2d'
)(cx)
cx= Conv3D(
    filters=8,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p1_cnn_3d'
)(cx)
cx= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c0p1_mp_3d'
)(cx)
cx= Reshape(
    target_shape=(10, -1),
    name='c0p1_reshape_b_att'
)(cx)
cx= Attention(
    name='c0p1_att'
)([cx,cx])
cx= Reshape(
    target_shape=(10, 78, 78, 8),
    name='c0p1_reshape_a_att'
)(cx)
# cx= Dropout(
#     rate=0.1,
#     name='c0p1_do'
# )(cx)
cx= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p2_cnn_2d'
)(cx)
cx= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c0p2_mp_2d'
)(cx)
cx= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p2_cnn_3d'
)(cx)
cx= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c0p2_mp_3d'
)(cx)
cx= Reshape(
    target_shape=(4, -1),
    name='c0p2_reshape_b_att'
)(cx)
cx= Attention(
    name='c0p2_att'
)([cx,cx])
cx= Reshape(
    target_shape=(4, 38, 38, 16),
    name='c0p2_reshape_a_att'
)(cx)
# cx= Dropout(
#     rate=0.1,
#     name='c0p2_do'
# )(cx)
cx= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p3_cnn_2d'
)(cx)
cx= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c0p3_mp_2d'
)(cx)
cx= Conv3D(
    filters=32,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p3_cnn_3d'
)(cx)
cx= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c0p3_mp_3d'
)(cx)
cx= Reshape(
    target_shape=(-1, 32),
    name='c0p3_reshape_b_att'
)(cx)
cx= Permute(
    dims=(2, 1),
    name='transpose_b'
)(cx)
cx= Attention(
    name='c0p3_att'
)([cx,cx])
# cx= Permute(
#     dims=(2, 1),
#     name='transpose_a'
# )(cx)
# cx= Reshape(
#     target_shape=(18, 18, 32),
#     name='c0p3_reshape_a_att'
# )(cx)
# cx= Dropout(
#     rate=0.1,
#     name='c0p2_do'
# )(cx)
#
#
#
#
#
#
#
#
# x= Conv2D(
#     filters=64,
#     kernel_size=(3,3),
#     strides=(1,1),
#     padding='valid',
#     data_format='channels_last',
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     name='p1_cnn_2d'
# )(cx)
# x= MaxPooling2D(
#     pool_size=(2,2),
#     padding='valid',
#     data_format='channels_last',
#     name='p1_mp_2d'
# )(x)
# x= Dropout(
#     rate=0.1,
#     name='p1_do'
# )(x)
# x= Conv2D(
#     filters=128,
#     kernel_size=(3,3),
#     strides=(1,1),
#     padding='valid',
#     data_format='channels_last',
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     name='p2_cnn_2d'
# )(x)
# x= Conv2D(
#     filters=10,
#     kernel_size=(3,3),
#     strides=(1,1),
#     padding='valid',
#     data_format='channels_last',
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     name='p3_cnn_2d'
# )(x)
x= Flatten(name='flat')(cx)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
