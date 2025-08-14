# from types import LambdaType
# from typing import Any
from keras.src.activations.activations import ReLU
from keras.src.layers import Concatenate, Conv2D, Conv3D, Dense, Flatten, Input, MaxPooling2D, MaxPooling3D, Dot, Reshape
from keras.src.activations import softmax
# from keras.ops import expand_dims
# from tensorflow import reshape, convert_to_tensor
# from keras.saving import register_keras_serializable
from numpy import float32, array


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid',
)
mask_channel0= array([[
    [1],
    [0],
    [0]
]], dtype=float32)
mask_channel1= array([[
    [1],
    [0],
    [0]
]], dtype=float32)
mask_channel2= array([[
    [1],
    [0],
    [0]
]], dtype=float32)


c0x= Reshape(
    target_shape=(-1, 3)
)(data_in)
c0x= Dot(
    axes=(-1, -2),
    normalize=False,
    name='channel_0'
)([c0x, mask_channel0])
c0x= Reshape(
    target_shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 1),
    name='channel_0_reshape'
)(c0x)
c0x= Conv3D(
    filters=8,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p1_cnn_2d'
)(c0x)
c0x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c0p1_mp_2d'
)(c0x)
c0x= Conv3D(
    filters=8,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p1_cnn_3d'
)(c0x)
c0x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c0p1_mp_3d'
)(c0x)
c0x= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p2_cnn_2d'
)(c0x)
c0x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c0p2_mp_2d'
)(c0x)
c0x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p2_cnn_3d'
)(c0x)
c0x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c0p2_mp_3d'
)(c0x)
c0x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p3_cnn_2d'
)(c0x)
c0x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c0p3_mp_2d'
)(c0x)
c0x= Conv3D(
    filters=1,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c0p3_cnn_3d'
)(c0x)
c0x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c0p3_mp_3d'
)(c0x)
c1x= Reshape(
    target_shape=(-1, 3)
)(data_in)
c1x= Dot(
    axes=(-1, -2),
    normalize=False,
    name='channel_1'
)([c1x, mask_channel1])
c1x= Reshape(
    target_shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 1),
    name='channel_1_reshape'
)(c1x)
c1x= Conv3D(
    filters=8,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c1p1_cnn_2d'
)(c1x)
c1x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c1p1_mp_2d'
)(c1x)
c1x= Conv3D(
    filters=8,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c1p1_cnn_3d'
)(c1x)
c1x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c1p1_mp_3d'
)(c1x)
c1x= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c1p2_cnn_2d'
)(c1x)
c1x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c1p2_mp_2d'
)(c1x)
c1x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c1p2_cnn_3d'
)(c1x)
c1x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c1p2_mp_3d'
)(c1x)
c1x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c1p3_cnn_2d'
)(c1x)
c1x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c1p3_mp_2d'
)(c1x)
c1x= Conv3D(
    filters=1,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c1p3_cnn_3d'
)(c1x)
c1x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c1p3_mp_3d'
)(c1x)
c2x= Reshape(
    target_shape=(-1, 3)
)(data_in)
c2x= Dot(
    axes=(-1, -2),
    normalize=False,
    name='channel_2'
)([c2x, mask_channel2])
c2x= Reshape(
    target_shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 1),
    name='channel_2_reshape'
)(c2x)
c2x= Conv3D(
    filters=8,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c2p1_cnn_2d'
)(c2x)
c2x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c2p1_mp_2d'
)(c2x)
c2x= Conv3D(
    filters=8,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c2p1_cnn_3d'
)(c2x)
c2x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c2p1_mp_3d'
)(c2x)
c2x= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c2p2_cnn_2d'
)(c2x)
c2x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c2p2_mp_2d'
)(c2x)
c2x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c2p2_cnn_3d'
)(c2x)
c2x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c2p2_mp_3d'
)(c2x)
c2x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c2p3_cnn_2d'
)(c2x)
c2x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='c2p3_mp_2d'
)(c2x)
c2x= Conv3D(
    filters=1,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='c2p3_cnn_3d'
)(c2x)
c2x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='c2p3_mp_3d'
)(c2x)


c0x= Reshape(
    target_shape=(18, 18, 1),
    name='channel_0_to2d'
)(c0x)
c1x= Reshape(
    target_shape=(18, 18, 1),
    name='channel_1_to2d'
)(c1x)
c2x= Reshape(
    target_shape=(18, 18, 1),
    name='channel_2_to2d'
)(c2x)
x= Concatenate(
    axis=-1,
    name='concat_1'
)([c0x, c1x, c2x]) # output shape is (50, 17, 17, 3)








x= Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p1_cnn_2d'
)(x)
x= MaxPooling2D(
    pool_size=(2,2),
    padding='valid',
    data_format='channels_last',
    name='p1_mp_2d'
)(x)
x= Conv2D(
    filters=128,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p2_cnn_2d'
)(x)
x= MaxPooling2D(
    pool_size=(2,2),
    padding='valid',
    data_format='channels_last',
    name='p2_mp_2d'
)(x)
x= Conv2D(
    filters=10,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p3_cnn_2d'
)(x)
x= Flatten(name='flat')(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
