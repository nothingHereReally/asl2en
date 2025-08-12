from types import LambdaType
from typing import Any
from keras.src.activations.activations import ReLU
from keras.src.layers import Concatenate, Conv3D, Dense, Dropout, Flatten, Input, Lambda, Layer, MaxPooling3D, Reshape
from keras.src.activations import softmax
from numpy import float32, uint8


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, T10_GLOSS


class getWhat(Layer):
    def __init__(self, function: LambdaType, name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.function: LambdaType= function
        self.trainable: bool= False
        self.name: str= name
        # self.weights: list= []
        # self.trainable_variables: list= []
        # self.non_trainable_weights: list= []
        # self.non_trainable_variables: list= []
    def get_config(self) -> dict:
        return super().get_config()
    def build(self, input_shape) -> None:
        pass
    def call(self, inputs):
        # print(f"_____________________________ shape {inputs.shape}")
        # print(f"_____________________________ shape {inputs[:,:,:,:,0:1].shape}")
        # print(f"_____________________________ type {type(inputs)}")
        return self.function(inputs)
        # return inputs[:, :, :, :, 0:1]


data_in= Input(
    shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)


c0x= getWhat(
    function=lambda x: x[:, :, :, :, 0:1],
    name='channel_0'
)(data_in)
c0x= Conv3D(
    filters=3,
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
    filters=32,
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
    filters=1,
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
c1x= getWhat(
    function=lambda x: x[:, :, :, :, 1:2],
    name='channel_1'
)(data_in)
c1x= Conv3D(
    filters=3,
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
    filters=32,
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
    filters=1,
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
c2x= getWhat(
    function=lambda x: x[:, :, :, :, 2:],
    name='channel_2'
)(data_in)
c2x= Conv3D(
    filters=3,
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
    filters=32,
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
    filters=1,
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
x= Concatenate(
    axis=-1,
    name='concat_1'
)([c0x, c1x, c2x]) # output shape is (50, 17, 17, 3)








x= Conv3D(
    filters=3,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p1_cnn_3d'
)(x) # output shape (48, 17, 17, 3)
x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='p1_mp_3d'
)(x) # output shape (24, 17, 17, 3)
x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p2_cnn_3d'
)(x) # output shape (22, 17, 17, 16)
x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p21_cnn_3d'
)(x) # output shape (20, 17, 17, 16)
x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='p2_mp_3d'
)(x) # output shape (10, 17, 17, 16)
x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p3_cnn_2d'
)(x) # output shape (10, 15, 15, 32)
x= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='p31_cnn_2d'
)(x) # output shape (10, 13, 13, 16)
x= Dropout(
    rate=0.1,
    name='do_1'
)(x)








x= Flatten(name='flat')(x)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(x)
