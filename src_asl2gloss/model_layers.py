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
    name='cnn2d_1'
)(data_in)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_1'
)(x)
x= Conv3D(
    filters=8,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn3d_1'
)(x)
x= MaxPooling3D(
    pool_size=(2,1,1),
    padding='valid',
    data_format='channels_last',
    name='mp3d_1'
)(x)
x= Dropout(
    rate=0.1,
    name='do_1'
)(x)




x= Conv3D(
    filters=16,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_2'
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_2'
)(x)
x= Conv3D(
    filters=16,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn3d_2'
)(x)
# x= MaxPooling3D(
#     pool_size=(2,1,1),
#     padding='valid',
#     data_format='channels_last',
#     name='mp3d_2'
# )
x= Dropout(
    rate=0.1,
    name='do_2'
)(x)




x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_3'
)(x)
x= MaxPooling3D(
    pool_size=(1,2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_3'
)(x)
x= Conv3D(
    filters=32,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn3d_3'
)(x)
x= Dropout(
    rate=0.1,
    name='do_3'
)(x)




x= Conv3D(
    filters=32,
    kernel_size=(1,3,3),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_4'
)(x)
x= MaxPooling3D(
    pool_size=(1,3,3),
    padding='valid',
    data_format='channels_last',
    name='mp2d_4'
)(x)
x= Conv3D(
    filters=32,
    kernel_size=(3,1,1),
    strides=(1,1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn3d_4'
)(x)
x= Flatten(name='flat')(x)


data_out = Dense(TOTAL_GLOSS_UNIQ, activation=softmax, name='batch_class')(x)
