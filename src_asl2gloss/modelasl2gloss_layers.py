from keras.src.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, ReLU
from keras.src.activations import softmax
from numpy import float32


from .lmark_constant import QUANTITY_FRAME, IMG_SIZE, TOTAL_GLOSS_UNIQ


data_in= Input(
    shape=(QUANTITY_FRAME*IMG_SIZE, IMG_SIZE, 3),
    dtype=float32,
    name='batch_vid'
)



x= Conv2D(
    filters=8,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_1'
)(data_in)
x= MaxPooling2D(
    pool_size=(2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_1'
)(x)
x= Dropout(
    rate=0.2,
    name='do_1'
)(x)
x= Conv2D(
    filters=16,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_2'
)(x)
x= MaxPooling2D(
    pool_size=(2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_2'
)(x)
x= Dropout(
    rate=0.2,
    name='do_2'
)(x)
x= Conv2D(
    filters=24,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_3'
)(x)
x= MaxPooling2D(
    pool_size=(2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_3'
)(x)
x= Dropout(
    rate=0.2,
    name='do_3'
)(x)
x= Conv2D(
    filters=24,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_4'
)(x)
x= MaxPooling2D(
    pool_size=(2,2),
    padding='valid',
    data_format='channels_last',
    name='mp2d_4'
)(x)
x= Dropout(
    rate=0.2,
    name='do_4'
)(x)
x= Conv2D(
    filters=24,
    kernel_size=(3,3),
    strides=(1,1),
    padding='valid',
    data_format='channels_last',
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='cnn2d_5'
)(x)
# x= MaxPooling2D(
#     pool_size=(2,2),
#     padding='valid',
#     data_format='channels_last',
#     name='mp2d_5'
# )(x)
# x= Dropout(
#     rate=0.2,
#     name='do_5'
# )(x)
# x= Conv2D(
#     filters=24,
#     kernel_size=(3,3),
#     strides=(1,1),
#     padding='valid',
#     data_format='channels_last',
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     name='cnn2d_6'
# )(x)
# x= MaxPooling2D(
#     pool_size=(2,2),
#     padding='valid',
#     data_format='channels_last',
#     name='mp2d_6'
# )(x)
# x= Dropout(
#     rate=0.2,
#     name='do_6'
# )(x)
# x= Conv2D(
#     filters=24,
#     kernel_size=(3,3),
#     strides=(1,1),
#     padding='valid',
#     data_format='channels_last',
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     name='cnn2d_7'
# )(x)
# x= MaxPooling2D(
#     pool_size=(2,2),
#     padding='valid',
#     data_format='channels_last',
#     name='mp2d_7'
# )(x)
# x= Dropout(
#     rate=0.2,
#     name='do_7'
# )(x)


x= Flatten(
    name='flat_1'
)(x)
data_out= Dense(
    units=TOTAL_GLOSS_UNIQ,
    activation=softmax,
    name='batch_class'
)(x)
