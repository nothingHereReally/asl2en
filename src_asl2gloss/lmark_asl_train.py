from keras.src.layers import Conv3D, Dense, Flatten, Input, MaxPooling3D, Rescaling
from keras.src.activations import relu, sigmoid, softmax
from keras.src.models import Model
from numpy import float32

from .lmark_constant import IMG_SIZE, QUANTITY_FRAME, TOTAL_GLOSS_UNIQ


if __name__=="__main__":
    data_in= Input(
        shape=(QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
        dtype=float32,
        name='batch_vid'
    )
    x= Rescaling(
        scale=1./255,
        offset=0.0,
        dtype='float32',
        name='scale_0.0_1.0'
    )(data_in)



    x= Conv3D(
        filters=16,
        kernel_size=(QUANTITY_FRAME,3,3),
        strides=(1,1,1),
        padding='valid',
        activation=relu,
        dtype=float32,
        name='conv_1'
    )(x)
    # x= MaxPooling3D(
    #     pool_size=(5,5,5),
    #     strides=(3,3,3),
    #     padding='valid',
    #     dtype=float32,
    #     name='maxpool_1'
    # )(x)
    # x= Conv3D(
    #     filters=64,
    #     kernel_size=(3,3,3),
    #     strides=(1,1,1),
    #     padding='valid',
    #     activation=relu,
    #     dtype=float32,
    #     name='conv_2'
    # )(x)
    # x= MaxPooling3D(
    #     pool_size=(7,7,7),
    #     strides=(5,5,5),
    #     padding='valid',
    #     dtype=float32,
    #     name='maxpool_2'
    # )(x)


    x= Flatten()(x)
    x= Dense(
        units=16,
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


    model: Model= Model(
        inputs=data_in,
        outputs=[data_out]
    )
    model.summary()
