# best model yet: model_v14 then model_v10 and model_v12
from math import ceil
# from os.path import exists
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam
# from keras.src.saving import load_model

from .lmark_constant import PROJ_ROOT, T10_TRAIN, T10_VAL, TRAIN_BATCH
from .lmark_essentials import getdata_10
from .model_layers import data_in, data_out
from .model_callbacks import d_lr, sTraining, tf_board


    

if __name__=="__main__":
    model: Model= Model(
        inputs=data_in,
        outputs=data_out
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        x=getdata_10(),
        epochs=1,
        callbacks=[d_lr, sTraining, tf_board],
        validation_data=getdata_10(isSimg=False, TrainVal='val'),
        steps_per_epoch=int(ceil((T10_TRAIN*2*7)/TRAIN_BATCH)),
        validation_steps=int(ceil(T10_VAL/TRAIN_BATCH)),
        validation_freq=1
    )
    print(f"proj_root {PROJ_ROOT}")
    model.save(f"{PROJ_ROOT}model/aslvid2gloss_v14.keras")
    # loadModel= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v14.keras")
