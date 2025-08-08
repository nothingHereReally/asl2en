# best model yet: model_v10 and model_v12
from math import ceil
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam
# from keras.src.saving import load_model

from .lmark_constant import TOTAL_GLOSS_UNIQ, TOTAL_TRAIN_FILE, TRAIN_BATCH, PROJ_ROOT
from .model_layers import data_in, data_out
from .lmark_essentials import getdata
from .model_callbacks import d_lr, sTraining, tf_board




if __name__=="__main__":
    model: Model= Model(
        inputs=data_in,
        outputs=data_out
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        x=getdata(),
        epochs=12,
        callbacks=[d_lr, sTraining, tf_board],
        validation_data=getdata(isSimg=False, TrainVal='val'),
        steps_per_epoch=int(ceil(TOTAL_TRAIN_FILE/TRAIN_BATCH)),
        validation_steps=int(TOTAL_GLOSS_UNIQ/TRAIN_BATCH),
        validation_freq=1
    )
    print(f"proj_root {PROJ_ROOT}")
    model.save(f"{PROJ_ROOT}model/aslvid2gloss_v13.keras")
    # loadModel= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v13.keras")
