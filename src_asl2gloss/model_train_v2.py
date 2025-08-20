# best model yet: model_v14 then model_v10 and model_v12
# best model top 4
# 1) model v15 (57% testing)
# 2) model v14 (56% testing)
# 3) model v12 (4% testing)
# 4) model v10 (4% testing)
# from os.path import exists
from typing import Any
# from keras.src.losses import sparse_categorical_crossentropy
# from keras.src.models import Model
# from keras.src.optimizers import Adam
from keras.src.saving import load_model

# from .lmark_constant import EPOCHS, PROJ_ROOT, TRAIN_STEPS, VAL_STEPS
from .lmark_constant_v2 import EPOCHS, PROJ_ROOT, TRAIN_STEPS, VAL_STEPS
from .lmark_g10_essentials_v2 import getdata
# from .model_layers_v2 import data_in, data_out
from .model_callbacks_v2 import d_lr, sTraining, tf_board


    

if __name__=="__main__":
    # model: Model= Model(
    #     inputs=data_in,
    #     outputs=data_out
    # )
    # model.compile(
    #     optimizer=Adam(learning_rate=0.0001),
    #     loss=sparse_categorical_crossentropy,
    #     metrics=['accuracy']
    # )
    model: Any= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v16.keras")
    model.summary()
    model.fit(
        x=getdata(TrainVal='train'),
        epochs=EPOCHS,
        callbacks=[d_lr, sTraining, tf_board],
        validation_data=getdata(TrainVal='val'),
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        validation_freq=1
    )
    print(f"proj_root {PROJ_ROOT}")
    model.save(f"{PROJ_ROOT}model/aslvid2gloss_v16.1.keras")
    # loadModel= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v16.1.keras")
