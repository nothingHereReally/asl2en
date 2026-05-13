# best model yet: model_v14 then model_v10 and model_v12
# best model
# 1) model v30 (83.8% train/val/test g22)
# 2) model v23 (81% testing g10)
# 3) model v22 (60% testing g10)
# 4) model v15 (57% testing g10)
# 5) model v14 (56% testing g10)
# 6) model v12 (4% testing)
# 7) model v10 (4% testing)
# from os.path import exists
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam
# from keras.src.saving import load_model

from .lmark_constant import EPOCHS, PROJ_ROOT, TRAIN_STEPS, VAL_STEPS
from .lmark_essentials import getdata_landmark
from .model_layers import data_in, data_out
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
        x=getdata_landmark(trainVal='train'),
        epochs=EPOCHS,
        callbacks=[d_lr, sTraining, tf_board],
        validation_data=getdata_landmark(trainVal='val'),
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VAL_STEPS,
        validation_freq=1
    )
    print(f"proj_root {PROJ_ROOT}")
    print(f"model quantity of outputs {model.output_shape[-1]}")
    model.save(f"{PROJ_ROOT}model/aslvid2gloss_v30.keras")
    # loadModel= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v30.keras")
