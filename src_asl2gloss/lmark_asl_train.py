from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam
# from json import load as jsonload

# from .lmark_constant import TOTAL_TRAIN_FILE, TRAIN_BATCH, PROJ_ROOT
from .lmark_constant import TOTAL_TRAIN_FILE, TRAIN_BATCH


from .modelasl2gloss_layers import data_in, data_out
from .lmark_essential_draw import getdata

if __name__=="__main__":
    model: Model= Model(
        inputs=data_in,
        outputs=[data_out]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={"batch_class": sparse_categorical_crossentropy},
        # loss=sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    # print(f"TOTAL_TRAIN_FILE//TRAIN_BATCH {TOTAL_TRAIN_FILE//TRAIN_BATCH}")
    model.fit(
        getdata(),
        epochs=7,
        steps_per_epoch=TOTAL_TRAIN_FILE//TRAIN_BATCH
    )
    # wlasl_ready= {}
    # with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
    #     wlasl_ready= jsonload(f)
    # print(f"type {type(wlasl_ready['train'][0]['video_id'])}")
    # print(f"type {type(wlasl_ready['train'][0]['gloss_id'])}")
