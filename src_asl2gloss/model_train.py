# best model yet: model_v10 and model_v12




from math import ceil
from os.path import exists
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from numpy import argmax, float32

from .lmark_constant import TOTAL_TRAIN_FILE, TRAIN_BATCH, PROJ_ROOT
from .lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, TOTAL_TRAIN_FILE, TRAIN_BATCH, WLASL_VID_DIR, wlasl_READY
from .lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, WLASL_VID_DIR, wlasl_READY


from .model_layers import data_in, data_out
from .lmark_essentials import getSkeletonFrames, getdata
from .lmark_essentials import getSkeletonFrames
from .checks.test_asl_vid2gloss_model import test


    

if __name__=="__main__":
    test(f"{PROJ_ROOT}model/aslvid2gloss_v9.keras")
    # model: Model= Model(
    #     inputs=data_in,
    #     outputs=[data_out]
    # )
    # d_lr= ReduceLROnPlateau(
    #     monitor='accuracy',
    #     factor=0.1,
    #     mode='max',
    #     patience=2,
    #     min_lr=1.0e-7
    # )
    # sTraining= EarlyStopping(
    #     monitor='loss',
    #     min_delta=0.001,
    #     patience=2,
    #     mode='min'
    # )
    # model.compile(
    #     optimizer=Adam(learning_rate=0.0001),
    #     loss=sparse_categorical_crossentropy,
    #     metrics=['accuracy']
    # )
    # model.summary()
    # # print(f"TOTAL_TRAIN_FILE//TRAIN_BATCH {TOTAL_TRAIN_FILE//TRAIN_BATCH}")
    # model.fit(
    #     getdata(),
    #     epochs=12,
    #     steps_per_epoch=int(ceil(TOTAL_TRAIN_FILE/TRAIN_BATCH)),
    #     callbacks=[d_lr, sTraining]
    # )
    # print(f"proj_root {PROJ_ROOT}")
    # model.save(f"{PROJ_ROOT}model/aslvid2gloss_v9.keras")
    # # loadModel= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v9.keras")
    # # loadModel.summary()
    # # shouldBeBook= loadModel.predict(getSkeletonFrames(f"{PROJ_ROOT}dataset/wlasl_dataset/videos/07092.mp4").reshape((1, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3)).astype(float32)/255.0)
    # # print(f"{wlasl_READY['label_id2gloss'][argmax(shouldBeBook[0], axis=-1)]} --> accuracy {shouldBeBook[0][argmax(shouldBeBook[0], axis=-1)]*100}%")
