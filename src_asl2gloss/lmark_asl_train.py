from os.path import exists
# from keras.src.losses import sparse_categorical_crossentropy
# from keras.src.models import Model
# from keras.src.optimizers import Adam
from keras.src.saving import load_model
from numpy import argmax, float32

# from .lmark_constant import TOTAL_TRAIN_FILE, TRAIN_BATCH, PROJ_ROOT
# from .lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, TOTAL_TRAIN_FILE, TRAIN_BATCH, WLASL_VID_DIR, wlasl_READY
from .lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, WLASL_VID_DIR, wlasl_READY


# from .modelasl2gloss_layers import data_in, data_out
# from .lmark_essential_draw import getSkeletonFrames, getdata
from .lmark_essential_draw import getSkeletonFrames


def test(modelFile: str) -> None:
    if exists(modelFile):
        loadedModel= load_model(modelFile)
        correct: int= 0
        print(f"testing over {len(wlasl_READY['test'])} files, processing...")
        for cur, i in zip(range(len(wlasl_READY['test'])), wlasl_READY['test']):
            print(f"current in progress idx {cur}")
            out: list= loadedModel.predict(getSkeletonFrames(
                f"{WLASL_VID_DIR}{i['video_id']}.mp4"
            ).reshape((1, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3)).astype(float32)/255.0)
            out= out[0]
            if int(argmax(out, axis=-1))==int(i['gloss_id']):
                correct +=1
                print(f"correct( should be {wlasl_READY['label_id2gloss'][i['gloss_id']]} ) calculated be {wlasl_READY['label_id2gloss'][argmax(out, axis=-1)]} {out[argmax(out, axis=-1)]*100}%")
            else:
                print(f"INcorrect( should be {wlasl_READY['label_id2gloss'][i['gloss_id']]} ) calculated be {wlasl_READY['label_id2gloss'][argmax(out, axis=-1)]} {out[argmax(out, axis=-1)]*100}% ________ {i['video_id']}")
        print(f"correct {correct} / {len(wlasl_READY['test'])} = {correct/float(len(wlasl_READY['test']))*100}%")
    else:
        print(f"model file: {modelFile} does not exist")
    

if __name__=="__main__":
    test(f"{PROJ_ROOT}model/aslvid2gloss_v1.keras")
    # model: Model= Model(
    #     inputs=data_in,
    #     outputs=[data_out]
    # )
    # model.compile(
    #     optimizer=Adam(learning_rate=0.001),
    #     loss={"batch_class": sparse_categorical_crossentropy},
    #     # loss=sparse_categorical_crossentropy,
    #     metrics=['accuracy']
    # )
    # model.summary()
    # # print(f"TOTAL_TRAIN_FILE//TRAIN_BATCH {TOTAL_TRAIN_FILE//TRAIN_BATCH}")
    # model.fit(
    #     getdata(),
    #     epochs=7,
    #     steps_per_epoch=TOTAL_TRAIN_FILE//TRAIN_BATCH
    # )
    # print(f"proj_root {PROJ_ROOT}")
    # model.save(f"{PROJ_ROOT}model/aslvid2gloss_v1.keras")
    # loadModel= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v1.keras")
    # shouldBeBook= loadModel.predict(getSkeletonFrames(f"{PROJ_ROOT}dataset/wlasl_dataset/videos/07092.mp4").reshape((1, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3)).astype(float32)/255.0)
    # print(f"{wlasl_READY['label_id2gloss'][argmax(shouldBeBook[0], axis=-1)]} --> accuracy {shouldBeBook[0][argmax(shouldBeBook[0], axis=-1)]*100}%")
