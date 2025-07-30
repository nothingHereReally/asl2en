from os.path import exists
from keras.src.saving import load_model
from numpy import argmax, float32

from ..lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, WLASL_VID_DIR, wlasl_READY


from ..lmark_essential_draw import getSkeletonFrames


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

