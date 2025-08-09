from os.path import exists
from keras.src.saving import load_model
from numpy import argmax, float32

from ..lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, WLASL_VID_DIR, wlasl_READY_10


from ..lmark_essentials import getSkeletonFrames


def test(modelFile: str) -> None:
    if exists(modelFile):
        loadedModel= load_model(modelFile)
        correct: int= 0
        total: int= 0
        print(f"testing over {len(wlasl_READY_10['test'])} files, processing...")
        for cur, i in zip(range(len(wlasl_READY_10['test'])), wlasl_READY_10['test']):
            print(f"current in progress idx {cur}")
            try:
                out: list= loadedModel.predict(getSkeletonFrames(
                    f"{WLASL_VID_DIR}{i['video_id']}.mp4"
                )[0].reshape((1, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3)).astype(float32)/255.0)
                out= out[0]
                if int(argmax(out, axis=-1))==int(i['gloss_id']):
                    correct +=1
                    print(f"correct( should be {wlasl_READY_10['label_id2gloss'][i['gloss_id']]} ) calculated be {wlasl_READY_10['label_id2gloss'][argmax(out, axis=-1)]} {out[argmax(out, axis=-1)]*100}%")
                else:
                    print(f"INcorrect( should be {wlasl_READY_10['label_id2gloss'][i['gloss_id']]} ) calculated be {wlasl_READY_10['label_id2gloss'][argmax(out, axis=-1)]} {out[argmax(out, axis=-1)]*100}% ________ {i['video_id']}")
                total+= 1
            except FileExistsError as e:
                del e
                print("skipping due to on all images had less than 60% has a hand")
        # print(f"correct {correct} / {len(wlasl_READY_10['test'])} = {correct/float(len(wlasl_READY_10['test']))*100}%")
        print(f"correct {correct} / {total} = {correct/total*100}%")
    else:
        print(f"model file: {modelFile} does not exist")
    

if __name__=="__main__":
    test(f"{PROJ_ROOT}model/aslvid2gloss_v14.keras")

