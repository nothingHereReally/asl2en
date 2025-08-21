from math import ceil
from os.path import exists, join as pjoin
from typing import Any, Generator
from keras.src.saving import load_model
from numpy import argmax, float32, load, ndarray, uint16, zeros

from ..lmark_constant_v2 import LM_NPZ_DIR, LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME, T10_TEST, wlasl_READY_10, PROJ_ROOT


def getdata(TrainVal: str= 'test', batch: int=4) -> Generator[tuple, None, None]:
    # glossDist= {
    #     0: {
    #         'gloss_id': 0,
    #         'quantity': int(on this gloss id( ie. gloss_id=0 is book ) how many training was done),
    #         'video_id': list(all video_id that training was done, later be processed as unique),
    #         'vid_q_uniq': int(quantity of video_id that are unique, ie. above right after processed unique)
    #       },
    #     1: {...},
    #     2: {...},
    #     ...
    #     9: {...},
    #     'split': str(  'train'|'val'  )
    #     'split_size': int(quantity of video_id on 'train'|'val' ie. len(wlasl_READY_10[TrainVal]))
    # }
    # print(len(wlasl_READY_10['label_id2gloss'])) # correct, it exist
    init_IDXbatch: int= 0
    while True:
        batch_vids: ndarray= zeros(
            (batch, QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
            dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)
        for i in range(batch):
            idx_USE: int= (init_IDXbatch +i) if (init_IDXbatch +i)<len(wlasl_READY_10[TrainVal]) else (init_IDXbatch +i-len(wlasl_READY_10[TrainVal]))
            vid_lmarkNPZ_file: str= str(pjoin(LM_NPZ_DIR, f"{wlasl_READY_10[TrainVal][idx_USE]['file']}.npy"))
            if exists(vid_lmarkNPZ_file):
                with open(vid_lmarkNPZ_file, 'rb') as f:
                    gloss_id: int= int(wlasl_READY_10[TrainVal][idx_USE]['gloss_id'])
                    batch_vids[i]= load(f)
                    batch_class[i]= gloss_id
            else:
                raise FileNotFoundError(f"file {vid_lmarkNPZ_file} does not exist")
        init_IDXbatch+= batch


        if len(wlasl_READY_10[TrainVal])<=init_IDXbatch:
            init_IDXbatch-= int(len(wlasl_READY_10[TrainVal]))
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))
    

if __name__=="__main__":
    modelfile: str= f"{PROJ_ROOT}model/aslvid2gloss_v17.keras"
    batch_size: int= 4
    gdata: Generator= getdata(TrainVal='test', batch=batch_size)
    model: Any= load_model(modelfile)
    count_correct: int= 0
    for i in range(ceil(T10_TEST/batch_size)):
        batch_vid, batch_class= next(gdata)
        preds= model.predict(batch_vid)
        for pred, shouldBe in zip(preds, batch_class):
            pred_y: int= int(argmax(pred, axis=-1))
            if pred_y==shouldBe:
                print(f"correct {pred_y}( true is {shouldBe} ) ---- {wlasl_READY_10['label_id2gloss'][pred_y]} {pred[pred_y]}% {wlasl_READY_10['label_id2gloss'][shouldBe]}")
                count_correct+= 1
            else:
                print(f"INcorrect {pred_y}( true is {shouldBe} ) ---- {wlasl_READY_10['label_id2gloss'][pred_y]} {pred[pred_y]}% {wlasl_READY_10['label_id2gloss'][shouldBe]}")
    print(f"correct is {count_correct}/{T10_TEST} = {count_correct/T10_TEST*100}%")

