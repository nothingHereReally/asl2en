from typing import Generator
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, cvtColor, destroyAllWindows, imwrite
from json import load as jsonload
from os.path import exists
from mediapipe.python.solutions.holistic import Holistic
from numpy import array, dtype, ndarray, uint16, uint8, zeros
from random import shuffle

from .lmark_essential_draw import drawFacePoseHand, getSkeletonFrames
from .lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, TRAIN_BATCH, WLASL_VID_DIR


def getdata(batch: int=TRAIN_BATCH) -> Generator[tuple, None, None]:
    wlasl_ready: dict= {}
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
        wlasl_ready= jsonload(f)
        # wlasl_ready['train']
        # wlasl_ready['val']
        # wlasl_ready['test']
        # wlasl_ready['label_id2gloss']
        # wlasl_ready['label_gloss2id']
    shuffle(wlasl_ready['train'])
    current_idxTRAIN: int= 0
    while current_idxTRAIN<len(wlasl_ready['train']):
        batch_vids: ndarray= zeros((batch, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        batch_class: ndarray= zeros((batch), dtype=uint16)
        for i in range(current_idxTRAIN, current_idxTRAIN+batch):
            # batch_vids[i, :, :, :, :]= getSkeletonFrames(f"{WLASL_VID_DIR}{wlasl_ready['train'][i]['video_id']}.mp4")
            batch_vids[i]= getSkeletonFrames(f"{WLASL_VID_DIR}{wlasl_ready['train'][i]['video_id']}.mp4")
            batch_class[i]= wlasl_ready['train'][i]['gloss_id']
        yield (
            {"batch_vid": batch_vids},
            {"batch_class": batch_class}
        )
        current_idxTRAIN +=batch

if __name__=="__main__":
    gen: Generator= getdata()
    batch1= next(gen)
    for i in range(len(batch1[0]['batch_vid'][0])):
        if (i+1)<10:
            imwrite(f"/tmp/wlasl_vid_test/00{i+1}.png", batch1[0]['batch_vid'][0][i])
        else:
            imwrite(f"/tmp/wlasl_vid_test/0{i+1}.png", batch1[0]['batch_vid'][0][i])
    print(batch1[0]['batch_vid'].shape)
    print(batch1[0]['batch_vid'][0].shape)
    print(batch1[1]['batch_class'])
    # wlasl_ready: dict= {}
    # with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
    #     wlasl_ready= jsonload(f)
    #     # wlasl_ready['train']
    #     # wlasl_ready['val']
    #     # wlasl_ready['test']
    #     # wlasl_ready['label_id2gloss']
    #     # wlasl_ready['label_gloss2id']
    # # shuffle(wlasl_ready['train'])

