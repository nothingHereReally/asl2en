from typing import Generator
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, cvtColor, destroyAllWindows, imwrite
from json import load as jsonload
from os.path import exists
from mediapipe.python.solutions.holistic import Holistic
from numpy import array, dtype, ndarray, uint16, uint8, zeros
from random import shuffle

from .lmark_essential_draw import drawFacePoseHand, getSkeletonFrames, getdata
from .lmark_constant import IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, TRAIN_BATCH, WLASL_VID_DIR



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

