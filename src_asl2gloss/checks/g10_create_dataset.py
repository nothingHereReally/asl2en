from os.path import exists
from os import makedirs

from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, imwrite


from ..lmark_essentials import getSkeletonFrames
from ..lmark_constant import IMG_SIZE, PROJ_ROOT, WLASL_VID_DIR, wlasl_READY_10


if __name__=='__main__':
    # for train in wlasl_READY_10['train']:
    #     print(f"{train['video_id']} ---- {train['gloss_id']}")
    # for val in wlasl_READY_10['val']:
    #     print(f"{val['video_id']} ---- {val['gloss_id']}")
    # for test in wlasl_READY_10['test']:
    #     print(f"{test['video_id']} ---- {test['gloss_id']}")


    shape_vidBatch: tuple= (IMG_SIZE, IMG_SIZE, 3)
    for train in wlasl_READY_10['train']:
        vidfile: str= f"{WLASL_VID_DIR}{train['video_id']}.mp4"
        if exists(vidfile):
            try:
                vid: VideoCapture= VideoCapture(vidfile)
                if vid.isOpened():
                    qframes: int= int(vid.get(CAP_PROP_FRAME_COUNT))
                    if 0<qframes:
                        makedirs(f"{PROJ_ROOT}dataset/wlasl_dataset/ins_images/{train['video_id']}", exist_ok=True)
                        for i in range(qframes):
                            isNotEmpty, insFrame= vid.read()
                            if isNotEmpty and 0<len(insFrame):
                                imwrite('sldjs', insFrame)
            except Exception as e:
                del e

