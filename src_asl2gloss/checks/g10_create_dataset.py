from os.path import exists
from os import makedirs

from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, imwrite
from numpy import ndarray


from ..lmark_essentials import getSkeletonFrames
from ..lmark_constant import PROJ_ROOT, WLASL_VID_DIR, wlasl_READY_10


if __name__=='__main__':
    # for train in wlasl_READY_10['train']:
    #     print(f"{train['video_id']} ---- {train['gloss_id']}")
    # for val in wlasl_READY_10['val']:
    #     print(f"{val['video_id']} ---- {val['gloss_id']}")
    # for test in wlasl_READY_10['test']:
    #     print(f"{test['video_id']} ---- {test['gloss_id']}")


    init_dir: str= f"{PROJ_ROOT}dataset/wlasl_dataset/"
    write_to: str= f"{init_dir}ins_images/"
    TrainValTest: list= ["train", "val", "test"]
    if exists(init_dir) and not exists(f"{write_to}"):
        makedirs(f"{write_to}")
        print(f"writing to {write_to}")
        for tvt in TrainValTest:
            print(f"processing {tvt}...")
            for trainValTest_ins in wlasl_READY_10[tvt]:
                vidfile: str= f"{WLASL_VID_DIR}{trainValTest_ins['video_id']}.mp4"
                if exists(vidfile):
                    try:
                        vid: VideoCapture= VideoCapture(vidfile)
                        if vid.isOpened():
                            images: ndarray= getSkeletonFrames(
                                fpath_vid=vidfile,
                                isSingleImg=False,
                                initGT=0,
                                TqFRAMES=int(vid.get(CAP_PROP_FRAME_COUNT))
                            )[0]
                            if 0<len(images):
                                makedirs(f"{write_to}{trainValTest_ins['video_id']}")
                                for i in range(len(images)):
                                    if i<9:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/0000{i+1}.png",
                                            img=images[i]
                                        )
                                    elif i<99:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/000{i+1}.png",
                                            img=images[i]
                                        )
                                    elif i<999:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/00{i+1}.png",
                                            img=images[i]
                                        )
                                    elif i<9999:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/0{i+1}.png",
                                            img=images[i]
                                        )
                                    else:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/{i+1}.png",
                                            img=images[i]
                                        )
                        vid.release()
                        del vid
                    except Exception as e:
                        print(f"err: {e}")
                        del e
    else:
        print(f"{init_dir} doesn't\nexist, please get the dataset 1st")
        print(f"or if do exist, please delete this\ndirectory {write_to}")

