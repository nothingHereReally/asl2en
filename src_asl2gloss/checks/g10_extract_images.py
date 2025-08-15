from os.path import exists
from os import makedirs
from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, imwrite


from ..lmark_constant import PROJ_ROOT, WLASL_VID_DIR, wlasl_READY_10


if __name__=='__main__':


    init_dir: str= f"{PROJ_ROOT}dataset/wlasl_dataset/"
    write_to: str= f"{init_dir}raw_images/"
    TrainValTest: list= ["train", "val", "test"]
    if exists(init_dir) and not exists(f"{write_to}"):
        makedirs(write_to)
        print(f"writing to {write_to}")
        for tvt in TrainValTest:
            print(f"processing {tvt}...")
            for trainValTest_ins in wlasl_READY_10[tvt]:
                vidfile: str= f"{WLASL_VID_DIR}{trainValTest_ins['video_id']}.mp4"
                if exists(vidfile):
                    try:
                        vid: VideoCapture= VideoCapture(vidfile)
                        oqFrames: int= int(vid.get(CAP_PROP_FRAME_COUNT))
                        if vid.isOpened() and 0<oqFrames:
                            makedirs(f"{write_to}{trainValTest_ins['video_id']}")
                            for i in range(oqFrames):
                                isNotEmpty, idvImg= vid.read()
                                if isNotEmpty and 0<len(idvImg):
                                    if i<9:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/0000{i+1}.png",
                                            img=idvImg
                                        )
                                    elif i<99:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/000{i+1}.png",
                                            img=idvImg
                                        )
                                    elif i<999:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/00{i+1}.png",
                                            img=idvImg
                                        )
                                    elif i<9999:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/0{i+1}.png",
                                            img=idvImg
                                        )
                                    else:
                                        imwrite(
                                            filename=f"{write_to}{trainValTest_ins['video_id']}/{i+1}.png",
                                            img=idvImg
                                        )
                        vid.release()
                        del vid
                    except Exception as e:
                        print(f"err: {e}")
                        del e
    else:
        print(f"{init_dir} doesn't\nexist, please get the dataset 1st")
        print(f"or if do exist, please delete this\ndirectory {write_to}")

