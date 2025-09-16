from os.path import exists
from os import makedirs
from json import load as jload, dump as jdump
from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, imwrite


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"


if __name__=='__main__':
    tmp_ready: dict= {}
    with open(f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.clean.json", 'r') as f:
        tmp_ready= jload(f)
    wlasl_clean: dict= tmp_ready.copy()
    del tmp_ready


    wlasl_tvt: dict= {
        'train': [],
        'val': [],
        'test': [],
        'label_id2gloss': [],
        'label_gloss2id': {}
    }
    for gloss, gloss_id in zip(wlasl_clean, range(len(wlasl_clean))):
        wlasl_tvt['label_id2gloss'].append( str(gloss['gloss']) )
        wlasl_tvt['label_gloss2id'][ str(gloss['gloss']) ]= gloss_id
        for video in gloss['instances']:
            wlasl_tvt[  video['split']  ].append({
                'gloss_id': int(gloss_id),
                'video_id': str(video['video_id'])
            })
    with open(f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.train_val_test.json", 'w') as f:
        jdump(wlasl_tvt, f, indent=4)


    # dataset/wlasl/raw_image/ is created for corrupted video files
    # be not a headache, due to here will just skip corrupted file
    # via try except on VideoCapture
    init_dir: str= f"{PROJ_ROOT}dataset/wlasl/"
    write_to: str= f"{init_dir}raw_image/"
    video_dir_from: str= f"{init_dir}videos/"
    TrainValTest: tuple= ("train", "val", "test")
    if exists(init_dir) and not exists(write_to) and exists(video_dir_from):
        makedirs(write_to)
        print(f"writing to {write_to}")
        for tvt in TrainValTest:
            # for tain, val, test ie. tvt

            print(f"processing {tvt}...\n")
            for trainValTest_ins in wlasl_tvt[tvt]:
                vidfile: str= f"{video_dir_from}{trainValTest_ins['video_id']}.mp4"
                if exists(vidfile):
                    try:
                        vid: VideoCapture= VideoCapture(vidfile)
                        oqFrames: int= int(vid.get(CAP_PROP_FRAME_COUNT))
                        if vid.isOpened() and 0<oqFrames:

                            # each video be having it's own folder
                            # video_id is set for name of folder
                            # ie. dataset/wlasl/raw_image/< video_id >/
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
                        print(f"error reading video: {e}")
                        del e
    else:
        print(f"{init_dir} doesn't\nexist, please get the dataset 1st")
        print(f"or if do exist, please delete this\ndirectory {write_to}")

