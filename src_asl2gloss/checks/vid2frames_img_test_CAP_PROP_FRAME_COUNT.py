from json import load as jsonload
from os.path import exists

from cv2 import imwrite
from numpy import ndarray

from ..lmark_constant import PROJ_ROOT, WLASL_VID_DIR
from ..lmark_essential_draw import getSkeletonFrames


if __name__=="__main__":
    wlasl_ready: dict= {}
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
        wlasl_ready= jsonload(f)
        # wlasl_ready['train']
        # wlasl_ready['val']
        # wlasl_ready['test']
        # wlasl_ready['label_id2gloss']
        # wlasl_ready['label_gloss2id']
    # shuffle(wlasl_ready['train'])



    # vid= f"{WLASL_VID_DIR}15144.mp4" # orig 9 frames
    # if exists(vid):
    #     all_frames: ndarray= getSkeletonFrames(fpath_vid=vid)
    #     for i, img in zip(range(len(all_frames)), all_frames):
    #         if (i+1)<10:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/00{i+1}.png",
    #                 img=img
    #             )
    #         else:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/0{i+1}.png",
    #                 img=img
    #             )
    # else:
    #     print(f"file [{vid}] does not exist")


    # vid= f"{WLASL_VID_DIR}17732.mp4" # orig 55 frames
    # if exists(vid):
    #     all_frames: ndarray= getSkeletonFrames(fpath_vid=vid)
    #     for i, img in zip(range(len(all_frames)), all_frames):
    #         if (i+1)<10:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/00{i+1}.png",
    #                 img=img
    #             )
    #         else:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/0{i+1}.png",
    #                 img=img
    #             )
    # else:
    #     print(f"file [{vid}] does not exist")


    # vid= f"{WLASL_VID_DIR}12312.mp4" # orig 147 frames
    # if exists(vid):
    #     all_frames: ndarray= getSkeletonFrames(fpath_vid=vid)
    #     for i, img in zip(range(len(all_frames)), all_frames):
    #         if (i+1)<10:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/00{i+1}.png",
    #                 img=img
    #             )
    #         else:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/0{i+1}.png",
    #                 img=img
    #             )
    # else:
    #     print(f"file [{vid}] does not exist")


    # vid= f"{WLASL_VID_DIR}61742.mp4" # some frames/img corrupted
    # # can be read, frames should be 101 original, but can only read
    # # about 98 frames, now fixed
    # if exists(vid):
    #     all_frames: ndarray= getSkeletonFrames(fpath_vid=vid)
    #     for i, img in zip(range(len(all_frames)), all_frames):
    #         if (i+1)<10:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/00{i+1}.png",
    #                 img=img
    #             )
    #         else:
    #             imwrite(
    #                 filename=f"/tmp/wlasl_vid_test/0{i+1}.png",
    #                 img=img
    #             )
    # else:
    #     print(f"file [{vid}] does not exist")


    vid= f"{WLASL_VID_DIR}70053.mp4" # orig 247 frames
    if exists(vid):
        all_frames: ndarray= getSkeletonFrames(fpath_vid=vid)
        for i, img in zip(range(len(all_frames)), all_frames):
            if (i+1)<10:
                imwrite(
                    filename=f"/tmp/wlasl_vid_test/00{i+1}.png",
                    img=img
                )
            else:
                imwrite(
                    filename=f"/tmp/wlasl_vid_test/0{i+1}.png",
                    img=img
                )
    else:
        print(f"file [{vid}] does not exist")

