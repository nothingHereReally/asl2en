from pathlib import Path
from json import load as loadJson
import numpy as np
from time import sleep


PROJ_ROOT: Path= Path(__file__).parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
LANDMARK_dir: Path= DS_DIR /"ds_landmark"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_LANDMARK: str= 'landmark'
def main() -> None:
    ds_landmark: list
    with open(DS_DIR /"ds_landmark.json", 'r') as f:
        ds_landmark= loadJson(f)
    # can RAM handle if we load all numpy data
    lm_data_npy: list= []
    for a_gloss in ds_landmark:
        # a_gloss[KEY_G]
        for a_video in a_gloss[KEY_VIDS]:
            for a_start_end in a_video[KEY_LANDMARK]:
                for an_img_details in a_start_end[KEY_LANDMARK]:
                    with open(f"{LANDMARK_dir /a_start_end[KEY_PFOLDER] /an_img_details[KEY_FILE]}", 'rb') as f:
                        lm_data_npy.append(
                            np.load(f)
                        )
    for a_landmark in lm_data_npy:
        print(f"--> {a_landmark.shape} -- {a_landmark[-1]}")
        blah= a_landmark*2.1
        del blah
    print(f"loaded numpy files {len(lm_data_npy)} --> now be sleeping for 10.5 seconds")
    sleep(10.5)
    print(len(ds_landmark))
if __name__=="__main__":
    main()
