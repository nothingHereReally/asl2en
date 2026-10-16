from json import dump as writeJson
import numpy as np
from typing import Any
import cv2
from pathlib import Path


from src_asl2gloss.create_own_dataset.p012_investigate_hybrid_landmarks import drawFacePoseHand, MPH_fph


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"glasl"
VIDEO_DIR: Path= DS_DIR /"glasl_v4"
LANDMARK_DIR: Path= DS_DIR /"glasl_v4_landmark"
SKELETON_DIR: Path= DS_DIR /"glasl_v4_skeleton"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
IMG_SIZE: int= 158
VIDEO_LIST: list= [
    {
        KEY_G: 'book',
        KEY_VIDS: [
            'book_0001.mp4',
            'book_0002.mp4',
            'book_0003.mp4',
            'book_0004.mp4',
            'book_0005.mp4',
            'book_0006.mp4',
            'book_0007.mp4',
            'book_0008.mp4',
            'book_0009.mp4',
            'book_0010.mp4',
        ]
    },
    {
        KEY_G: 'drink',
        KEY_VIDS: [
            'drink_0001.mp4',
            'drink_0002.mp4',
            'drink_0003.mp4',
            'drink_0004.mp4',
            'drink_0005.mp4',
            'drink_0006.mp4',
            'drink_0007.mp4',
            'drink_0008.mp4',
            'drink_0009.mp4',
            'drink_0010.mp4',
        ]
    },
    {
        KEY_G: 'computer',
        KEY_VIDS: [
            'computer_0001.mp4',
            'computer_0002.mp4',
            'computer_0003.mp4',
            'computer_0004.mp4',
            'computer_0005.mp4',
            'computer_0006.mp4',
            'computer_0007.mp4',
            'computer_0008.mp4',
            'computer_0009.mp4',
            'computer_0010.mp4',
        ]
    },
    {
        KEY_G: 'before',
        KEY_VIDS: [
            'before_0001.mp4',
            'before_0002.mp4',
            'before_0003.mp4',
            'before_0004.mp4',
            'before_0005.mp4',
            'before_0006.mp4',
            'before_0007.mp4',
            'before_0008.mp4',
            'before_0009.mp4',
            'before_0010.mp4',
        ]
    },
    {
        KEY_G: 'chair',
        KEY_VIDS: [
            'chair_0001.mp4',
            'chair_0002.mp4',
            'chair_0003.mp4',
            'chair_0004.mp4',
            'chair_0005.mp4',
            'chair_0006.mp4',
            'chair_0007.mp4',
            'chair_0008.mp4',
            'chair_0009.mp4',
            'chair_0010.mp4',
        ]
    },
    {
        KEY_G: 'go',
        KEY_VIDS: [
            'go_0001.mp4',
            'go_0002.mp4',
            'go_0003.mp4',
            'go_0004.mp4',
            'go_0005.mp4',
            'go_0006.mp4',
            'go_0007.mp4',
            'go_0008.mp4',
            'go_0009.mp4',
            'go_0010.mp4',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_VIDS: [
            'clothes_0001.mp4',
            'clothes_0002.mp4',
            'clothes_0003.mp4',
            'clothes_0004.mp4',
            'clothes_0005.mp4',
            'clothes_0006.mp4',
            'clothes_0007.mp4',
            'clothes_0008.mp4',
            'clothes_0009.mp4',
            'clothes_0010.mp4',
        ]
    },
    {
        KEY_G: 'who',
        KEY_VIDS: [
            'who_0001.mp4',
            'who_0002.mp4',
            'who_0003.mp4',
            'who_0004.mp4',
            'who_0005.mp4',
            'who_0006.mp4',
            'who_0007.mp4',
            'who_0008.mp4',
            'who_0009.mp4',
            'who_0010.mp4',
        ]
    },
    {
        KEY_G: 'candy',
        KEY_VIDS: [
            'candy_0001.mp4',
            'candy_0002.mp4',
            'candy_0003.mp4',
            'candy_0004.mp4',
            'candy_0005.mp4',
            'candy_0006.mp4',
            'candy_0007.mp4',
            'candy_0008.mp4',
            'candy_0009.mp4',
            'candy_0010.mp4',
        ]
    },
    {
        KEY_G: 'cousin',
        KEY_VIDS: [
            'cousin_0001.mp4',
            'cousin_0002.mp4',
            'cousin_0003.mp4',
            'cousin_0004.mp4',
            'cousin_0005.mp4',
            'cousin_0006.mp4',
            'cousin_0007.mp4',
            'cousin_0008.mp4',
            'cousin_0009.mp4',
            'cousin_0010.mp4',
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_VIDS: [
            'mine_my_0001.mp4',
            'mine_my_0002.mp4',
            'mine_my_0003.mp4',
            'mine_my_0004.mp4',
            'mine_my_0005.mp4',
            'mine_my_0006.mp4',
            'mine_my_0007.mp4',
            'mine_my_0008.mp4',
            'mine_my_0009.mp4',
            'mine_my_0010.mp4',
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_VIDS: [
            'me_i_0001.mp4',
            'me_i_0002.mp4',
            'me_i_0003.mp4',
            'me_i_0004.mp4',
            'me_i_0005.mp4',
            'me_i_0006.mp4',
            'me_i_0007.mp4',
            'me_i_0008.mp4',
            'me_i_0009.mp4',
            'me_i_0010.mp4',
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_VIDS: [
            'stomach_0001.mp4',
            'stomach_0002.mp4',
            'stomach_0003.mp4',
            'stomach_0004.mp4',
            'stomach_0005.mp4',
            'stomach_0006.mp4',
            'stomach_0007.mp4',
            'stomach_0008.mp4',
            'stomach_0009.mp4',
            'stomach_0010.mp4',
        ]
    },
    {
        KEY_G: 'have',
        KEY_VIDS: [
            'have_0001.mp4',
            'have_0002.mp4',
            'have_0003.mp4',
            'have_0004.mp4',
            'have_0005.mp4',
            'have_0006.mp4',
            'have_0007.mp4',
            'have_0008.mp4',
            'have_0009.mp4',
            'have_0010.mp4',
        ]
    },
    {
        KEY_G: 'need',
        KEY_VIDS: [
            'need_0001.mp4',
            'need_0002.mp4',
            'need_0003.mp4',
            'need_0004.mp4',
            'need_0005.mp4',
            'need_0006.mp4',
            'need_0007.mp4',
            'need_0008.mp4',
            'need_0009.mp4',
            'need_0010.mp4',
        ]
    },
    {
        KEY_G: 'see',
        KEY_VIDS: [
            'see_0001.mp4',
            'see_0002.mp4',
            'see_0003.mp4',
            'see_0004.mp4',
            'see_0005.mp4',
            'see_0006.mp4',
            'see_0007.mp4',
            'see_0008.mp4',
            'see_0009.mp4',
            'see_0010.mp4',
        ]
    },
    {
        KEY_G: 'feel',
        KEY_VIDS: [
            'feel_0001.mp4',
            'feel_0002.mp4',
            'feel_0003.mp4',
            'feel_0004.mp4',
            'feel_0005.mp4',
            'feel_0006.mp4',
            'feel_0007.mp4',
            'feel_0008.mp4',
            'feel_0009.mp4',
            'feel_0010.mp4',
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_VIDS: [
            'hurt_0001.mp4',
            'hurt_0002.mp4',
            'hurt_0003.mp4',
            'hurt_0004.mp4',
            'hurt_0005.mp4',
            'hurt_0006.mp4',
            'hurt_0007.mp4',
            'hurt_0008.mp4',
            'hurt_0009.mp4',
            'hurt_0010.mp4',
        ]
    },
    {
        KEY_G: 'fever',
        KEY_VIDS: [
            'fever_0001.mp4',
            'fever_0002.mp4',
            'fever_0003.mp4',
            'fever_0004.mp4',
            'fever_0005.mp4',
            'fever_0006.mp4',
            'fever_0007.mp4',
            'fever_0008.mp4',
            'fever_0009.mp4',
            'fever_0010.mp4',
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_VIDS: [
            'dizzy_0001.mp4',
            'dizzy_0002.mp4',
            'dizzy_0003.mp4',
            'dizzy_0004.mp4',
            'dizzy_0005.mp4',
            'dizzy_0006.mp4',
            'dizzy_0007.mp4',
            'dizzy_0008.mp4',
            'dizzy_0009.mp4',
            'dizzy_0010.mp4',
        ]
    },
    {
        KEY_G: 'headache',
        KEY_VIDS: [
            'headache_0001.mp4',
            'headache_0002.mp4',
            'headache_0003.mp4',
            'headache_0004.mp4',
            'headache_0005.mp4',
            'headache_0006.mp4',
            'headache_0007.mp4',
            'headache_0008.mp4',
            'headache_0009.mp4',
            'headache_0010.mp4',
        ]
    },
    {
        KEY_G: 'doctor',
        KEY_VIDS: [
            'doctor_0001.mp4',
            'doctor_0002.mp4',
            'doctor_0003.mp4',
            'doctor_0004.mp4',
            'doctor_0005.mp4',
            'doctor_0006.mp4',
            'doctor_0007.mp4',
            'doctor_0008.mp4',
            'doctor_0009.mp4',
            'doctor_0010.mp4',
        ]
    }
]
def init_dirs() -> None:
    if LANDMARK_DIR.exists():
        raise NotImplementedError(f"Please delete {LANDMARK_DIR}, this will be created for you.")
    if SKELETON_DIR.exists():
        raise NotImplementedError(f"Please delete {SKELETON_DIR}, this will be created for you.")
    LANDMARK_DIR.mkdir()
    SKELETON_DIR.mkdir()
def process_videos_ds() -> tuple:
    ds_landmark: list= []
    ds_skeleton: list= []
    for a_gloss in VIDEO_LIST:
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS]:
            ds_landmark[-1][KEY_VIDS].append({
                KEY_FILE: a_video,
                KEY_LANDMARK: [],
            })
            ds_skeleton[-1][KEY_VIDS].append({
                KEY_FILE: a_video,
                KEY_SKELETON: [],
            })
            landmark_abs_folder: Path= LANDMARK_DIR /a_video[:-4]
            skeleton_abs_folder: Path= SKELETON_DIR /a_video[:-4]
            landmark_abs_folder.mkdir()
            skeleton_abs_folder.mkdir()
            video_data: cv2.VideoCapture= cv2.VideoCapture(f"{VIDEO_DIR /a_video}")
            for idx in range(int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))):
                has_img, image= video_data.read()
                skeleton_landmark_filename: str= str(idx+1).zfill(8)
                if has_img:
                    image= cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
                    mph_fph_result: Any= MPH_fph.process(image)
                    img_skeleton, lmark_face_pose_lr_hand= drawFacePoseHand(
                        img_write_to=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
                        lmark_mph=mph_fph_result,
                        orig_shape=image.shape,
                    )
                    ds_landmark[-1][KEY_VIDS][-1][KEY_LANDMARK].append({
                        KEY_FILE: f"{skeleton_landmark_filename}.npy",
                        KEY_FACE: mph_fph_result.face_landmarks != None,
                        KEY_POSE: mph_fph_result.pose_landmarks != None,
                        KEY_LHAND: mph_fph_result.left_hand_landmarks != None,
                        KEY_RHAND: mph_fph_result.right_hand_landmarks != None,
                    })
                    ds_skeleton[-1][KEY_VIDS][-1][KEY_SKELETON].append({
                        KEY_FILE: f"{skeleton_landmark_filename}.jpg",
                        KEY_FACE: mph_fph_result.face_landmarks != None,
                        KEY_POSE: mph_fph_result.pose_landmarks != None,
                        KEY_LHAND: mph_fph_result.left_hand_landmarks != None,
                        KEY_RHAND: mph_fph_result.right_hand_landmarks != None,
                    })
                    with open(f"{landmark_abs_folder /skeleton_landmark_filename}.npy", 'wb') as f:
                        np.save(f, lmark_face_pose_lr_hand)
                    cv2.imwrite(
                        f"{skeleton_abs_folder /skeleton_landmark_filename}.jpg",
                        img_skeleton,
                    )
    return (ds_landmark, ds_skeleton)
def main() -> None:
    init_dirs()
    ds_landmark, ds_skeleton= process_videos_ds()
    with open(f"{DS_DIR /"glasl_v4_landmark.json"}", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{DS_DIR /"glasl_v4_skeleton.json"}", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    main()
