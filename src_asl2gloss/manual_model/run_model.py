from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
MODEL_LANDMARK_DIR: Path= DS_DIR /"ds_landmark"
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
KEY_SKELETON: str= 'skeleton'
# ---------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
VERIFIED_DS_DIR: list= [
    {
        KEY_G: 'book',
        KEY_LANDMARK: [
            'book_07071_00002_001_018_from_gt_p2',
            'book_07071_00004_001_018_from_gt_p2',
            'book_68011_00112_006_040_from_gt_p2',
            'book_68011_00122_006_040_from_gt_p3',
            'book_07076_00254_019_067_from_gt_p2',
            'book_07076_00264_019_067_from_gt_p3',
            'book_07097_00218_019_043_from_gt_p2',
            'book_07097_00220_019_043_from_gt_p2',
        ]
    },
]
def main() -> None:
    print('hello world')
if __name__=="__main__":
    main()
