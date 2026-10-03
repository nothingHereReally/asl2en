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
    {
        KEY_G: 'drink',
        KEY_LANDMARK: [
            'drink_17724_00020_001_030_from_gt_p2',
            'drink_68538_00030_032_055_from_gt_p1',
            'drink_69302_00065_022_057_from_gt_p2',
            'drink_17729_00107_020_048_from_gt_p1',
            'drink_17725_00162_020_045_from_gt_p1',
            'drink_17726_00177_023_062_from_gt_p1',
            'drink_17727_00195_019_053_from_gt_p1',
            'drink_17727_00198_019_053_from_gt_p2',
        ]
    },
    {
        KEY_G: 'computer',
        KEY_LANDMARK: [
            'computer_12306_00002_001_052_from_gt_p2',
            'computer_68028_00055_023_063_from_gt_p2',
            'computer_12331_00183_024_051_from_gt_p1',
            'computer_12331_00189_024_051_from_gt_p2',
            'computer_12312_00237_047_081_from_gt_p1',
            'computer_12326_00262_013_073_from_gt_p1',
            'computer_computer_0004_00335_019_086_from_gt_p1',
            'computer_computer_0005_00413_013_082_from_gt_p1',
        ]
    },
    {
        KEY_G: 'before',
        KEY_LANDMARK: [
            'before_05730_00001_008_025_from_gt_p1',
            'before_05733_00013_025_083_from_gt_p1',
            'before_05729_00068_013_036_from_gt_p1',
            'before_05727_00077_024_055_from_gt_p1',
            'before_05739_00089_001_052_from_gt_p1',
            'before_05750_00178_036_058_from_gt_p1',
            'before_05744_00200_024_047_from_gt_p1',
            'before_65167_00209_015_032_from_gt_p1',
        ]
    },
]
def main() -> None:
    print('hello world')
if __name__=="__main__":
    main()
