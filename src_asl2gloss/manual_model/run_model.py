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
    {
        KEY_G: 'chair',
        KEY_LANDMARK: [
            'chair_09853_00002_001_020_from_gt_p2',
            'chair_09848_00018_018_060_from_gt_p2',
            'chair_09869_00051_040_065_from_gt_p1',
            'chair_09847_00114_001_052_from_gt_p1',
            'chair_68580_00166_013_040_from_gt_p1',
            'chair_70263_00188_033_081_from_gt_p1',
            'chair_68019_00219_017_047_from_gt_p1',
            'chair_09866_00280_028_064_from_gt_p1',
        ]
    },
    {
        KEY_G: 'go',
        KEY_LANDMARK: [
            'go_24948_00001_001_018_from_gt_p1',
            'go_68292_00022_019_038_from_gt_p2',
            'go_24965_00044_017_037_from_gt_p1',
            'go_24954_00062_001_061_from_gt_p1',
            'go_24857_00135_001_035_from_gt_p1',
            'go_69345_00160_015_039_from_gt_p1',
            'go_24941_00172_016_032_from_gt_p1',
            'go_24969_00182_027_060_from_gt_p1',
            'go_24970_00203_028_060_from_gt_p1',
            'go_24971_00221_021_039_from_gt_p2',
        ]
    },
    {
        KEY_G: 'clothes',
        KEY_LANDMARK: [
            'clothes_11326_00001_002_023_from_gt_p1',
            'clothes_11311_00021_027_053_from_gt_p1',
            'clothes_11314_00040_022_033_from_gt_p1',
            'clothes_11316_00051_022_049_from_gt_p1',
            'clothes_11305_00073_001_052_from_gt_p1',
            'clothes_68870_00128_012_035_from_gt_p2',
            'clothes_11328_00190_001_029_from_gt_p1',
            'clothes_11329_00215_001_028_from_gt_p1',
        ]
    },
    {
        KEY_G: 'who',
        KEY_LANDMARK: [
            'who_63240_00001_025_044_from_gt_p1',
            'who_63228_00025_013_029_from_gt_p1',
            'who_63229_00035_019_032_from_gt_p1',
            'who_63234_00048_025_043_from_gt_p1',
            'who_63232_00109_065_116_from_gt_p1',
            'who_70380_00161_001_027_from_gt_p1',
            'who_who_0003_00180_020_072_from_gt_p1',
            'who_who_0005_00238_018_068_from_gt_p1',
        ]
    },
    {
        KEY_G: 'candy',
        KEY_LANDMARK: [
            'candy_08923_00001_001_039_from_gt_p1',
            'candy_70326_00045_067_099_from_gt_p1',
            'candy_08929_00062_020_041_from_gt_p1',
            'candy_68790_00095_074_105_from_gt_p2',
            'candy_08916_00140_035_062_from_gt_p1',
            'candy_08919_00162_027_050_from_gt_p1',
            'candy_08921_00171_006_035_from_gt_p1',
            'candy_08927_00218_015_046_from_gt_p1',
        ]
    },
]
def main() -> None:
    print('hello world')
if __name__=="__main__":
    main()
