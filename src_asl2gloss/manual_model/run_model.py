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
    {
        KEY_G: 'cousin',
        KEY_LANDMARK: [
            'cousin_13640_00047_001_059_from_gt_p1',
            'cousin_65415_00102_025_043_from_gt_p1',
            'cousin_68592_00116_005_031_from_gt_p1',
            'cousin_13647_00135_022_045_from_gt_p1',
            'cousin_13648_00144_021_040_from_gt_p1',
            'cousin_13632_00161_027_052_from_gt_p2',
            'cousin_13633_00175_043_067_from_gt_p1',
            'cousin_67535_00187_017_032_from_gt_p1',
        ]
    },
    {
        KEY_G: 'mine_my',
        KEY_LANDMARK: [
            'mine_my_37472_00001_001_026_from_gt_p1',
            'mine_my_37474_00016_010_093_from_gt_p1',
            'mine_my_37477_00123_031_049_from_gt_p1',
            'mine_my_36139_00137_012_050_from_gt_p1',
            'mine_my_36145_00214_018_048_from_gt_p1',
            'mine_my_36146_00246_029_049_from_gt_p1',
            'mine_my_37464_00264_004_033_from_gt_p1',
            'mine_my_69404_00293_015_043_from_gt_p1',
        ]
    },
    {
        KEY_G: 'me_i',
        KEY_LANDMARK: [
            'me_i_35267_00001_001_035_from_gt_p1',
            'me_i_35267_00003_001_035_from_gt_p2',
            'me_i_35544_00026_018_042_from_gt_p1',
            'me_i_35545_00048_024_041_from_gt_p1',
            'me_i_35546_00060_004_058_from_gt_p1',
            'me_i_35547_00133_021_055_from_gt_p1',
            'me_i_35548_00254_001_030_from_gt_p1',
            'me_i_me_i_0001_00283_018_097_from_gt_p1',
        ]
    },
    {
        KEY_G: 'stomach',
        KEY_LANDMARK: [
            'stomach_54870_00001_017_049_from_gt_p1',
            'stomach_54885_00018_009_042_from_gt_p1',
            'stomach_66560_00039_015_034_from_gt_p1',
            'stomach_54878_00055_031_063_from_gt_p1',
            'stomach_54880_00072_001_051_from_gt_p1',
            'stomach_54881_00117_006_019_from_gt_p1',
            'stomach_54883_00130_001_060_from_gt_p1',
            'stomach_54884_00196_013_040_from_gt_p1',
        ]
    },
    {
        KEY_G: 'have',
        KEY_LANDMARK: [
            'have_69088_00030_001_032_from_gt_p1',
            'have_26776_00064_012_045_from_gt_p1',
            'have_26757_00085_001_035_from_gt_p1',
            'have_68069_00124_030_064_from_gt_p1',
            'have_70221_00149_027_055_from_gt_p1',
            'have_68984_00174_013_040_from_gt_p1',
            'have_26775_00196_014_043_from_gt_p1',
            'have_26766_00225_011_035_from_gt_p1',
        ]
    },
    {
        KEY_G: 'need',
        KEY_LANDMARK: [
            'need_37891_00026_024_050_from_gt_p1',
            'need_37892_00045_017_046_from_gt_p1',
            'need_68544_00075_004_036_from_gt_p2',
            'need_37885_00091_001_055_from_gt_p1',
            'need_37887_00164_017_056_from_gt_p1',
            'need_37889_00222_017_036_from_gt_p1',
            'need_37881_00239_028_057_from_gt_p2',
            'need_67925_00267_013_039_from_gt_p1',
        ]
    },
    {
        KEY_G: 'see',
        KEY_LANDMARK: [
            'see_50125_00001_001_030_from_gt_p1',
            'see_68444_00030_023_050_from_gt_p1',
            'see_50128_00061_020_033_from_gt_p1',
            'see_50107_00074_017_053_from_gt_p1',
            'see_50120_00117_039_059_from_gt_p1',
            'see_67178_00135_010_025_from_gt_p1',
            'see_50123_00143_001_055_from_gt_p1',
            'see_50126_00216_034_050_from_gt_p1',
        ]
    },
    {
        KEY_G: 'feel',
        KEY_LANDMARK: [
            'feel_67653_00001_012_029_from_gt_p1',
            'feel_21434_00013_018_059_from_gt_p1',
            'feel_69319_00042_023_056_from_gt_p1',
            'feel_21425_00063_001_049_from_gt_p1',
            'feel_21438_00094_021_033_from_gt_p1',
            'feel_21439_00106_023_050_from_gt_p1',
            'feel_21441_00153_013_048_from_gt_p1',
            'feel_21436_00202_001_053_from_gt_p1',
        ]
    },
    {
        KEY_G: 'hurt',
        KEY_LANDMARK: [
            'hurt_28450_00039_029_058_from_gt_p1',
            'hurt_28445_00068_001_055_from_gt_p1',
            'hurt_28441_00141_016_042_from_gt_p1',
            'hurt_28443_00160_001_053_from_gt_p1',
            'hurt_28444_00218_001_053_from_gt_p1',
            'hurt_28454_00276_025_045_from_gt_p1',
            'hurt_28452_00294_026_047_from_gt_p1',
            'hurt_28438_00314_001_033_from_gt_p1',
        ]
    },
    {
        KEY_G: 'fever',
        KEY_LANDMARK: [
            'fever_78219_00001_007_043_from_gt_p1',
            'fever_78220_00036_007_028_from_gt_p1',
            'fever_78221_00056_002_019_from_gt_p1',
            'fever_78222_00068_004_014_from_gt_p1',
            'fever_78223_00079_005_089_from_gt_p2',
            'fever_78224_00196_023_116_from_gt_p1',
            'fever_78225_00339_001_050_from_gt_p1',
            'fever_78226_00377_024_050_from_gt_p1',
        ]
    },
    {
        KEY_G: 'dizzy',
        KEY_LANDMARK: [
            'dizzy_16980_00001_001_091_from_gt_p1',
            'dizzy_16982_00109_017_049_from_gt_p1',
            'dizzy_16986_00126_001_057_from_gt_p1',
            'dizzy_16987_00167_023_058_from_gt_p1',
            'dizzy_16988_00198_017_053_from_gt_p1',
            'dizzy_78227_00233_022_052_from_gt_p1',
            'dizzy_78228_00265_048_160_from_gt_p1',
            'dizzy_78229_00380_019_130_from_gt_p1',
        ]
    },
    {
        KEY_G: 'headache',
        KEY_LANDMARK: [
            'headache_26832_00001_014_038_from_gt_p1',
            'headache_26835_00013_031_061_from_gt_p1',
            'headache_67747_00045_014_028_from_gt_p1',
            'headache_26846_00082_029_061_from_gt_p1',
            'headache_26839_00099_001_046_from_gt_p1',
            'headache_26838_00176_001_041_from_gt_p1',
            'headache_26841_00199_020_038_from_gt_p1',
            'headache_65881_00213_014_032_from_gt_p1',
        ]
    },
    {
        KEY_G: 'doctor',
        KEY_LANDMARK: [
            'doctor_17007_00035_001_043_from_gt_p2',
            'doctor_70049_00067_026_054_from_gt_p1',
            'doctor_17017_00096_024_058_from_gt_p2',
            'doctor_65504_00119_015_041_from_gt_p2',
            'doctor_17023_00136_022_048_from_gt_p1',
            'doctor_17014_00155_029_053_from_gt_p1',
            'doctor_17020_00171_001_055_from_gt_p2',
            'doctor_17022_00242_028_064_from_gt_p2',
        ]
    },
]
def main() -> None:
    print('hello world')
if __name__=="__main__":
    main()
