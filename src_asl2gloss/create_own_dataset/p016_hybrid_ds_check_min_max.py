from pathlib import Path
from json import load as loadjson


KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
# -------------------
KEY_LANDMARK: str= 'landmark'
PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
# ---------------------
KEY_SPLIT: str= 'split'
KEY_TRAIN: str= 'train'
KEY_TEST: str= 'test'
ds_landmark: list= []
with open(f"{PROJ_ROOT /"dataset" /"clean_dataset" /"ds_landmark.json"}", "r") as f:
    ds_landmark= loadjson(f)


def min_images(a_gloss: dict) -> dict:
    details: dict= {
        'min': 999_999,
        'index_video': 0,
        'index_se': 0,
    }
    for idx_vid, a_video in enumerate(a_gloss[KEY_VIDS]):
        for idx_se, a_start_end_video in enumerate(a_video[KEY_LANDMARK]):
            if len(a_start_end_video[KEY_LANDMARK])<details['min']:
                details= {
                    'min': len(a_start_end_video[KEY_LANDMARK]),
                    'index_video': idx_vid,
                    'index_se': idx_se
                }

    return details
def max_images(a_gloss: dict) -> dict:
    details: dict= {
        'max': 0,
        'index_video': 0,
        'index_se': 0,
    }
    for idx_vid, a_video in enumerate(a_gloss[KEY_VIDS]):
        for idx_se, a_start_end_video in enumerate(a_video[KEY_LANDMARK]):
            if details['max']<len(a_start_end_video[KEY_LANDMARK]):
                details= {
                    'max': len(a_start_end_video[KEY_LANDMARK]),
                    'index_video': idx_vid,
                    'index_se': idx_se
                }

    return details
def min_images_hand(a_gloss: dict) -> dict:
    details: dict= {
        'min': 999_999,
        'index_video': 0,
        'index_se': 0,
    }
    for idx_vid, a_video in enumerate(a_gloss[KEY_VIDS]):
        for idx_se, a_start_end_video in enumerate(a_video[KEY_LANDMARK]):
            q_images_hand: int= list(filter(
                lambda x: (x[KEY_LHAND] or x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
                a_start_end_video[KEY_LANDMARK]
            )).__len__()
            if q_images_hand<details['min']:
                details= {
                    'min': q_images_hand,
                    'index_video': idx_vid,
                    'index_se': idx_se
                }

    return details
def max_images_hand(a_gloss: dict) -> dict:
    details: dict= {
        'max': 0,
        'index_video': 0,
        'index_se': 0,
    }
    for idx_vid, a_video in enumerate(a_gloss[KEY_VIDS]):
        for idx_se, a_start_end_video in enumerate(a_video[KEY_LANDMARK]):
            q_images_hand: int= list(filter(
                lambda x: (x[KEY_LHAND] or x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
                a_start_end_video[KEY_LANDMARK]
            )).__len__()
            if details['max']<q_images_hand:
                details= {
                    'max': q_images_hand,
                    'index_video': idx_vid,
                    'index_se': idx_se
                }

    return details
def min_images_2hand(a_gloss: dict) -> dict:
    details: dict= {
        'min': 999_999,
        'index_video': 0,
        'index_se': 0,
    }
    for idx_vid, a_video in enumerate(a_gloss[KEY_VIDS]):
        for idx_se, a_start_end_video in enumerate(a_video[KEY_LANDMARK]):
            q_images_hand: int= list(filter(
                lambda x: (x[KEY_LHAND] and x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
                a_start_end_video[KEY_LANDMARK]
            )).__len__()
            if q_images_hand<details['min']:
                details= {
                    'min': q_images_hand,
                    'index_video': idx_vid,
                    'index_se': idx_se
                }

    return details
def max_images_2hand(a_gloss: dict) -> dict:
    details: dict= {
        'max': 0,
        'index_video': 0,
        'index_se': 0,
    }
    for idx_vid, a_video in enumerate(a_gloss[KEY_VIDS]):
        for idx_se, a_start_end_video in enumerate(a_video[KEY_LANDMARK]):
            q_images_hand: int= list(filter(
                lambda x: (x[KEY_LHAND] and x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
                a_start_end_video[KEY_LANDMARK]
            )).__len__()
            if details['max']<q_images_hand:
                details= {
                    'max': q_images_hand,
                    'index_video': idx_vid,
                    'index_se': idx_se
                }

    return details


def main() -> None:
    for a_gloss in ds_landmark:
        min_images_gloss: dict= min_images(a_gloss)
        max_images_gloss: dict= max_images(a_gloss)

        min_images_gloss_hand: dict= min_images_hand(a_gloss)
        max_images_gloss_hand: dict= max_images_hand(a_gloss)

        min_images_gloss_2hand: dict= min_images_2hand(a_gloss)
        max_images_gloss_2hand: dict= max_images_2hand(a_gloss)

        print(f"------------- {a_gloss[KEY_G]} -------------")
        print(f"minimum images on a video on: {min_images_gloss['min']} at --> {
            a_gloss[KEY_VIDS][min_images_gloss['index_video']][KEY_LANDMARK][min_images_gloss['index_se']][KEY_PFOLDER]
        }")
        print(f"maximum images on a video on: {max_images_gloss['max']} --> {
            a_gloss[KEY_VIDS][max_images_gloss['index_video']][KEY_LANDMARK][max_images_gloss['index_se']][KEY_PFOLDER]
        }")
        print(f"minimum images( at least 1 hand ) on a video on: {min_images_gloss_hand['min']} --> {
            a_gloss[KEY_VIDS][min_images_gloss_hand['index_video']][KEY_LANDMARK][min_images_gloss_hand['index_se']][KEY_PFOLDER]
        }")
        print(f"maximum images( at least 1 hand ) on a video on: {max_images_gloss_hand['max']} --> {
            a_gloss[KEY_VIDS][max_images_gloss_hand['index_video']][KEY_LANDMARK][max_images_gloss_hand['index_se']][KEY_PFOLDER]
        }")
        print(f"minimum images( 2 hand ) on a video on: {min_images_gloss_2hand['min']} --> {
            a_gloss[KEY_VIDS][min_images_gloss_2hand['index_video']][KEY_LANDMARK][min_images_gloss_2hand['index_se']][KEY_PFOLDER]
        }")
        print(f"maximum images( 2 hand ) on a video on: {max_images_gloss_2hand['max']} --> {
            a_gloss[KEY_VIDS][max_images_gloss_2hand['index_video']][KEY_LANDMARK][max_images_gloss_2hand['index_se']][KEY_PFOLDER]
        }")
        print('\n')
if __name__=="__main__":
    main()
