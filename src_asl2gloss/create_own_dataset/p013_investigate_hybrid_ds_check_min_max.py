from pathlib import Path
from json import load as loadjson


KEY_G: str=     'gloss'
KEY_VID: str=   'videos'
KEY_VFILE: str= 'video_file'
KEY_LMARK: str= 'landmark'
KEY_FACE: str=  'face'
KEY_POSE: str=  'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
hybrid_ds_investigate: list= []
with open(f"{PROJ_ROOT /"dataset" /"clean_dataset" /"hybrid_data.investigate.landmark.json"}", "r") as f:
    hybrid_ds_investigate= loadjson(f)


def min_images(split: list) -> dict:
    details: dict= {
        'min': 999_999,
        'index': 0,
    }
    for idx, el in enumerate(split):
        if len(el[KEY_LMARK])<details['min']:
            details= {
                'min': len(el[KEY_LMARK]),
                'index': idx
            }

    return details
def max_images(split: list) -> dict:
    details: dict= {
        'max': 0,
        'index': 0,
    }
    for idx, el in enumerate(split):
        if details['max']<len(el[KEY_LMARK]):
            details= {
                'max': len(el[KEY_LMARK]),
                'index': idx
            }

    return details
def min_images_hand(split: list) -> dict:
    details: dict= {
        'min': 999_999,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] or x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            el[KEY_LMARK]
        )).__len__()
        if q_images_hand<details['min']:
            details= {
                'min': q_images_hand,
                'index': idx
            }

    return details
def max_images_hand(split: list) -> dict:
    details: dict= {
        'max': 0,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] or x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            el[KEY_LMARK]
        )).__len__()
        if details['max']<q_images_hand:
            details= {
                'max': q_images_hand,
                'index': idx
            }

    return details
def min_images_2hand(split: list) -> dict:
    details: dict= {
        'min': 999_999,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] and x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            el[KEY_LMARK]
        )).__len__()
        if q_images_hand<details['min']:
            details= {
                'min': q_images_hand,
                'index': idx
            }

    return details
def max_images_2hand(split: list) -> dict:
    details: dict= {
        'max': 0,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] and x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            el[KEY_LMARK]
        )).__len__()
        if details['max']<q_images_hand:
            details= {
                'max': q_images_hand,
                'index': idx
            }

    return details


def main() -> None:
    for a_gloss in hybrid_ds_investigate:
        min_images_gloss: dict= min_images(a_gloss[KEY_VID])
        max_images_gloss: dict= max_images(a_gloss[KEY_VID])

        min_images_gloss_hand: dict= min_images_hand(a_gloss[KEY_VID])
        max_images_gloss_hand: dict= max_images_hand(a_gloss[KEY_VID])

        min_images_gloss_2hand: dict= min_images_2hand(a_gloss[KEY_VID])
        max_images_gloss_2hand: dict= max_images_2hand(a_gloss[KEY_VID])

        print(f"------------- {a_gloss[KEY_G]} -------------")
        print(f"minimum images on a video on: {min_images_gloss['min']} at index {min_images_gloss['index']} --> {
            a_gloss[KEY_VID][min_images_gloss['index']][KEY_VFILE]
        }")
        print(f"maximum images on a video on: {max_images_gloss['max']} at index {max_images_gloss['index']} --> {
            a_gloss[KEY_VID][max_images_gloss['index']][KEY_VFILE]
        }")
        print(f"minimum images( at least 1 hand ) on a video on: {min_images_gloss_hand['min']} at index {min_images_gloss_hand['index']} --> {
            a_gloss[KEY_VID][min_images_gloss_hand['index']][KEY_VFILE]
        }")
        print(f"maximum images( at least 1 hand ) on a video on: {max_images_gloss_hand['max']} at index {max_images_gloss_hand['index']} --> {
            a_gloss[KEY_VID][max_images_gloss_hand['index']][KEY_VFILE]
        }")
        print(f"minimum images( 2 hand ) on a video on: {min_images_gloss_2hand['min']} at index {min_images_gloss_2hand['index']} --> {
            a_gloss[KEY_VID][min_images_gloss_2hand['index']][KEY_VFILE]
        }")
        print(f"maximum images( 2 hand ) on a video on: {max_images_gloss_2hand['max']} at index {max_images_gloss_2hand['index']} --> {
            a_gloss[KEY_VID][max_images_gloss_2hand['index']][KEY_VFILE]
        }")
        print('\n')
if __name__=="__main__":
    main()
