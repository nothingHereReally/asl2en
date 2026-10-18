from pathlib import Path
from json import load as loadjson


KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'         # -- list
# -------------------
KEY_PFOLDER: str= 'parent_folder'
KEY_LANDMARK: str= 'landmark'   # -- list
# -------------------
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
# -------------------
PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
ds_landmark: list= []
with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl_v4_landmark.json"}", "r") as f:
    ds_landmark= loadjson(f)


def min_images(a_gloss: dict) -> dict:
    details: dict= {
        'min': 999_999,
        'index_video': -1,
    }
    for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
        if len(a_video[KEY_LANDMARK])<details['min']:
            details= {
                'min': len(a_video[KEY_LANDMARK]),
                'index_video': idx_video,
            }

    return details
def max_images(a_gloss: dict) -> dict:
    details: dict= {
        'max': 0,
        'index_video': -1,
    }
    for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
        if len(a_video[KEY_LANDMARK])>details['max']:
            details= {
                'max': len(a_video[KEY_LANDMARK]),
                'index_video': idx_video,
            }

    return details
def min_images_hand(a_gloss: dict) -> dict:
    details: dict= {
        'min': 999_999,
        'index_video': -1,
    }
    for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] or x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            a_video[KEY_LANDMARK],
        )).__len__()
        if q_images_hand<details['min']:
            details= {
                'min': q_images_hand,
                'index_video': idx_video,
            }

    return details
def max_images_hand(a_gloss: dict) -> dict:
    details: dict= {
        'max': 0,
        'index_video': -1,
    }
    for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] or x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            a_video[KEY_LANDMARK],
        )).__len__()
        if q_images_hand>details['max']:
            details= {
                'max': q_images_hand,
                'index_video': idx_video,
            }

    return details
def min_images_2hand(a_gloss: dict) -> dict:
    details: dict= {
        'min': 999_999,
        'index_video': -1,
    }
    for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] and x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            a_video[KEY_LANDMARK],
        )).__len__()
        if q_images_hand<details['min']:
            details= {
                'min': q_images_hand,
                'index_video': idx_video,
            }

    return details
def max_images_2hand(a_gloss: dict) -> dict:
    details: dict= {
        'max': 0,
        'index_video': -1,
    }
    for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
        q_images_hand: int= list(filter(
            lambda x: (x[KEY_LHAND] and x[KEY_RHAND]) and x[KEY_FACE] and x[KEY_POSE],
            a_video[KEY_LANDMARK],
        )).__len__()
        if q_images_hand>details['max']:
            details= {
                'max': q_images_hand,
                'index_video': idx_video,
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
        if min_images_gloss['index_video']!=-1:
            print(
                f"minimum images on a video on: " +
                f"{min_images_gloss['min']} --> " +
                f"{a_gloss[KEY_VIDS][min_images_gloss['index_video']][KEY_PFOLDER]}"
            )
        else:
            print('NO minimum images on a video')


        if max_images_gloss['index_video']!=-1:
            print(
                f"maximum images on a video on: " +
                f"{max_images_gloss['max']} --> " +
                f"{a_gloss[KEY_VIDS][max_images_gloss['index_video']][KEY_PFOLDER]}"
            )
        else:
            print('NO maximum images on a video')


        if min_images_gloss_hand['index_video']!=-1:
            print(
                f"minimum images( at least 1 hand ) on a video on: " +
                f"{min_images_gloss_hand['min']} --> " +
                f"{a_gloss[KEY_VIDS][min_images_gloss_hand['index_video']][KEY_PFOLDER]}"
            )
        else:
            print('NO minimum images( at least 1 hand ) on a video')


        if max_images_gloss_hand['index_video']!=-1:
            print(
                f"maximum images( at least 1 hand ) on a video on: " +
                f"{max_images_gloss_hand['max']} --> " +
                f"{a_gloss[KEY_VIDS][max_images_gloss_hand['index_video']][KEY_PFOLDER]}"
            )
        else:
            print('NO maximum images( at least 1 hand ) on a video')


        if min_images_gloss_2hand['index_video']!=-1:
            print(
                f"minimum images( 2 hand ) on a video on: " +
                f"{min_images_gloss_2hand['min']} --> " +
                f"{a_gloss[KEY_VIDS][min_images_gloss_2hand['index_video']][KEY_PFOLDER]}"
            )
        else:
            print('NO minimum images( 2 hand ) on a video')


        if max_images_gloss_2hand['index_video']!=-1:
            print(
                f"maximum images( 2 hand ) on a video on: " +
                f"{max_images_gloss_2hand['max']} --> " +
                f"{a_gloss[KEY_VIDS][max_images_gloss_2hand['index_video']][KEY_PFOLDER]}"
            )
        else:
            print('NO maximum images( 2 hand ) on a video')
        print('\n')
if __name__=="__main__":
    main()
