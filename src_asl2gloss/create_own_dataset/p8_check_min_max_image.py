from pathlib import Path
from json import load as loadjson


KEY_TRAIN: str= 'train'   # landmark is face, then pose, then left_had, then right hand
KEY_VAL: str= 'val'       # face full is (468, 2) --> face worthy is (36, 2)
KEY_TEST: str= 'test'     # pose full is (33, 2) --> pose worthy is (8, 2)
KEY_LMARK: str= 'landmark'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
glasl_landmark: dict= {}
with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl.annotation.landmark.json"}", "r") as f:
    glasl_landmark= loadjson(f)


def min_split(split: list) -> dict:
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
def max_split(split: list) -> dict:
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
def min_split_hand(split: list) -> dict:
    details: dict= {
        'min': 999_999,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: x[KEY_LHAND] or x[KEY_RHAND],
            el[KEY_LMARK]
        )).__len__()
        if q_images_hand<details['min']:
            details= {
                'min': q_images_hand,
                'index': idx
            }

    return details
def max_split_hand(split: list) -> dict:
    details: dict= {
        'max': 0,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: x[KEY_LHAND] or x[KEY_RHAND],
            el[KEY_LMARK]
        )).__len__()
        if details['max']<q_images_hand:
            details= {
                'max': q_images_hand,
                'index': idx
            }

    return details
def min_split_2hand(split: list) -> dict:
    details: dict= {
        'min': 999_999,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: x[KEY_LHAND] and x[KEY_RHAND],
            el[KEY_LMARK]
        )).__len__()
        if q_images_hand<details['min']:
            details= {
                'min': q_images_hand,
                'index': idx
            }

    return details
def max_split_2hand(split: list) -> dict:
    details: dict= {
        'max': 0,
        'index': 0,
    }
    for idx, el in enumerate(split):
        q_images_hand: int= list(filter(
            lambda x: x[KEY_LHAND] and x[KEY_RHAND],
            el[KEY_LMARK]
        )).__len__()
        if details['max']<q_images_hand:
            details= {
                'max': q_images_hand,
                'index': idx
            }

    return details


def main() -> None:
    min_images_train: dict= min_split(glasl_landmark[KEY_TRAIN])
    min_images_val: dict= min_split(glasl_landmark[KEY_VAL])
    min_images_test: dict= min_split(glasl_landmark[KEY_TEST])
    print(f"minimum images on a video on --'train'--: {min_images_train['min']} at index {min_images_train['index']}")
    print(f"minimum images on a video on --'val'----: {min_images_val['min']  } at index {min_images_val['index']}")
    print(f"minimum images on a video on --'test'---: {min_images_test['min'] } at index {min_images_test['index']}")
    print(f"-----------------------------------------------------------")
    max_images_train: dict= max_split(glasl_landmark[KEY_TRAIN])
    max_images_val: dict= max_split(glasl_landmark[KEY_VAL])
    max_images_test: dict= max_split(glasl_landmark[KEY_TEST])
    print(f"maximum images on a video on --'train'--: {max_images_train['max']} at index {max_images_train['index']}")
    print(f"maximum images on a video on --'val'----: {max_images_val['max']  } at index {max_images_val['index']}")
    print(f"maximum images on a video on --'test'---: {max_images_test['max'] } at index {max_images_test['index']}")

    print()
    print(f"------------------------------------------------------------------------------")
    min_images_train_hand: dict= min_split_hand(glasl_landmark[KEY_TRAIN])
    min_images_val_hand: dict= min_split_hand(glasl_landmark[KEY_VAL])
    min_images_test_hand: dict= min_split_hand(glasl_landmark[KEY_TEST])
    print(f"minimum images( at least 1 hand ) on a video on --'train'--: {min_images_train_hand['min']} at index {min_images_train_hand['index']}")
    print(f"minimum images( at least 1 hand ) on a video on --'val'----: {min_images_val_hand['min']  } at index {min_images_val_hand['index']}")
    print(f"minimum images( at least 1 hand ) on a video on --'test'---: {min_images_test_hand['min'] } at index {min_images_test_hand['index']}")
    print(f"------------------------------------------------------------------------------")
    max_images_train_hand: dict= max_split_hand(glasl_landmark[KEY_TRAIN])
    max_images_val_hand: dict= max_split_hand(glasl_landmark[KEY_VAL])
    max_images_test_hand: dict= max_split_hand(glasl_landmark[KEY_TEST])
    print(f"maximum images( at least 1 hand ) on a video on --'train'--: {max_images_train_hand['max']} at index {max_images_train_hand['index']}")
    print(f"maximum images( at least 1 hand ) on a video on --'val'----: {max_images_val_hand['max']  } at index {max_images_val_hand['index']}")
    print(f"maximum images( at least 1 hand ) on a video on --'test'---: {max_images_test_hand['max'] } at index {max_images_test_hand['index']}")

    print()
    print(f"------------------------------------------------------------------------------")
    min_images_train_2hand: dict= min_split_2hand(glasl_landmark[KEY_TRAIN])
    min_images_val_2hand: dict= min_split_2hand(glasl_landmark[KEY_VAL])
    min_images_test_2hand: dict= min_split_2hand(glasl_landmark[KEY_TEST])
    print(f"minimum images( 2 hand ) on a video on --'train'--: {min_images_train_2hand['min']} at index {min_images_train_2hand['index']}")
    print(f"minimum images( 2 hand ) on a video on --'val'----: {min_images_val_2hand['min']  } at index {min_images_val_2hand['index']}")
    print(f"minimum images( 2 hand ) on a video on --'test'---: {min_images_test_2hand['min'] } at index {min_images_test_2hand['index']}")
    print(f"------------------------------------------------------------------------------")
    max_images_train_2hand: dict= max_split_2hand(glasl_landmark[KEY_TRAIN])
    max_images_val_2hand: dict= max_split_2hand(glasl_landmark[KEY_VAL])
    max_images_test_2hand: dict= max_split_2hand(glasl_landmark[KEY_TEST])
    print(f"maximum images( 2 hand ) on a video on --'train'--: {max_images_train_2hand['max']} at index {max_images_train_2hand['index']}")
    print(f"maximum images( 2 hand ) on a video on --'val'----: {max_images_val_2hand['max']  } at index {max_images_val_2hand['index']}")
    print(f"maximum images( 2 hand ) on a video on --'test'---: {max_images_test_2hand['max'] } at index {max_images_test_2hand['index']}")
if __name__=="__main__":
    main()
