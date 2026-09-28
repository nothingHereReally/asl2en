from pathlib import Path
from json import load as loadJson


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
MODEL_LANDMARK_DIR: Path= DS_DIR /"ds_landmark"
MODEL_SKELETON_DIR: Path= DS_DIR /"ds_skeleton"
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
# ---------------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'


def min_images(a_gloss: dict) -> dict:
    return min(
        (
            {
                "min": len(el[KEY_LANDMARK]),
                KEY_PFOLDER: el[KEY_PFOLDER],
                "index": idx,
            }
            for idx, el in enumerate(a_gloss[KEY_LANDMARK])
        ),
        key=lambda el: el["min"],
    )
def max_images(a_gloss: dict) -> dict:
    return max(
        (
            {
                "max": len(el[KEY_LANDMARK]),
                KEY_PFOLDER: el[KEY_PFOLDER],
                "index": idx,
            }
            for idx, el in enumerate(a_gloss[KEY_LANDMARK])
        ),
        key=lambda el: el["max"],
    )


def min_images_hand(a_gloss: dict) -> dict:
    return min(
        (
            {
                "min": list(filter(
                    lambda el: el[KEY_FACE] and el[KEY_POSE] and (el[KEY_LHAND] or el[KEY_RHAND]),
                    a_start_end[KEY_LANDMARK]
                )).__len__(),
                KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                "index": idx,
            }
            for idx, a_start_end in enumerate(a_gloss[KEY_LANDMARK])
        ),
        key=lambda el: el["min"],
    )
def max_images_hand(a_gloss: dict) -> dict:
    return max(
        (
            {
                "max": list(filter(
                    lambda el: el[KEY_FACE] and el[KEY_POSE] and (el[KEY_LHAND] or el[KEY_RHAND]),
                    a_start_end[KEY_LANDMARK]
                )).__len__(),
                KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                "index": idx,
            }
            for idx, a_start_end in enumerate(a_gloss[KEY_LANDMARK])
        ),
        key=lambda el: el["max"],
    )


def min_images_2hand(a_gloss: dict) -> dict:
    return min(
        (
            {
                "min": list(filter(
                    lambda el: el[KEY_FACE] and el[KEY_POSE] and el[KEY_LHAND] and el[KEY_RHAND],
                    a_start_end[KEY_LANDMARK]
                )).__len__(),
                KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                "index": idx,
            }
            for idx, a_start_end in enumerate(a_gloss[KEY_LANDMARK])
        ),
        key=lambda el: el["min"],
    )
def max_images_2hand(a_gloss: dict) -> dict:
    return max(
        (
            {
                "max": list(filter(
                    lambda el: el[KEY_FACE] and el[KEY_POSE] and el[KEY_LHAND] and el[KEY_RHAND],
                    a_start_end[KEY_LANDMARK]
                )).__len__(),
                KEY_PFOLDER: a_start_end[KEY_PFOLDER],
                "index": idx,
            }
            for idx, a_start_end in enumerate(a_gloss[KEY_LANDMARK])
        ),
        key=lambda el: el["max"],
    )


def split_counts(a_gloss: dict) -> dict:
    return {
        VAL_TRAIN: list(filter(
            lambda el: el[KEY_SPLIT]==VAL_TRAIN,
            a_gloss[KEY_LANDMARK],
        )).__len__(),
        VAL_TEST: list(filter(
            lambda el: el[KEY_SPLIT]==VAL_TEST,
            a_gloss[KEY_LANDMARK],
        )).__len__(),
    }


def main() -> None:
    ds_annotation: list
    with open(f"{DS_DIR /'ds_landmark.json'}", 'r') as f:
        ds_annotation= loadJson(f)
    ds_overall: dict= {
        'min_img': {
            'val': 999_999,
            KEY_PFOLDER: '',
            'index': -1,
        },
        'max_img': {
            'val': -1,
            KEY_PFOLDER: '',
            'index': -1,
        },
        'min_img_hand': {
            'val': 999_999,
            KEY_PFOLDER: '',
            'index': -1,
        },
        'max_img_hand': {
            'val': -1,
            KEY_PFOLDER: '',
            'index': -1,
        },
        'min_img2hand': {
            'val': 999_999,
            KEY_PFOLDER: '',
            'index': -1,
        },
        'max_img2hand': {
            'val': -1,
            KEY_PFOLDER: '',
            'index': -1,
        },
        'train_count': 0,
        'test_count': 0,
    }
    for a_gloss in ds_annotation:
        split_count: dict= split_counts(a_gloss)
        min_img: dict= min_images(a_gloss)
        max_img: dict= max_images(a_gloss)
        min_img_hand: dict= min_images_hand(a_gloss)
        max_img_hand: dict= max_images_hand(a_gloss)
        min_img2hand: dict= min_images_2hand(a_gloss)
        max_img2hand: dict= max_images_2hand(a_gloss)

        ds_overall['train_count']+= split_count[VAL_TRAIN]
        ds_overall['test_count']+= split_count[VAL_TEST]
        if min_img["min"] < ds_overall["min_img"]["val"]:
            ds_overall["min_img"]= {
                "val": min_img["min"],
                KEY_PFOLDER: min_img[KEY_PFOLDER],
                "index": min_img["index"],
            }
        if ds_overall["max_img"]["val"] < max_img["max"]:
            ds_overall["max_img"]= {
                "val": max_img["max"],
                KEY_PFOLDER: max_img[KEY_PFOLDER],
                "index": max_img["index"],
            }
        if min_img_hand["min"] < ds_overall["min_img_hand"]["val"]:
            ds_overall["min_img_hand"]= {
                "val": min_img_hand["min"],
                KEY_PFOLDER: min_img_hand[KEY_PFOLDER],
                "index": min_img_hand["index"],
            }
        if ds_overall["max_img_hand"]["val"] < max_img_hand["max"]:
            ds_overall["max_img_hand"]= {
                "val": max_img_hand["max"],
                KEY_PFOLDER: max_img_hand[KEY_PFOLDER],
                "index": max_img_hand["index"],
            }
        if min_img2hand["min"] < ds_overall["min_img2hand"]["val"]:
            ds_overall["min_img2hand"]= {
                "val": min_img2hand["min"],
                KEY_PFOLDER: min_img2hand[KEY_PFOLDER],
                "index": min_img2hand["index"],
            }
        if ds_overall["max_img2hand"]["val"] < max_img2hand["max"]:
            ds_overall["max_img2hand"]= {
                "val": max_img2hand["max"],
                KEY_PFOLDER: max_img2hand[KEY_PFOLDER],
                "index": max_img2hand["index"],
            }

        print(f"\nGloss: {a_gloss[KEY_G]}")
        print(f"  train count: {split_count[VAL_TRAIN]}")
        print(f"  test  count: {split_count[VAL_TEST]}")
        print(f"  min images      : {min_img}")
        print(f"  max images      : {max_img}")
        print(f"  min images hand : {min_img_hand}")
        print(f"  max images hand : {max_img_hand}")
        print(f"  min images 2hand: {min_img2hand}")
        print(f"  max images 2hand: {max_img2hand}")
    print(f"minimum images on a video: {ds_overall['min_img']}")
    print(f"maximum images on a video: {ds_overall['max_img']}")
    print(f"minimum images( at least 1 hand ) on a video: {ds_overall['min_img_hand']}")
    print(f"maximum images( at least 1 hand ) on a video: {ds_overall['max_img_hand']}")
    print(f"minimum images( 2 hands ) on a video: {ds_overall['min_img2hand']}")
    print(f"maximum images( 2 hands ) on a video: {ds_overall['max_img2hand']}")
    print(f"\ntrain count: {ds_overall['train_count']}")
    print(f"test count: {ds_overall['test_count']}")
if __name__=="__main__":
    main()
