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
    for a_gloss in ds_annotation:
        print(f"\nGloss: {a_gloss[KEY_G]}")
        print(f"  train count: {split_counts(a_gloss)[VAL_TRAIN]}")
        print(f"  test  count: {split_counts(a_gloss)[VAL_TEST]}")
        print(f"  min images      : {min_images(a_gloss)}")
        print(f"  max images      : {max_images(a_gloss)}")
        print(f"  min images hand : {min_images_hand(a_gloss)}")
        print(f"  max images hand : {max_images_hand(a_gloss)}")
        print(f"  min images 2hand: {min_images_2hand(a_gloss)}")
        print(f"  max images 2hand: {max_images_2hand(a_gloss)}")
if __name__=="__main__":
    main()
