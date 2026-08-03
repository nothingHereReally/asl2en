from os.path import exists
from json import load as jsonload, dump as jsonsave
from pathlib import Path


# ------------------------
# ------------------------
# ---- contants start ----
PROJ_ROOT= Path(__file__).resolve().parent.parent.parent
GLASL_DIR: Path= PROJ_ROOT /"dataset" /"glasl"
VIDEO_DIR: Path= GLASL_DIR /"video"
IMAGE_dir: Path= GLASL_DIR /"image"
LANDMARK_dir: Path= GLASL_DIR /"landmark"
SKELETON_dir: Path= GLASL_DIR /"skeleton"
KEY_TRAIN: str= "train"
KEY_VAL: str= "val"
KEY_TEST: str= "test"
KEY_GID: str= "gloss_id"
KEY_ID2G: str= "id2gloss"
KEY_G2ID: str= "gloss2id"
KEY_RH_MANDATORY: str= "right_hand_mandatory"
# ---- contants end ------
# ------------------------
# ------------------------




def mandatory_all_2exist() -> None:
    if not exists(LANDMARK_dir):
        raise FileExistsError(f"please this folder {LANDMARK_dir} exist and has dataset contents.")
    if not exists(SKELETON_dir):
        raise FileExistsError(f"please this folder {SKELETON_dir} exist and has dataset contents.")


def load_dataset() -> tuple:
    glasl_LANDMARK: dict= {}
    with open(f"{GLASL_DIR /"glasl.annotation.landmark.99videos.json"}", 'r') as f:
        glasl_LANDMARK= jsonload(f)
    glasl_SKELETON: dict= {}
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.99videos.json"}", 'r') as f:
        glasl_SKELETON= jsonload(f)
    return (glasl_LANDMARK, glasl_SKELETON)


def group_by_gloss_id(arr_videos: list[dict], len_gloss: int) -> list:
    out: list= list(range(len_gloss))
    for id in range(len_gloss):
        out[id]= list(filter(lambda el: el[KEY_GID]==id, arr_videos))
    return out
def get_details(landmark, skeleton) -> tuple:
    len_gloss: int= len(landmark[KEY_G2ID])
    landmark_details: dict= {
        KEY_TRAIN: group_by_gloss_id(landmark[KEY_TRAIN], len_gloss),
        KEY_VAL: group_by_gloss_id(landmark[KEY_VAL], len_gloss),
        KEY_TEST: group_by_gloss_id(landmark[KEY_TEST], len_gloss),
        KEY_ID2G: list(landmark[KEY_ID2G]),
        KEY_G2ID: dict(landmark[KEY_G2ID]),
        KEY_RH_MANDATORY: list(landmark[KEY_RH_MANDATORY]),
    }
    skeleton_details: dict= {
        KEY_TRAIN: group_by_gloss_id(skeleton[KEY_TRAIN], len_gloss),
        KEY_VAL: group_by_gloss_id(skeleton[KEY_VAL], len_gloss),
        KEY_TEST: group_by_gloss_id(skeleton[KEY_TEST], len_gloss),
        KEY_ID2G: list(skeleton[KEY_ID2G]),
        KEY_G2ID: dict(skeleton[KEY_G2ID]),
        KEY_RH_MANDATORY: list(skeleton[KEY_RH_MANDATORY]),
    }
    return (landmark_details, skeleton_details)
def get45videos(landmark, skeleton) -> tuple:
    out_landmark: dict= {
        KEY_TRAIN: list(),
        KEY_VAL: list(),
        KEY_TEST: list(),
        KEY_ID2G: list(landmark[KEY_ID2G]),
        KEY_G2ID: dict(landmark[KEY_G2ID]),
        KEY_RH_MANDATORY: list(landmark[KEY_RH_MANDATORY]),
    }
    out_skeleton: dict= {
        KEY_TRAIN: list(),
        KEY_VAL: list(),
        KEY_TEST: list(),
        KEY_ID2G: list(skeleton[KEY_ID2G]),
        KEY_G2ID: dict(skeleton[KEY_G2ID]),
        KEY_RH_MANDATORY: list(skeleton[KEY_RH_MANDATORY]),
    }
    for a_gloss in landmark[KEY_TRAIN]:
        out_landmark[KEY_TRAIN].extend(a_gloss[:40])
        out_landmark[KEY_VAL].extend(a_gloss[40:43])
        out_landmark[KEY_TEST].extend(a_gloss[43:45])
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[:40])
        out_skeleton[KEY_VAL].extend(a_gloss[40:43])
        out_skeleton[KEY_TEST].extend(a_gloss[43:45])
    return (out_landmark, out_skeleton)
def get50videos(landmark, skeleton) -> tuple:
    out_landmark: dict= {
        KEY_TRAIN: list(),
        KEY_VAL: list(),
        KEY_TEST: list(),
        KEY_ID2G: list(landmark[KEY_ID2G]),
        KEY_G2ID: dict(landmark[KEY_G2ID]),
        KEY_RH_MANDATORY: list(landmark[KEY_RH_MANDATORY]),
    }
    out_skeleton: dict= {
        KEY_TRAIN: list(),
        KEY_VAL: list(),
        KEY_TEST: list(),
        KEY_ID2G: list(skeleton[KEY_ID2G]),
        KEY_G2ID: dict(skeleton[KEY_G2ID]),
        KEY_RH_MANDATORY: list(skeleton[KEY_RH_MANDATORY]),
    }
    for a_gloss in landmark[KEY_TRAIN]:
        out_landmark[KEY_TRAIN].extend(a_gloss[45:90])
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[45:90])

    for a_gloss in landmark[KEY_VAL]:
        out_landmark[KEY_VAL].extend(a_gloss[:3])
    for a_gloss in skeleton[KEY_VAL]:
        out_skeleton[KEY_VAL].extend(a_gloss[:3])

    for a_gloss in landmark[KEY_TEST]:
        out_landmark[KEY_TEST].extend(a_gloss[:2])
    for a_gloss in skeleton[KEY_TEST]:
        out_skeleton[KEY_TEST].extend(a_gloss[:2])
    return (out_landmark, out_skeleton)


def main() -> None:
    mandatory_all_2exist()
    glasl_LANDMARK, glasl_SKELETON= load_dataset()

    landmark_details, skeleton_details= get_details(glasl_LANDMARK, glasl_SKELETON)
    glasl_LANDMARK_45, glasl_SKELETON_45= get45videos(landmark_details, skeleton_details)
    glasl_LANDMARK_50, glasl_SKELETON_50= get50videos(landmark_details, skeleton_details)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.45videos.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_45, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.45videos.json"}", "w") as f:
        jsonsave(glasl_SKELETON_45, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.landmark.50videos.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_50, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.50videos.json"}", "w") as f:
        jsonsave(glasl_SKELETON_50, f, indent=4)


if __name__=='__main__':
    main()
