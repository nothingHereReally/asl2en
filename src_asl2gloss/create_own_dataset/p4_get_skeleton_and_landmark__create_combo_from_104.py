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
        raise FileExistsError(f"Missing folder {LANDMARK_dir} and dataset contents.")
    if not exists(SKELETON_dir):
        raise FileExistsError(f"Missing folder {SKELETON_dir} and dataset contents.")


def load_dataset() -> tuple:
    glasl_LANDMARK: dict= {}
    with open(f"{GLASL_DIR /"glasl.annotation.landmark.104videos.json"}", 'r') as f:
        glasl_LANDMARK= jsonload(f)
    glasl_SKELETON: dict= {}
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.104videos.json"}", 'r') as f:
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
def get9videos(landmark, skeleton) -> tuple:
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
        out_landmark[KEY_TRAIN].extend(a_gloss[90:94])
        out_landmark[KEY_VAL].extend(a_gloss[94:97])
        out_landmark[KEY_TEST].extend(a_gloss[97:])
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[90:94])
        out_skeleton[KEY_VAL].extend(a_gloss[94:97])
        out_skeleton[KEY_TEST].extend(a_gloss[97:])
    return (out_landmark, out_skeleton)


# ----------------------------------------------------------
# ----------------------------------------------------------
def get19videos(landmark, skeleton) -> tuple:
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
        out_landmark[KEY_TRAIN].extend(a_gloss[:5])
        out_landmark[KEY_TRAIN].extend(a_gloss[45:50])
        out_landmark[KEY_TRAIN].extend(a_gloss[90:94])
        out_landmark[KEY_VAL].extend(a_gloss[94:97])
        out_landmark[KEY_TEST].extend(a_gloss[97:])
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[:5])
        out_skeleton[KEY_TRAIN].extend(a_gloss[45:50])
        out_skeleton[KEY_TRAIN].extend(a_gloss[90:94])
        out_skeleton[KEY_VAL].extend(a_gloss[94:97])
        out_skeleton[KEY_TEST].extend(a_gloss[97:])
    return (out_landmark, out_skeleton)
def get95is45p50videos(landmark, skeleton) -> tuple:
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
        out_landmark[KEY_TRAIN].extend(a_gloss[:90])
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[:90])

    for a_gloss in landmark[KEY_VAL]:
        out_landmark[KEY_VAL].extend(a_gloss[:3])
    for a_gloss in skeleton[KEY_VAL]:
        out_skeleton[KEY_VAL].extend(a_gloss[:3])

    for a_gloss in landmark[KEY_TEST]:
        out_landmark[KEY_TEST].extend(a_gloss[:2])
    for a_gloss in skeleton[KEY_TEST]:
        out_skeleton[KEY_TEST].extend(a_gloss[:2])
    return (out_landmark, out_skeleton)
def get54is45p9videos(landmark, skeleton) -> tuple:
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
        out_landmark[KEY_TRAIN].extend(a_gloss[:45])   # ie. +45
        out_landmark[KEY_TRAIN].extend(a_gloss[90:94]) # ie. +4
        out_landmark[KEY_VAL].extend(a_gloss[94:97])   # ie. +3
        out_landmark[KEY_TEST].extend(a_gloss[97:])    # ie. +2
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[:45])   # ie. +45
        out_skeleton[KEY_TRAIN].extend(a_gloss[90:94]) # ie. +4
        out_skeleton[KEY_VAL].extend(a_gloss[94:97])   # ie. +3
        out_skeleton[KEY_TEST].extend(a_gloss[97:])    # ie. +2
    return (out_landmark, out_skeleton)
def get59is50p9videos(landmark, skeleton) -> tuple:
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
    for a_gloss_train, a_gloss_val, a_gloss_test in zip(landmark[KEY_TRAIN], landmark[KEY_VAL], landmark[KEY_TEST]):
        out_landmark[KEY_TRAIN].extend(a_gloss_train[45:90]) # ie. +45
        out_landmark[KEY_TRAIN].extend(a_gloss_val)          # ie. +3
        out_landmark[KEY_TRAIN].extend(a_gloss_test)         # ie. +2
    for a_gloss_train, a_gloss_val, a_gloss_test in zip(skeleton[KEY_TRAIN], skeleton[KEY_VAL], skeleton[KEY_TEST]):
        out_skeleton[KEY_TRAIN].extend(a_gloss_train[45:90]) # ie. +45
        out_skeleton[KEY_TRAIN].extend(a_gloss_val)          # ie. +3
        out_skeleton[KEY_TRAIN].extend(a_gloss_test)         # ie. +2

    for a_gloss in landmark[KEY_TRAIN]:
        out_landmark[KEY_TRAIN].extend(a_gloss[90:94]) # ie. +4
        out_landmark[KEY_VAL].extend(a_gloss[94:97])   # ie. +3
        out_landmark[KEY_TEST].extend(a_gloss[97:])    # ie. +2
    for a_gloss in skeleton[KEY_TRAIN]:
        out_skeleton[KEY_TRAIN].extend(a_gloss[90:94]) # ie. +4
        out_skeleton[KEY_VAL].extend(a_gloss[94:97])   # ie. +3
        out_skeleton[KEY_TEST].extend(a_gloss[97:])    # ie. +2
    return (out_landmark, out_skeleton)
# ----------------------------------------------------------
# ----------------------------------------------------------


def main() -> None:
    mandatory_all_2exist()
    glasl_LANDMARK, glasl_SKELETON= load_dataset()

    landmark_details, skeleton_details= get_details(glasl_LANDMARK, glasl_SKELETON)
    glasl_LANDMARK_45, glasl_SKELETON_45= get45videos(landmark_details, skeleton_details)
    glasl_LANDMARK_50, glasl_SKELETON_50= get50videos(landmark_details, skeleton_details)
    glasl_LANDMARK_9, glasl_SKELETON_9= get9videos(landmark_details, skeleton_details)
    # -----------------------------------------------------------------------------------
    glasl_LANDMARK_19, glasl_SKELETON_19= get19videos(landmark_details, skeleton_details)
    glasl_LANDMARK_95is45p50, glasl_SKELETON_95is45p50= get95is45p50videos(landmark_details, skeleton_details)
    glasl_LANDMARK_54is45p9, glasl_SKELETON_54is45p9= get54is45p9videos(landmark_details, skeleton_details)
    glasl_LANDMARK_59is50p9, glasl_SKELETON_59is50p9= get59is50p9videos(landmark_details, skeleton_details)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.45videos.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_45, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.45videos.json"}", "w") as f:
        jsonsave(glasl_SKELETON_45, f, indent=4)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.50videos.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_50, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.50videos.json"}", "w") as f:
        jsonsave(glasl_SKELETON_50, f, indent=4)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.9videos.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_9, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.9videos.json"}", "w") as f:
        jsonsave(glasl_SKELETON_9, f, indent=4)

    # -----------------------------------------------------------------------------
    with open(f"{GLASL_DIR /"glasl.annotation.landmark.19videos.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_19, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.19videos.json"}", "w") as f:
        jsonsave(glasl_SKELETON_19, f, indent=4)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.95videos_is45p50.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_95is45p50, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.95videos_is45p50.json"}", "w") as f:
        jsonsave(glasl_SKELETON_95is45p50, f, indent=4)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.54videos_is45p9.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_54is45p9, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.54videos_is45p9.json"}", "w") as f:
        jsonsave(glasl_SKELETON_54is45p9, f, indent=4)

    with open(f"{GLASL_DIR /"glasl.annotation.landmark.59videos_is50p9.json"}", "w") as f:
        jsonsave(glasl_LANDMARK_59is50p9, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.59videos_is50p9.json"}", "w") as f:
        jsonsave(glasl_SKELETON_59is50p9, f, indent=4)


if __name__=='__main__':
    main()
