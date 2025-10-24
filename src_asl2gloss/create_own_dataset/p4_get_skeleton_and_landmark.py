from os.path import exists
from os import makedirs
from json import load as jsonload, dump as jsonsave
from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture, imwrite


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
GLASL_DIR: str= f"{PROJ_ROOT}dataset/glasl/"
VIDEO_DIR: str= f"{GLASL_DIR}video/"
SKELETON_dir: str= f"{GLASL_DIR}skeleton/"
LANDMARK_dir: str= f"{GLASL_DIR}landmark/"

T_TRAIN: str= "train"
T_VAL: str= "val"
T_TEST: str= "test"


if __name__=='__main__':
    glasl_clean: dict= {}
    with open(f"{GLASL_DIR}glasl.annotation.clean.json", 'r') as f:
        glasl_clean= jsonload(f)


    glasl_SKELETON: dict= {
        T_TRAIN: [],
        T_VAL: [],
        T_TEST: [],
        "id2gloss": [ins["gloss"] for ins in glasl_clean],
        "gloss2id": {glasl_clean[i]["gloss"]: i for i in range(len(glasl_clean))}
    }
    glasl_LANDMARK: dict= {
        T_TRAIN: [],
        T_VAL: [],
        T_TEST: [],
        "id2gloss": [ins["gloss"] for ins in glasl_clean],
        "gloss2id": {glasl_clean[i]["gloss"]: i for i in range(len(glasl_clean))}
    }
