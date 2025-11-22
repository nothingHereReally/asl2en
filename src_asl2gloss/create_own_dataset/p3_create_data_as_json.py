from json import dump as savejson
from os.path import exists


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
VIDEO_DIR: str= f"{PROJ_ROOT}dataset/glasl/video/"
T_TRAIN: str= "train"
T_VAL: str= "val"
T_TEST: str= "test"
glasl: list= [
    {
        "gloss": "book",
        "instances": []
    },
    {
        "gloss": "drink",
        "instances": []
    },
    {
        "gloss": "computer",
        "instances": []
    },
    {
        "gloss": "before",
        "instances": []
    },
    {
        "gloss": "chair",
        "instances": []
    },
    {
        "gloss": "go",
        "instances": []
    },
    {
        "gloss": "clothes",
        "instances": []
    },
    {
        "gloss": "who",
        "instances": []
    },
    {
        "gloss": "candy",
        "instances": []
    },
    {
        "gloss": "cousin",
        "instances": []
    },
]
LEN_CLASS: int= len(glasl)
for i in range(1,46):
    if i<10:
        # loop LEN_CLASS times, due to LEN_CLASS categories
        for ii in range(LEN_CLASS):
            glasl[ii]["instances"].append({
                "split": T_TRAIN,
                "video_file": f"{glasl[ii]["gloss"]}_000{i}.mp4"
            })
    elif i<41:
        # loop LEN_CLASS times, due to LEN_CLASS categories
        for ii in range(LEN_CLASS):
            glasl[ii]["instances"].append({
                "split": T_TRAIN,
                "video_file": f"{glasl[ii]["gloss"]}_00{i}.mp4"
            })
    elif i<44:
        # loop LEN_CLASS times, due to LEN_CLASS categories
        for ii in range(LEN_CLASS):
            glasl[ii]["instances"].append({
                "split": T_VAL,
                "video_file": f"{glasl[ii]["gloss"]}_00{i}.mp4"
            })
    else:
        # loop LEN_CLASS times, due to LEN_CLASS categories
        for ii in range(LEN_CLASS):
            glasl[ii]["instances"].append({
                "split": T_TEST,
                "video_file": f"{glasl[ii]["gloss"]}_00{i}.mp4"
            })


for i in range(45):
    for ii in range(LEN_CLASS):
        if not exists(f"{VIDEO_DIR}{glasl[ii]["instances"][i]["video_file"]}"):
            raise FileNotFoundError(f"video {glasl[ii]["instances"][i]["video_file"]} Does Not Exist.")
print("all videos on dataset glasl Exists")


with open(f"{PROJ_ROOT}src_asl2gloss/create_own_dataset/glasl.annotation.clean.json", "w") as f:
    savejson(glasl, f, indent=4)
