from json import dump as savejson
from os.path import exists
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
GLASL_DIR: Path= PROJ_ROOT /"dataset" /"glasl"
VIDEO_DIR: Path= GLASL_DIR /"video"
T_TRAIN: str= "train"
T_VAL: str= "val"
T_TEST: str= "test"
GLOSS_IN_ORDER: list[str]= [
    "book",
    "drink",
    "computer",
    "before",
    "chair",
    "go",
    "clothes",
    "who",
    "candy",
    "cousin",
    "mine_my",
    "me_i",
    "stomach",
    "have",
    "need",
    "see",
    "feel",
    "hurt",
    "fever",
    "dizzy",
    "headache",
    "doctor",
]


def main() -> None:
    glasl: list[dict]= [{
        "gloss": glossWord,
        "instances": []
    } for glossWord in GLOSS_IN_ORDER]
    glasl_45: list[dict]= [{
        "gloss": glossWord,
        "instances": []
    } for glossWord in GLOSS_IN_ORDER]
    glasl_50: list[dict]= [{
        "gloss": glossWord,
        "instances": []
    } for glossWord in GLOSS_IN_ORDER]
    LEN_CLASS: int= len(glasl)
    for i in range(1,96):
        if i<41:
            # loop LEN_CLASS times, due to LEN_CLASS categories
            for ii in range(LEN_CLASS):
                glasl_45[ii]["instances"].append({
                    "split": T_TRAIN,
                    "video_file": f"{glasl_45[ii]["gloss"]}_00{str(i).zfill(2)}.mp4"
                })
        elif i<44:
            # loop LEN_CLASS times, due to LEN_CLASS categories
            for ii in range(LEN_CLASS):
                glasl_45[ii]["instances"].append({
                    "split": T_VAL,
                    "video_file": f"{glasl_45[ii]["gloss"]}_00{i}.mp4"
                })
        elif i<46:
            # loop LEN_CLASS times, due to LEN_CLASS categories
            for ii in range(LEN_CLASS):
                glasl_45[ii]["instances"].append({
                    "split": T_TEST,
                    "video_file": f"{glasl_45[ii]["gloss"]}_00{i}.mp4"
                })
        # -----------------------------------------------------------
        if i<91:
            # loop LEN_CLASS times, due to LEN_CLASS categories
            for ii in range(LEN_CLASS):
                glasl[ii]["instances"].append({
                    "split": T_TRAIN,
                    "video_file": f"{glasl[ii]["gloss"]}_00{str(i).zfill(2)}.mp4"
                })
            if 45<i:
                for ii in range(LEN_CLASS):
                    glasl_50[ii]["instances"].append({
                        "split": T_TRAIN,
                        "video_file": f"{glasl_50[ii]["gloss"]}_00{str(i).zfill(2)}.mp4"
                    })
        elif i<94:
            # loop LEN_CLASS times, due to LEN_CLASS categories
            for ii in range(LEN_CLASS):
                glasl[ii]["instances"].append({
                    "split": T_VAL,
                    "video_file": f"{glasl[ii]["gloss"]}_00{i}.mp4"
                })
                glasl_50[ii]["instances"].append({
                    "split": T_VAL,
                    "video_file": f"{glasl_50[ii]["gloss"]}_00{i}.mp4"
                })
        else:
            # loop LEN_CLASS times, due to LEN_CLASS categories
            for ii in range(LEN_CLASS):
                glasl[ii]["instances"].append({
                    "split": T_TEST,
                    "video_file": f"{glasl[ii]["gloss"]}_00{i}.mp4"
                })
                glasl_50[ii]["instances"].append({
                    "split": T_TEST,
                    "video_file": f"{glasl_50[ii]["gloss"]}_00{i}.mp4"
                })
    for i in range(1,5):
        for ii in range(LEN_CLASS):
            glasl[ii]["instances"].append({
                "split": T_TRAIN,
                "video_file": f"{glasl[ii]["gloss"]}__long_000{i}.mp4"
            })
    glasl_19: list[dict]= [{
        "gloss": glossWord,
        "instances": []
    } for glossWord in GLOSS_IN_ORDER]
    for ii in range(LEN_CLASS):
        glasl_19[ii]["instances"].extend( glasl[ii]["instances"][:5] )
        glasl_19[ii]["instances"].extend( glasl[ii]["instances"][45:45+5] )
        glasl_19[ii]["instances"].extend( glasl[ii]["instances"][-4:] )
        glasl_19[ii]["instances"].extend( glasl[ii]["instances"][90:95] )


    for idxVideo in range(99):
        for idxCat in range(LEN_CLASS):
            if not exists(f"{VIDEO_DIR /glasl[idxCat]["instances"][idxVideo]["video_file"]}"):
                raise FileNotFoundError(f"video {VIDEO_DIR /glasl[idxCat]["instances"][idxVideo]["video_file"]} Does Not Exist.")
    print("all videos on dataset glasl Exists")


    with open(f"{GLASL_DIR /"glasl.annotation.clean.45videos.json"}", "w") as f:
        savejson(glasl_45, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.clean.50videos.json"}", "w") as f:
        savejson(glasl_50, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.clean.99videos.json"}", "w") as f:
        savejson(glasl, f, indent=4)

    with open(f"{GLASL_DIR /"glasl.annotation.clean.19videos.json"}", "w") as f:
        savejson(glasl_19, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.clean.json"}", "w") as f:
        savejson(glasl_19, f, indent=4)

if __name__=="__main__":
    main()
