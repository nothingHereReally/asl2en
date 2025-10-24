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
for i in range(1,46):
    if i<10:
        glasl[0]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[0]["gloss"]}_000{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[1]["gloss"]}_000{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[2]["gloss"]}_000{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[3]["gloss"]}_000{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[4]["gloss"]}_000{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[5]["gloss"]}_000{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[6]["gloss"]}_000{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[7]["gloss"]}_000{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[8]["gloss"]}_000{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[9]["gloss"]}_000{i}.mp4"
        })
    elif i<41:
        glasl[0]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[0]["gloss"]}_00{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[1]["gloss"]}_00{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[2]["gloss"]}_00{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[3]["gloss"]}_00{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[4]["gloss"]}_00{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[5]["gloss"]}_00{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[6]["gloss"]}_00{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[7]["gloss"]}_00{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[8]["gloss"]}_00{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"{glasl[9]["gloss"]}_00{i}.mp4"
        })
    elif i<44:
        glasl[0]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[0]["gloss"]}_00{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[1]["gloss"]}_00{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[2]["gloss"]}_00{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[3]["gloss"]}_00{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[4]["gloss"]}_00{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[5]["gloss"]}_00{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[6]["gloss"]}_00{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[7]["gloss"]}_00{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[8]["gloss"]}_00{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_VAL,
            "video_file": f"{glasl[9]["gloss"]}_00{i}.mp4"
        })
    else:
        glasl[0]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[0]["gloss"]}_00{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[1]["gloss"]}_00{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[2]["gloss"]}_00{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[3]["gloss"]}_00{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[4]["gloss"]}_00{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[5]["gloss"]}_00{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[6]["gloss"]}_00{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[7]["gloss"]}_00{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[8]["gloss"]}_00{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_TEST,
            "video_file": f"{glasl[9]["gloss"]}_00{i}.mp4"
        })


for i in range(45):
    if not exists(f"{VIDEO_DIR}{glasl[0]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[0]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[1]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[1]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[2]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[2]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[3]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[3]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[4]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[4]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[5]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[5]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[6]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[6]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[7]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[7]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[8]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[8]["instances"][i]["video_file"]} Does Not Exist.")
    if not exists(f"{VIDEO_DIR}{glasl[9]["instances"][i]["video_file"]}"):
        raise FileNotFoundError(f"video {glasl[9]["instances"][i]["video_file"]} Does Not Exist.")
print("all videos on dataset glasl Exists")


with open(f"{PROJ_ROOT}src_asl2gloss/create_own_dataset/glasl.annotation.clean.json", "w") as f:
    savejson(glasl, f, indent=4)
