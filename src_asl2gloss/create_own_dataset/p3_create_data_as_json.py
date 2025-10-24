from json import dump as savejson


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
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
            "video_file": f"book_000{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"drink_000{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"computer_000{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"before_000{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"chair_000{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"go_000{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"clothes_000{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"who_000{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"candy_000{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"cousin_000{i}.mp4"
        })
    elif i<41:
        glasl[0]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"book_00{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"drink_00{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"computer_00{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"before_00{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"chair_00{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"go_00{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"clothes_00{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"who_00{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"candy_00{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_TRAIN,
            "video_file": f"cousin_00{i}.mp4"
        })
    elif i<44:
        glasl[0]["instances"].append({
            "split": T_VAL,
            "video_file": f"book_00{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_VAL,
            "video_file": f"drink_00{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_VAL,
            "video_file": f"computer_00{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_VAL,
            "video_file": f"before_00{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_VAL,
            "video_file": f"chair_00{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_VAL,
            "video_file": f"go_00{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_VAL,
            "video_file": f"clothes_00{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_VAL,
            "video_file": f"who_00{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_VAL,
            "video_file": f"candy_00{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_VAL,
            "video_file": f"cousin_00{i}.mp4"
        })
    else:
        glasl[0]["instances"].append({
            "split": T_TEST,
            "video_file": f"book_00{i}.mp4"
        })
        glasl[1]["instances"].append({
            "split": T_TEST,
            "video_file": f"drink_00{i}.mp4"
        })
        glasl[2]["instances"].append({
            "split": T_TEST,
            "video_file": f"computer_00{i}.mp4"
        })
        glasl[3]["instances"].append({
            "split": T_TEST,
            "video_file": f"before_00{i}.mp4"
        })
        glasl[4]["instances"].append({
            "split": T_TEST,
            "video_file": f"chair_00{i}.mp4"
        })
        glasl[5]["instances"].append({
            "split": T_TEST,
            "video_file": f"go_00{i}.mp4"
        })
        glasl[6]["instances"].append({
            "split": T_TEST,
            "video_file": f"clothes_00{i}.mp4"
        })
        glasl[7]["instances"].append({
            "split": T_TEST,
            "video_file": f"who_00{i}.mp4"
        })
        glasl[8]["instances"].append({
            "split": T_TEST,
            "video_file": f"candy_00{i}.mp4"
        })
        glasl[9]["instances"].append({
            "split": T_TEST,
            "video_file": f"cousin_00{i}.mp4"
        })


with open(f"{PROJ_ROOT}src_asl2gloss/create_own_dataset/glasl.annotation.clean.json", "w") as f:
    savejson(glasl, f, indent=4)
