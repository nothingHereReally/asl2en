from json import load
from ..lmark_constant import PROJ_ROOT


data_jsondict: dict= {}
with open(f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.skeleton_image.train_val_test.json", 'r') as f:
    data_jsondict= load(f)


tvt: tuple= ("train", "val", "test")
data_str: str= "#!/bin/bash\n\n"
past_gloss: int= -1
for tvt_idv in tvt:
    data_str= f"{data_str}# --------------------\n# {tvt_idv}\n"
    for each_video in data_jsondict[tvt_idv]:
        if past_gloss!=each_video['gloss_id']:
            past_gloss= each_video['gloss_id']
            data_str= f"{data_str}# gloss_id={past_gloss}\n"
        data_str= f"{data_str}imv dataset/wlasl/skeleton_image/{each_video['video_id']}\n"

with open(f"{PROJ_ROOT}src_asl2gloss/view_image_dataset_script/view_image_skeleton.sh", "w") as f:
    f.write(data_str)

