from json import load as loadjson
from os.path import exists


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
wlasl_ds: dict= {}
with open(f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.clean.json", 'r') as f:
    wlasl_ds= loadjson(f)
text_out2file: str= ""

for i in range(10):
    text_out2file= f"{text_out2file}# {wlasl_ds[i]['gloss']}, quantity_videos( {len(wlasl_ds[i]['instances'])} ) --> {i}\n"
    for each_video in wlasl_ds[i]['instances']:
        if not exists(f"{PROJ_ROOT}dataset/wlasl/videos/{each_video['video_id']}.mp4"):
            raise FileExistsError(f"file does not exist dataset/wlasl/videos/{each_video['video_id']}.mp4")
        text_out2file= f"{text_out2file}mpv ./dataset/wlasl/videos/{each_video['video_id']}.mp4\n"
    text_out2file= f"{text_out2file}\n\n"

with open(f"{PROJ_ROOT}src_asl2gloss/create_own_dataset/p2_play_video_commands.txt", "w") as f:
    f.write(text_out2file)
