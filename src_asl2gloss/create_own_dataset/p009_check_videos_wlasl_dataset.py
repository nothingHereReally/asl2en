from json import load
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
def get_video_exist(videos: list, gloss: str) -> list:
    videos_exist: list= []
    for a_video in videos:
        if not (PROJ_ROOT /"dataset" /"wlasl" /"videos" /f"{a_video["video_id"]}.mp4").exists():
            print(f"file NOT exist -- {gloss} -- {a_video["video_id"]}.mp4")
        else:
            videos_exist.append(f"dataset/wlasl/videos{a_video["video_id"]}.mp4")
    return videos_exist
def main():
    wlasl_dataset: list= []
    command2play_video: str= ""
    with open(f"{PROJ_ROOT /"dataset" /"wlasl" /"wlasl.annotation.clean.json"}", 'r') as f:
        wlasl_dataset= load(f)
    for a_gloss in wlasl_dataset:
        videos_exist: list= get_video_exist(a_gloss['instances'], a_gloss['gloss'])
        command2play_video= f"{command2play_video}# -- {a_gloss['gloss']}\nmpv {" ".join(videos_exist)}"
        command2play_video= f"{command2play_video}\n\n"
    with open(f"{PROJ_ROOT /"dataset" /"wlasl" /"wlasl.annotation.videos.play.sh"}", 'w', encoding='utf-8') as f:
        f.write(command2play_video)
if __name__=="__main__":
    main()
