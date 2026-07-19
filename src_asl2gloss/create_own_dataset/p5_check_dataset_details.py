from json import load as jsonload
from os.path import exists
from pathlib import Path


KEY_TRAIN: str= "train"
KEY_VAL: str= "val"
KEY_TEST: str= "test"
G_ID: str= "gloss_id"
V_ID: str= "video_id"
G_: str= "gloss"
KEY_ID2G: str= "id2gloss"
KEY_G2ID: str= "gloss2id"
PROJ_ROOT= Path(__file__).resolve().parent.parent.parent
GLASL_DIR: Path= PROJ_ROOT /"dataset" /"glasl"



def count_dataset_is_it_correct(gl_clean: dict, gl_landmark: dict, gl_skeleton: dict) -> tuple:
    g2id: dict= dict(gl_landmark[KEY_G2ID])
    errors: list= []
    MIN_hands_quantity_min1hand: int= 99999
    MIN_hands_quantity_2hands:   int= 99999
    for gloss in gl_clean:
        if len(gloss["instances"])!=(
            len(tuple(filter(lambda x: x[G_ID]==g2id[gloss[G_]], tuple(gl_landmark[KEY_TRAIN]))))+
            len(tuple(filter(lambda x: x[G_ID]==g2id[gloss[G_]], tuple(gl_landmark[KEY_VAL]))))+
            len(tuple(filter(lambda x: x[G_ID]==g2id[gloss[G_]], tuple(gl_landmark[KEY_TEST]))))
        )*1:
            errors.append({'landmark': f"quantity dataset doesn't match clean dataset, {gloss[G_]} -- {g2id[gloss[G_]]}"})
        if len(gloss["instances"])!=(
            len(tuple(filter(lambda x: x[G_ID]==g2id[gloss[G_]], tuple(gl_skeleton[KEY_TRAIN]))))+
            len(tuple(filter(lambda x: x[G_ID]==g2id[gloss[G_]], tuple(gl_skeleton[KEY_VAL]))))+
            len(tuple(filter(lambda x: x[G_ID]==g2id[gloss[G_]], tuple(gl_skeleton[KEY_TEST]))))
        ):
            errors.append({'skeleton': f"quantity dataset doesn't match clean dataset, {gloss[G_]} -- {g2id[gloss[G_]]}"})
        if gl_landmark[KEY_G2ID]!=gl_skeleton[KEY_G2ID]:
            errors.append({KEY_G2ID: f"landmark and skeleton doesn't have the same {KEY_G2ID} values"})
        if gl_landmark[KEY_ID2G]!=gl_skeleton[KEY_ID2G]:
            errors.append({KEY_ID2G: f"landmark and skeleton doesn't have the same {KEY_ID2G} values"})

        for each_video in gloss["instances"]:
            landmark_video= list(filter(lambda x: x[V_ID]==each_video['video_file'][:-4], gl_landmark[KEY_TRAIN]))
            landmark_video= landmark_video if len(landmark_video)!=0 else list(filter(lambda x: x[V_ID]==each_video['video_file'][:-4], gl_landmark[KEY_VAL]))
            landmark_video= landmark_video if len(landmark_video)!=0 else list(filter(lambda x: x[V_ID]==each_video['video_file'][:-4], gl_landmark[KEY_TEST]))
            if len(landmark_video)!=1:
                errors.append({each_video['video_file'][:-4]: f"video {each_video['video_file']} shouldOnlyBe 1 but {len(landmark_video)} on landmark"})

            skeleton_video= list(filter(lambda x: x[V_ID]==each_video['video_file'][:-4], gl_skeleton[KEY_TRAIN]))
            skeleton_video= skeleton_video if len(skeleton_video)!=0 else list(filter(lambda x: x[V_ID]==each_video['video_file'][:-4], gl_skeleton[KEY_VAL]))
            skeleton_video= skeleton_video if len(skeleton_video)!=0 else list(filter(lambda x: x[V_ID]==each_video['video_file'][:-4], gl_skeleton[KEY_TEST]))
            if len(skeleton_video)!=1:
                errors.append({each_video['video_file'][:-4]: f"video {each_video['video_file']} shouldOnlyBe 1 but {len(skeleton_video)} on skeleton"})
            else:
                landmark_video= dict(landmark_video[0])
                skeleton_video= dict(skeleton_video[0])
                if landmark_video[G_ID]!=skeleton_video[G_ID]:
                    errors.append({f"{each_video[G_]}_{each_video['video_file'][:-4]}": f"landmark gloss id != skeleton gloss id --> {landmark_video[G_ID]}!={skeleton_video[G_ID]}"})
                if landmark_video[V_ID]!=skeleton_video[V_ID]:
                    errors.append({f"{each_video[G_]}_{each_video['video_file'][:-4]}": f"landmark video id != skeleton video id --> {landmark_video[V_ID]}!={skeleton_video[V_ID]}"})
                if len(landmark_video['landmark'])!=len(skeleton_video['skeleton']):
                    errors.append({f"{each_video[G_]}_{each_video['video_file'][:-4]}": f"landmark len(landmark) != skeleton len(skeleton) --> {len(landmark_video['landmark'])}!={len(skeleton_video['skeleton'])}"})
                else:
                    count_hands_min1hand: int= 0
                    count_hands_2hands: int= 0
                    for lm, sn in zip(landmark_video['landmark'], skeleton_video['skeleton']):
                        if lm['face']!=sn['face']:
                            errors.append({f"{each_video['video_file'][:-4]}": f"problem at face {lm['face']}!={sn['face']}"})
                        if lm['pose']!=sn['pose']:
                            errors.append({f"{each_video['video_file'][:-4]}": f"problem at pose {lm['pose']}!={sn['pose']}"})
                        if lm['left_hand']!=sn['left_hand']:
                            errors.append({f"{each_video['video_file'][:-4]}": f"problem at left_hand {lm['left_hand']}!={sn['left_hand']}"})
                        if lm['right_hand']!=sn['right_hand']:
                            errors.append({f"{each_video['video_file'][:-4]}": f"problem at right_hand {lm['right_hand']}!={sn['right_hand']}"})
                        if lm['left_hand'] or lm['right_hand']:
                            count_hands_min1hand+= 1
                        elif lm['left_hand'] and lm['right_hand']:
                            count_hands_2hands+= 1
                    if count_hands_min1hand==0:
                        errors.append({f"{each_video['video_file'][:-4]}": f"no hand not even just single 1"})
                    if count_hands_min1hand<MIN_hands_quantity_min1hand:
                        MIN_hands_quantity_min1hand= count_hands_min1hand
                    if count_hands_2hands<MIN_hands_quantity_2hands:
                        MIN_hands_quantity_2hands= count_hands_2hands
    if len(errors)!=0:
        for err in errors:
            print(f"{tuple(err.keys())[0]}: {err[tuple(err.keys())[0]]}")
        raise NotImplementedError("____ sorry incorrect implementation on p4, due to here p5 didn't pass ____")
    return (True, MIN_hands_quantity_min1hand, MIN_hands_quantity_2hands)


def mandatory3exist(clean_file: Path, landmark_file: Path, skeleton_file: Path) -> None:
    if not exists(clean_file):
        raise FileNotFoundError(f"file doesn't exist {clean_file}")
    if not exists(landmark_file):
        raise FileNotFoundError(f"file doesn't exist {landmark_file}")
    if not exists(skeleton_file):
        raise FileNotFoundError(f"file doesn't exist {skeleton_file}")


def main():
    glasl_clean: dict= {}
    glasl_landmark: dict= {}
    glasl_skeleton: dict= {}
    glasl_clean_file_str: Path= GLASL_DIR /"glasl.annotation.clean.json"
    glasl_landmark_file_str: Path= GLASL_DIR /"glasl.annotation.landmark.json"
    glasl_skeleton_file_str: Path= GLASL_DIR /"glasl.annotation.skeleton.json"
    mandatory3exist(
        clean_file=glasl_clean_file_str,
        landmark_file=glasl_landmark_file_str,
        skeleton_file=glasl_skeleton_file_str
    )
    with open(glasl_clean_file_str, 'r') as f:
        glasl_clean= jsonload(f)
    with open(glasl_landmark_file_str, 'r') as f:
        glasl_landmark= jsonload(f)
    with open(glasl_skeleton_file_str, 'r') as f:
        glasl_skeleton= jsonload(f)


    ds_ok, min1hand, has2hands= count_dataset_is_it_correct(glasl_clean, glasl_landmark, glasl_skeleton)
    if ds_ok:
        print("--> quantity dataset: train, val, test")
        for tvt in (KEY_TRAIN, KEY_VAL, KEY_TEST):
            print(f"{tvt} quantity videos: {len(glasl_landmark[tvt])}")
        print("-------------------------------------------")
        print("--> quantity dataset on class")
        for gloss in glasl_clean:
            print(f"{gloss[G_]}( {glasl_landmark[KEY_G2ID][gloss[G_]]} ) quantity videos: {len(gloss["instances"])}")
        print("-------------------------------------------")
        print(f"min quantity of at least 1 hand on single video: {min1hand}")
        print(f"min quantity of         2 hands on single video: {has2hands}")
        print( "<< ------------------------------------------------------------------ >>")
        print(f"-- passed: all have same quantity of video on clean/landmark/skeleton --")
        print( "<< ------------------------------------------------------------------ >>")


if __name__=='__main__':
    main()
