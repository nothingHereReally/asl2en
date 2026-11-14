from itertools import groupby
from json import load as loadJson
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
DS1_22_LANDMARK_DIR: Path= PROJ_ROOT / 'dataset' / 'clean_dataset' / 'fix_recent_n_classic_landmark'
DS2_16_LANDMARK_DIR: Path= PROJ_ROOT / 'dataset' / 'asl_citizen' / 'selected_start_end_landmark'
DS1_22_cleaned_wlasl: Path= PROJ_ROOT / 'dataset' / 'clean_dataset' / 'fix_recent_n_classic_landmark.json'
DS2_16_cleaned_aslct: Path= PROJ_ROOT / 'dataset' / 'asl_citizen' / 'selected_start_end_landmark.json'

EPOCHS: int= 20
BATCH: int= 16
KEY_G: str= 'gloss'
KEY_GID: str= 'gloss_id'
KEY_VIDS: str= 'videos'
KEY_LANDMARK: str= 'landmark'
KEY_PFOLDER_FROM: str= 'parent_folder_from'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
RATIO4TRAIN: float= 0.8
Q_VIDS_PER_G: int= 10

WORTHY_FACE_IDX: tuple[int, ...]= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)
WORTHY_POSE_IDX: tuple[int, ...]= (11, 12, 13, 14, 15, 16, 23, 24)
Q_W_FACE: int= len(WORTHY_FACE_IDX)
Q_W_POSE: int= len(WORTHY_POSE_IDX)
Q_W_HAND: int= 21
LANDMARK_SHAPE: tuple[int, int]= (Q_W_FACE +Q_W_POSE +Q_W_HAND*2 +1, 2)

# -----------------------------
LM_IDX_TOP_LIP: int        =  0
LM_IDX_TIP_OF_FOREHEAD: int=  3
LM_IDX_RIGHT_MOST_EYE: int =  7
LM_IDX_RIGHT_MOST_LIP: int =  8
LM_IDX_CHIN_TIP: int       = 17
LM_IDX_RIGHT_EAR: int      = 19
LM_IDX_UP_TIP_NOSE: int    = 20
LM_IDX_LEFT_MOST_EYE: int  = 23
LM_IDX_LEFT_MOST_LIP: int  = 24
LM_IDX_LEFT_EAR: int       = 34
LMs_IDXs_FACE_GOOD: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_RIGHT_MOST_EYE,
    LM_IDX_RIGHT_MOST_LIP,
    LM_IDX_CHIN_TIP,
    LM_IDX_RIGHT_EAR,
    LM_IDX_UP_TIP_NOSE,
    LM_IDX_LEFT_MOST_EYE,
    LM_IDX_LEFT_MOST_LIP,
    LM_IDX_LEFT_EAR,
)
# -----------------------------
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
LM_IDX_LEFT_INDEXF_TIP: int= 52
LM_IDX_RIGHT_INDEXF_TIP:int= 73
# -----------------------------
LMs_IDXs_LEFT_H_NOT_INDEXF: tuple= (
    44, 45, 46, 47, 48,
    49, 50, 51,
    53, 54, 55, 56, 57,
    58, 59, 60, 61, 62,
    63, 64,
)
LMs_IDXs_RIGHT_H_NOT_INDEXF: tuple= (
    65, 66, 67, 68, 69,
    70, 71, 72,
    74, 75, 76, 77, 78,
    79, 80, 81, 82, 83,
    84, 85,
)


def has_atleast1hand(annotation: dict) -> bool:
    return bool(annotation[KEY_FACE] and annotation[KEY_POSE] and
                (annotation[KEY_LHAND] or annotation[KEY_RHAND]))
def load_dataset_annotations(show_dist: bool=False, name: bool=False) -> list[dict]:
    with open(DS1_22_cleaned_wlasl, 'r') as f:
        ds1_wlasl: list= loadJson(f)
    with open(DS2_16_cleaned_aslct, 'r') as f:
        ds2_aslct: list= loadJson(f)
    output: list= []
    for idx_g, a_gloss in enumerate(ds1_wlasl):
        if a_gloss[KEY_G] != 'who':
            output.append({
                KEY_G: a_gloss[KEY_G],
                KEY_VIDS: [],
            })
            for a_video in a_gloss[KEY_VIDS]:
                for start_end in a_video[KEY_LANDMARK]:
                    output[-1][KEY_VIDS].append({
                        KEY_GID: idx_g,
                        KEY_PFOLDER_FROM: DS1_22_LANDMARK_DIR.name if name else DS1_22_LANDMARK_DIR,
                        KEY_PFOLDER: start_end[KEY_PFOLDER],
                        KEY_LANDMARK: [],
                    })
                    for idx in range(len(start_end[KEY_LANDMARK])):
                        if idx == 0 or has_atleast1hand(start_end[KEY_LANDMARK][idx]):
                            output[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                dict(start_end[KEY_LANDMARK][idx]).copy()
                            )
                        else:
                            # to fix no hand img
                            output[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                output[-1][KEY_VIDS][-1][KEY_LANDMARK][-1].copy()
                            )
            output[-1][KEY_VIDS]= output[-1][KEY_VIDS][:Q_VIDS_PER_G]
        else:
            output.append({
                KEY_G: a_gloss[KEY_G],
                KEY_VIDS: [],
            })
            for a_video in a_gloss[KEY_VIDS][Q_VIDS_PER_G:]:
                for start_end in a_video[KEY_LANDMARK]:
                    output[-1][KEY_VIDS].append({
                        KEY_GID: idx_g,
                        KEY_PFOLDER_FROM: DS1_22_LANDMARK_DIR.name if name else DS1_22_LANDMARK_DIR,
                        KEY_PFOLDER: start_end[KEY_PFOLDER],
                        KEY_LANDMARK: [],
                    })
                    for idx in range(len(start_end[KEY_LANDMARK])):
                        if idx == 0 or has_atleast1hand(start_end[KEY_LANDMARK][idx]):
                            output[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                dict(start_end[KEY_LANDMARK][idx]).copy()
                            )
                        else:
                            # to fix no hand img
                            output[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                output[-1][KEY_VIDS][-1][KEY_LANDMARK][-1].copy()
                            )
        assert len(output[-1][KEY_VIDS]) == Q_VIDS_PER_G

    past_g_total: int= len(output)
    for idx, a_gloss in enumerate(ds2_aslct):
        output.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS][:Q_VIDS_PER_G]:
            output[-1][KEY_VIDS].append({
                KEY_GID: idx +past_g_total,
                KEY_PFOLDER_FROM: DS2_16_LANDMARK_DIR.name if name else DS2_16_LANDMARK_DIR,
                KEY_PFOLDER: a_video[KEY_PFOLDER],
                KEY_LANDMARK: a_video[KEY_LANDMARK],
            })
        assert len(output[-1][KEY_VIDS]) == Q_VIDS_PER_G
    if show_dist:
        for a_gloss in output:
            print(f'{a_gloss[KEY_G]} --> {len(a_gloss[KEY_VIDS])}')
    return output
def do_all_have_hands(dataset: list, show_msg: bool=False) -> bool:
    all_atleast1hand: bool= True
    for a_gloss in dataset:
        for a_video in a_gloss[KEY_VIDS]:
            for an_img in a_video[KEY_LANDMARK]:
                if not (an_img[KEY_LHAND] or an_img[KEY_RHAND]) or not (
                    an_img[KEY_FACE] and an_img[KEY_POSE]
                ):
                    all_atleast1hand= False
                    if show_msg:
                        print(
                            f"gloss: {a_gloss[KEY_G]}\n" +
                            f"  parent folder: {a_video[KEY_PFOLDER_FROM].name}/{a_video[KEY_PFOLDER]}\n" +
                            f"    image: {an_img[KEY_FILE]}\n" +
                            f"    face: {an_img[KEY_FACE]}\n" +
                            f"    pose: {an_img[KEY_POSE]}\n" +
                            f"    left hand: {an_img[KEY_LHAND]}\n" +
                            f"    right hand: {an_img[KEY_RHAND]}\n"
                        )
                    else:
                        return all_atleast1hand
    if show_msg:
        if all_atleast1hand:
            print("---- All good, all has at least 1 hand ----")
        else:
            print("---- Has parts where NO HAND EXISTS, not good ---")
    return all_atleast1hand
def check_ds_is_good(dataset: list[dict], q_videos_per_g: int, show_msg=False) -> bool:
    '''
    check for quantity videos
    and that files exists
    '''
    is_good: bool= True
    err_msg: list[str]= []
    for a_gloss in dataset:
        if len(a_gloss[KEY_VIDS]) != q_videos_per_g:
            err_msg.append(
                f"{a_gloss[KEY_G]} has {len(a_gloss[KEY_VIDS])} elements, expected2be {q_videos_per_g}"
            )
            is_good= False
        for a_video in a_gloss[KEY_VIDS]:
            for an_img_detail in a_video[KEY_LANDMARK]:
                if not (a_video[KEY_PFOLDER_FROM] /a_video[KEY_PFOLDER] /an_img_detail[KEY_FILE]).exists():
                    err_msg.append(
                        f"{a_video[KEY_PFOLDER]} has file NOT EXISTS: " +
                        f"{an_img_detail[KEY_FILE]}"
                    )
                    is_good= False
    if show_msg:
        print()
        print('---- result checking dataset ----')
        print('----------------------------')
        if is_good:
            print(
                f"dataset obeys {q_videos_per_g} quantity videos per gloss\n" +
                f"    and ALL files exists, so all good"
            )
        else:
            for msg in err_msg:
                print(f"problem: {msg}")
        print('----------------------------')
        print()

    return is_good
def median_what(dataset: list, show_dist: bool=False) -> int:
    length_list: list= []
    for a_gloss in dataset:
        for a_video in a_gloss[KEY_VIDS]:
            length_list.append({
                KEY_G: a_gloss[KEY_G],
                'len': len(a_video[KEY_LANDMARK])
            })
    length_list= sorted(
        length_list,
        key=lambda el: el['len'],
    )
    median_out: int= -1
    if len(length_list)%2 == 0:
        quantity: int= len(length_list)
        median_out= int(
            (length_list[quantity//2-1]['len'] +length_list[quantity//2]['len'])/2.0
        )
    else:
        median_out= length_list[len(length_list)//2]['len']
    if show_dist:
        for a_len, glosses in groupby(length_list, key=lambda el: el['len']):
            print(f"{a_len}  {',  '.join(el[KEY_G] for el in tuple(glosses))}")
        print(f"quantity groups( by number of images ): {len(tuple(groupby(length_list, key=lambda el: el['len'])))}")
        print(f"quantity videos: {len(length_list)}")
    return median_out


DS_ANNOTATIONS: list[dict]= load_dataset_annotations(
    show_dist=False,
    name=False,
)
QUANTITY_FRAME: int= median_what(dataset=DS_ANNOTATIONS, show_dist=False)
LEN_GLOSS: int= len(DS_ANNOTATIONS)


if __name__=='__main__':
    print(
        f"is there at least 1 hand on every image landmarks? " +
        "yes" if do_all_have_hands(DS_ANNOTATIONS, show_msg=False) else "NO"
    )
    print(
        f"are all dataset good? " +
        "yes" if check_ds_is_good(
            dataset=DS_ANNOTATIONS,
            q_videos_per_g=Q_VIDS_PER_G,
            show_msg=True,
        ) else "NO"
    )
