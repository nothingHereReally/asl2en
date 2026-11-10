from json import load as loadJson, dump as writeJson
from itertools import groupby
from math import ceil
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
'''
average human being from https://media.istockphoto.com/id/965662288/photo/a-young-bearded-man-in-casual-wear-stands-with-hands-on-his-sides-on-a-white-background.jpg?s=612x612&w=0&k=20&c=QNRZO_MJP-hANRVdtyJ3OeVx30JM5IDrCjPwLbw8NjQ=
'''
DS1_22_LANDMARK_DIR: Path= PROJ_ROOT /'dataset' /'clean_dataset' /'fix_recent_n_classic_landmark'
DS2_16_LANDMARK_DIR: Path= PROJ_ROOT /'dataset' /'asl_citizen' /'selected_start_end_landmark'
DS1_22_cleaned_wlasl: Path= PROJ_ROOT /'dataset' /'clean_dataset' /'fix_recent_n_classic_landmark.json'
DS2_16_cleaned_aslct: Path= PROJ_ROOT /'dataset' /'asl_citizen' /'selected_start_end_landmark.json'
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
Q_VIDS_PER_G: int= 10
KEY_LANDMARK: str= 'landmark'
# ---------------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
RATIO4TRAIN: float= 0.8
# ---------------------------
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_PFOLDER_FROM: str= 'parent_folder_from'
KEY_PFOLDER: str= 'parent_folder'


def has_atleast1hand(annotation: dict) -> bool:
    return (annotation[KEY_LHAND] or annotation[KEY_RHAND]) and \
            annotation[KEY_FACE] and annotation[KEY_POSE]
def load_dataset(show_dist: bool=False) -> list:
    with open(DS1_22_cleaned_wlasl, 'r') as f:
        ds1_wlasl: list= loadJson(f)
    with open(DS2_16_cleaned_aslct, 'r') as f:
        ds2_aslct: list= loadJson(f)
    out: list= []
    for a_gloss in ds1_wlasl:
        if a_gloss[KEY_G] != 'who':
            out.append({
                KEY_G: a_gloss[KEY_G],
                KEY_VIDS: [],
            })
            for a_video in a_gloss[KEY_VIDS]:
                for start_end in a_video[KEY_LANDMARK]:
                    out[-1][KEY_VIDS].append({
                        KEY_PFOLDER_FROM: DS1_22_LANDMARK_DIR,
                        KEY_PFOLDER: start_end[KEY_PFOLDER],
                        KEY_LANDMARK: [],
                    })
                    for idx in range(len(start_end[KEY_LANDMARK])):
                        if idx == 0 or has_atleast1hand(start_end[KEY_LANDMARK][idx]):
                            out[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                dict(start_end[KEY_LANDMARK][idx]).copy()
                            )
                        else:
                            out[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                out[-1][KEY_VIDS][-1][KEY_LANDMARK][-1].copy()
                            )
            out[-1][KEY_VIDS]= out[-1][KEY_VIDS][:Q_VIDS_PER_G]
        else:
            # -- ie. for gloss who --
            out.append({
                KEY_G: a_gloss[KEY_G],
                KEY_VIDS: [],
            })
            for a_video in a_gloss[KEY_VIDS][:Q_VIDS_PER_G]:
                for start_end in a_video[KEY_LANDMARK]:
                    out[-1][KEY_VIDS].append({
                        KEY_PFOLDER_FROM: DS1_22_LANDMARK_DIR,
                        KEY_PFOLDER: start_end[KEY_PFOLDER],
                        KEY_LANDMARK: [],
                    })
                    for idx in range(len(start_end[KEY_LANDMARK])):
                        if idx == 0 or has_atleast1hand(start_end[KEY_LANDMARK][idx]):
                            out[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                dict(start_end[KEY_LANDMARK][idx]).copy()
                            )
                        else:
                            out[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                out[-1][KEY_VIDS][-1][KEY_LANDMARK][-1].copy()
                            )
            out.append({
                KEY_G: a_gloss[KEY_G],
                KEY_VIDS: [],
            })
            for a_video in a_gloss[KEY_VIDS][Q_VIDS_PER_G:]:
                for start_end in a_video[KEY_LANDMARK]:
                    out[-1][KEY_VIDS].append({
                        KEY_PFOLDER_FROM: DS1_22_LANDMARK_DIR,
                        KEY_PFOLDER: start_end[KEY_PFOLDER],
                        KEY_LANDMARK: [],
                    })
                    for idx in range(len(start_end[KEY_LANDMARK])):
                        if idx == 0 or has_atleast1hand(start_end[KEY_LANDMARK][idx]):
                            out[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                dict(start_end[KEY_LANDMARK][idx]).copy()
                            )
                        else:
                            out[-1][KEY_VIDS][-1][KEY_LANDMARK].append(
                                out[-1][KEY_VIDS][-1][KEY_LANDMARK][-1].copy()
                            )
    videos_per_gloss: int= int(len(out[-1][KEY_VIDS]))
    for a_gloss in ds2_aslct:
        out.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS][:videos_per_gloss]:
            out[-1][KEY_VIDS].append({
                KEY_PFOLDER_FROM: DS2_16_LANDMARK_DIR,
                KEY_PFOLDER: a_video[KEY_PFOLDER],
                KEY_LANDMARK: a_video[KEY_LANDMARK],
            })
    if show_dist:
        for a_gloss in out:
            print(f'{a_gloss[KEY_G]} --> {len(a_gloss[KEY_VIDS])}')
    return out
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
    return all_atleast1hand
def process4quantity_frame(dataset: list, q_frame: int, show_dist: bool=False) -> list:
    output: list= []
    min_element_count_on_gloss: int= 999_999
    min_element_count_on_video: int= 999_999
    if show_dist:
        print("---- dataset elements(obeys quantity frame) count ----")
    for a_gloss in dataset:
        output.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        per_gloss_per_video_elements: list[int]= []
        for a_video in a_gloss[KEY_VIDS]:
            if len(a_video[KEY_LANDMARK]) < q_frame:
                # part 1
                # spread out via ceil( ... )
                output[-1][KEY_VIDS].append({
                    KEY_PFOLDER_FROM: a_video[KEY_PFOLDER_FROM],
                    KEY_PFOLDER: a_video[KEY_PFOLDER],
                    KEY_LANDMARK: [],
                })
                ratio: int= ceil(q_frame/len(a_video[KEY_LANDMARK]))
                for an_img_detail in a_video[KEY_LANDMARK]:
                    output[-1][KEY_VIDS][-1][KEY_LANDMARK].extend(
                        [an_img_detail for _ in range(ratio)]
                    )
                output[-1][KEY_VIDS][-1][KEY_LANDMARK]= output[-1][KEY_VIDS][-1][
                    KEY_LANDMARK
                ][:q_frame]

                # part 2
                # use again loop ie. [x1, x2, ..., xn, x1, x2, ...] till q_frame
                output[-1][KEY_VIDS].append({
                    KEY_PFOLDER_FROM: a_video[KEY_PFOLDER_FROM],
                    KEY_PFOLDER: a_video[KEY_PFOLDER],
                    KEY_LANDMARK: []
                })
                for _ in range(ratio):
                    output[-1][KEY_VIDS][-1][KEY_LANDMARK].extend(
                        a_video[KEY_LANDMARK]
                    )
                output[-1][KEY_VIDS][-1][KEY_LANDMARK]= output[-1][KEY_VIDS][-1][
                    KEY_LANDMARK
                ][:q_frame]

                # part 3
                # use last ie. [x1, x2, ..., xn, xn, xn, ...]
                output[-1][KEY_VIDS].append({
                    KEY_PFOLDER_FROM: a_video[KEY_PFOLDER_FROM],
                    KEY_PFOLDER: a_video[KEY_PFOLDER],
                    KEY_LANDMARK: a_video[KEY_LANDMARK],
                })
                output[-1][KEY_VIDS][-1][KEY_LANDMARK].extend(
                    [ output[-1][KEY_VIDS][-1][KEY_LANDMARK][-1] ] *(
                        q_frame -len(output[-1][KEY_VIDS][-1][KEY_LANDMARK])
                    )
                )
                if 3 < min_element_count_on_video:
                    min_element_count_on_video= 3
                per_gloss_per_video_elements.append(3)

            elif len(a_video[KEY_LANDMARK]) == q_frame:
                # part 1
                output[-1][KEY_VIDS].append({
                    KEY_PFOLDER_FROM: a_video[KEY_PFOLDER_FROM],
                    KEY_PFOLDER: a_video[KEY_PFOLDER],
                    KEY_LANDMARK: a_video[KEY_LANDMARK],
                })
                if 1 < min_element_count_on_video:
                    min_element_count_on_video= 1
                per_gloss_per_video_elements.append(1)

            else:
                for init_idx in range(len(a_video[KEY_LANDMARK]) -q_frame +1):
                    output[-1][KEY_VIDS].append({
                        KEY_PFOLDER_FROM: a_video[KEY_PFOLDER_FROM],
                        KEY_PFOLDER: a_video[KEY_PFOLDER],
                        KEY_LANDMARK: a_video[KEY_LANDMARK][init_idx:init_idx+q_frame],
                    })
                if (len(a_video[KEY_LANDMARK]) -q_frame +1) < min_element_count_on_video:
                    min_element_count_on_video= len(a_video[KEY_LANDMARK]) -q_frame +1
                per_gloss_per_video_elements.append(
                    len(a_video[KEY_LANDMARK]) -q_frame +1
                )
        if show_dist:
            print(f"-- gloss( {output[-1][KEY_G]} ) -- quantity elements: {len(output[-1][KEY_VIDS])}")
            for idx_video, elements_count_on_a_video in enumerate(per_gloss_per_video_elements):
                print(f"    {str(idx_video+1).zfill(3)}: {elements_count_on_a_video}")
            print()
            if len(output[-1][KEY_VIDS]) < min_element_count_on_gloss:
                min_element_count_on_gloss= len(output[-1][KEY_VIDS])
    if show_dist:
        print('------------------------')
        print(f"minimum elements count on a gloss: {min_element_count_on_gloss}")
        print(f"minimum elements count on a video: {min_element_count_on_video}")
        print('------------------------')
    return output
def split4train_test(dataset: list, q_per_video: int) -> tuple:
    out_train: list= []
    out_test: list= []
    for a_gloss in dataset:
        out_train.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        out_test.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        record_not_used4train: list= []
        record_not_used4test: list= []
        group_by_video: list[list[dict]]= [
            list(group) for _, group in groupby(
                a_gloss[KEY_VIDS], key=lambda el: el[KEY_PFOLDER]
            )
        ]
        assert len(group_by_video) == Q_VIDS_PER_G
        for idx_video, a_video_elements in enumerate(group_by_video):
            if idx_video < RATIO4TRAIN*Q_VIDS_PER_G:
                '''
                due2could be less than q_per_video
                '''
                out_train[-1][KEY_VIDS].extend(
                    a_video_elements[:q_per_video]
                    if len(a_video_elements) >= q_per_video
                    else a_video_elements
                )
            else:
                '''
                due2could be less than q_per_video
                '''
                out_test[-1][KEY_VIDS].extend(
                    a_video_elements[:q_per_video]
                    if len(a_video_elements) >= q_per_video
                    else a_video_elements
                )

            if len(a_video_elements) > q_per_video:
                if idx_video < RATIO4TRAIN*Q_VIDS_PER_G:
                    record_not_used4train.extend(a_video_elements[q_per_video:])
                else:
                    record_not_used4test.extend(a_video_elements[q_per_video:])
        q_for_train_per_g: int= int(q_per_video *RATIO4TRAIN*Q_VIDS_PER_G)
        q_for_test_per_g: int= ceil(q_per_video *(1-RATIO4TRAIN)*Q_VIDS_PER_G) # due to 3.99999...
        if len(out_train[-1][KEY_VIDS]) < q_for_train_per_g:
            missing_quantity: int=        q_for_train_per_g -len(out_train[-1][KEY_VIDS])
            assert missing_quantity <= len(record_not_used4train)
            out_train[-1][KEY_VIDS].extend(
                record_not_used4train[:missing_quantity]
            )
        if len(out_test[-1][KEY_VIDS]) < q_for_test_per_g:
            missing_quantity: int=       q_for_test_per_g -len(out_test[-1][KEY_VIDS])
            assert missing_quantity <= len(record_not_used4test)
            out_test[-1][KEY_VIDS].extend(
                record_not_used4test[:missing_quantity]
            )
        assert int(len(out_train[-1][KEY_VIDS])) == q_for_train_per_g
        assert int(len(out_test[-1][KEY_VIDS]))  == q_for_test_per_g
    return (out_train, out_test)
def main() -> None:
    dataset_annotations: list= load_dataset(show_dist=False)
    QUANTITY_FRAME: int= median_what(dataset_annotations, show_dist=False) # median 26
    print(
        f"is there at least 1 hand on every image landmarks? " +
        "yes" if do_all_have_hands(dataset_annotations, show_msg=False) else "NO"
    )
    dataset_annotations= process4quantity_frame(
        dataset=dataset_annotations,
        q_frame=QUANTITY_FRAME,
        show_dist=False,
    )
    ds_annotations_train, ds_annotations_test= split4train_test(dataset_annotations, 2)
    '''
    TODO:
    ✔ QUANTITY_FRAME be median blah blahhh
    ✔ check mandatory all has atleast 1 hand blah
    use sklearn KNN blah for model
    do ML stats blah
    '''
    # with open('/tmp/asdf.json', 'w') as f:
    #     writeJson(dataset_annotations, f, indent=4)
if __name__=="__main__":
    main()
