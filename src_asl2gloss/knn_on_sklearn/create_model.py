from json import load as loadJson
from itertools import groupby
from math import ceil, floor
import numpy as np
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
# -------------------------------
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)
Q_HAND_LMARK: int= 21
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


def flip_vertical_y_axis(landmarks) -> np.ndarray:
    '''
    landmarks be of shape (INT, ..., 86, 2)
    '''
    new_landmark= np.array(landmarks).copy()
    new_landmark[..., 0]= 1 -new_landmark[..., 0]
    lhand2rhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_HAND_LMARK
    ,:].copy()
    rhand2lhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_HAND_LMARK:
    ,:].copy()
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_HAND_LMARK
    ,:]= rhand2lhand # ie. past was right hand, now be left hand
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_HAND_LMARK:
    ,:]= lhand2rhand # ie. past was left hand, now be right hand

    indices2flip: tuple[tuple, ...]= (
        # -- face --
        ( 12, 28 ),
        ( 19, 34 ),
        ( 11, 27 ),
        ( 21, 35 ),
        ( 14, 30 ),
        ( 13, 29 ),
        ( 10, 26 ),
        ( 15, 31 ),
        ( 18, 33 ),
        (  7, 23 ),
        ( 16, 32 ),
        (  9, 25 ),
        (  8, 24 ),

        # -- pose --
        ( 37, 36 ),
        ( 43, 42 ),
        ( 39, 38 ),
        ( 41, 40 ),
    )
    # -- flip index pair face
    # -- ( 12, 28 )
    for p_a, p_b in indices2flip:
        tmp= new_landmark[..., p_a:p_a+1, :].copy()
        new_landmark[..., p_a:p_a+1, :]= new_landmark[..., p_b:p_b+1, :].copy()
        new_landmark[..., p_b:p_b+1, :]= tmp

    return new_landmark
def has_atleast1hand(annotation: dict) -> bool:
    return (annotation[KEY_LHAND] or annotation[KEY_RHAND]) and \
            annotation[KEY_FACE] and annotation[KEY_POSE]
def load_dataset_annotations(show_dist: bool=False) -> list:
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
def q_lessthan_p1_spread(video_annotation: dict, q_frame: int) -> list[dict]:
    # spread out via ceil( ... )
    assert len(video_annotation[KEY_LANDMARK]) < q_frame

    ratio: int= ceil(q_frame/len(video_annotation[KEY_LANDMARK]))
    q2be_created: int= len(video_annotation[KEY_LANDMARK]) -ceil(q_frame/ratio) +1
    output: list[dict]= []
    for init_idx in range(q2be_created):
        output.append({
            KEY_PFOLDER_FROM: video_annotation[KEY_PFOLDER_FROM],
            KEY_PFOLDER: video_annotation[KEY_PFOLDER],
            KEY_LANDMARK: [],
        })
        for an_img_detail in video_annotation[KEY_LANDMARK][
            init_idx:init_idx +ceil(q_frame/ratio)
        ]:
            output[-1][KEY_LANDMARK].extend(
                [an_img_detail for _ in range(ratio)]
            )
        output[-1][KEY_LANDMARK]= output[-1][KEY_LANDMARK][:q_frame]
        assert len(output[-1][KEY_LANDMARK]) == q_frame
    assert len(output) == q2be_created

    return output
def q_lessthan_p2_loop(video_annotation: dict, q_frame: int) -> list[dict]:
    # use again loop ie. [x1, x2, ..., xn, x1, x2, ...] till q_frame
    assert len(video_annotation[KEY_LANDMARK]) < q_frame

    output: list[dict]= []
    q2be_created: int= len(video_annotation[KEY_LANDMARK])
    for init_idx in range(q2be_created):
        output.append({
            KEY_PFOLDER_FROM: video_annotation[KEY_PFOLDER_FROM],
            KEY_PFOLDER: video_annotation[KEY_PFOLDER],
            KEY_LANDMARK: video_annotation[KEY_LANDMARK][init_idx:],
        })
        while len(output[-1][KEY_LANDMARK]) < q_frame:
            output[-1][KEY_LANDMARK].extend(
                list(video_annotation[KEY_LANDMARK]).copy()
            )
        output[-1][KEY_LANDMARK]= output[-1][KEY_LANDMARK][:q_frame]
        assert len(output[-1][KEY_LANDMARK]) == q_frame
    assert len(output) == q2be_created

    return output
def q_lessthan_p3_stretch_startend(video_annotation: dict, q_frame: int) -> list[dict]:
    # use 1st  ie. [x1, x1, ..., x1, x2, ..., xn]
    # use last ie. [x1, x2, ..., xn, xn, xn, ...]
    assert len(video_annotation[KEY_LANDMARK]) < q_frame

    output: list[dict]= []
    missing_what: int= q_frame -len(video_annotation[KEY_LANDMARK])

    # on start stretch
    output.append({
        KEY_PFOLDER_FROM: video_annotation[KEY_PFOLDER_FROM],
        KEY_PFOLDER: video_annotation[KEY_PFOLDER],
        KEY_LANDMARK: [
            video_annotation[KEY_LANDMARK][0].copy()
            for _ in range(missing_what)
        ],
    })
    output[-1][KEY_LANDMARK].extend(video_annotation[KEY_LANDMARK])
    assert len(output[-1][KEY_LANDMARK]) == q_frame

    # on end stretch
    output.append({
        KEY_PFOLDER_FROM: video_annotation[KEY_PFOLDER_FROM],
        KEY_PFOLDER: video_annotation[KEY_PFOLDER],
        KEY_LANDMARK: video_annotation[KEY_LANDMARK],
    })
    output[-1][KEY_LANDMARK].extend([
            video_annotation[KEY_LANDMARK][-1].copy()
            for _ in range(missing_what)
        ])
    assert len(output[-1][KEY_LANDMARK]) == q_frame

    return output
def q_equalto_p4_loop(video_annotation: dict, q_frame: int) -> list[dict]:
    # use again loop ie. [x1, x2, ..., xn],
    # then also [x2, x3, ..., xn, x1],
    # ...,
    # [xn, x1, x2, ..., x(n-1)] till len q_frame
    assert len(video_annotation[KEY_LANDMARK]) == q_frame

    output: list[dict]= []
    for idx_init in range(len(video_annotation[KEY_LANDMARK])):
        output.append({
            KEY_PFOLDER_FROM: video_annotation[KEY_PFOLDER_FROM],
            KEY_PFOLDER: video_annotation[KEY_PFOLDER],
            KEY_LANDMARK: video_annotation[KEY_LANDMARK][idx_init:],
        })
        if len(output[-1][KEY_LANDMARK]) < q_frame:
            output[-1][KEY_LANDMARK].extend(
                video_annotation[KEY_LANDMARK]
            )
            output[-1][KEY_LANDMARK]= output[-1][KEY_LANDMARK][:q_frame]
            assert len(output[-1][KEY_LANDMARK]) == q_frame
    assert len(output) == len(video_annotation[KEY_LANDMARK])

    return output
def q_greaterthan_p5_loop(video_annotation: dict, q_frame: int) -> list[dict]:
    # loop till last ie. [x1, x2, ..., x_qf],
    # then also [x2, x3, ..., x_qf, x_qf+1],
    # ...,
    # [xn, x1, x2, ..., x_qf-1] till len q_frame, due to on orig last is xn
    assert len(video_annotation[KEY_LANDMARK]) > q_frame

    output: list[dict]= []
    for idx_init in range(len(video_annotation[KEY_LANDMARK])):
        output.append({
            KEY_PFOLDER_FROM: video_annotation[KEY_PFOLDER_FROM],
            KEY_PFOLDER: video_annotation[KEY_PFOLDER],
            KEY_LANDMARK: video_annotation[KEY_LANDMARK][
                idx_init:idx_init+q_frame
            ],
        })
        if len(output[-1][KEY_LANDMARK]) < q_frame:
            output[-1][KEY_LANDMARK].extend(
                video_annotation[KEY_LANDMARK]
            )
            output[-1][KEY_LANDMARK]= output[-1][KEY_LANDMARK][:q_frame]
            assert len(output[-1][KEY_LANDMARK]) == q_frame
    assert len(output) == len(video_annotation[KEY_LANDMARK])

    return output
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
                q_added: int= 0

                # part 1
                elements: list[dict]= q_lessthan_p1_spread(a_video, q_frame)
                output[-1][KEY_VIDS].extend(elements)
                q_added+= len(elements)

                # part 2
                elements= q_lessthan_p2_loop(a_video, q_frame)
                output[-1][KEY_VIDS].extend(elements)
                q_added+= len(elements)

                # part 3
                elements= q_lessthan_p3_stretch_startend(a_video, q_frame)
                output[-1][KEY_VIDS].extend(elements)
                q_added+= len(elements)

                if q_added < min_element_count_on_video:
                    min_element_count_on_video= q_added
                per_gloss_per_video_elements.append(q_added)
            elif len(a_video[KEY_LANDMARK]) == q_frame:
                # part 4
                elements: list[dict]= q_equalto_p4_loop(a_video, q_frame)
                output[-1][KEY_VIDS].extend(elements)

                if len(elements) < min_element_count_on_video:
                    min_element_count_on_video= len(elements)
                per_gloss_per_video_elements.append(len(elements))

            else:
                # part 5
                elements: list[dict]= q_greaterthan_p5_loop(a_video, q_frame)
                output[-1][KEY_VIDS].extend(elements)

                if len(elements) < min_element_count_on_video:
                    min_element_count_on_video= len(elements)
                per_gloss_per_video_elements.append(len(elements))
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
def split4train_test(dataset: list, suggested_q_per_video: int, show_dist: bool=False) -> tuple:
    out_train: list= []
    out_test: list= []

    min_elements_per_g4train: int= 9_999_999
    min_elements_per_g4test: int=  9_999_999
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


        if show_dist:
            print(f"-- gloss( {a_gloss[KEY_G]} ) -- quantity elements: {len(a_gloss[KEY_VIDS])}")
            total_elements_train: int= sum([
                len(el)
                for el in group_by_video[:ceil(RATIO4TRAIN*Q_VIDS_PER_G)
            ]])
            total_elements_test: int=  sum([
                len(el)
                for el in group_by_video[ceil(RATIO4TRAIN*Q_VIDS_PER_G):]
            ])
            if total_elements_train < min_elements_per_g4train:
                min_elements_per_g4train= total_elements_train
            if total_elements_test < min_elements_per_g4test:
                min_elements_per_g4test= total_elements_test
            print(f"    from train max count: {total_elements_train}")
            print(f"    from test max count: {total_elements_test}")
        for idx_video, a_video_elements in enumerate(group_by_video):
            if idx_video < RATIO4TRAIN*Q_VIDS_PER_G:
                '''
                due2could be less than suggested_q_per_video
                '''
                out_train[-1][KEY_VIDS].extend(
                    a_video_elements[:suggested_q_per_video]
                    if len(a_video_elements) >= suggested_q_per_video
                    else a_video_elements
                )
            else:
                '''
                due2could be less than suggested_q_per_video
                '''
                out_test[-1][KEY_VIDS].extend(
                    a_video_elements[:suggested_q_per_video]
                    if len(a_video_elements) >= suggested_q_per_video
                    else a_video_elements
                )

            if len(a_video_elements) > suggested_q_per_video:
                if idx_video < RATIO4TRAIN*Q_VIDS_PER_G:
                    record_not_used4train.extend(a_video_elements[suggested_q_per_video:])
                else:
                    record_not_used4test.extend(a_video_elements[suggested_q_per_video:])

            if show_dist:
                print(f"    {str(idx_video+1).zfill(3)}: {len(a_video_elements)}")
        q_for_train_per_g: int= int(suggested_q_per_video *RATIO4TRAIN*Q_VIDS_PER_G)
        q_for_test_per_g: int= ceil(suggested_q_per_video *(1-RATIO4TRAIN)*Q_VIDS_PER_G) # due to 3.99999...
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

        if show_dist:
            print()
    if show_dist:
        print('------------------------')
        # print(f"minimum elements count on a gloss for train: {min_elements_per_g4train}")
        # print(f"minimum elements count on a gloss for test: {min_elements_per_g4test}")
        total_possible_per_gloss: int= floor(min(
            ceil(min_elements_per_g4train / RATIO4TRAIN),
            ceil(min_elements_per_g4test / (1-RATIO4TRAIN)),
        ))
        print(f"max possible 80% train elements on a gloss: {ceil(total_possible_per_gloss *RATIO4TRAIN)}")
        print(f"max possible 20% test elements on a gloss: {ceil(total_possible_per_gloss *(1-RATIO4TRAIN))}")
        print(f"max possible suggested elements per video: {floor(total_possible_per_gloss/Q_VIDS_PER_G)}")
        print('------------------------')
    return (out_train, out_test)
def check_ds_is_good(dataset: list[dict], q_elements_per_g: int, q_frame: int, show_msg=False) -> bool:
    is_good: bool= True
    err_msg: list[str]= []
    for a_gloss in dataset:
        if len(a_gloss[KEY_VIDS]) != q_elements_per_g:
            err_msg.append(
                f"{a_gloss[KEY_G]} has {len(a_gloss[KEY_VIDS])} elements, expected2be {q_elements_per_g}"
            )
            is_good= False
        for a_ds_element in a_gloss[KEY_VIDS]:
            if len(a_ds_element[KEY_LANDMARK]) != q_frame:
                err_msg.append(
                    f"{a_ds_element[KEY_PFOLDER]} has {len(a_ds_element[KEY_LANDMARK])} QUANTITY_FRAME, expected2be {q_frame}"
                )
                is_good= False
            for an_img_detail in a_ds_element[KEY_LANDMARK]:
                if not (a_ds_element[KEY_PFOLDER_FROM] /a_ds_element[KEY_PFOLDER] /an_img_detail[KEY_FILE]).exists():
                    err_msg.append(
                        f"{a_ds_element[KEY_PFOLDER]} has file NOT EXISTS: " +
                        f"{an_img_detail[KEY_FILE]}"
                    )
                    is_good= False
    if show_msg:
        print()
        print('---- result checking dataset ----')
        print('----------------------------')
        if is_good:
            print(
                f"dataset obeys {q_elements_per_g} quantity elements per gloss\n" +
                f"    has {q_frame} QUANTITY_FRAME per dataset element\n" +
                f"    and ALL files exists, so all good"
            )
        else:
            for msg in err_msg:
                print(f"problem: {msg}")
        print('----------------------------')
        print()

    return is_good
def loadnpfiles(annotation: dict) -> list[np.ndarray]:
    '''
    annotation is dict having keys
    -- KEY_PFOLDER_FROM: Path
    -- KEY_PFOLDER: str
    -- KEY_LANDMARK: list[dict]
    '''
    output: list[np.ndarray]= []
    for an_img_detail in annotation[KEY_LANDMARK]:
        with open(
            annotation[KEY_PFOLDER_FROM] /annotation[KEY_PFOLDER] /an_img_detail[KEY_FILE],
            'rb',
        ) as f:
            output.append(np.load(f))

    return output
def get2dots_frombody(input: np.ndarray) -> np.ndarray:
    '''
    input -- shape of (..., 86, 2)
    -------------------------------
    returned shape is (...,  2, 2)
    '''
    shoulder_center: np.ndarray= (
        input[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
        input[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
    )/2.0
    hip_center: np.ndarray= (
        input[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
        input[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
    )/2.0
    shoulder_hip_center: np.ndarray= (shoulder_center +hip_center)/2.0
    '''
    shoulder_center and hip_center, both has shape (..., 1, 2)
    so as shoulder_hip_center has shape (..., 1, 2)
    '''
    return np.concatenate([
        (shoulder_center+shoulder_hip_center)/2.0,
        (shoulder_hip_center+hip_center)/2.0,
    ], axis=-2)
def extract_features(landmarks: np.ndarray) -> np.ndarray:
    '''
    landmarks be of shape (BATCH, 86, 2)
    ---- to calculate distance from ----
    10 face landmarks
     8 pose landmarks + 2 landmarks/dots
    20 left hand landmarks
    20 right hand landmarks
    '''
    assert landmarks.shape[1:] == (86, 2)

    left_index_finger_diff: np.ndarray=  landmarks[..., LM_IDX_LEFT_INDEXF_TIP:LM_IDX_LEFT_INDEXF_TIP+1, :]
    right_index_finger_diff: np.ndarray= landmarks[..., LM_IDX_RIGHT_INDEXF_TIP:LM_IDX_RIGHT_INDEXF_TIP+1, :]
    the_rest: np.ndarray= np.concatenate(
        [
            landmarks[..., LMs_IDXs_FACE_GOOD, :],
            get2dots_frombody(landmarks),
            landmarks[..., len(WORTHY_FACE_IDX):len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX), :],
            landmarks[..., LMs_IDXs_LEFT_H_NOT_INDEXF,  :],
            landmarks[..., LMs_IDXs_RIGHT_H_NOT_INDEXF, :],
        ],
        axis=-2,
    )
    left_index_finger_diff= (left_index_finger_diff -the_rest)**2
    right_index_finger_diff= (right_index_finger_diff -the_rest)**2
    left_index_finger_diff= np.sum(left_index_finger_diff, axis=-1, keepdims=False)**0.5
    right_index_finger_diff= np.sum(right_index_finger_diff, axis=-1, keepdims=False)**0.5
    return np.concatenate(
        [
            left_index_finger_diff,
            right_index_finger_diff,
        ],
        axis=-1,
    )
def get_dataset(dataset: list[dict]) -> tuple[np.ndarray, list[int]]:
    ds_elements: list|np.ndarray= []
    ds_elements_label: list[int]= []
    for idx_gloss, a_gloss in enumerate(dataset):
        for a_ds_element in a_gloss[KEY_VIDS]:
            a_ds_element_landmarks: list[np.ndarray]= loadnpfiles(a_ds_element)
            # a_ds_element_landmarks shape (QUANTITY_FRAME, 86, 2)

            ds_elements.append(a_ds_element_landmarks)
            ds_elements_label.append(idx_gloss)

            ds_elements.append(flip_vertical_y_axis(a_ds_element_landmarks))
            ds_elements_label.append(idx_gloss)
    ds_elements= np.array(ds_elements)
    orig_shape: tuple= ds_elements.shape
    ds_elements= extract_features(np.reshape(
        ds_elements,
        shape=(-1, 86, 2),
    ))
    ds_elements= np.reshape(ds_elements, shape=(orig_shape[0], orig_shape[1], -1))
    assert ds_elements.shape == (len(ds_elements_label), orig_shape[1], 120)

    return (
        np.reshape(
            ds_elements,
            shape=(orig_shape[0], -1),
        ),
        ds_elements_label,
    )
def main() -> None:
    dataset_annotations: list= load_dataset_annotations(show_dist=False)
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
    '''
    current on 23+16=39 gloss
    minimum elements count on a gloss: 188
    minimum elements count on a video: 9
    '''
    q_pervideo_notstrict: int= 18
    ds_annotations_train, ds_annotations_test= split4train_test(
        dataset=dataset_annotations,
        suggested_q_per_video=q_pervideo_notstrict,
        show_dist=False,
    )
    '''
    max possible 80% train elements on a gloss: 144
    max possible 20% test elements on a gloss: 36
    max possible suggested elements per video: 18
    -- and is using
    144 dataset elements per gloss on Train
        from 1st 8 videos
    36 dataset elements per gloss on Test
        from last 2 videos
    180=144+36, dataset elements per gloss
        80% of 180 is 144
        20% of 180 is  36
        -----------------
        180*39=7020 total elements
            7020*.8=5616 elements for train
            7020*.2=2484 elements for test
    '''
    check_ds_is_good(
        dataset=ds_annotations_train,
        q_elements_per_g=int(
            q_pervideo_notstrict*ceil(Q_VIDS_PER_G*RATIO4TRAIN)
        ),
        q_frame=QUANTITY_FRAME,
        show_msg=False,
    )
    check_ds_is_good(
        dataset=ds_annotations_test,
        q_elements_per_g=int(
            q_pervideo_notstrict*ceil(Q_VIDS_PER_G*(1-RATIO4TRAIN))
        ), # due to python: 10*(1-0.8) = 1.9999999999999996
        q_frame=QUANTITY_FRAME,
        show_msg=False,
    )
    '''
    max possible 80% train elements on a gloss: 144
    max possible 20% test elements on a gloss: 36
    max possible suggested elements per video: 18
    '''
    '''
    TODO:
    ✔ QUANTITY_FRAME be median blah blahhh
    ✔ check mandatory all has atleast 1 hand blah
    use sklearn KNN blah for model
    do ML stats blah
    '''
if __name__=="__main__":
    main()
