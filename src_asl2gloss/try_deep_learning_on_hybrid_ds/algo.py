from math import ceil
import numpy as np
from random import shuffle
from typing import Generator


from .constant import (
    DS_ANNOTATIONS,
    BATCH,

    KEY_G,
    KEY_GID,
    KEY_VIDS,
    KEY_PFOLDER,
    KEY_PFOLDER_FROM,
    KEY_LANDMARK,
    LANDMARK_SHAPE,

    Q_W_FACE,
    Q_W_POSE,
    Q_W_HAND,

    KEY_FILE,
    KEY_LHAND,
    KEY_RHAND,

    Q_VIDS_PER_G,
    Q_PERVIDEO_NOTSTRICT,
    QUANTITY_FRAME,

    RATIO4TRAIN,
    VAL_TEST,
    VAL_TRAIN,
)


def q_lessthan_p1_spread(video_annotation: dict, q_frame: int) -> list[dict]:
    # spread out via ceil( ... )
    assert len(video_annotation[KEY_LANDMARK]) < q_frame

    ratio: int= ceil(q_frame/len(video_annotation[KEY_LANDMARK]))
    q2be_created: int= len(video_annotation[KEY_LANDMARK]) -ceil(q_frame/ratio) +1
    output: list[dict]= []
    for init_idx in range(q2be_created):
        output.append({
            KEY_GID: video_annotation[KEY_GID],
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
            KEY_GID: video_annotation[KEY_GID],
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
        KEY_GID: video_annotation[KEY_GID],
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
        KEY_GID: video_annotation[KEY_GID],
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
            KEY_GID: video_annotation[KEY_GID],
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
            KEY_GID: video_annotation[KEY_GID],
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
def flatten_and_check_dataset(dataset: list[dict]) -> list[dict]:
    '''
    flattens dataset to just dataset elements and checks file that they exists
    input( dataset ) be of [
        {
            KEY_G: str,
            KEY_VIDS: [...]
        }
    ]
    -----------------------
    returns list[dict] of [
        {
            KEY_GID: int,
            KEY_PFOLDER_FROM: Path,
            KEY_PFOLDER: str,
            KEY_LANDMARK: [ -- length of KEY_LANDMARK --> QUANTITY_FRAME
                {
                    KEY_FILE: str,
                    KEY_FACE: bool,
                    KEY_POSE: bool,
                    KEY_LHAND: bool,
                    KEY_RHAND: bool,
                },
                ...
            ]
        }
    ]
    '''
    output: list[dict]= []
    for a_gloss in dataset:
        for a_video in a_gloss[KEY_VIDS]:
            if len(a_video[KEY_LANDMARK]) != QUANTITY_FRAME:
                raise ValueError(f"{a_video[KEY_PFOLDER]} has {len(a_video[KEY_LANDMARK])} image, should be {QUANTITY_FRAME}")

            if not (a_video[KEY_PFOLDER_FROM] /a_video[KEY_PFOLDER]).exists():
                raise FileNotFoundError(f"error: {a_video[KEY_PFOLDER]} doesn't exists")

            for an_img_detail in a_video[KEY_LANDMARK]:
                if not (a_video[KEY_PFOLDER_FROM] /a_video[KEY_PFOLDER] /an_img_detail[KEY_FILE]).exists():
                    raise FileNotFoundError(f"error: {a_video[KEY_PFOLDER]}/{an_img_detail[KEY_FILE]} doesn't exists")

            output.append({
                KEY_GID: a_video[KEY_GID],
                KEY_PFOLDER_FROM: a_video[KEY_PFOLDER_FROM],
                KEY_PFOLDER: a_video[KEY_PFOLDER],
                KEY_LANDMARK: a_video[KEY_LANDMARK],
            })
    return output
def extract_ds_annotation_split(split: str, q_pervideo_notstrict: int, show_dist: bool=False) -> list[dict]:
    '''
    returns list[dict] of [
        {
            KEY_GID: int,
            KEY_PFOLDER_FROM: Path,
            KEY_PFOLDER: str,
            KEY_LANDMARK: [ -- length of KEY_LANDMARK --> QUANTITY_FRAME
                {
                    KEY_FILE: str,
                    KEY_FACE: bool,
                    KEY_POSE: bool,
                    KEY_LHAND: bool,
                    KEY_RHAND: bool,
                },
                ...
            ]
        }
    ]
    '''
    # assert split in (VAL_TRAIN, VAL_TEST), due2already check on get_data_landmark(...)

    output: list[dict]= []
    min_element_count_on_gloss: int= 999_999
    min_element_count_on_video: int= 999_999
    if show_dist:
        print(f"---- dataset {split} -- elements(obeys QUANTITY_FRAME) count ----")
    for a_gloss in DS_ANNOTATIONS:
        output.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })

        # -- to follow 80% train, 20% test
        assert len(a_gloss[KEY_VIDS]) == Q_VIDS_PER_G
        q_elements_per_g: int= -1
        if split == VAL_TRAIN:
            output[-1][KEY_VIDS]= a_gloss[KEY_VIDS][:ceil(RATIO4TRAIN*Q_VIDS_PER_G)]
            q_elements_per_g= q_pervideo_notstrict*ceil(RATIO4TRAIN*Q_VIDS_PER_G)
        elif split == VAL_TEST:
            output[-1][KEY_VIDS]= a_gloss[KEY_VIDS][ceil(RATIO4TRAIN*Q_VIDS_PER_G):]
            q_elements_per_g= q_pervideo_notstrict*ceil((1-RATIO4TRAIN)*Q_VIDS_PER_G)

        ds_elements_per_gloss: list[dict]= []
        ds_elements_notused: list[dict]= []
        pergloss_pervideo_count: list[int]= []
        for a_video in output[-1][KEY_VIDS]:
            tmp_elements_a_video: list[dict]= []
            if len(a_video[KEY_LANDMARK]) < QUANTITY_FRAME:
                q_added: int= 0

                # part 1
                elements: list[dict]= q_lessthan_p1_spread(a_video, q_frame=QUANTITY_FRAME)
                tmp_elements_a_video.extend(elements)
                q_added+= len(elements)

                # part 2
                elements= q_lessthan_p2_loop(a_video, q_frame=QUANTITY_FRAME)
                tmp_elements_a_video.extend(elements)
                q_added+= len(elements)

                # part 3
                elements= q_lessthan_p3_stretch_startend(a_video, q_frame=QUANTITY_FRAME)
                tmp_elements_a_video.extend(elements)
                q_added+= len(elements)

                if q_added < min_element_count_on_video:
                    min_element_count_on_video= q_added
                pergloss_pervideo_count.append(q_added)
            elif len(a_video[KEY_LANDMARK]) == QUANTITY_FRAME:
                # part 4
                elements: list[dict]= q_equalto_p4_loop(a_video, q_frame=QUANTITY_FRAME)
                tmp_elements_a_video.extend(elements)

                if len(elements) < min_element_count_on_video:
                    min_element_count_on_video= len(elements)
                pergloss_pervideo_count.append(len(elements))
            else:
                # part 5
                elements: list[dict]= q_greaterthan_p5_loop(a_video, q_frame=QUANTITY_FRAME)
                tmp_elements_a_video.extend(elements)

                if len(elements) < min_element_count_on_video:
                    min_element_count_on_video= len(elements)
                pergloss_pervideo_count.append(len(elements))


            if len(tmp_elements_a_video) <= q_pervideo_notstrict:
                ds_elements_per_gloss.extend(tmp_elements_a_video)
            else:
                ds_elements_per_gloss.extend(tmp_elements_a_video[:q_pervideo_notstrict])
                ds_elements_notused.extend(tmp_elements_a_video[q_pervideo_notstrict:])


        # -- fill missing due2target --> q_elements_per_g, ie. elements per gloss
        if len(ds_elements_per_gloss) < q_elements_per_g:
            q_missing_elements: int= q_elements_per_g -len(ds_elements_per_gloss)
            assert q_missing_elements <= len(ds_elements_notused)
            ds_elements_per_gloss.extend(ds_elements_notused[:q_missing_elements])
        assert len(ds_elements_per_gloss) == q_elements_per_g
        output[-1][KEY_VIDS]= ds_elements_per_gloss

        if show_dist:
            print(f"-- gloss( {output[-1][KEY_G]} ) -- quantity elements max: {sum(pergloss_pervideo_count)} --(to)--> {len(output[-1][KEY_VIDS])}")
            for idx_video, elements_count_on_a_video in enumerate(pergloss_pervideo_count):
                print(f"    {str(idx_video+1).zfill(3)}: {elements_count_on_a_video}")
            print()
            if len(output[-1][KEY_VIDS]) < min_element_count_on_gloss:
                min_element_count_on_gloss= len(output[-1][KEY_VIDS])
    if show_dist:
        print('------------------------')
        print(f"minimum elements count on a gloss: {min_element_count_on_gloss}")
        print(f"minimum elements count on a video: {min_element_count_on_video}")
        print('------------------------')

    output= flatten_and_check_dataset(output)

    return output


def flip_vertical_y_axis(landmarks) -> np.ndarray:
    '''
    landmarks be of shape (INT, ..., 86, 2)
    '''
    new_landmark= np.array(landmarks).copy()
    new_landmark[..., 0]= 1 -new_landmark[..., 0]
    lhand2rhand: np.ndarray= new_landmark[...,
        Q_W_FACE +Q_W_POSE:
        Q_W_FACE +Q_W_POSE +Q_W_HAND
    ,:].copy()
    rhand2lhand: np.ndarray= new_landmark[...,
        Q_W_FACE +Q_W_POSE +Q_W_HAND:
    ,:].copy()
    new_landmark[...,
        Q_W_FACE +Q_W_POSE:
        Q_W_FACE +Q_W_POSE +Q_W_HAND
    ,:]= rhand2lhand # ie. past was right hand, now be left hand
    new_landmark[...,
        Q_W_FACE +Q_W_POSE +Q_W_HAND:
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


def loadnpfiles(annotation: dict) -> list[np.ndarray]:
    '''
    annotation is dict having keys
    -- KEY_GID: int
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


def from86extract87andflip(annotation: dict) -> list[np.ndarray]:
    '''
    annotation dict has keys
    -- KEY_GID: int
    -- KEY_PFOLDER_FROM: Path
    -- KEY_PFOLDER: str
    -- KEY_LANDMARK: list[dict] of length QUANTITY_FRAME
    '''
    output: list[list[np.ndarray]]|np.ndarray= [loadnpfiles(annotation)]
    output.append(flip_vertical_y_axis(output[0]).tolist())
    # output of shape (2, QUANTITY_FRAME, 86, 2)
    output= np.pad(
        array=np.array(output),
        pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), # due2on axis=-2 after
        mode='constant',
        constant_values=0.0,
    )
    for a_video in dataset:
        idx_init: int= idx_init_has_hand(a_video[KEY_LANDMARK])
        if idx_init!=-1:
            if len(a_video[KEY_LANDMARK])<=QUANTITY_FRAME:
                total_DS+= 1
            else:
                len_available_images: int= len(a_video[KEY_LANDMARK])-idx_init
                o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
                total_DS+= 1 # for part 1
                if QUANTITY_FRAME<len_available_images:
                    total_DS+= int(
                        o2t_mod *(len(a_video[KEY_LANDMARK]) -(idx_init+ QUANTITY_FRAME*o2t_mod))
                    ) # for part 2
                    for work4mod_what in MOD_PART_3: # for part 3
                        if 0<int(len_available_images//work4mod_what):
                            total_DS+= (len_available_images//work4mod_what) -QUANTITY_FRAME +1
                        else:
                            total_DS+= 1
                else:
                    total_DS+= 1
    return int(ceil(total_DS/float(batch_size)))


def correct_hand_position(landmarks: np.ndarray, left_or_right: str) -> np.ndarray:
    move_howmuch: float= 0.02
    assert landmarks.shape == (86, 2)
    if left_or_right == KEY_LHAND:
        target= landmarks[LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] + move_howmuch
        move= target - landmarks[LM_IDX_LEFT_HAND_WRIST:LM_IDX_LEFT_HAND_WRIST+1, :]
        landmarks[
            LEN_W_FACE+LEN_W_POSE :LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK,
            :
        ]= landmarks[
            LEN_W_FACE+LEN_W_POSE :LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK,
            :
        ] + move
        target[:, 1]= target[:, 1] - move_howmuch
        landmarks[LM_IDX_LEFT_ARM_TIP]= target.copy()
        landmarks[LM_IDX_LEFT_ELBOW, :]= (
            landmarks[LM_IDX_LEFT_SHOULDER, :] +
            target
        )/2.0
    elif left_or_right == KEY_RHAND:
        target= landmarks[LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :] - move_howmuch
        move= target - landmarks[LM_IDX_RIGHT_HAND_WRIST:LM_IDX_RIGHT_HAND_WRIST+1, :]
        landmarks[
            LEN_W_FACE +LEN_W_POSE +QUANTITY_HAND_LMARK:,
            :
        ]= landmarks[
            LEN_W_FACE +LEN_W_POSE +QUANTITY_HAND_LMARK:,
            :
        ] + move
        target[:, 1]= target[:, 1] - move_howmuch
        landmarks[LM_IDX_RIGHT_ARM_TIP]= target.copy()
        landmarks[LM_IDX_RIGHT_ELBOW, :]= (
            landmarks[LM_IDX_RIGHT_SHOULDER, :] +
            target
        )/2.0
    else:
        raise ValueError(f'left_or_right can only be: {KEY_LHAND} or {KEY_RHAND}')
    return landmarks
def load_npfiles(landmark: dict) -> list[np.ndarray]:
    out: list= []
    for an_img_detail in landmark[KEY_LANDMARK]:
        assert an_img_detail[KEY_FACE] and an_img_detail[KEY_POSE] and \
            ( an_img_detail[KEY_LHAND] or an_img_detail[KEY_RHAND] )
        with open(f"{landmark[KEY_PFOLDER_FROM] /landmark[KEY_PFOLDER] /an_img_detail[KEY_FILE]}", 'rb') as f:
            if not an_img_detail[KEY_LHAND] or not an_img_detail[KEY_RHAND]:
                with open(LEFT_HAND_DIR_FILE, 'rb') as lhf:
                    left_hand_data= np.load(lhf)
                tmp: np.ndarray= np.load(f)
                if an_img_detail[KEY_LHAND]:
                    tmp[LEN_W_FACE+LEN_W_POSE:LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK, :]= left_hand_data
                else:
                    left_hand_data[:, 0]= 1- left_hand_data[:, 0]
                    tmp[LEN_W_FACE+LEN_W_POSE+QUANTITY_HAND_LMARK:, :]= left_hand_data
                out.append(correct_hand_position(
                    landmarks=tmp,
                    left_or_right=KEY_LHAND if an_img_detail[KEY_LHAND] else KEY_RHAND,
                ))
            else:
                out.append(np.load(f))
    return out
def load_elements_npfiles(elements: list) -> list:
    out: list= []
    for el in elements:
        out.append(load_npfiles(el))
    return out


def has_atleast_1hand(annotation: dict) -> bool:
    return (annotation[KEY_LHAND] or annotation[KEY_RHAND]) and \
            annotation[KEY_FACE] and annotation[KEY_POSE]
def idx_init_has_hand(landmarks: list) -> int:
    for idx in range(len(landmarks)):
        if has_atleast_1hand(landmarks[idx]):
            return idx
    return -1
def q_imgs_less_than_or_equal(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> dict:
    annotations: dict= {
        KEY_PFOLDER_FROM: parent_folder_from,
        KEY_PFOLDER: parent_folder,
        KEY_LANDMARK: [],
    }

    idx_init: int= idx_init_has_hand(landmarks)
    ratio: int= int(ceil(QUANTITY_FRAME / (len(landmarks)-idx_init)))
    for landmark in landmarks[idx_init:]:
        if has_atleast_1hand(landmark) and len(annotations[KEY_LANDMARK])<QUANTITY_FRAME:
            annotations[KEY_LANDMARK].extend([
                {
                    KEY_FILE:  landmark[KEY_FILE],
                    KEY_FACE:  landmark[KEY_FACE],
                    KEY_POSE:  landmark[KEY_POSE],
                    KEY_LHAND: landmark[KEY_LHAND],
                    KEY_RHAND: landmark[KEY_RHAND],
                }
            ] * min(ratio, QUANTITY_FRAME -len(annotations[KEY_LANDMARK])))
        elif len(annotations[KEY_LANDMARK])<QUANTITY_FRAME:
            annotations[KEY_LANDMARK].extend([
                {
                    KEY_FILE:  annotations[KEY_LANDMARK][-1][KEY_FILE],
                    KEY_FACE:  annotations[KEY_LANDMARK][-1][KEY_FACE],
                    KEY_POSE:  annotations[KEY_LANDMARK][-1][KEY_POSE],
                    KEY_LHAND: annotations[KEY_LANDMARK][-1][KEY_LHAND],
                    KEY_RHAND: annotations[KEY_LANDMARK][-1][KEY_RHAND],
                }
            ] * min(ratio, QUANTITY_FRAME -len(annotations[KEY_LANDMARK])))

    return annotations
def greater_than_qf_p1(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> dict:
    assert has_atleast_1hand(landmarks[0])
    annotations: dict= {
        KEY_PFOLDER_FROM: parent_folder_from,
        KEY_PFOLDER: parent_folder,
        KEY_LANDMARK: [],
    }

    ratio: int= floor(len(landmarks) / QUANTITY_FRAME)
    for idx in range(QUANTITY_FRAME):
        a_landmark = landmarks[idx * ratio]
        if has_atleast_1hand(a_landmark):
            annotations[KEY_LANDMARK].append({
                KEY_FILE: a_landmark[KEY_FILE],
                KEY_FACE: a_landmark[KEY_FACE],
                KEY_POSE: a_landmark[KEY_POSE],
                KEY_LHAND: a_landmark[KEY_LHAND],
                KEY_RHAND: a_landmark[KEY_RHAND],
            })
        else:
            annotations[KEY_LANDMARK].append({
                KEY_FILE:  annotations[KEY_LANDMARK][-1][KEY_FILE],
                KEY_FACE:  annotations[KEY_LANDMARK][-1][KEY_FACE],
                KEY_POSE:  annotations[KEY_LANDMARK][-1][KEY_POSE],
                KEY_LHAND: annotations[KEY_LANDMARK][-1][KEY_LHAND],
                KEY_RHAND: annotations[KEY_LANDMARK][-1][KEY_RHAND],
            })

    return annotations
def greater_than_qf_p2(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> list:
    assert has_atleast_1hand(landmarks[0])
    annotations_many: list= []

    how_many_qfs: int= floor(len(landmarks)/QUANTITY_FRAME)
    remains: int=      len(landmarks) - how_many_qfs*QUANTITY_FRAME
    mod_list: tuple=   tuple(range(how_many_qfs))
    past_notation: dict= {
        KEY_FILE: landmarks[0][KEY_FILE],
        KEY_FACE: landmarks[0][KEY_FACE],
        KEY_POSE: landmarks[0][KEY_POSE],
        KEY_LHAND: landmarks[0][KEY_LHAND],
        KEY_RHAND: landmarks[0][KEY_RHAND],
    }
    for idx_init_only in range(remains+1):
        tmp_annotations: list= [{
            KEY_PFOLDER_FROM: parent_folder_from,
            KEY_PFOLDER: parent_folder,
            KEY_LANDMARK: [],
        } for _ in range(how_many_qfs)]
        for idx in range(
            idx_init_only,
            idx_init_only +(how_many_qfs*QUANTITY_FRAME)
        ):
            for mod_what in mod_list:
                if (idx%how_many_qfs)==mod_what:
                    if has_atleast_1hand(landmarks[idx]):
                        tmp_annotations[mod_what][KEY_LANDMARK].append({
                            KEY_FILE:  landmarks[idx][KEY_FILE],
                            KEY_FACE:  landmarks[idx][KEY_FACE],
                            KEY_POSE:  landmarks[idx][KEY_POSE],
                            KEY_LHAND: landmarks[idx][KEY_LHAND],
                            KEY_RHAND: landmarks[idx][KEY_RHAND],
                        })
                        past_notation= {
                            KEY_FILE:  landmarks[idx][KEY_FILE],
                            KEY_FACE:  landmarks[idx][KEY_FACE],
                            KEY_POSE:  landmarks[idx][KEY_POSE],
                            KEY_LHAND: landmarks[idx][KEY_LHAND],
                            KEY_RHAND: landmarks[idx][KEY_RHAND],
                        }
                    else:
                        tmp_annotations[mod_what][KEY_LANDMARK].append(past_notation.copy())
        annotations_many.extend(tmp_annotations)
    return annotations_many
def greater_than_qf_p3(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> list:
    assert has_atleast_1hand(landmarks[0])

    annotations_many: list= []
    groups_notation: list= [[] for _ in MOD_PART_3]
    past_annotation: dict= {
        KEY_FILE:  landmarks[0][KEY_FILE],
        KEY_FACE:  landmarks[0][KEY_FACE],
        KEY_POSE:  landmarks[0][KEY_POSE],
        KEY_LHAND: landmarks[0][KEY_LHAND],
        KEY_RHAND: landmarks[0][KEY_RHAND],
    }

    for idx, a_landmark in enumerate(landmarks):
        for idx_mod, what_mod in enumerate(MOD_PART_3):
            if ((idx + 1) % what_mod) == 0:
                if has_atleast_1hand(a_landmark):
                    past_annotation= {
                        KEY_FILE: a_landmark[KEY_FILE],
                        KEY_FACE: a_landmark[KEY_FACE],
                        KEY_POSE: a_landmark[KEY_POSE],
                        KEY_LHAND: a_landmark[KEY_LHAND],
                        KEY_RHAND: a_landmark[KEY_RHAND],
                    }
                groups_notation[idx_mod].append(past_annotation.copy())

    for group in groups_notation:
        if len(group) <= QUANTITY_FRAME:
            annotations: list= []
            ratio: int= ceil(QUANTITY_FRAME / len(group))

            for annotation in group:
                if len(annotations)<QUANTITY_FRAME:
                    annotations.extend(
                        [annotation.copy()] * min(
                            ratio, QUANTITY_FRAME - len(annotations)
                        )
                    )

            annotations_many.append({
                KEY_PFOLDER_FROM: parent_folder_from,
                KEY_PFOLDER: parent_folder,
                KEY_LANDMARK: annotations,
            })
        else:
            for idx_init in range(len(group) - QUANTITY_FRAME + 1):
                annotations_many.append({
                    KEY_PFOLDER_FROM: parent_folder_from,
                    KEY_PFOLDER: parent_folder,
                    KEY_LANDMARK: group[
                        idx_init:idx_init + QUANTITY_FRAME
                    ],
                })

    return annotations_many
def q_imgs_greater_than(
    landmarks: list,
    parent_folder_from: Path,
    parent_folder: str,
) -> list:
    annotations_many: list= []

    idx_init: int= idx_init_has_hand(landmarks)
    q_imgs_available: int= len(landmarks)-idx_init
    if q_imgs_available<=QUANTITY_FRAME:
        annotation: dict= q_imgs_less_than_or_equal(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.append(annotation)
    else:
        # ---- part 1 ----
        annotation: dict= greater_than_qf_p1(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.append(annotation)

        # ---- part 2 ----
        annotations: list= greater_than_qf_p2(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.extend(annotations)

        # ---- part 3 ----
        annotations= greater_than_qf_p3(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.extend(annotations)
    return annotations_many
def get_data_landmark(
    train_test: str=VAL_TRAIN,
    batch_size: int=BATCH
) -> Generator:
    dataset: list= extract_ds_annotation_split(
        wlasl_or_glasl='wlasl',
        split=train_test,
    )
    # dataset.extend(extract_ds_annotation_split(
    #     wlasl_or_glasl='glasl',
    #     split=train_test,
    # ))
    dataset_idxs: tuple= tuple(sample(
        range(len(dataset)),
        len(dataset)
    ))

    batch_ds_elements: list= []
    batch_ds_class: list= []
    while True:
        for idx_ds in dataset_idxs:
            an_element_ds: dict= dataset[idx_ds]
            if len(an_element_ds[KEY_LANDMARK])<=QUANTITY_FRAME:
                batch_ds_class.append(an_element_ds[KEY_GID])
                batch_ds_elements.append(q_imgs_less_than_or_equal(
                    landmarks=an_element_ds[KEY_LANDMARK],
                    parent_folder_from=an_element_ds[KEY_PFOLDER_FROM],
                    parent_folder=an_element_ds[KEY_PFOLDER],
                ))
            else:
                result_from_many: list= q_imgs_greater_than(
                    landmarks=an_element_ds[KEY_LANDMARK],
                    parent_folder_from=an_element_ds[KEY_PFOLDER_FROM],
                    parent_folder=an_element_ds[KEY_PFOLDER],
                )
                batch_ds_class.extend( [an_element_ds[KEY_GID]] * len(result_from_many) )
                batch_ds_elements.extend(result_from_many)
            # ---- now ready ----
            while batch_size<=len(batch_ds_class):
                out_inputs: np.ndarray= np.array(
                    load_elements_npfiles(batch_ds_elements[:batch_size]),
                )
                out_inputs= np.concatenate([
                    out_inputs,
                    flip_vertical_x_axis(out_inputs)
                ], axis=0)
                out_expected_outputs: np.ndarray= np.array(batch_ds_class[:batch_size])
                out_expected_outputs= np.concatenate([
                    out_expected_outputs,
                    out_expected_outputs,
                ], axis=0)
                assert out_inputs.shape==(batch_size*2, QUANTITY_FRAME, 86, 2)
                assert out_expected_outputs.shape==(batch_size*2,)
                assert out_inputs.shape[0]==out_expected_outputs.shape[0]

                batch_ds_elements= batch_ds_elements[batch_size:]
                batch_ds_class= batch_ds_class[batch_size:]

                yield (out_inputs, out_expected_outputs)
