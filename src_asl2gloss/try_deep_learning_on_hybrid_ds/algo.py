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
    for idx, an_img_detail in enumerate(annotation[KEY_LANDMARK]):
        if an_img_detail[KEY_LHAND]:
            output[0, idx, 86, 0]= 1.0
            output[1, idx, 86, 1]= 1.0
        else:
            output[0, idx, 86, 0]= 0.0
            output[1, idx, 86, 1]= 0.0

        if an_img_detail[KEY_RHAND]:
            output[0, idx, 86, 1]= 1.0
            output[1, idx, 86, 0]= 1.0
        else:
            output[0, idx, 86, 1]= 0.0
            output[1, idx, 86, 0]= 0.0

    return [output[0], output[1]]


def get_data_landmark(train_test: str=VAL_TRAIN, batch_size: int=BATCH) -> Generator:
    assert train_test in (VAL_TRAIN, VAL_TEST)
    assert 0 < batch_size

    dataset: list[dict]= extract_ds_annotation_split(
        split=train_test,
        q_pervideo_notstrict=Q_PERVIDEO_NOTSTRICT,
        show_dist=False,
    )
    dataset_idxs: list[int]= list(range(len(dataset)))
    shuffle(dataset_idxs)

    batch_annotations: list[dict]= []
    while True:
        for idx_ds in dataset_idxs:
            batch_annotations.append(dataset[idx_ds])
            while batch_size <= len(batch_annotations):
                x_ds_elements_batch: list[np.ndarray]|np.ndarray= []
                y_ds_elements_batch: list[int]= []
                for a_ds_element in batch_annotations[:batch_size]:
                    x_ds_elements_batch.extend(from86extract87andflip(a_ds_element))
                    y_ds_elements_batch.append(a_ds_element[KEY_GID])
                    y_ds_elements_batch.append(a_ds_element[KEY_GID])
                batch_annotations= batch_annotations[batch_size:]

                x_ds_elements_batch= np.array(x_ds_elements_batch)
                assert x_ds_elements_batch.shape == (batch_size*2, QUANTITY_FRAME, *LANDMARK_SHAPE)
                yield (
                    np.array(x_ds_elements_batch),
                    np.array(y_ds_elements_batch),
                )


if __name__=='__main__':
    pass
    '''
    ds_train: list[dict]= extract_ds_annotation_split(
        split=VAL_TRAIN,
        q_pervideo_notstrict=Q_PERVIDEO_NOTSTRICT,
        show_dist=True,
    )
    ds_test: list[dict]= extract_ds_annotation_split(
        split=VAL_TEST,
        q_pervideo_notstrict=Q_PERVIDEO_NOTSTRICT,
        show_dist=True,
    )
    output: list[dict]= []
    for a_ds_element in ds_train:
        output.append({
            KEY_GID: a_ds_element[KEY_GID],
            KEY_PFOLDER_FROM: a_ds_element[KEY_PFOLDER_FROM].name,
            KEY_PFOLDER: a_ds_element[KEY_PFOLDER],
            KEY_LANDMARK: a_ds_element[KEY_LANDMARK],
        })
    ds_train= output
    output= []
    for a_ds_element in ds_test:
        output.append({
            KEY_GID: a_ds_element[KEY_GID],
            KEY_PFOLDER_FROM: a_ds_element[KEY_PFOLDER_FROM].name,
            KEY_PFOLDER: a_ds_element[KEY_PFOLDER],
            KEY_LANDMARK: a_ds_element[KEY_LANDMARK],
        })
    ds_test= output
    with open('/tmp/ds_train_flatten.json', 'w') as f:
        writeJson(ds_train, f, indent=4)
    with open('/tmp/ds_test_flatten.json', 'w') as f:
        writeJson(ds_test, f, indent=4)
    '''
