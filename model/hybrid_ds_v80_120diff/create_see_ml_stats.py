from json import load as loadJson
from math import ceil
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS1_22_LANDMARK_DIR: Path= PROJ_ROOT / 'dataset' / 'clean_dataset' / 'fix_recent_n_classic_landmark'
DS2_16_LANDMARK_DIR: Path= PROJ_ROOT / 'dataset' / 'asl_citizen' / 'selected_start_end_landmark'
DS1_22_cleaned_wlasl: Path= PROJ_ROOT / 'dataset' / 'clean_dataset' / 'fix_recent_n_classic_landmark.json'
DS2_16_cleaned_aslct: Path= PROJ_ROOT / 'dataset' / 'asl_citizen' / 'selected_start_end_landmark.json'
STATS_PATH: Path= PROJ_ROOT /"model" /"hybrid_ds_v80_120diff" /"ml_stats_image"

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
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
QUANTITY_FRAME: int= 26
RATIO4TRAIN: float= 0.8
Q_VIDS_PER_G: int= 10
Q_PERVIDEO_NOTSTRICT= 18


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


def save_stats(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    create_files: bool=False,
) -> None:
    class_ids: list[int]= list(range(len(labels)))
    scores: dict[str, float]= {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, labels=class_ids, average="macro",
        ),
        "recall": recall_score(
            y_true, y_pred, labels=class_ids, average="macro",
        ),
        "f1_score": f1_score(
            y_true, y_pred, labels=class_ids, average="macro",
        ),
    }
    if not STATS_PATH.exists() and create_files:
        STATS_PATH.mkdir()

    print(f"\n-------------------------")
    print(f"---- {split} ----")
    for metric, score in scores.items():
        # if create_files:
        #     image= np.full((120, 520, 3), 255, dtype=np.uint8)
        #     cv2.putText(image, f"{split.capitalize()} {metric.replace('_', '-')}:",
        #                 (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
        #     cv2.putText(image, f"{score * 100:.2f}%", (20, 96),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 110, 40), 2)
        #     image_path= STATS_PATH / f"{split}_{metric}.png"
        #     if not cv2.imwrite(str(image_path), image):
        #         raise OSError(f"Could not write statistics image: {image_path}")
        print(f"{metric} on {split}: {score * 100:.5f} %")
    print()

    if create_files:
        matrix= confusion_matrix(y_true, y_pred, labels=class_ids)
        cell_size: int= max(28, min(56, 1200 // max(len(class_ids), 1)))
        label_width: int= 180
        header_height: int= 50
        image= np.full((header_height +len(class_ids) *cell_size,
                        label_width +len(class_ids) *cell_size, 3), 255, dtype=np.uint8)
        cv2.putText(
            img=image,
            text="Actual row / predicted column",
            org=(5, 28),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(35, 35, 35),
            thickness=1,
        )
        for row, gloss in enumerate(labels):
            y0= header_height +row *cell_size
            cv2.putText(image, f"{row}: {gloss}", (5, y0 +cell_size // 2 +5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (35, 35, 35), 1)
            for col in range(len(class_ids)):
                x0= label_width +col *cell_size
                cv2.putText(image, str(col), (x0 +3, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (35, 35, 35), 1)
                intensity= int(255 *(1 -matrix[row, col] / max(matrix.max(), 1)))
                cv2.rectangle(image, (x0, y0), (x0 +cell_size, y0 +cell_size),
                              (intensity, 220, intensity), thickness=-1)
                cv2.rectangle(image, (x0, y0), (x0 +cell_size, y0 +cell_size),
                              (180, 180, 180), thickness=1)
                cv2.putText(image, str(matrix[row, col]),
                            (x0 +3, y0 +cell_size // 2 +5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)
        matrix_path= STATS_PATH / f"{split}_confusion_matrix.png"
        if not cv2.imwrite(str(matrix_path), image):
            raise OSError(f"Could not write confusion matrix image: {matrix_path}")


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
def extract_ds_annotation_split(
    dataset: list[dict],
    split: str,
    q_pervideo_notstrict: int,
    show_dist: bool=False,
) -> list[dict]:
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
    # assert split in (VAL_TRAIN, VAL_TEST), due2already check on dataset_generator(...)

    output: list[dict]= []
    min_element_count_on_gloss: int= 999_999
    min_element_count_on_video: int= 999_999
    if show_dist:
        print(f"---- dataset {split} -- elements(obeys QUANTITY_FRAME) count ----")
    for a_gloss in dataset:
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


def main() -> None:
    ds_annotations: list[dict]= load_dataset_annotations(
        show_dist=False,
        name=False,
    )


if __name__=='__main__':
    main()
