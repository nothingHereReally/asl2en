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


def main() -> None:
    ds_annotations: list[dict]= load_dataset_annotations(
        show_dist=False,
        name=False,
    )


if __name__=='__main__':
    main()
