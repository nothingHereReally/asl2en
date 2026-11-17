import cv2
from json import load as loadJson
from keras.src.activations.activations import ReLU
from keras.src.saving import load_model
from keras.src.layers import Dense, LayerNormalization, MultiHeadAttention, Layer
from keras.src import ops
from keras.src.saving import register_keras_serializable
from math import ceil
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from typing import Any


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS1_22_LANDMARK_DIR: Path= PROJ_ROOT / 'dataset' / 'clean_dataset' / 'fix_recent_n_classic_landmark'
DS2_16_LANDMARK_DIR: Path= PROJ_ROOT / 'dataset' / 'asl_citizen' / 'selected_start_end_landmark'
DS1_22_cleaned_wlasl: Path= PROJ_ROOT / 'dataset' / 'clean_dataset' / 'fix_recent_n_classic_landmark.json'
DS2_16_cleaned_aslct: Path= PROJ_ROOT / 'dataset' / 'asl_citizen' / 'selected_start_end_landmark.json'
MODEL_PATH: Path= PROJ_ROOT /"model" /"hybrid_ds_v82_120diff_111k_params" /"aslvid2gloss_v82.keras"
LABEL_PATH: Path= PROJ_ROOT /"model" /"hybrid_ds_v82_120diff_111k_params" /"labels.json"
STATS_PATH: Path= PROJ_ROOT /"model" /"hybrid_ds_v82_120diff_111k_params" /"ml_stats_image"
# ---- test GLASL ----
DS3_22_LANDMARK_DIR: Path= PROJ_ROOT /'dataset' /'glasl_verified' /'ds_landmark'
DS3_22_cleaned_glasl: Path= PROJ_ROOT /'dataset' /'glasl_verified' /'ds_landmark.json'
# ---- test GLASL ----

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
BATCH: int= 32


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


# -- for custom layers needed by model -- start -----------
@register_keras_serializable()
class EncoderTransformer( Layer ):
    def __init__(
        self,
        num_heads: int=4,  # 8 on 'attention is all u need' (2017)
        key_dim: int=43,   # 64 on 'attention is all u need' (2017)
        value_dim: int=43, # --> so that 43*4= 172= 86*2
        dropout: float=0.1,
        ann_units: int=2048,
        name: str='encoder_transformer',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads= num_heads
        self.key_dim=   key_dim
        self.value_dim= value_dim
        self.dropout=   dropout
        self.ann_units= ann_units
        self.name=      name
    def build(self, input_shape):
        # assert len(input_shape)==3
        d_model= input_shape[-1]
        self.multiHeadAtt= MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            dropout=self.dropout,
            use_bias=True,
        )
        self.normAfterMHA= LayerNormalization()
        self.ann= Dense(
            units=self.ann_units,
            activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
        )
        self.annBringBackShape= Dense(
            units=d_model,
            activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
        )
        self.normAfterAnn= LayerNormalization()
    def call(self, data_input):
        data= self.multiHeadAtt(
            query=data_input,
            value=data_input,
            key=data_input,
        )
        data= ops.add(data, data_input)
        data_after_mha= self.normAfterMHA(data)

        data= self.ann(data_after_mha)
        data= self.annBringBackShape(data)
        data= ops.add(data, data_after_mha)
        data= self.normAfterAnn(data)

        return data
    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "value_dim": self.value_dim,
            "dropout":   self.dropout,
            "ann_units": self.ann_units,
            "name":      self.name,
        }
@register_keras_serializable()
class NTimesEncoderTransformer(Layer):
    def __init__(
        self,
        n_times: int=6,
        num_heads: int=4,  # 8 on 'attention is all u need' (2017)
        key_dim: int=43,   # 64 on 'attention is all u need' (2017)
        value_dim: int=43, # --> so that 43*4= 172= 86*2
        dropout: float=0.1,
        ann_units: int=2048,
        name: str='multi_encoder_in_sequence',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_times=   n_times
        self.num_heads= num_heads
        self.key_dim=   key_dim
        self.value_dim= value_dim
        self.dropout=   dropout
        self.ann_units= ann_units
        self.name=      name
    def build(self, input_shape):
        # assert len(input_shape)==3
        del input_shape
        self.encoder_seq= [EncoderTransformer(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            dropout=self.dropout,
            ann_units=self.ann_units,
            name=f"enc_{idx}",
        ) for idx in range(self.n_times)]
    def call(self, data_input):
        data= data_input
        for encoder in self.encoder_seq:
            data= encoder(data)
        return data
    def get_config(self):
        return {
            **super().get_config(),
            "n_times":   self.n_times,
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "value_dim": self.value_dim,
            "dropout":   self.dropout,
            "ann_units": self.ann_units,
            "name":      self.name,
        }
@register_keras_serializable()
class AttentionPooling(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scoring_layer= Dense(1, activation="tanh")
        del input_shape
    def call(self, inputs):
        # -- inputs shape (B, T, D)
        scores= self.scoring_layer(inputs) # be shape (B, T, 1)
        weights= ops.softmax(scores, axis=1) # norm via softmax still shape (B, T, 1)
        # weights, accross what time is it most important
        return ops.sum(inputs * weights, axis=1) # shape (B, D)
@register_keras_serializable()
class ExtractFeatures(Layer):
    def __init__(self, name: str='extract_features', **kwargs):
        super().__init__(**kwargs)
        self.name= name
    def build(self, input_shape):
        del input_shape
    def _get2dots_frombody(self, landmarks):
        '''
        input -- shape of (..., be >= 86, 2)
        -------------------------------
        returned shape is (...,  2, 2)
        '''
        shoulder_center= (
            landmarks[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
            landmarks[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
        )/2.0
        hip_center= (
            landmarks[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
            landmarks[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
        )/2.0
        shoulder_hip_center= (shoulder_center +hip_center)/2.0
        '''
        shoulder_center and hip_center, both has shape (..., 1, 2)
        so as shoulder_hip_center has shape (..., 1, 2)
        '''
        return ops.concatenate([
            (shoulder_center+shoulder_hip_center)/2.0,
            (shoulder_hip_center+hip_center)/2.0,
        ], axis=-2)
    def call(self, input):
        left_indexf_tip: Any= input[..., LM_IDX_LEFT_INDEXF_TIP:LM_IDX_LEFT_INDEXF_TIP+1, :]
        right_indexf_tip: Any= input[..., LM_IDX_RIGHT_INDEXF_TIP:LM_IDX_RIGHT_INDEXF_TIP+1, :]
        the_rest: Any= ops.concatenate(
            [
                ops.take(input, indices=LMs_IDXs_FACE_GOOD, axis=-2),
                self._get2dots_frombody(input),
                input[..., Q_W_FACE:Q_W_FACE+Q_W_POSE, :],
                ops.take(input, indices=LMs_IDXs_LEFT_H_NOT_INDEXF, axis=-2),
                ops.take(input, indices=LMs_IDXs_RIGHT_H_NOT_INDEXF, axis=-2),
            ],
            axis=-2,
        )
        '''
        left_indexf_tip  shape (B, QUANTITY_FRAME, 1, 2)
        right_indexf_tip shape (B, QUANTITY_FRAME, 1, 2)
        the_rest of shape (B, QUANTITY_FRAME, 10 +2 +8 +20 +20, 2) --> (B, QF, 60, 2)
        '''
        left_indexf_tip= (the_rest - left_indexf_tip)**2
        right_indexf_tip= (the_rest - right_indexf_tip)**2
        left_indexf_tip= ops.sqrt(ops.sum(left_indexf_tip, axis=-1, keepdims=False))
        right_indexf_tip= ops.sqrt(ops.sum(right_indexf_tip, axis=-1, keepdims=False))
        '''
        left_indexf_tip  shape (B, QUANTITY_FRAME, 60)
        right_indexf_tip shape (B, QUANTITY_FRAME, 60)
        '''
        left_indexf_tip=  left_indexf_tip  * input[..., -1:, 0]
        right_indexf_tip= right_indexf_tip * input[..., -1:, 1]

        return ops.concatenate([
            left_indexf_tip,
            right_indexf_tip,
        ], axis=-1)
    def get_config(self):
        return {
            **super().get_config(),
            "name":      self.name,
        }
# -- for custom layers needed by model -- end -------------
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
def load_glasl_test_annotations(training_annotations: list, show_dist: bool=False) -> list[dict]:
    with open(DS3_22_cleaned_glasl, 'r') as f:
        ds3_glasl: list[dict]= loadJson(f)

    assert len(ds3_glasl) <= len(training_annotations)
    output: list[dict]= []
    for idx_g, a_gloss in enumerate(ds3_glasl):
        assert a_gloss[KEY_G] == training_annotations[idx_g][KEY_G]
        output.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS][:Q_VIDS_PER_G]:
            output[-1][KEY_VIDS].append({
                KEY_GID: idx_g,
                KEY_PFOLDER_FROM: DS3_22_LANDMARK_DIR,
                KEY_PFOLDER: a_video[KEY_PFOLDER],
                KEY_LANDMARK: a_video[KEY_LANDMARK],
            })
    if show_dist:
        for a_gloss in output:
            print(f'{a_gloss[KEY_G]} --> {len(a_gloss[KEY_VIDS])}')
    return output


def save_stats(
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    untill: int,
    create_files: bool=False,
) -> None:
    class_ids: list[int]= list(range(len(labels)))
    scores: dict[str, float]= {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, labels=class_ids[:untill], average="macro",
        ),
        "recall": recall_score(
            y_true, y_pred, labels=class_ids[:untill], average="macro",
        ),
        "f1_score": f1_score(
            y_true, y_pred, labels=class_ids[:untill], average="macro",
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
        elif split == 'all':
            output[-1][KEY_VIDS]= a_gloss[KEY_VIDS]
            q_elements_per_g= q_pervideo_notstrict*Q_VIDS_PER_G

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
def part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # xs, ys= zip(*landmarks)
    xs= list(map(lambda el: el[0], landmarks))
    ys= list(map(lambda el: el[1], landmarks))
    min_x, min_y= min(xs), min(ys)
    xNeedForward: bool= min_x<0.0
    yNeedForward: bool= min_y<0.0
    if xNeedForward or yNeedForward:
        landmarks= [(
            x -min_x    if xNeedForward    else x,
            y -min_y    if yNeedForward    else y
        ) for x, y in landmarks]
    del xs, ys, min_x, min_y, xNeedForward, yNeedForward

    max_xy: float= max(
        max(x for x, _ in landmarks),
        max(y for _, y in landmarks)
    )
    if max_xy>1:
        landmarks= [(
            x/max_xy,
            y/max_xy
        ) for x, y in landmarks]

    return landmarks
def part2_beSquareRatioOnImage(landmarks: list[tuple[float, float]], original_shape: tuple[int, int]) -> list[tuple[float, float]]:
    height, width= original_shape
    if height==width:
        return landmarks

    if width<height: # portrait, change2withRespect2Height
        scale: float= width/height
        return [(
            x*scale,
            y
        ) for x, y in landmarks]

    # landscape, change2withRespect2Width
    scale: float= height/width
    return [(
        x,
        y*scale
    ) for x, y in landmarks]
def part3_zoomInOutForPadding(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    # zoom in/out for padding be 10% each side with respect to original aspect ratio
    # ie.:
    # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
    # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
    # pad: float= 0.05
    pad: float= 4.0/158.0
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    # xs= list(filter(lambda el: el!=0, xs))
    # ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    min_x, min_y=    min(xs), min(ys)
    max_x, max_y=    max(xs), max(ys)
    scale: float= (1  -2*pad)/max(
        max_x -min_x,
        max_y -min_y
    )
    return [(
        # (x -min_x)    *scale    +pad  if x!=0 else x,
        # (y -min_y)    *scale    +pad  if y!=0 else y
        (x -min_x)    *scale    +pad,
        (y -min_y)    *scale    +pad
    ) for x, y in landmarks]
def part4_centerLandmarkVerticallyHorizontally(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 3) center landmark with same aspect ratio as original
    # center horizontally and vertically, since done padding then just
    # move to right/down
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    # xs= list(filter(lambda el: el!=0, xs))
    # ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    shift_x: float=  0.5    -(min(xs) +max(xs))  /2
    shift_y: float=  0.5    -(min(ys) +max(ys))  /2

    return [(
        # x +shift_x  if x!=0 else x,
        # y +shift_y  if y!=0 else y
        x +shift_x,
        y +shift_y
    ) for x, y in landmarks]
def normalizeLandmarks(landmarks: list[tuple[float, float]], original_shape: tuple) -> list[tuple[float, float]]:
    '''
    landmarks is an array eg. of shape (86, 2)
    original_shape is tuple (HEIGHT, WIDTH)
    '''
    # lmark_fph.face_landmarks.landmark
    # lmark_fph.pose_landmarks.landmark
    # lmark_fph.left_hand_landmarks.landmark
    # lmark_fph.right_hand_landmarks.landmark
    # logic resize to --> 480 x 480 x 3
    #     0) all coords be greater than|= 0.0 and less than|= 1.0
    #         a) lmarks x,y overwrite to [0.0, 1.0] only
    #         b) has x < 0.0 then ALL_x+abs(min(x_neg)), ie. move right
    #         b) has y < 0.0 then ALL_y+abs(min(y_neg)), ie. move down
    #         c) ALL_coords_x_y/highest_value
    #         d) eg. 1.74 then ALL_coords_x_y/1.74
    #         e) for all be scaled down with same aspect ratio as orig
    #         f) NOW all( x, y ) are 0.0 to 1.0 value only
    #     1) from old img ratio to new square img ratio
    #         a) if owx < ohy: all_x= all_x* (480*owx/ohy)/480
    #         b) if ohy < owx: all_y= all_y* (480*ohy/owx)/480
    #     2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    #         a) if too far zoom in
    #         b) if too close zoom out
    #         c) goal lowest val 0.05 both(x,y) .ie padding
    #         d) goal highest val 0.95 both(x,y)
    #         e) ie. max(lm_wx, lm_hy) == 0.9
    #     3) center landmark with same aspect ratio as original
    #         a) min_wx_hy= min( wx, hy ); max_wx_hy= max( wx, hy )
    #         b) min_wx_hy as mn; max_wx_hy as mx
    #         c) if mn is wx, all X +( (mx-mn)/(mx*2) )
    #         d) if mn is hy, all Y +( (mx-mn)/(mx*2) )
    assert 1<len(original_shape) # incorrect use of normalizeLandmarks(...), mandatory 1<len(original_shape)
    landmarks= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks)
    landmarks= part2_beSquareRatioOnImage(
        landmarks,
        (original_shape[0], original_shape[1])
    )
    landmarks= part3_zoomInOutForPadding(landmarks)
    landmarks= part4_centerLandmarkVerticallyHorizontally(landmarks)


    return landmarks
def normalizeWorthyLandmarkWrapper(
    landmarks: list[tuple[float, float]], # order --> face_pose_left_right_hand
    original_shape: tuple,
    hasLandmarks: dict
) -> list[tuple[float, float]]:
    if not (hasLandmarks["face"] or hasLandmarks["pose"] or hasLandmarks["left_hand"] or hasLandmarks["right_hand"]):
        return landmarks
    norm: list= []
    if hasLandmarks['face']:
        norm.extend(landmarks[
            :len(WORTHY_FACE_IDX)
        ])
    if hasLandmarks['pose']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
        ])
    if hasLandmarks['left_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_W_HAND
        ])
    if hasLandmarks['right_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_W_HAND:
        ])
    norm= normalizeLandmarks(landmarks=norm, original_shape=original_shape)

    out_landmark: list= []
    # -- face landmarks --
    if hasLandmarks["face"]:
        out_landmark.extend(norm[:len(WORTHY_FACE_IDX)])
    else:
        out_landmark.extend(np.zeros(
            (
                len(WORTHY_FACE_IDX),
                2
            )).tolist()
        )

    # -- pose landmarks --
    if hasLandmarks["pose"]:
        if hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
            ])
        else:
            out_landmark.extend(norm[
                :len(WORTHY_POSE_IDX)
            ])
    else:
        out_landmark.extend(np.zeros(
            (
                len(WORTHY_POSE_IDX),
                2
            )).tolist())

    # -- left hand landmarks --
    if hasLandmarks["left_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_W_HAND
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +Q_W_HAND
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
                len(WORTHY_POSE_IDX) +Q_W_HAND
            ])
        else:
            out_landmark.extend(norm[
                :Q_W_HAND
            ])
    else:
        out_landmark.extend(np.zeros(
            (
                Q_W_HAND,
                2
            )).tolist())

    # -- right hand landmarks --
    if hasLandmarks["right_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +Q_W_HAND:
            ])
        elif hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["face"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +Q_W_HAND:
            ])
        elif hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX) +Q_W_HAND:
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                Q_W_HAND:
            ])
        else:
            out_landmark.extend(norm)
    else:
        out_landmark.extend(np.zeros(
            (
                Q_W_HAND,
                2
            )).tolist())

    # out_landmark shape:
    # (
    #     len(WORTHY_FACE_IDX)
    #     +len(WORTHY_POSE_IDX)
    #     +Q_W_HAND *2,
    #     2
    # )
    return out_landmark # ie. output shape (86, 2)
def rm_left_hand(landmarks: list[np.ndarray], annotations: list[dict]) -> list[np.ndarray]:
    output: list[np.ndarray]= []
    for idx, an_img_detail in enumerate(annotations):
        tmp: np.ndarray= np.array(landmarks[idx]).copy()
        tmp[
            Q_W_FACE +Q_W_POSE:
            Q_W_FACE +Q_W_POSE +Q_W_HAND
        ]= 0.0
        tmp= np.array(normalizeWorthyLandmarkWrapper(
            landmarks=tmp.tolist(),
            original_shape=(300, 300),
            hasLandmarks={
                KEY_FACE: an_img_detail[KEY_FACE],
                KEY_POSE: an_img_detail[KEY_POSE],
                KEY_LHAND: False,
                KEY_RHAND: an_img_detail[KEY_RHAND],
            },
        ))
        output.append(tmp)
    return output
def rm_left_hand_from86extract87andflip(annotation: dict) -> list[np.ndarray]:
    '''
    annotation dict has keys
    -- KEY_GID: int
    -- KEY_PFOLDER_FROM: Path
    -- KEY_PFOLDER: str
    -- KEY_LANDMARK: list[dict] of length QUANTITY_FRAME
    '''
    output: list[list[np.ndarray]]|np.ndarray= [rm_left_hand(
        landmarks=loadnpfiles(annotation),
        annotations=annotation[KEY_LANDMARK],
    )]
    output.append(flip_vertical_y_axis(output[0]).tolist())
    # output of shape (2, QUANTITY_FRAME, 86, 2)
    output= np.pad(
        array=np.array(output),
        pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), # due2on axis=-2 after
        mode='constant',
        constant_values=0.0,
    )
    for idx, an_img_detail in enumerate(annotation[KEY_LANDMARK]):
        # -- for left hand, removed
        output[0, idx, 86, 0]= 0.0 # due to rm left hand be 0.0
        output[1, idx, 86, 1]= 0.0 # due to rm left hand be 0.0, due2for flip

        if an_img_detail[KEY_RHAND]:
            output[0, idx, 86, 1]= 1.0
            output[1, idx, 86, 0]= 1.0
        else:
            output[0, idx, 86, 1]= 0.0
            output[1, idx, 86, 0]= 0.0

    return [output[0], output[1]]


def model_prediction(
    model,
    dataset: list[dict],
    batch_size: int=BATCH,
) -> tuple[list[int], list[int]]:
    out_predicted: list[int]= []
    out_actual: list[int]= []

    for idx_start in range(0, len(dataset), batch_size):
        batch_x_ds_elements__nums: list[np.ndarray]= []

        for a_ds_element in dataset[idx_start:idx_start+batch_size]:
            batch_x_ds_elements__nums.extend(from86extract87andflip(a_ds_element))
            out_actual.extend([a_ds_element[KEY_GID], a_ds_element[KEY_GID]])
        out_predicted.extend(np.argmax(model.predict(
            x=np.array(batch_x_ds_elements__nums),
            batch_size=len(batch_x_ds_elements__nums)
        ), axis=-1).tolist())

    return (out_actual, out_predicted)


def model_prediction_rm_left(
    model,
    labels: list[str],
    dataset: list[dict],
    batch_size: int=BATCH,
) -> tuple[list[int], list[int]]:
    out_predicted: list[int]= []
    out_actual: list[int]= []
    rm_left_hand: tuple[str, ...]= (
        'drink',
        'who',
        'candy',
        'cousin',
        'mine_my',
        'me_i',
        'stomach',
        'need',
        'see',
        'feel',
        'fever',
    )
    rm_left_hand_g_id: list[int]= []
    for idx, gloss in enumerate(labels):
        if gloss in rm_left_hand:
            rm_left_hand_g_id.append(idx)

    for idx_start in range(0, len(dataset), batch_size):
        batch_x_ds_elements__nums: list[np.ndarray]= []

        for a_ds_element in dataset[idx_start:idx_start+batch_size]:
            if a_ds_element[KEY_GID] not in rm_left_hand_g_id:
                batch_x_ds_elements__nums.extend(from86extract87andflip(a_ds_element))
            else:
                batch_x_ds_elements__nums.extend(rm_left_hand_from86extract87andflip(a_ds_element))

            out_actual.extend([a_ds_element[KEY_GID], a_ds_element[KEY_GID]])
        out_predicted.extend(np.argmax(model.predict(
            x=np.array(batch_x_ds_elements__nums),
            batch_size=len(batch_x_ds_elements__nums)
        ), axis=-1).tolist())

    return (out_actual, out_predicted)


def main() -> None:
    ds_annotations: list[dict]= load_dataset_annotations(
        show_dist=False,
        name=False,
    )
    glasl_test_annotations: list[dict]= load_glasl_test_annotations(
        ds_annotations,
        show_dist=False,
    )
    glasl_total_class: int= len(glasl_test_annotations)


    ds_train_annotations: list[dict]= extract_ds_annotation_split(
        dataset=ds_annotations,
        split=VAL_TRAIN,
        q_pervideo_notstrict=Q_PERVIDEO_NOTSTRICT,
        show_dist=False,
    )
    ds_test_annotations: list[dict]= extract_ds_annotation_split(
        dataset=ds_annotations,
        split=VAL_TEST,
        q_pervideo_notstrict=Q_PERVIDEO_NOTSTRICT,
        show_dist=False,
    )
    glasl_test_annotations: list[dict]= extract_ds_annotation_split(
        dataset=glasl_test_annotations,
        split='all',
        q_pervideo_notstrict=Q_PERVIDEO_NOTSTRICT,
        show_dist=False,
    )
    model= load_model(str(MODEL_PATH))
    with open(LABEL_PATH, 'r') as f:
        labels: list[str]= loadJson(f)
    actual_y_train, pred_y_train= model_prediction(model, ds_train_annotations)
    actual_y_test, pred_y_test= model_prediction(model, ds_test_annotations)
    actual_y_glasl, pred_y_glasl= model_prediction_rm_left(
        model,
        labels=labels[:glasl_total_class],
        dataset=glasl_test_annotations,
    )


    save_stats(
        split='train',
        y_true=np.array(actual_y_train),
        y_pred=np.array(pred_y_train),
        labels=labels,
        untill=len(labels),
        create_files=True,
    )
    save_stats(
        split='test',
        y_true=np.array(actual_y_test),
        y_pred=np.array(pred_y_test),
        labels=labels,
        untill=len(labels),
        create_files=True,
    )
    save_stats(
        split='glasl_test',
        y_true=np.array(actual_y_glasl),
        y_pred=np.array(pred_y_glasl),
        labels=labels,
        untill=glasl_total_class,
        create_files=True,
    )


if __name__=='__main__':
    '''
    -------------------------
    ---- train ----
    accuracy on train: 99.14108 %
    precision on train: 99.15813 %
    recall on train: 99.14108 %
    f1_score on train: 99.13783 %


    -------------------------
    ---- test ----
    accuracy on test: 84.39327 %
    precision on test: 84.27814 %
    recall on test: 84.39327 %
    f1_score on test: 82.29653 %


    -------------------------
    ---- glasl_test ----
    accuracy on glasl_test: 90.15152 %
    precision on glasl_test: 95.08070 %
    recall on glasl_test: 90.15152 %
    f1_score on glasl_test: 89.88954 %
    '''
    main()
