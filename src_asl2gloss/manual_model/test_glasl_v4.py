from json import load as loadJson
import numpy as np
import jax.numpy as jnp
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT / "dataset" / "glasl"
LANDMARK_DIR: Path= DS_DIR / "glasl_v4_landmark_ds"
MODEL_DIR: Path= PROJ_ROOT /"src_asl2gloss" / "manual_model" / "model"
KEY_G: str= 'gloss'
KEY_LANDMARK: str= 'landmark'
KEY_PFOLDER: str= 'parent_folder'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_SKELETON: str= 'skeleton'
QUANTITY_FRAME: int= 8
# ---------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
# -----------------------------------------------------
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)
QUANTITY_HAND_LMARK: int= 21
IMGS_LM_SHAPE: tuple= (
    QUANTITY_FRAME, # 8
    len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK*2, # 86
    2
)
LM_IDX_TOP_LIP: int       =  0
LM_IDX_TIP_OF_FOREHEAD:int=  3
LM_IDX_RIGHT_MOST_EYE: int=  7
LM_IDX_RIGHT_MOST_LIP: int=  8
LM_IDX_RIGHT_EAR: int     = 19
LM_IDX_UP_TIP_NOSE: int   = 20
LM_IDX_LEFT_MOST_EYE: int = 23
LM_IDX_LEFT_MOST_LIP: int = 24
LM_IDX_LEFT_EAR: int      = 34
LM_IDX_LEFT_ELBOW: int    = 38
LM_IDX_RIGHT_ELBOW: int   = 39
LM_IDX_LEFT_ARM_TIP: int  = 40
LM_IDX_RIGHT_ARM_TIP: int = 41
# -----------------------------
LM_IDX_LEFT_INDEXF_TIP: int= 52
LM_IDX_RIGHT_INDEXF_TIP:int= 73
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# -----------------------------
WORTHY_LM_IMPORTANT_IDX_V1: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_RIGHT_MOST_EYE,
    LM_IDX_RIGHT_MOST_LIP,
    LM_IDX_RIGHT_EAR,
    LM_IDX_LEFT_MOST_EYE,
    LM_IDX_LEFT_MOST_LIP,
    LM_IDX_LEFT_EAR,
    LM_IDX_LEFT_ELBOW,
    LM_IDX_RIGHT_ELBOW,
)
WORTHY_LM_IMPORTANT_IDX: tuple= (
    LM_IDX_TOP_LIP,
    LM_IDX_TIP_OF_FOREHEAD,
    LM_IDX_UP_TIP_NOSE,
    # -- on later process here to insert up and left body dots
    LM_IDX_LEFT_ARM_TIP,
    LM_IDX_RIGHT_ARM_TIP,
)


def load_model() -> tuple:
    with open(MODEL_DIR / "ds_samples.npy", "rb") as f:
        model= np.load(f)
    with open(MODEL_DIR / "mask.npy", "rb") as f:
        mask= np.load(f)
    with open(MODEL_DIR / "body2dots.npy", "rb") as f:
        body2dots= np.load(f)
    with open(MODEL_DIR / "weights.npy", "rb") as f:
        weights= np.load(f)
    with open(MODEL_DIR / "labels.json", "r") as f:
        labels: list= loadJson(f)
    return (
        jnp.array(model),
        jnp.array(mask),
        jnp.array(body2dots),
        jnp.array(weights),
        labels,
    )


def loadnpfiles(parent_folder: str, landmarks: list) -> np.ndarray:
    out: list= []
    for landmark in landmarks:
        with open(LANDMARK_DIR /parent_folder /landmark[KEY_FILE], "rb") as f:
            out.append(np.load(f))
    return np.asarray(out)


def flip_vertical_x_axis(landmarks) -> np.ndarray:
    '''
    landmarks be of shape (INT, ..., 86, 2)
    '''
    new_landmark= np.array(landmarks).copy()
    new_landmark[..., 0]= 1 -new_landmark[..., 0]
    lhand2rhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ,:].copy()
    rhand2lhand: np.ndarray= new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ,:].copy()
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ,:]= rhand2lhand # ie. past was right hand, now be left hand
    new_landmark[...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ,:]= lhand2rhand # ie. past was left hand, now be right hand

    # -- flip index pair face
    # -- ( 12, 28 )
    tmp= new_landmark[..., 12:13, :].copy()
    new_landmark[..., 12:13, :]= new_landmark[..., 28:29, :].copy()
    new_landmark[..., 28:29, :]= tmp

    # -- ( 19, 34 )
    tmp = new_landmark[..., 19:20, :].copy()
    new_landmark[..., 19:20, :] = new_landmark[..., 34:35, :].copy()
    new_landmark[..., 34:35, :] = tmp

    # -- ( 11, 27 )
    tmp = new_landmark[..., 11:12, :].copy()
    new_landmark[..., 11:12, :] = new_landmark[..., 27:28, :].copy()
    new_landmark[..., 27:28, :] = tmp

    # -- ( 21, 35 )
    tmp = new_landmark[..., 21:22, :].copy()
    new_landmark[..., 21:22, :] = new_landmark[..., 35:36, :].copy()
    new_landmark[..., 35:36, :] = tmp

    # -- ( 14, 30 )
    tmp = new_landmark[..., 14:15, :].copy()
    new_landmark[..., 14:15, :] = new_landmark[..., 30:31, :].copy()
    new_landmark[..., 30:31, :] = tmp

    # -- ( 13, 29 )
    tmp = new_landmark[..., 13:14, :].copy()
    new_landmark[..., 13:14, :] = new_landmark[..., 29:30, :].copy()
    new_landmark[..., 29:30, :] = tmp

    # -- ( 10, 26 )
    tmp = new_landmark[..., 10:11, :].copy()
    new_landmark[..., 10:11, :] = new_landmark[..., 26:27, :].copy()
    new_landmark[..., 26:27, :] = tmp

    # -- ( 15, 31 )
    tmp = new_landmark[..., 15:16, :].copy()
    new_landmark[..., 15:16, :] = new_landmark[..., 31:32, :].copy()
    new_landmark[..., 31:32, :] = tmp

    # -- ( 18, 33 )
    tmp = new_landmark[..., 18:19, :].copy()
    new_landmark[..., 18:19, :] = new_landmark[..., 33:34, :].copy()
    new_landmark[..., 33:34, :] = tmp

    # -- (  7, 23 )
    tmp = new_landmark[..., 7:8, :].copy()
    new_landmark[..., 7:8, :] = new_landmark[..., 23:24, :].copy()
    new_landmark[..., 23:24, :] = tmp

    # -- ( 16, 32 )
    tmp = new_landmark[..., 16:17, :].copy()
    new_landmark[..., 16:17, :] = new_landmark[..., 32:33, :].copy()
    new_landmark[..., 32:33, :] = tmp

    # -- (  9, 25 )
    tmp = new_landmark[..., 9:10, :].copy()
    new_landmark[..., 9:10, :] = new_landmark[..., 25:26, :].copy()
    new_landmark[..., 25:26, :] = tmp

    # -- (  8, 24 )
    tmp = new_landmark[..., 8:9, :].copy()
    new_landmark[..., 8:9, :] = new_landmark[..., 24:25, :].copy()
    new_landmark[..., 24:25, :] = tmp




    # -- flip index pair pose
    # -- ( 37, 36 )
    tmp = new_landmark[..., 37:38, :].copy()
    new_landmark[..., 37:38, :] = new_landmark[..., 36:37, :].copy()
    new_landmark[..., 36:37, :] = tmp

    # -- ( 43, 42 )
    tmp = new_landmark[..., 43:44, :].copy()
    new_landmark[..., 43:44, :] = new_landmark[..., 42:43, :].copy()
    new_landmark[..., 42:43, :] = tmp

    # -- ( 39, 38 )
    tmp = new_landmark[..., 39:40, :].copy()
    new_landmark[..., 39:40, :] = new_landmark[..., 38:39, :].copy()
    new_landmark[..., 38:39, :] = tmp

    # -- ( 41, 40 )
    tmp = new_landmark[..., 41:42, :].copy()
    new_landmark[..., 41:42, :] = new_landmark[..., 40:41, :].copy()
    new_landmark[..., 40:41, :] = tmp


    return new_landmark


def get_left_hand(input: jnp.ndarray) -> jnp.ndarray:
    if input.shape[-1]!=2:
        return input[
            ...,
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
        ].copy()
    return input[
        ...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK,
        :
    ].copy()
def get_right_hand(input: jnp.ndarray) -> jnp.ndarray:
    if input.shape[-1]!=2:
        return input[
            ...,
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
        ].copy()
    return input[
        ...,
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:,
        :
    ].copy()


def get_worthy5landmarks(input: jnp.ndarray) -> jnp.ndarray:
    '''
    input can be of shape (..., 86, 2) or (..., 86)
    ------------------------------
    if input shape is (..., 86, 2)
    then returned shape is (..., 5, 2)
    else returned shape is (..., 5)
    -------------------------------
    meaning does not include 2 dots
    '''
    if input.shape[-1]!=2:
        return input[
            ...,
            WORTHY_LM_IMPORTANT_IDX
        ]
    return input[
        ...,
        WORTHY_LM_IMPORTANT_IDX,
        :
    ]
def get2dots_frombody(input: jnp.ndarray) -> jnp.ndarray:
    '''
    input -- shape of (..., 86, 2)
    -------------------------------
    returned shape is (...,  2, 2)
    '''
    shoulder_center: jnp.ndarray= (
        input[..., LM_IDX_LEFT_SHOULDER:LM_IDX_LEFT_SHOULDER+1, :] +
        input[..., LM_IDX_RIGHT_SHOULDER:LM_IDX_RIGHT_SHOULDER+1, :]
    )/2.0
    hip_center: jnp.ndarray= (
        input[..., LM_IDX_LEFT_HIP:LM_IDX_LEFT_HIP+1, :] +
        input[..., LM_IDX_RIGHT_HIP:LM_IDX_RIGHT_HIP+1, :]
    )/2.0
    shoulder_hip_center: jnp.ndarray= (shoulder_center +hip_center)/2.0
    '''
    shoulder_center and hip_center, both has shape (..., 1, 2)
    so as shoulder_hip_center has shape (..., 1, 2)
    '''
    return jnp.concatenate([
        (shoulder_center+shoulder_hip_center)/2.0,
        (shoulder_hip_center+hip_center)/2.0,
    ], axis=-2)
def mean_ignore_zero(input: jnp.ndarray, keepdims: bool=False) -> jnp.ndarray:
    out: jnp.ndarray= jnp.sum(input, axis=-1, keepdims=keepdims)
    count: jnp.ndarray= jnp.sum(input!=0.0, axis=-1, keepdims=keepdims)
    # count= jnp.where(count==0, 1, count)
    count= jnp.maximum(count, 1)
    return out/count
def weights_edges(input: jnp.ndarray) -> jnp.ndarray:
    return input *jnp.ones(input.shape[-1]) \
                             .at[0].set(0.05) \
                             .at[1].set(0.1) \
                             .at[-2].set(0.1) \
                             .at[-1].set(0.05)
def hand_mse(
    input: jnp.ndarray,
    model: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    '''
    input -- shape of ( BATCH_SIZE,              1,                1, QUANTITY_FRAME, 86, 2)
    model -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    86 landmarks mandatory to follow this order
    [
        ...face_landmarks_36...,
        ...pose_landmarks_8...,
        ...left_hand_landmarks_21...,
        ...right_hand_landmarks_21...,
    ]
    ----------------------------------
    output, has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    left_hand_diff: jnp.ndarray= (
        get_left_hand(model) -get_left_hand(input)
    )**2
    right_hand_diff: jnp.ndarray= (
        get_right_hand(model) -get_right_hand(input)
    )**2
    left_hand_diff= jnp.sum( # due2now c^2, from c^2= a^2 +b^2
        left_hand_diff,
        axis=-1,
        keepdims=False
    ) # now be shape (..., QUANTITY_FRAME, 21), ie. removed X, Y cords axis
    right_hand_diff= jnp.sum(
        right_hand_diff,
        axis=-1,
        keepdims=False
    ) # now be shape (..., QUANTITY_FRAME, 21), ie. removed X, Y cords axis
    left_hand_diff= jnp.sqrt(left_hand_diff) *jnp.expand_dims(mask[..., 0], axis=-1)
    right_hand_diff= jnp.sqrt(right_hand_diff) *jnp.expand_dims(mask[..., 1], axis=-1)
    '''
    now ----
    left_hand_diff
    right_hand_diff
    ---------------
    both has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 21)
    '''
    left_hand_diff= jnp.mean(left_hand_diff, axis=-1, keepdims=False)
    right_hand_diff= jnp.mean(right_hand_diff, axis=-1, keepdims=False)
    '''
    now ----
    left_hand_diff
    right_hand_diff
    ---------------
    both has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME)
    '''
    left_hand_diff= weights_edges(left_hand_diff)
    right_hand_diff= weights_edges(right_hand_diff)

    # left_hand_diff= jnp.mean(left_hand_diff, axis=-1, keepdims=True, where=left_hand_diff!=0.0)
    # right_hand_diff= jnp.mean(right_hand_diff, axis=-1, keepdims=True, where=right_hand_diff!=0.0)
    left_hand_diff= mean_ignore_zero(left_hand_diff, keepdims=True)
    right_hand_diff= mean_ignore_zero(right_hand_diff, keepdims=True)
    '''
    now ----
    left_hand_diff
    right_hand_diff
    ---------------
    both has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, 1)
    '''''
    out: jnp.ndarray= mean_ignore_zero(
        jnp.concatenate([left_hand_diff, right_hand_diff], axis=-1),
        keepdims=False,
    )
    '''
    out, has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    return out
def relativity_mse(
    input: jnp.ndarray,
    model: jnp.ndarray,
    mask: jnp.ndarray,
    model2dots: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    '''
    input -- shape of ( BATCH_SIZE,              1,                1, QUANTITY_FRAME, 86, 2)
    model -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask  -- shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (          1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 7)
    --------------------------------------------------------------
    output shape is (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    input_lhand_index: jnp.ndarray= jnp.expand_dims(
        input[..., LM_IDX_LEFT_INDEXF_TIP, :],
        axis=-2,
    )  # shape (..., 1, 2)
    input_rhand_index: jnp.ndarray= jnp.expand_dims(
        input[..., LM_IDX_RIGHT_INDEXF_TIP, :],
        axis=-2,
    ) # shape (..., 1, 2)
    input_7landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy5landmarks(input),
        get2dots_frombody(input),
    ], axis=-2) # shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7, 2)
    input_7landmarks= jnp.concatenate([
        input_7landmarks[..., :3,  :],
        input_7landmarks[..., 5:,  :], # body2dots
        input_7landmarks[..., 3:5, :], # for left and right arm tip
    ], axis=-2)
    # ----------------------------------------------
    model_lhand_index: jnp.ndarray= jnp.expand_dims(
        model[..., LM_IDX_LEFT_INDEXF_TIP, :],
        axis=-2
    )  # shape (..., 1, 2)
    model_rhand_index: jnp.ndarray= jnp.expand_dims(
        model[..., LM_IDX_RIGHT_INDEXF_TIP, :],
        axis=-2
    ) # shape (..., 1, 2)
    model_7landmarks: jnp.ndarray= jnp.concatenate([
        get_worthy5landmarks(model),
        model2dots,
    ], axis=-2) # shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7, 2)
    model_7landmarks= jnp.concatenate([
        model_7landmarks[..., :3,  :],
        model_7landmarks[..., 5:,  :], # body2dots
        model_7landmarks[..., 3:5, :], # for left and right arm tip
    ], axis=-2)
    '''
    below are
    ---------
    now do difference/subtraction on left/right hand on 7 landmarks
    then multiply to weights
    then multiply to mask4left/right hand
    -------------------------------------
    distance( c^2= a^2 +b^2 ) on left/right hand( index finger tip only ) to 7 landmarks chosen
    '''
    distance_input_lhand: jnp.ndarray= (input_7landmarks -input_lhand_index)**2
    distance_input_rhand: jnp.ndarray= (input_7landmarks -input_rhand_index)**2
    distance_model_lhand: jnp.ndarray= (model_7landmarks -model_lhand_index)**2
    distance_model_rhand: jnp.ndarray= (model_7landmarks -model_rhand_index)**2
    '''
    distance_input_lhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7, 2)
    distance_input_rhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7, 2)
    distance_model_lhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7, 2)
    distance_model_rhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7, 2)
    '''
    distance_input_lhand= jnp.sqrt(jnp.sum(distance_input_lhand, axis=-1, keepdims=False))
    distance_input_rhand= jnp.sqrt(jnp.sum(distance_input_rhand, axis=-1, keepdims=False))
    distance_model_lhand= jnp.sqrt(jnp.sum(distance_model_lhand, axis=-1, keepdims=False))
    distance_model_rhand= jnp.sqrt(jnp.sum(distance_model_rhand, axis=-1, keepdims=False))
    '''
    distance_input_lhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7)
    distance_input_rhand shape of (BATCH_SIZE,              1,                1, QUANTITY_FRAME, 7)
    distance_model_lhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    distance_model_rhand shape of (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    '''
    model_input_lhand: jnp.ndarray= jnp.absolute(distance_input_lhand -distance_model_lhand)
    model_input_rhand: jnp.ndarray= jnp.absolute(distance_input_rhand -distance_model_rhand)
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    weights, shape of          (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 2, 7)
    mask, shape of             (         1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 2)
    '''
    model_input_lhand= model_input_lhand *weights[..., 0, :] *mask[..., 0:1]
    model_input_rhand= model_input_rhand *weights[..., 1, :] *mask[..., 1:2]
    '''
    model_input_lhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    model_input_rhand shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 7)
    '''
    model_input_lhand= mean_ignore_zero(
        input=jnp.stack([ # jnp.stack(...) gives shape of (..., 7, 2)
            model_input_lhand,
            model_input_rhand,
        ], axis=-1),
        keepdims=False,
    ) # result shape (..., QUANTITY_FRAME, 7)
    del model_input_rhand
    model_input_lhand= mean_ignore_zero(
        input=model_input_lhand,
        keepdims=False,
    ) # result shape (..., QUANTITY_FRAME)
    model_input_lhand= weights_edges(model_input_lhand)
    model_input_lhand=mean_ignore_zero(
        input=model_input_lhand,
        keepdims=False,
    ) # result shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)

    return model_input_lhand
def comparison_score(
    input: np.ndarray,
    model: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    '''
    input, shape of -------------------------- (BATCH_SIZE, QUANTITY_FRAME, 86, 2)
    model, shape of - (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 7)
    -- on weights axis=-2 of value 2, for left and right hand repsectively
    -- on weights axis=-1 of value 7 for 7 landmarks
    -------------------------------------------------------------
    output shape is (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    '''
    assert len(input.shape)==(len(IMGS_LM_SHAPE)+1)
    assert len(model.shape)==(len(IMGS_LM_SHAPE)+3)
    assert input.shape[1:]==IMGS_LM_SHAPE
    assert model.shape[-1*len(IMGS_LM_SHAPE):]==IMGS_LM_SHAPE
    assert model.shape[:-2]==mask.shape[:-1]
    input_exp: jnp.ndarray= jnp.expand_dims(input, axis=(1, 2))
    left_right_hand_mse: jnp.ndarray= hand_mse(
        input=input_exp,
        model=model,
        mask=mask,
    ) # shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    hand2body_mse: jnp.ndarray= relativity_mse(
        input=input_exp,
        model=model,
        mask=mask,
        model2dots=body2dots,
        weights=weights,
    ) # shape of (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    return mean_ignore_zero(
        input=jnp.stack([left_right_hand_mse, hand2body_mse], axis=-1),
        keepdims=False,
    )


def sort_score(score: jnp.ndarray) -> list:
    '''
    score has shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
    ----------
    return is a list of shape (BATCH_SIZE, QUANTITY_GLOSS)
    '''
    result: jnp.ndarray= jnp.min(score, axis=-1)
    return jnp.argsort(result, axis=-1, stable=True).tolist()


def test_all_data(
    model: jnp.ndarray,
    mask: jnp.ndarray,
    body2dots: jnp.ndarray,
    weights: jnp.ndarray,
    model_annotations: list[str],
    annotations: list,
) -> None:
    '''
    model, shape of - (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME, 86, 2)
    mask, shape of -- (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2)
    body2dots, shape  (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 2)
    weights, shape of (1, QUANTITY_GLOSS, SAMPLE_PER_GLOSS, QUANTITY_FRAME,  2, 7)
    -- on weights axis=-2 of value 2, for left and right hand repsectively
    -- on weights axis=-1 of value 7 for 7 landmarks
    '''
    batch_size: int= 32
    ds_elements: list= [] # be of shape (INT, QUANTITY_FRAME, 86, 2)
    ds_labels: list[int]= [] # loadnpfiles(..)
    ds_pfolders: list[str]= []
    result_details: list= []
    for a_gloss in annotations:
        result_details.append({
            KEY_G: a_gloss[KEY_G],
            'correct': 0,
            'count': 0,
            'incorrect_pfolder': [],
        })

    def test_batch(
        batch_elements: list,
        batch_labels: list[int],
        batch_pfolders: list[str],
    ) -> None:
        result= comparison_score(
            input=np.array(batch_elements),
            model=model,
            mask=mask,
            body2dots=body2dots,
            weights=weights,
        ) # returns jnp.ndarray shape (BATCH_SIZE, QUANTITY_GLOSS, SAMPLE_PER_GLOSS)
        index_result: list= sort_score(result) # returns list shape (BATCH_SIZE, QUANTITY_GLOSS)
        for idx_batch, ds_element_result in enumerate(index_result):
            expected_idx: int= batch_labels[idx_batch]
            predicted_idx: int= ds_element_result[0]
            result_details[expected_idx]['count']+= 1
            if model_annotations[predicted_idx]==annotations[expected_idx][KEY_G]:
                result_details[expected_idx]['correct']+= 1
            else:
                result_details[expected_idx]['incorrect_pfolder'].append(batch_pfolders[idx_batch])

    for idx_gloss, a_gloss in enumerate(annotations):
        for a_start_end in a_gloss[KEY_LANDMARK]:
            while batch_size<=len(ds_labels):
                test_batch(
                    batch_elements=ds_elements[:batch_size],
                    batch_labels=ds_labels[:batch_size],
                    batch_pfolders=ds_pfolders[:batch_size],
                )
                ds_elements= ds_elements[batch_size:]
                ds_labels=   ds_labels[batch_size:]
                ds_pfolders= ds_pfolders[batch_size:]
            ds_elements.append(loadnpfiles(
                parent_folder=a_start_end[KEY_PFOLDER],
                landmarks=a_start_end[KEY_LANDMARK],
            ))
            ds_labels.append(idx_gloss)
            ds_pfolders.append(a_start_end[KEY_PFOLDER])
            # -- for flip
            ds_elements.append(flip_vertical_x_axis(ds_elements[-1]))
            ds_labels.append(idx_gloss)
            ds_pfolders.append(f"{a_start_end[KEY_PFOLDER]}_flip")
    if len(ds_labels)!=0:
        test_batch(
            batch_elements=ds_elements,
            batch_labels=ds_labels,
            batch_pfolders=ds_pfolders,
        )
    del ds_elements
    del ds_labels
    del ds_pfolders

    total_correct: int= 0
    total_count: int= 0
    for a_result in result_details:
        total_correct+= a_result['correct']
        total_count+= a_result['count']
        incorrect: int= a_result['count'] -a_result['correct']
        print(
            f"\n{a_result[KEY_G]}\n" +
            f"    correct: {a_result['correct']}/{a_result['count']} -- {a_result['correct']/a_result['count']*100}%\n" +
            f"    incorrect: {incorrect}/{a_result['count']} -- {incorrect/a_result['count']*100}%"
        )
        if incorrect!=0:
            print(f"{a_result[KEY_G]} incorrect_pfolder")
            for an_incorrect_pfolder in a_result['incorrect_pfolder']:
                print(f"    {an_incorrect_pfolder}")
    total_incorrect: int= total_count -total_correct
    print("-------------------------------")
    print("-------------------------------")
    print(f"total correct: {total_correct}/{total_count} = {total_correct/total_count *100}%")
    print(f"total incorrect: {total_incorrect}/{total_count} = {total_incorrect/total_count *100}%")
    print("-------------------------------")
    print("-------------------------------")
    for a_result in result_details:
        incorrect: int= a_result['count'] -a_result['correct']
        print(
            f"\n{a_result[KEY_G]}\n" +
            f"    correct: {a_result['correct']}/{a_result['count']} -- {a_result['correct']/a_result['count']*100}%\n" +
            f"    incorrect: {incorrect}/{a_result['count']} -- {incorrect/a_result['count']*100}%"
        )


def main() -> None:
    model, mask, body2dots, weights, labels= load_model()
    with open(DS_DIR /"glasl_v4_landmark_ds.json", "r") as f:
        annotations= loadJson(f)
    test_all_data(
        model=model,
        mask=mask,
        body2dots=body2dots,
        weights=weights,
        model_annotations=labels,
        annotations=annotations,
    )
if __name__=="__main__":
    main()
