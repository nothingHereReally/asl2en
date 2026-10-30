from json import dump as writeJson, load as loadJson
from math import ceil, floor
from pathlib import Path
import cv2
import numpy as np


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"glasl_verified"
VIDEO_DIR: Path= DS_DIR /"videos"
IN_LANDMARK_DIR: Path= DS_DIR /"ds_landmark"
IN_SKELETON_DIR: Path= DS_DIR /"ds_skeleton"
OUT_LANDMARK_DIR: Path= DS_DIR /"qf8_ds_landmark"
OUT_SKELETON_DIR: Path= DS_DIR /"qf8_ds_skeleton"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
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
KEY_IMG_VALID: str= 'valid_images'
KEY_IMGSTART: str= 'start' # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_IMGEND: str= 'end'     # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_PFOLDER_FROM: str= 'parent_folder_from'
KEY_PFOLDER: str= 'parent_folder'
IMG_SIZE: int= 158
QUANTITY_FRAME: int= 8
MOD_PART_3: tuple= (2, 3, 4, 5, 6)
FACE_CONNECTIONS: tuple= (
    (3, 28), (28, 34), (34, 27), (27, 35), (35, 17), # left oval face
    (3, 12), (12, 19), (19, 11), (11, 21), (21, 17), # right oval face

    (26, 29), (29, 30), # left eyebrow

    (23, 32), (32, 31), # left eye down
    (31, 33), (33, 23), # left eye up

    (10, 13), (13, 14), # right eyebrow

    (7, 16), (16, 15), # right eye down
    (15, 18), (18, 7), # rght eye up

    (20, 22), (22, 2), # nose vertical line
    (2, 25), (25, 1), # left half nose
    (1, 9), (9, 2), # rigth half nose

    # mouth
    (8, 6), (6, 24), # down lip edge down
    (24, 0), (0, 8), # up lip edge up
    (8, 5), (5, 24), # up/down lip inner a
    (24, 4), (4, 8), # up/down lip inner b
)
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)

# before use of POSE_CONNECTIONS modify landmark 1st
# modify to use index to be used only: 11,12,13,14,15,16,23,24
# so new index: 0,1,2,3,4,5,6,7
POSE_CONNECTIONS: tuple= ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6), (1, 7))
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)

HAND_CONNECTIONS: tuple= (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), # palm connections
    (1, 2), (2, 3), (3, 4),           # thumb finger connections
    (5, 6), (6, 7), (7, 8),           # index finger connections
    (9, 10), (10, 11), (11, 12),      # middle finger connections
    (13, 14), (14, 15), (15, 16),     # ring finger connections
    (17, 18), (18, 19), (19, 20)      # pinky finger connections
)
QUANTITY_HAND_LMARK: int= 21
def get_face(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., :36, :].copy()
def get_pose(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., 36:36+8, :].copy()
def get_left_hand(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., 36+8:36+8+21, :].copy()
def get_right_hand(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., -21:, :].copy()
def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: np.ndarray, \
                    lmark_coordinates: list, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> np.ndarray:
    img_wh: dict= {"wx": image.shape[1], "hy": image.shape[0]}


    # drawing the lines between 2 landmark connections
    if color_line!=None or color_dot!=None:
        for lmark_idx_pair in connections_idxs:
            pA: tuple= (
                lmark_coordinates[  lmark_idx_pair[0]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[0]  ][1]  # y
            )
            pB: tuple= (
                lmark_coordinates[  lmark_idx_pair[1]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[1]  ][1]  # y
            )
            if isOkPlot(pA) and isOkPlot(pB):
                if color_dot!=None:
                    cv2.circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    cv2.circle(
                        img=image,
                        center=(
                            int(pB[0]*img_wh['wx']),
                            int(pB[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                if color_line!=None:
                    cv2.line(
                        img=image,
                        pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                        pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                        color=color_line,
                        thickness=thick
                    )
            else:
                raise NotImplementedError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0, on both x and y coordinates")
            del pA
            del pB
    return image
def drawFacePoseHandFromLM(img_write_to: np.ndarray, landmark: list[tuple[float, float]], hasLandmark: dict) -> np.ndarray:
    # ---- face landmarks ----
    if hasLandmark[KEY_FACE]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_face(np.array(landmark)).tolist(),
            connections_idxs=FACE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 153, 0), # 153/255= 0.6
        )

    # ---- pose landmarks ----
    if hasLandmark[KEY_POSE]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_pose(np.array(landmark)).tolist(),
            connections_idxs=POSE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 0, 153), # 153/255= 0.6
        )

    # ---- left hand landmarks ----
    if hasLandmark[KEY_LHAND]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_left_hand(np.array(landmark)).tolist(),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(255, 255, 255)
        )

    # ---- right hand landmarks ----
    if hasLandmark[KEY_RHAND]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_right_hand(np.array(landmark)).tolist(),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(153, 204, 204), # 204/255= 0.8
        )

    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to
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
    '''
    return dict of structure
    {
        KEY_PFOLDER_FROM: Path,
        KEY_PFOLDER: str,
        KEY_LANDMARK: list,
    }
    '''
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
) -> tuple:
    '''
    return (annotations_many, process_from)
    annotations_many is list[dict], dict has structure
    {
        KEY_PFOLDER_FROM: Path,
        KEY_PFOLDER: str,
        KEY_LANDMARK: list[dict],
    }
    process_from is list[str], elements be 'from_lt_p0'
    or 'from_gt_p1' or 'from_gt_p2' or 'from_gt_p3'
    '''
    annotations_many: list= []
    process_from: list[str]= []

    idx_init: int= idx_init_has_hand(landmarks)
    q_imgs_available: int= len(landmarks)-idx_init
    if q_imgs_available<=QUANTITY_FRAME:
        annotation: dict= q_imgs_less_than_or_equal(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.append(annotation)
        process_from.append('from_lt_p0')
    else:
        # ---- part 1 ----
        annotation: dict= greater_than_qf_p1(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.append(annotation)
        process_from.append('from_gt_p1')

        # ---- part 2 ----
        annotations: list= greater_than_qf_p2(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.extend(annotations)
        process_from.extend(['from_gt_p2' for _ in range(len(annotations))])

        # ---- part 3 ----
        annotations= greater_than_qf_p3(
            landmarks=landmarks[idx_init:],
            parent_folder_from=parent_folder_from,
            parent_folder=parent_folder,
        )
        annotations_many.extend(annotations)
        process_from.extend(['from_gt_p3' for _ in range(len(annotations))])
    return (annotations_many, process_from)




def init_dirs() -> None:
    err: list[str]= []
    if not IN_LANDMARK_DIR.exists():
        err.append(f"Missing {IN_LANDMARK_DIR} folder.")
    if not IN_SKELETON_DIR.exists():
        err.append(f"Missing {IN_SKELETON_DIR} folder.")

    if not OUT_LANDMARK_DIR.exists():
        err.append(f"Please delete {OUT_LANDMARK_DIR}, this will be created for you.")
    if not OUT_SKELETON_DIR.exists():
        err.append(f"Please delete {OUT_SKELETON_DIR}, this will be created for you.")
    OUT_LANDMARK_DIR.mkdir()
    OUT_SKELETON_DIR.mkdir()
def get_past_annotations() -> list:
    with open(DS_DIR /"ds_landmark.json", 'r') as f:
        ds_landmark= loadJson(f)
    return ds_landmark
def has_atleast1hand(annotations: dict) -> bool:
    return annotations[KEY_FACE] and \
           annotations[KEY_POSE] and \
           (annotations[KEY_LHAND] or \
           annotations[KEY_RHAND])
def get_landmarks_data(
    abs_root_path: Path,
    parent_folder: str,
    landmarks: list[dict],
) -> list:
    '''
    returns (landmarks, annotations) landmarks is a list[numpy.ndarray]
    annotations is a list[dict]
    '''
    out_landmarks: list= []
    for idx, an_img_landmark in enumerate(landmarks):
        if has_atleast1hand(an_img_landmark) or idx == 0:
            with open(abs_root_path /parent_folder /an_img_landmark[KEY_FILE], 'rb') as f:
                out_landmarks.append( np.load(f) )
        else:
            out_landmarks.append(out_landmarks[-1].copy())

    return out_landmarks
def process_videos_ds() -> tuple:
    ds_landmark: list= []
    ds_skeleton: list= []
    '''
    TODO: later blah blah
    del left hand
        drink
        who
        candy
        cousin
        feel
        mine_my
        me_i
        stomach
        need
        see
        feel
        fever
    '''
    past_landmark: list= get_past_annotations()
    for a_gloss in past_landmark:
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS]:
            many_sets_annotations: list= []
            many_sets_from: list[str]= []
            if len(a_video[KEY_LANDMARK]) <= QUANTITY_FRAME:
                many_sets_annotations.append(q_imgs_less_than_or_equal(
                    landmarks=a_video[KEY_LANDMARK],
                    parent_folder_from=IN_LANDMARK_DIR,
                    parent_folder=a_video[KEY_PFOLDER],
                ))
                many_sets_from.append('from_lt_p0')
            else:
                many_annotations, many_from= q_imgs_greater_than(
                    landmarks=a_video[KEY_LANDMARK],
                    parent_folder_from=IN_LANDMARK_DIR,
                    parent_folder=a_video[KEY_PFOLDER],
                )
                many_sets_annotations.extend(many_annotations)
                many_sets_from.extend(many_from)
            for idx_a_qf_set, a_qf_set in enumerate(many_sets_annotations):
                pfolder_name: str= f"{a_qf_set[KEY_PFOLDER][:-10]}_{str(idx_a_qf_set+1).zfill(4)}"
                pfolder_name= f"{pfolder_name}_{a_qf_set[KEY_PFOLDER][-9:]}"
                pfolder_name= f"{pfolder_name}_{many_sets_from[idx_a_qf_set]}"

                landmark_abs_folder: Path= OUT_LANDMARK_DIR /pfolder_name
                skeleton_abs_folder: Path= OUT_SKELETON_DIR /pfolder_name
                landmark_abs_folder.mkdir()
                skeleton_abs_folder.mkdir()
                ds_landmark[-1][KEY_VIDS].append({
                    KEY_PFOLDER: pfolder_name,
                    KEY_SPLIT: a_video[KEY_SPLIT],
                    KEY_LANDMARK: [],
                })
                ds_skeleton[-1][KEY_VIDS].append({
                    KEY_PFOLDER: pfolder_name,
                    KEY_SPLIT: a_video[KEY_SPLIT],
                    KEY_SKELETON: [],
                })
                landmarks_data_numpy: list= get_landmarks_data(
                    abs_root_path=a_qf_set[KEY_PFOLDER_FROM],
                    parent_folder=a_qf_set[KEY_PFOLDER],
                    landmarks=a_qf_set[KEY_LANDMARK]
                )
                for idx_init0, an_annotation_img in enumerate(a_qf_set[KEY_LANDMARK]):
                    landmark_skeleton_prefix: str= str(idx_init0+1).zfill(8)
                    with open(f"{landmark_abs_folder /landmark_skeleton_prefix}.npy", 'wb') as f:
                        np.save(f, landmarks_data_numpy[idx_init0])
                    cv2.imwrite(
                        filename=f"{skeleton_abs_folder /landmark_skeleton_prefix}.jpg",
                        img=drawFacePoseHandFromLM(
                            img_write_to=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
                            landmark=landmarks_data_numpy[idx_init0],
                            hasLandmark=an_annotation_img,
                        ),
                    )
                    ds_landmark[-1][KEY_VIDS][-1][KEY_LANDMARK].append({
                        KEY_FILE: f"{landmark_skeleton_prefix}.npy",
                        KEY_FACE:  an_annotation_img[KEY_FACE],
                        KEY_POSE:  an_annotation_img[KEY_POSE],
                        KEY_LHAND: an_annotation_img[KEY_LHAND],
                        KEY_RHAND: an_annotation_img[KEY_RHAND],
                    })
                    ds_skeleton[-1][KEY_VIDS][-1][KEY_SKELETON].append({
                        KEY_FILE: f"{landmark_skeleton_prefix}.jpg",
                        KEY_FACE:  an_annotation_img[KEY_FACE],
                        KEY_POSE:  an_annotation_img[KEY_POSE],
                        KEY_LHAND: an_annotation_img[KEY_LHAND],
                        KEY_RHAND: an_annotation_img[KEY_RHAND],
                    })
    return (ds_landmark, ds_skeleton)
def main() -> None:
    init_dirs()
    ds_landmark, ds_skeleton= process_videos_ds()
    with open(f"{DS_DIR /"qf8_ds_landmark.json"}", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{DS_DIR /"qf8_ds_skeleton.json"}", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    '''
    took approx. 5 minutes to finish
    '''
    main()
