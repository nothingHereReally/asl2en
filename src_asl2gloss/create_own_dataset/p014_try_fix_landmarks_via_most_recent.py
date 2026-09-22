from json import dump as writeJson, load
from cv2 import circle, imwrite, line
import numpy as np
from pathlib import Path


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
OLD_LANDMARK_DIR: Path= DS_DIR /"investigate_landmark"
LANDMARK_DIR: Path= DS_DIR /"fixed_landmark_most_recent"
SKELETON_DIR: Path= DS_DIR /"fixed_skeleton_most_recent"
IMG_SIZE: int= 158
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_VFILE: str= 'video_file'
KEY_LMARK: str= 'landmark'
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
old_ds_annotation: list= []
with open(f"{DS_DIR /"hybrid_data.investigate.landmark.json"}", 'r') as f:
    old_ds_annotation: list= load(f)
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
def get_lmark_face(landmark):
    return landmark[:len(WORTHY_FACE_IDX)]
def get_lmark_pose(landmark):
    return landmark[
        len(WORTHY_FACE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
    ]
def get_lmark_lhand(landmark):
    return landmark[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
    ]
def get_lmark_rhand(landmark):
    return landmark[
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
    ]
def check_folders():
    is_valid: bool= True
    if not OLD_LANDMARK_DIR.exists():
        is_valid= False
    if LANDMARK_DIR.exists():
        is_valid= False
    if SKELETON_DIR.exists():
        is_valid= False
    if not is_valid:
        print(f"Should exists: {OLD_LANDMARK_DIR}")
        print(f"------------- Please Delete Folders -------------")
        print(LANDMARK_DIR)
        print(SKELETON_DIR)
        raise FileNotFoundError(f"Please delete these folders: {LANDMARK_DIR.stem}, {SKELETON_DIR.stem}")
    LANDMARK_DIR.mkdir()
    SKELETON_DIR.mkdir()
def record_past_lmark(past_lmark: dict, landmark: np.ndarray, notation: dict) -> dict:
    if notation[KEY_FACE]:
        past_lmark[KEY_FACE]= get_lmark_face(landmark)
    if notation[KEY_POSE]:
        past_lmark[KEY_POSE]= get_lmark_pose(landmark)
    if notation[KEY_LHAND]:
        past_lmark[KEY_LHAND]= get_lmark_lhand(landmark)
    if notation[KEY_RHAND]:
        past_lmark[KEY_RHAND]= get_lmark_rhand(landmark)
    return past_lmark
def redo4new_lmark(landmark: np.ndarray, notation: dict, past_lmark_solution: dict) -> tuple:
    new_landmark: list= []
    if notation[KEY_FACE]:
        new_landmark.extend(get_lmark_face(landmark))
    elif 0<len(past_lmark_solution[KEY_FACE]):
        new_landmark.extend(past_lmark_solution[KEY_FACE])
        notation[KEY_FACE]= True
    else:
        new_landmark.extend(  np.zeros((len(WORTHY_FACE_IDX), 2))  )

    if notation[KEY_POSE]:
        new_landmark.extend(get_lmark_pose(landmark))
    elif 0<len(past_lmark_solution[KEY_POSE]):
        new_landmark.extend(past_lmark_solution[KEY_POSE])
        notation[KEY_POSE]= True
    else:
        new_landmark.extend(  np.zeros((len(WORTHY_POSE_IDX), 2))  )

    if notation[KEY_LHAND]:
        new_landmark.extend(get_lmark_lhand(landmark))
    elif 0<len(past_lmark_solution[KEY_LHAND]):
        new_landmark.extend(past_lmark_solution[KEY_LHAND])
        notation[KEY_LHAND]= True
    else:
        new_landmark.extend(  np.zeros((QUANTITY_HAND_LMARK, 2))  )

    if notation[KEY_RHAND]:
        new_landmark.extend(get_lmark_rhand(landmark))
    elif 0<len(past_lmark_solution[KEY_RHAND]):
        new_landmark.extend(past_lmark_solution[KEY_RHAND])
        notation[KEY_RHAND]= True
    else:
        new_landmark.extend(  np.zeros((QUANTITY_HAND_LMARK, 2))  )
    return (
        np.array(new_landmark),
        notation,
    )
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
                    circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    circle(
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
                    line(
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
def drawFacePoseLRHand(image: np.ndarray, landmark: list, notation: dict) -> np.ndarray:
    # ---- face landmarks ----
    if notation[KEY_FACE]:
        image= drawSkeletonImg(
            image=image,
            lmark_coordinates=landmark[
                :len(WORTHY_FACE_IDX)
            ],
            connections_idxs=FACE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 153, 0), # 153/255= 0.6
        )

    # ---- pose landmarks ----
    if notation[KEY_POSE]:
        image= drawSkeletonImg(
            image=image,
            lmark_coordinates=landmark[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
            ],
            connections_idxs=POSE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 0, 153), # 153/255= 0.6
        )

    # ---- left hand landmarks ----
    if notation[KEY_LHAND]:
        image= drawSkeletonImg(
            image=image,
            lmark_coordinates=landmark[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ],
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(255, 255, 255)
        )

    # ---- right hand landmarks ----
    if notation[KEY_RHAND]:
        image= drawSkeletonImg(
            image=image,
            lmark_coordinates=landmark[
                len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK:
            ],
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(153, 204, 204), # 204/255= 0.8
        )
    return image
def process_each_gloss() -> tuple:
    fixed_landmark_notation: list= []
    fixed_skeleton_notation: list= []
    for a_gloss in old_ds_annotation:
        fixed_landmark_notation.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        fixed_skeleton_notation.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for a_video in a_gloss[KEY_VIDS]:
            old_vid_folder: str= f"{a_gloss[KEY_G]}_{Path(a_video[KEY_VFILE]).stem}"
            vid_folder: str= f"{a_gloss[KEY_G][3:]}_{Path(a_video[KEY_VFILE]).stem}"
            (LANDMARK_DIR /vid_folder).mkdir()
            (SKELETON_DIR /vid_folder).mkdir()
            past_lmark_parts: dict= {
                KEY_FACE: [], KEY_POSE: [], KEY_LHAND: [], KEY_RHAND: [],
            }
            fixed_landmark_notation[-1][KEY_VIDS].append({
                KEY_VFILE: a_video[KEY_VFILE],
                KEY_LMARK: [],
            })
            fixed_skeleton_notation[-1][KEY_VIDS].append({
                KEY_VFILE: a_video[KEY_VFILE],
                KEY_LMARK: [],
            })
            for an_image_lmark in a_video[KEY_LMARK]:
                old_img_lmark: np.ndarray
                with open(f"{OLD_LANDMARK_DIR /old_vid_folder /an_image_lmark[KEY_FILE]}", 'rb') as f:
                    old_img_lmark= np.load(f)
                new_lmark, new_notation= redo4new_lmark(
                    landmark=old_img_lmark,
                    notation=an_image_lmark,
                    past_lmark_solution=past_lmark_parts,
                )
                new_skeleton: np.ndarray= drawFacePoseLRHand(
                    image=np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                    landmark=new_lmark.tolist(),
                    notation=new_notation,
                )
                jpeg_image_format: str= f"{Path(new_notation[KEY_FILE]).stem}.jpg"
                with open(f"{LANDMARK_DIR /vid_folder /new_notation[KEY_FILE]}", 'wb') as f:
                    np.save(f, new_lmark)
                imwrite(f"{SKELETON_DIR /vid_folder /jpeg_image_format}", new_skeleton)
                fixed_landmark_notation[-1][KEY_VIDS][-1][KEY_LMARK].append(new_notation)
                fixed_skeleton_notation[-1][KEY_VIDS][-1][KEY_LMARK].append({
                    KEY_FILE: jpeg_image_format,
                    KEY_FACE: new_notation[KEY_FACE],
                    KEY_POSE: new_notation[KEY_POSE],
                    KEY_LHAND: new_notation[KEY_LHAND],
                    KEY_RHAND: new_notation[KEY_RHAND],
                })
                past_lmark_parts= record_past_lmark(
                    past_lmark=past_lmark_parts,
                    landmark=new_lmark,
                    notation=new_notation,
                ) # goal: use past( individual face/pose/lhand/rhand ) if current not exist
    return (fixed_landmark_notation, fixed_skeleton_notation)
def main() -> None:
    check_folders()
    fixed_landmark_notation, fixed_skeleton_notation= process_each_gloss()
    with open(f"{DS_DIR /'fixed_data_most_recent.investigate.landmark.json'}", 'w') as f:
        writeJson(fixed_landmark_notation, f, indent=4)
    with open(f"{DS_DIR /'fixed_data_most_recent.investigate.skeleton.json'}", 'w') as f:
        writeJson(fixed_skeleton_notation, f, indent=4)
if __name__=="__main__":
    main()
