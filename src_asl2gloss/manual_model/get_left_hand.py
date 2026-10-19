from pathlib import Path
import numpy as np
import cv2


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"glasl"
LHAND_FOLDER_ABS: Path= DS_DIR /"extracted_left_hand"
wholebody_landmark_path: Path= DS_DIR /"glasl_v4_landmark_recent" /"fever_0010" /"00000007.npy"
IMG_SIZE: int= 158
HAND_CONNECTIONS: tuple= (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), # palm connections
    (1, 2), (2, 3), (3, 4),           # thumb finger connections
    (5, 6), (6, 7), (7, 8),           # index finger connections
    (9, 10), (10, 11), (11, 12),      # middle finger connections
    (13, 14), (14, 15), (15, 16),     # ring finger connections
    (17, 18), (18, 19), (19, 20)      # pinky finger connections
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
                    color_dot: tuple|None=None, \
                    highlight_index: int|None=None) -> np.ndarray:
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
                        color=tuple(
                            color_dot
                            if highlight_index!=None and highlight_index==lmark_idx_pair[0]
                            else tuple(color_dot[-1*(idx+1)] for idx in range(len(color_dot)))
                        ),
                        thickness=int(
                            thick*10
                            if highlight_index!=None and highlight_index==lmark_idx_pair[0]
                            else thick*4
                        )
                    )
                    cv2.circle(
                        img=image,
                        center=(
                            int(pB[0]*img_wh['wx']),
                            int(pB[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=tuple(
                            color_dot
                            if highlight_index!=None and highlight_index==lmark_idx_pair[1]
                            else tuple(color_dot[-1*(idx+1)] for idx in range(len(color_dot)))
                        ),
                        thickness=int(
                            thick*10
                            if highlight_index!=None and highlight_index==lmark_idx_pair[1]
                            else thick*4
                        )
                    )

                if color_line!=None:
                    cv2.line(
                        img=image,
                        pt1=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        pt2=(
                            int(pB[0]*img_wh['wx']),
                            int(pB[1]*img_wh['hy'])
                        ),
                        color=color_line,
                        thickness=thick
                    )
            else:
                raise NotImplementedError(
                    "Has landmark_coordinate<0.0 or "
                    "1.0<landmark_coordinate which is not allowed, "
                    "it should be 0.0<= landmark_coordinate <=1.0, "
                    "on both x and y coordinates"
                )
            del pA
            del pB
    return image
def init_dirs() -> None:
    assert wholebody_landmark_path.exists()
    if LHAND_FOLDER_ABS.exists():
        raise NotImplementedError(
            f"Please delete {LHAND_FOLDER_ABS}" +
            ", this will be created for you"
        )
    LHAND_FOLDER_ABS.mkdir()
def main() -> None:
    init_dirs()
    with open(wholebody_landmark_path, 'rb') as f:
        wholebody_lm= np.load(f)
    left_hand_lm= wholebody_lm[36+8:-21, :]
    # ---- left hand landmarks ----
    left_hand_skeleton= drawSkeletonImg(
        image=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
        lmark_coordinates=left_hand_lm,
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=None,
        color_line=(110, 190, 245),
        highlight_index=None
    )
    with open(f"{LHAND_FOLDER_ABS /'left_hand.py'}", 'wb') as f:
        np.save(f, left_hand_lm)
    cv2.imwrite(
        filename=f"{LHAND_FOLDER_ABS /'left_hand.jpg'}",
        img=left_hand_skeleton,
    )
    print(f"created {LHAND_FOLDER_ABS}")
if __name__=="__main__":
    main()
