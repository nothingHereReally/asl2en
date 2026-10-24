from cv2 import circle, imwrite, line
from tempfile import gettempdir
import jax.numpy as jnp
import numpy as np
from pathlib import Path


# ------------------------
# ------------------------
# ---- contants start ----
PROJ_ROOT: Path= Path(__file__).resolve().parent.parent.parent
GET_SHOULDER_FROM: Path= PROJ_ROOT /"dataset" /"clean_dataset" /"investigate_landmark" /"book_07072" /"00000001.npy"
def get_tmp_folder() -> Path:
    out: Path= Path(gettempdir()).resolve() /f"see_shoulder__{np.random.randint(1, 1000)}"
    while out.exists():
        out= Path(gettempdir()).resolve() /f"see_shoulder__{np.random.randint(1, 4_000)}"
    out.mkdir()
    return out
IMG_SIZE: int= 1_500
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
LM_IDX_LEFT_SHOULDER: int  = 36
LM_IDX_RIGHT_SHOULDER: int = 37
LM_IDX_LEFT_HIP: int       = 42
LM_IDX_RIGHT_HIP: int      = 43
# ---- contants end ------
# ------------------------
# ------------------------








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
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
        ])
    if hasLandmarks['right_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
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
            ), dtype=np.float32).tolist()
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
            ), dtype=np.float32).tolist())

    # -- left hand landmarks --
    if hasLandmarks["left_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +QUANTITY_HAND_LMARK
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
                len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ])
        else:
            out_landmark.extend(norm[
                :QUANTITY_HAND_LMARK
            ])
    else:
        out_landmark.extend(np.zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            ), dtype=np.float32).tolist())

    # -- right hand landmarks --
    if hasLandmarks["right_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["face"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
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
                QUANTITY_HAND_LMARK:
            ])
        else:
            out_landmark.extend(norm)
    else:
        out_landmark.extend(np.zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            ), dtype=np.float32).tolist())

    # out_landmark shape:
    # (
    #     len(WORTHY_FACE_IDX)
    #     +len(WORTHY_POSE_IDX)
    #     +QUANTITY_HAND_LMARK *2,
    #     2
    # )
    return out_landmark # ie. output shape (86, 2)
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
    if color_line!=None and color_dot!=None:
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
                    circle(
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
                    line(
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
def extract_landmarks(lmark_mph, orig_shape: tuple) -> list:
    landmark__face_pose_left_right_hand: list= np.zeros((
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +(QUANTITY_HAND_LMARK*2),
        2
    ), dtype=np.float32).tolist()
    if lmark_mph.face_landmarks!=None \
        or lmark_mph.pose_landmarks!=None \
        or lmark_mph.left_hand_landmarks!=None \
        or lmark_mph.right_hand_landmarks!=None:
        landmark__face_pose_left_right_hand= list()
        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0

        # ---- face landmarks ----
        if lmark_mph.face_landmarks != None:
            for idx, el in enumerate(lmark_mph.face_landmarks.landmark):
                if idx in WORTHY_FACE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((len(WORTHY_FACE_IDX), 2)).tolist())

        # ---- pose landmarks ----
        if lmark_mph.pose_landmarks != None:
            for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
                if idx in WORTHY_POSE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((len(WORTHY_POSE_IDX), 2)).tolist())

        # ---- left hand landmarks ----
        if lmark_mph.left_hand_landmarks != None:
            for el in lmark_mph.left_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((QUANTITY_HAND_LMARK, 2)).tolist())

        # ---- right hand landmarks ----
        if lmark_mph.right_hand_landmarks != None:
            for el in lmark_mph.right_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((QUANTITY_HAND_LMARK, 2)).tolist())


        landmark__face_pose_left_right_hand= normalizeWorthyLandmarkWrapper(
            landmark__face_pose_left_right_hand,
            orig_shape,
            {
                'face': lmark_mph.face_landmarks != None,
                'pose': lmark_mph.pose_landmarks != None,
                'left_hand': lmark_mph.left_hand_landmarks != None,
                'right_hand': lmark_mph.right_hand_landmarks != None,
            }
        )
    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return landmark__face_pose_left_right_hand
def drawFacePoseHandAnd2dots(
    img_write_to: np.ndarray,
    landmarks: list,
    landmarks2dots: list,
    color_up_dot: tuple=(23, 23, 255),
    color_down_dot: tuple=(110, 190, 245),
) -> np.ndarray:
    # ---- face landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[:len(WORTHY_FACE_IDX)],
        connections_idxs=FACE_CONNECTIONS,
        thick=1,
        color_dot=color_up_dot,
        color_line=color_down_dot,
        highlight_index=None,
    )

    # ---- pose landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[
            len(WORTHY_FACE_IDX): len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)
        ],
        connections_idxs=POSE_CONNECTIONS,
        thick=1,
        color_dot=color_up_dot,
        color_line=color_down_dot,
        highlight_index=None
    )

    # ---- left hand landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[
            len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX): len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK
        ],
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=color_up_dot,
        color_line=color_down_dot,
        highlight_index=None
    )

    # ---- right hand landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[ len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK: ],
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=color_up_dot,
        color_line=color_down_dot,
        highlight_index=None
    )

    # -- line for 2 dots on body --
    line(
        img=img_write_to,
        pt1=(
            int(landmarks2dots[0][0]*img_write_to.shape[0]),
            int(landmarks2dots[0][1]*img_write_to.shape[0]),
        ),
        pt2=(
            int(landmarks2dots[1][0]*img_write_to.shape[0]),
            int(landmarks2dots[1][1]*img_write_to.shape[0]),
        ),
        color=color_down_dot,
        thickness=4
    )

    # -- upper dot --
    assert img_write_to.shape[0]==img_write_to.shape[1]
    circle(
        img=img_write_to,
        center=(
            int(landmarks2dots[0][0]*img_write_to.shape[0]),
            int(landmarks2dots[0][1]*img_write_to.shape[0]),
        ),
        radius=0,
        color=color_up_dot,
        thickness=25
    )

    # -- down dot --
    circle(
        img=img_write_to,
        center=(
            int(landmarks2dots[1][0]*img_write_to.shape[0]),
            int(landmarks2dots[1][1]*img_write_to.shape[0]),
        ),
        radius=0,
        color=color_down_dot,
        thickness=10
    )


    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to
def create_moved_lm_images(landmarks: list[tuple[float, float]], abs_folder: Path) -> None:
    # ---- for manually moving parts, left hand ----
    left_hand_skeleton= drawSkeletonImg(
        image=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
        lmark_coordinates=landmarks[36+8:-21],
        connections_idxs=HAND_CONNECTIONS,
        thick=2,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    left_hand_lm: jnp.ndarray= jnp.array(landmarks[36+8:-21])
    left_hand_lm= left_hand_lm +(
        left_hand_lm[0:1, :] -0.5
    )*-1.0
    # ---- left hand landmarks ----
    left_hand_skeleton= drawSkeletonImg(
        image=left_hand_skeleton,
        lmark_coordinates=left_hand_lm.tolist(),
        connections_idxs=HAND_CONNECTIONS,
        thick=6,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    imwrite(
        filename=f"{abs_folder /"human_03_left_hand_from_origin_to_center.jpg"}",
        img=left_hand_skeleton,
    )

    # ---- for manually moving parts, right hand ----
    right_hand_skeleton= drawSkeletonImg(
        image=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
        lmark_coordinates=landmarks[-21:],
        connections_idxs=HAND_CONNECTIONS,
        thick=2,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    right_hand_lm: jnp.ndarray= jnp.array(landmarks[-21:])
    right_hand_lm= right_hand_lm +(
        right_hand_lm[0:1, :] -0.5
    )*-1.0
    # ---- right hand landmarks ----
    right_hand_skeleton= drawSkeletonImg(
        image=right_hand_skeleton,
        lmark_coordinates=right_hand_lm.tolist(),
        connections_idxs=HAND_CONNECTIONS,
        thick=6,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    imwrite(
        filename=f"{abs_folder /"human_04_right_hand_from_origin_to_center.jpg"}",
        img=right_hand_skeleton,
    )

    # ---- for manually moving parts, left and right hand ----
    left_right_hand_skeleton= drawSkeletonImg(
        image=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
        lmark_coordinates=landmarks[36+8:-21],
        connections_idxs=HAND_CONNECTIONS,
        thick=2,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    left_right_hand_skeleton= drawSkeletonImg(
        image=left_right_hand_skeleton,
        lmark_coordinates=left_hand_lm.tolist(),
        connections_idxs=HAND_CONNECTIONS,
        thick=6,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    left_right_hand_skeleton= drawSkeletonImg(
        image=left_right_hand_skeleton,
        lmark_coordinates=landmarks[-21:],
        connections_idxs=HAND_CONNECTIONS,
        thick=2,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    left_right_hand_skeleton= drawSkeletonImg(
        image=left_right_hand_skeleton,
        lmark_coordinates=right_hand_lm.tolist(),
        connections_idxs=HAND_CONNECTIONS,
        thick=6,
        color_dot=(23, 56, 90),
        color_line=(90, 23, 56),
        highlight_index=None
    )
    imwrite(
        filename=f"{abs_folder /"human_05_left_right_hand_from_origin_to_center.jpg"}",
        img=left_right_hand_skeleton,
    )
def create_images_for2dots(
    landmarks: list[tuple[float, float]],
    landmarks_flip: list[tuple[float, float]],
    abs_folder: Path,
) -> None:
    image: np.ndarray= drawFacePoseHandAnd2dots(
        img_write_to=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
        landmarks=landmarks,
        landmarks2dots=get2dots_frombody(jnp.array(landmarks)).tolist(),
    )
    image_flip: np.ndarray= drawFacePoseHandAnd2dots(
        img_write_to=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
        landmarks=landmarks_flip,
        landmarks2dots=get2dots_frombody(jnp.array(landmarks_flip)).tolist(),
    )
    imwrite(
        filename=f"{abs_folder /"human_01_origin_with2dots_on_body.jpg"}",
        img=image,
    )
    imwrite(
        filename=f"{abs_folder /"human_02_flip_with2dots_on_body.jpg"}",
        img=image_flip,
    )

    create_moved_lm_images(landmarks, abs_folder)
def drawFacePoseHand2color(
    img_write_to: np.ndarray,
    landmarks: list,
    index: int,
    color_dot: tuple=(23, 23, 255),
    color_the_rest: tuple=(110, 190, 245),
) -> np.ndarray:
    # ---- face landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[:len(WORTHY_FACE_IDX)],
        connections_idxs=FACE_CONNECTIONS,
        thick=1,
        color_dot=color_dot,
        color_line=color_the_rest,
        highlight_index=index if index<len(WORTHY_FACE_IDX) else None,
    )

    # ---- pose landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[
            len(WORTHY_FACE_IDX): len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)
        ],
        connections_idxs=POSE_CONNECTIONS,
        thick=1,
        color_dot=color_dot,
        color_line=color_the_rest,
        highlight_index=int(index-len(WORTHY_FACE_IDX)) if len(WORTHY_FACE_IDX)<=index and index<int(
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
        ) else None
    )

    # ---- left hand landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[
            len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX): len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK
        ],
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=color_dot,
        color_line=color_the_rest,
        highlight_index=int(
            index -len(WORTHY_FACE_IDX) -len(WORTHY_POSE_IDX)
        ) if int(len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX))<=index and index<int(
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
        ) else None
    )

    # ---- right hand landmarks ----
    img_write_to= drawSkeletonImg(
        image=img_write_to,
        lmark_coordinates=landmarks[ len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK: ],
        connections_idxs=HAND_CONNECTIONS,
        thick=1,
        color_dot=color_dot,
        color_line=color_the_rest,
        highlight_index=int(
            index -len(WORTHY_FACE_IDX) -len(WORTHY_POSE_IDX) -QUANTITY_HAND_LMARK
        ) if int(len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK)<=index else None
    )

    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to
def create_images2folder(landmarks: list[tuple[float, float]], folder: Path) -> None:
    for idx in range(len(landmarks)):
        file_name: str= f"{str(idx).zfill(3)}"
        if idx<len(WORTHY_FACE_IDX):
            file_name= f"{file_name}_face.jpg"
        elif idx<int(len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)):
            file_name= f"{file_name}_pose.jpg"
        elif idx<int(len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK):
            file_name= f"{file_name}_left_hand.jpg"
        else:
            file_name= f"{file_name}_right_hand.jpg"
        img_by_landmark: np.ndarray= drawFacePoseHand2color(
            img_write_to=np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            landmarks=landmarks,
            index=idx,
        )
        imwrite(
            filename=f"{folder /file_name}",
            img=img_by_landmark
        )
def rotate2same_y(
    right_shoulder: jnp.ndarray,
    left_shoulder: jnp.ndarray,
    others: jnp.ndarray|None,
) -> tuple:
    '''
    right_shoulder be of shape (BATCH, 2)
    left_shoulder be of shape (BATCH, 2)
    others be of shape (BATCH, INT, 2)
    ----------------------------------
    returns (
        updated_left_shoulder,
        updated_others
    )
    '''
    assert right_shoulder.shape == left_shoulder.shape
    assert len(right_shoulder.shape) == 2 and right_shoulder.shape[-1] == 2
    assert jnp.all(right_shoulder[..., 0] < left_shoulder[..., 0])

    rs_x= jnp.expand_dims(right_shoulder[..., 0:1], axis=1)
    rs_y= jnp.expand_dims(right_shoulder[..., 1:], axis=1)
    ls_x= jnp.expand_dims(left_shoulder[..., 0:1], axis=1)
    ls_y= jnp.expand_dims(left_shoulder[..., 1:], axis=1)
    dx: jnp.ndarray= ls_x -rs_x
    dy: jnp.ndarray= ls_y -rs_y
    distance: jnp.ndarray= (dx**2 + dy**2)**0.5
    if others == None:
        return (
            jnp.concatenate([
                rs_x +distance,
                rs_y
            ], axis=-1),
            None,
        )

    assert len(others.shape) == 3
    assert others.shape[0] == left_shoulder.shape[0]

    theta_x: jnp.ndarray= dx / distance
    theta_y: jnp.ndarray= dy / distance
    new_left_shoulder: jnp.ndarray= jnp.concatenate([
        rs_x +distance,
        rs_y
    ], axis=-1)
    others_x: jnp.ndarray= others[..., 0:1]
    others_y: jnp.ndarray= others[..., 1:]
    new_others: jnp.ndarray= jnp.concatenate([
        rs_x +(others_x-rs_x)*theta_x +(others_y-rs_y)*theta_y,
        rs_y -(others_x-rs_x)*theta_y +(others_y-rs_y)*theta_x,
    ], axis=-1)
    return (
        new_left_shoulder,
        new_others,
    )
def move2ab(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    dots: jnp.ndarray
) -> jnp.ndarray:
    '''
    a, b, c, d, all 4 of shape (BATCH, 2)
    dots of shape (BATCH, INT, 2)
    '''
    assert jnp.all(a[..., 0] < b[..., 0])
    assert jnp.all(c[..., 0] < d[..., 0])
    assert jnp.all(a[..., 1] == b[..., 1])
    assert jnp.all(c[..., 1] == d[..., 1])
    assert a.shape == b.shape and \
        b.shape == c.shape and \
        c.shape == d.shape and \
        len(d.shape) == 2 and \
        d.shape[-1] == 2
    assert dots.shape[-1] == 2 and \
        d.shape[0] == dots.shape[0] and \
        len(dots.shape) == 3
    # new_c: np.ndarray= a.copy()
    # new_b: np.ndarray= b.copy()
    expand_a= jnp.expand_dims(a, axis=1)
    expand_b= jnp.expand_dims(b, axis=1)
    expand_c= jnp.expand_dims(c, axis=1)
    expand_d= jnp.expand_dims(d, axis=1)
    new_dots= expand_a + (
        expand_b[..., 0:1] - expand_a[..., 0:1]
    )/(
        expand_d[..., 0:1] - expand_c[..., 0:1]
    )*(dots - expand_c)
    return new_dots
def stationary_be_shoulder(landmarks: jnp.ndarray) -> jnp.ndarray:
    assert landmarks.shape[-2:] == (86, 2)
    left_shoulder, others= rotate2same_y(
        right_shoulder=landmarks[..., (LM_IDX_RIGHT_SHOULDER,), :],
        left_shoulder=landmarks[..., (LM_IDX_LEFT_SHOULDER,), :],
        others=jnp.expand_dims(jnp.concatenate([
            landmarks[..., :LM_IDX_LEFT_SHOULDER, :],
            landmarks[..., LM_IDX_RIGHT_SHOULDER+1:, :]
        ], axis=-2), axis=0),
    )
    others= jnp.concatenate([
        others[0, :LM_IDX_LEFT_SHOULDER, :],
        left_shoulder[0],
        landmarks[..., (LM_IDX_RIGHT_SHOULDER,), :],
        others[0, LM_IDX_LEFT_SHOULDER:, :],
    ], axis=-2)

    return others
def main() -> None:
    landmarks: np.ndarray
    with open(GET_SHOULDER_FROM, 'rb') as f:
        landmarks= np.load(f)
    landmarks_shoulder_same_y: jnp.ndarray= stationary_be_shoulder(jnp.array(landmarks))
    imgs_folder: Path= get_tmp_folder()
    imgs_folder_r: Path= imgs_folder.parent /f"{imgs_folder.name}_rotated_shoulder"
    imgs_folder_r.mkdir()
    create_images2folder(
        landmarks=landmarks.tolist(),
        folder=imgs_folder,
    )
    create_images2folder(
        landmarks=landmarks_shoulder_same_y.tolist(),
        folder=imgs_folder_r
    )
    print(f"images folder origin --> {imgs_folder}")
    print(f"images folder rotated shoulder --> {imgs_folder_r}")
if __name__=="__main__":
    main()
