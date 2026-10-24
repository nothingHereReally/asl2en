import cv2
from mediapipe.python.solutions.holistic import Holistic
import numpy as np
from pathlib import Path
import random
import sys
from tempfile import gettempdir


IMG_SIZE: int= 158
MPH_fph: Holistic= Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)
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
            )).tolist())

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
            )).tolist())

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
def drawFacePoseHand(img_write_to: np.ndarray, lmark_mph, orig_shape: tuple) -> tuple:
    landmark__face_pose_left_right_hand: list= np.zeros((
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +(QUANTITY_HAND_LMARK*2),
        2
    )).tolist()
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


        # ---- face landmarks ----
        if lmark_mph.face_landmarks != None:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=landmark__face_pose_left_right_hand[:len(WORTHY_FACE_IDX)],
                connections_idxs=FACE_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(0, 153, 0), # 153/255= 0.6
            )

        # ---- pose landmarks ----
        if lmark_mph.pose_landmarks != None:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=landmark__face_pose_left_right_hand[
                    len(WORTHY_FACE_IDX): len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)
                ],
                connections_idxs=POSE_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(0, 0, 153), # 153/255= 0.6
            )

        # ---- left hand landmarks ----
        if lmark_mph.left_hand_landmarks != None:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=landmark__face_pose_left_right_hand[
                    len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX): len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK
                ],
                connections_idxs=HAND_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(255, 255, 255)
            )

        # ---- right hand landmarks ----
        if lmark_mph.right_hand_landmarks != None:
            img_write_to= drawSkeletonImg(
                image=img_write_to,
                lmark_coordinates=landmark__face_pose_left_right_hand[ len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK: ],
                connections_idxs=HAND_CONNECTIONS,
                thick=1,
                color_dot=None,
                color_line=(153, 204, 204), # 204/255= 0.8
            )

        # HERE ORDER OF LANDMARKS
        # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return (img_write_to, landmark__face_pose_left_right_hand)
def get_tmp_folder() -> Path:
    out: Path= Path(gettempdir()).resolve() /f"see_skeleton_video__{random.randint(1, 1000)}"
    while out.exists():
        out= Path(gettempdir()).resolve()   /f"see_skeleton_video__{random.randint(1, 4_000)}"
    out.mkdir()
    return out
def main() -> None:
    if len(sys.argv)<2:
        print("Please provide file.mp4 path")
    else:
        video_abs_path: Path= Path(sys.argv[1]).resolve()
        folder4video: Path= get_tmp_folder()
        video: cv2.VideoCapture= cv2.VideoCapture(f"{video_abs_path}")
        for idx in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            has_img, image= video.read()
            if has_img:
                image, _= drawFacePoseHand(
                    np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                    MPH_fph.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                    image.shape
                )
                cv2.imwrite(f"{folder4video /str(idx+1).zfill(8)}.jpg", image)
        print(f"images at --> {folder4video}")
if __name__=="__main__":
    main()
