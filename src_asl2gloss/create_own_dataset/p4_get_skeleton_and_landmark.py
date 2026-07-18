from os.path import exists
from os import makedirs
from json import load as jsonload, dump as jsonsave
from typing import Any
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, circle, cvtColor, imwrite, line
from sys import stderr
from numpy import array, float32, ndarray, uint8, zeros, save as numpysave
from mediapipe.python.solutions.holistic import Holistic
from pathlib import Path


# ------------------------
# ------------------------
# ---- contants start ----
PROJ_ROOT= Path(__file__).resolve().parent.parent.parent
GLASL_DIR: Path= PROJ_ROOT /"dataset" /"glasl"
VIDEO_DIR: Path= GLASL_DIR /"video"
IMAGE_dir: Path= GLASL_DIR /"image"
LANDMARK_dir: Path= GLASL_DIR /"landmark"
SKELETON_dir: Path= GLASL_DIR /"skeleton"
MPH_fph: Holistic= Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
KEY_TRAIN: str= "train"
KEY_VAL: str= "val"
KEY_TEST: str= "test"
KEY_ID2G: str= "id2gloss"
KEY_G2ID: str= "gloss2id"
IMG_SIZE: int= 158
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
# ---- contants end ------
# ------------------------
# ------------------------




def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
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
    xs= list(map(lambda el: el[0], landmarks))
    ys= list(map(lambda el: el[1], landmarks))
    xs= list(filter(lambda el: el!=0, xs))
    ys= list(filter(lambda el: el!=0, ys))
    min_x, min_y=    min(xs), min(ys)
    max_x, max_y=    max(xs), max(ys)
    scale: float= (1  -2*pad)/max(
        max_x -min_x,
        max_y -min_y
    )
    return [(
        (x -min_x)    *scale    +pad  if x!=0 else x,
        (y -min_y)    *scale    +pad  if y!=0 else y
    ) for x, y in landmarks]
def part4_centerLandmarkVerticallyHorizontally(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 3) center landmark with same aspect ratio as original
    # center horizontally and vertically, since done padding then just
    # move to right/down
    # xs, ys = zip(*landmarks)
    xs= list(map(lambda el: el[0], landmarks))
    ys= list(map(lambda el: el[1], landmarks))
    xs= list(filter(lambda el: el!=0, xs))
    ys= list(filter(lambda el: el!=0, ys))
    shift_x: float=  0.5    -(min(xs) +max(xs))  /2
    shift_y: float=  0.5    -(min(ys) +max(ys))  /2

    return [(
        x +shift_x  if x!=0 else x,
        y +shift_y  if y!=0 else y
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
    assert len(landmarks)==len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+QUANTITY_HAND_LMARK*2
    landmarks= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks)
    landmarks= part2_beSquareRatioOnImage(
        landmarks,
        (original_shape[0], original_shape[1])
    )
    landmarks= part3_zoomInOutForPadding(landmarks)
    landmarks= part4_centerLandmarkVerticallyHorizontally(landmarks)


    return landmarks
def drawSkeletonImg(image: ndarray, \
                    lmark_coordinates: list, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> ndarray:
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
def drawFacePoseHand(img_write_to: ndarray, lmark_mph, orig_shape: tuple) -> tuple:
    landmark__face_pose_left_right_hand: ndarray|list= zeros((
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +(QUANTITY_HAND_LMARK*2),
        2
    ), dtype=float32)
    landmark__face_pose_left_right_hand= landmark__face_pose_left_right_hand.tolist()
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
            landmark__face_pose_left_right_hand.extend(zeros((len(WORTHY_FACE_IDX), 2)).tolist())

        # ---- pose landmarks ----
        if lmark_mph.pose_landmarks != None:
            for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
                if idx in WORTHY_POSE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(zeros((len(WORTHY_POSE_IDX), 2)).tolist())

        # ---- left hand landmarks ----
        if lmark_mph.left_hand_landmarks != None:
            for el in lmark_mph.left_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(zeros((QUANTITY_HAND_LMARK, 2)).tolist())

        # ---- right hand landmarks ----
        if lmark_mph.right_hand_landmarks != None:
            for el in lmark_mph.right_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(zeros((QUANTITY_HAND_LMARK, 2)).tolist())


        landmark__face_pose_left_right_hand= normalizeLandmarks(
            landmark__face_pose_left_right_hand,
            orig_shape
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


def get_images_from_video(split_vid_dict: dict) -> ndarray:
    video_abs_file_dir: Path= VIDEO_DIR /split_vid_dict["video_file"]
    if exists(video_abs_file_dir):
        try:
            video_ocv: VideoCapture= VideoCapture(str(video_abs_file_dir))
            frames_on_video: list= []
            if video_ocv.isOpened():
                for _ in range(  int(video_ocv.get(CAP_PROP_FRAME_COUNT))  ):
                    isNotEmpty, obj_image= video_ocv.read()
                    if isNotEmpty and 0<len(obj_image):
                        frames_on_video.append(array(obj_image, dtype=uint8))
                if len(frames_on_video)<1:
                    raise ValueError(f"Video {VIDEO_DIR /split_vid_dict["video_file"]} has No images exist.")
                return array(frames_on_video, dtype=uint8)


        except Exception as e:
            print(f"error at video {VIDEO_DIR /split_vid_dict['video_file']}: {e}", file=stderr)
    raise FileNotFoundError(f"Video {split_vid_dict["video_file"]} Does Not Exist --> No such file {video_abs_file_dir}")


def get_video_details(split_vid_dict: dict) -> tuple:
    allImg_human: ndarray= get_images_from_video(split_vid_dict)
    allImg_landmark: list= []
    allImg_skeleton: list= []
    allImg_details: list= []
    for img in allImg_human:
        fph_lmark: Any= MPH_fph.process(cvtColor(src=img, code=COLOR_BGR2RGB))
        skeleton__image, landmark__fpLhRh= drawFacePoseHand(
            img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
            lmark_mph=fph_lmark,
            orig_shape=img.shape
        )
        allImg_landmark.append(landmark__fpLhRh)
        allImg_skeleton.append(skeleton__image)
        allImg_details.append({
            'face': fph_lmark.face_landmarks != None,
            'pose': fph_lmark.face_landmarks != None,
            'left_hand': fph_lmark.left_hand_landmarks != None,
            'right_hand': fph_lmark.right_hand_landmarks != None,
            'width': IMG_SIZE,
            'height': IMG_SIZE
        })

    if len(allImg_human)!=len(allImg_landmark) or len(allImg_landmark)!=len(allImg_skeleton) or len(allImg_skeleton)!=len(allImg_details):
        print(f"len allImg_human --> {len(allImg_human)}", file=stderr)
        print(f"len allImg_landmark --> {len(allImg_landmark)}", file=stderr)
        print(f"len allImg_skeleton --> {len(allImg_skeleton)}", file=stderr)
        print(f"len allImg_details --> {len(allImg_details)}", file=stderr)
        raise NotImplementedError("Incorrect implementation due to mandatory all 4 be having same quantity of elements")
    return (
        array(allImg_human, dtype=uint8),
        array(allImg_landmark, dtype=float32),
        array(allImg_skeleton, dtype=uint8),
        allImg_details,
    )


def mandatory_all_3_notExist() -> None:
    if exists(IMAGE_dir):
        raise FileExistsError(f"please delete this folder {IMAGE_dir}, will be the one to create it for you.")
    if exists(LANDMARK_dir):
        raise FileExistsError(f"please delete this folder {LANDMARK_dir}, will be the one to create it for you.")
    if exists(SKELETON_dir):
        raise FileExistsError(f"please delete this folder {SKELETON_dir}, will be the one to create it for you.")

    makedirs(IMAGE_dir)
    makedirs(LANDMARK_dir)
    makedirs(SKELETON_dir)


def processDataForTrainingLater(glasl_clean: list, glasl_LANDMARK: dict, glasl_SKELETON: dict) -> tuple:
    for idxGloss, gloss_ds in enumerate(glasl_clean): # for each gloss ie. book, drink, computer, ...
        print(f"currently processing( {gloss_ds['gloss']} ) completed: {round(idxGloss/len(glasl_clean), 3)*100}%")
        for gloss_instance in gloss_ds["instances"]: # on each gloss has many videos, now for each videos
            imgs_human_rgb, imgs_landmark, imgs_skeleton, imgs_details= get_video_details(gloss_instance)
            # don't extract images due to takes too much space, ie. 10gloss about 46GiB
            # makedirs(f"{IMAGE_dir /gloss_instance["video_file"][:-4]}")
            makedirs(f"{LANDMARK_dir /gloss_instance["video_file"][:-4]}")
            makedirs(f"{SKELETON_dir /gloss_instance["video_file"][:-4]}")
            glasl_LANDMARK[ gloss_instance["split"] ].append({
                "gloss_id": int(glasl_LANDMARK[KEY_G2ID][gloss_ds["gloss"]]),
                "video_id": gloss_instance["video_file"][:-4],
                "landmark": [],
            })
            glasl_SKELETON[ gloss_instance["split"] ].append({
                "gloss_id": int(glasl_SKELETON[KEY_G2ID][gloss_ds["gloss"]]),
                "video_id": gloss_instance["video_file"][:-4],
                "skeleton": [],
            })
            for i in range(len(imgs_human_rgb)): # each video has many images, now for each images
                file2create: str= str(i+1).zfill(5)
                # filename_abs_human: Path= IMAGE_dir /gloss_instance["video_file"][:-4] /f"{file2create}.png"
                filename_abs_landmark: Path= LANDMARK_dir /gloss_instance["video_file"][:-4] /f"{file2create}.npy"
                filename_abs_skeleton: Path= SKELETON_dir /gloss_instance["video_file"][:-4] /f"{file2create}.png"
                # imwrite(filename=str(filename_abs_human), img=imgs_human_rgb[i])
                with open(str(filename_abs_landmark), "wb") as f:
                    # lanmarks order is face, then pose, then left hand, then right hand
                    # see `HERE ORDER OF LANDMARKS`
                    numpysave(file=f, arr=imgs_landmark[i])
                imwrite(filename=str(filename_abs_skeleton), img=imgs_skeleton[i])
                glasl_LANDMARK[ gloss_instance["split"] ][-1]["landmark"].append({
                    "file": f"{file2create}.npy",
                    "face": imgs_details[i]["face"],
                    "pose": imgs_details[i]["pose"],
                    "left_hand": imgs_details[i]["left_hand"],
                    "right_hand": imgs_details[i]["right_hand"],
                })
                glasl_SKELETON[ gloss_instance["split"] ][-1]["skeleton"].append({
                    "file": f"{file2create}.png",
                    "face": imgs_details[i]["face"],
                    "pose": imgs_details[i]["pose"],
                    "left_hand": imgs_details[i]["left_hand"],
                    "right_hand": imgs_details[i]["right_hand"],
                    "width": imgs_details[i]["width"],
                    "height": imgs_details[i]["height"],
                })
    return (glasl_LANDMARK, glasl_SKELETON)


def init_vars() -> tuple:
    glasl_clean: list= []
    with open(f"{GLASL_DIR /"glasl.annotation.clean.json"}", 'r') as f:
        glasl_clean= jsonload(f)
    glasl_LANDMARK: dict= {
        KEY_TRAIN: [],
        KEY_VAL: [],
        KEY_TEST: [],
        KEY_ID2G: [ins["gloss"] for ins in glasl_clean],
        KEY_G2ID: {glasl_clean[i]["gloss"]: i for i in range(len(glasl_clean))}
    }
    glasl_SKELETON: dict= {
        KEY_TRAIN: [],
        KEY_VAL: [],
        KEY_TEST: [],
        KEY_ID2G: [ins["gloss"] for ins in glasl_clean],
        KEY_G2ID: {glasl_clean[i]["gloss"]: i for i in range(len(glasl_clean))}
    }
    return (glasl_clean, glasl_LANDMARK, glasl_SKELETON)


def main() -> None:
    mandatory_all_3_notExist()
    glasl_clean, glasl_LANDMARK, glasl_SKELETON= init_vars()


    glasl_LANDMARK, glasl_SKELETON= processDataForTrainingLater(glasl_clean, glasl_LANDMARK, glasl_SKELETON)
    with open(f"{GLASL_DIR /"glasl.annotation.landmark.json"}", "w") as f:
        jsonsave(glasl_LANDMARK, f, indent=4)
    with open(f"{GLASL_DIR /"glasl.annotation.skeleton.json"}", "w") as f:
        jsonsave(glasl_SKELETON, f, indent=4)


if __name__=='__main__':
    main()
