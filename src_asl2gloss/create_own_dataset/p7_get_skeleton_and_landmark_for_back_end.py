from os.path import exists
from os import makedirs
from json import load as jsonload, dump as jsonsave
from cv2 import circle, imwrite, line
from sys import stderr
from numpy import array, load as npload, ndarray, uint8, zeros, save as numpysave


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"
GLASL_DIR: str= f"{PROJ_ROOT}dataset/glasl/"
LANDMARK_origin: str= f"{GLASL_DIR}landmark/"
LANDMARK_dir: str= f"{GLASL_DIR}image_landmark/"
SKELETON_dir: str= f"{GLASL_DIR}image_skeleton/"
KEY_TRAIN: str= "train"
KEY_VAL: str= "val"
KEY_TEST: str= "test"
KEY_ID2G: str= "id2gloss"
KEY_G2ID: str= "gloss2id"
IMG_SIZE: int= 240
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




def drawSkeletonImg(img_orig: ndarray, \
                    lmark_cords: tuple, \
                    conn_idxs_list: tuple, \
                    thick: int=2, \
                    color_conn: tuple=(255,0,255), \
                    color_lmark: tuple=(255,255,0), \
                    drawJoint: bool=True) -> ndarray:
    def isOKplt(coord: tuple) -> bool:
        # x and y coordinates
        # mandatory be greater than or equal to Zero
        # and less than or equal to One
        return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
    img: ndarray= img_orig.copy()
    img_wh: dict= {"wx": img.shape[1], "hy": img.shape[0]}


    # drawing the lines between 2 landmark connections
    for l in conn_idxs_list:
        pA: tuple= (
            lmark_cords[  l[0]  ][0], # x
            lmark_cords[  l[0]  ][1]  # y
        )
        pB: tuple= (
            lmark_cords[  l[1]  ][0], # x
            lmark_cords[  l[1]  ][1]  # y
        )
        if isOKplt(pA) and isOKplt(pB):
            line(
                img=img,
                pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                color=color_conn,
                thickness=thick
            )
        else:
            raise ValueError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0")
        del pA
        del pB


    # drawing joints as cricles
    if drawJoint:
        for o in lmark_cords:
            if isOKplt(o):
                circle(
                    img=img,
                    center=(
                        int(o[0]*img_wh['wx']),
                        int(o[1]*img_wh['hy'])
                    ),
                    radius=0,
                    color=color_lmark,
                    thickness=thick*2
                )
            else:
                raise ValueError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0")
    return img
def drawFacePoseHand(
        img_write_to: ndarray,
        landmark_numpy: ndarray,
        has_face: bool,
        has_pose: bool,
        has_left_hand: bool,
        has_right_hand: bool
    ) -> ndarray:
    def recalcDrawFace(img_orig: ndarray, lmark_face: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_face,
            conn_idxs_list=FACE_CONNECTIONS,
            thick=1,
            color_conn=(0, 153, 0), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawPose(img_orig: ndarray, lmark_pose: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_pose,
            conn_idxs_list=POSE_CONNECTIONS,
            thick=1,
            color_conn=(0, 0, 153), # 153/255= 0.6
            drawJoint=False
        )
    def recalcDrawLeftHands(img_orig: ndarray, lmark_lhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_lhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(255, 255, 255),
            drawJoint=False
        )
    def recalcDrawRightHands(img_orig: ndarray, lmark_rhand: tuple) -> ndarray:
        img: ndarray= img_orig.copy()
        return drawSkeletonImg(
            img_orig=img,
            lmark_cords=lmark_rhand,
            conn_idxs_list=HAND_CONNECTIONS,
            thick=1,
            color_conn=(204, 204, 204), # 204/255= 0.8
            drawJoint=False
        )
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
    quantity_lm_face: int= len(WORTHY_FACE_IDX)
    quantity_lm_pose: int= len(WORTHY_POSE_IDX)
    if has_face:
        # print(f"{array(landmark_numpy[:len(WORTHY_FACE_IDX)]).shape}")
        img_write_to= recalcDrawFace(img_write_to, tuple(landmark_numpy.tolist()[:quantity_lm_face]))
    if has_pose:
        # print(f"{array(landmark_numpy[len(WORTHY_FACE_IDX):len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)]).shape}")
        img_write_to= recalcDrawPose(img_write_to, tuple(landmark_numpy.tolist()[quantity_lm_face:quantity_lm_face+quantity_lm_pose]))
    if has_left_hand:
        # print(f"{array(landmark_numpy[len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX):len(WORTHY_FACE_IDX)+len(WORTHY_POSE_IDX)+21]).shape}")
        img_write_to= recalcDrawLeftHands(img_write_to, tuple(landmark_numpy.tolist()[quantity_lm_face+quantity_lm_pose:quantity_lm_face+quantity_lm_pose+21]))
    if has_right_hand:
        # print(f"{array(landmark_numpy[-21:]).shape}")
        img_write_to= recalcDrawRightHands(img_write_to, tuple(landmark_numpy.tolist()[-21:]))

    return img_write_to


def get_video_skeletons(gloss_video_lmark: dict) -> ndarray:
    allImg_skeleton: list= []
    for lmark_np in gloss_video_lmark['landmark']:
        skeleton__image= drawFacePoseHand(
            img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
            landmark_numpy=npload(f"{LANDMARK_origin}{gloss_video_lmark['video_id']}/{lmark_np['file']}"),
            has_face=lmark_np['face'],
            has_pose=lmark_np['pose'],
            has_left_hand=lmark_np['left_hand'],
            has_right_hand=lmark_np['right_hand'],
        )
        allImg_skeleton.append(skeleton__image)

    if len(gloss_video_lmark['landmark'])!=len(allImg_skeleton):
        print(f"len gloss_video_lmark['landmark'] --> {len(gloss_video_lmark['landmark'])}", file=stderr)
        print(f"len allImg_skeleton --> {len(allImg_skeleton)}", file=stderr)
        raise NotImplementedError("Incorrect implementation due to mandatory all 4 be having same quantity of elements")
    return array(allImg_skeleton, dtype=uint8)


def mandatory_all_2_notExist() -> None:
    if exists(LANDMARK_dir):
        raise FileExistsError(f"please delete this folder {LANDMARK_dir}, will be the one to create it for you.")
    if exists(SKELETON_dir):
        raise FileExistsError(f"please delete this folder {SKELETON_dir}, will be the one to create it for you.")

    makedirs(LANDMARK_dir)
    makedirs(SKELETON_dir)


def init_vars() -> tuple:
    glasl_clean_landmark: list= []
    with open(f"{GLASL_DIR}glasl.annotation.landmark.json", 'r') as f:
        glasl_clean_landmark= jsonload(f)
    glasl_LANDMARK: dict= {
        KEY_TRAIN: [],
        KEY_VAL: [],
        KEY_TEST: [],
        KEY_ID2G: [ins for ins in glasl_clean_landmark[KEY_ID2G]],
        KEY_G2ID: {glasl_clean_landmark[KEY_ID2G][i]: i for i in range(len(glasl_clean_landmark[KEY_ID2G]))},
    }
    glasl_SKELETON: dict= {
        KEY_TRAIN: [],
        KEY_VAL: [],
        KEY_TEST: [],
        KEY_ID2G: [ins for ins in glasl_clean_landmark[KEY_ID2G]],
        KEY_G2ID: {glasl_clean_landmark[KEY_ID2G][i]: i for i in range(len(glasl_clean_landmark[KEY_ID2G]))},
    }
    return (glasl_clean_landmark, glasl_LANDMARK, glasl_SKELETON)


if __name__=='__main__':
    mandatory_all_2_notExist()
    glasl_c_landmark, glasl_LANDMARK, glasl_SKELETON= init_vars()


    for data_split in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        for gloss_video in glasl_c_landmark[data_split]: # on each split has many videos, now for each videos
            # gloss_video['gloss_id']
            # gloss_video['video_id']
            # gloss_video['landmark']
            # ---- gloss_video['landmark'][...] is has key
            # -------- gloss_video['landmark'][...]['file']
            # -------- gloss_video['landmark'][...]['face']
            # -------- gloss_video['landmark'][...]['pose']
            # -------- gloss_video['landmark'][...]['left_hand']
            # -------- gloss_video['landmark'][...]['right_hand']
            imgs_skeleton= get_video_skeletons(gloss_video)
            glasl_LANDMARK[ data_split ].append({
                "gloss_id": gloss_video['gloss_id'],
                "video_id": gloss_video['video_id'],
                "landmark": [],
            })
            glasl_SKELETON[ data_split ].append({
                "gloss_id": gloss_video['gloss_id'],
                "video_id": gloss_video['video_id'],
                "skeleton": [],
            })
            for i in range(imgs_skeleton.shape[0]): # each video has many images, now for each images
                file2create: str= f"{gloss_video['video_id']}_{gloss_video['landmark'][i]['file'][:-4]}"
                filename_abs_landmark_w: str= f"{LANDMARK_dir}{file2create}.npy"
                filename_abs_skeleton_w: str= f"{SKELETON_dir}{file2create}.png"
                with open(filename_abs_landmark_w, "wb") as f:
                    numpysave(file=f, arr=npload(f"{LANDMARK_origin}{gloss_video['video_id']}/{gloss_video['landmark'][i]['file']}"))
                imwrite(filename=filename_abs_skeleton_w, img=imgs_skeleton[i])
                glasl_LANDMARK[ data_split ][-1]["landmark"].append({
                    "file": f"{file2create}.npy",
                    "face": gloss_video['landmark'][i]['face'],
                    "pose": gloss_video['landmark'][i]['pose'],
                    "left_hand": gloss_video['landmark'][i]['left_hand'],
                    "right_hand": gloss_video['landmark'][i]['right_hand'],
                })
                glasl_SKELETON[ data_split ][-1]["skeleton"].append({
                    "file": f"{file2create}.png",
                    "face": gloss_video['landmark'][i]['face'],
                    "pose": gloss_video['landmark'][i]['pose'],
                    "left_hand": gloss_video['landmark'][i]['left_hand'],
                    "right_hand": gloss_video['landmark'][i]['right_hand'],
                    "width": IMG_SIZE,
                    "height": IMG_SIZE,
                })
    with open(f"{GLASL_DIR}glasl.annotation.image_landmark.json", "w") as f:
        jsonsave(glasl_LANDMARK, f, indent=4)
    with open(f"{GLASL_DIR}glasl.annotation.image_skeleton.json", "w") as f:
        jsonsave(glasl_SKELETON, f, indent=4)
