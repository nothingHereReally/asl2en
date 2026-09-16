from typing import Any
from cv2 import CAP_PROP_FRAME_COUNT, COLOR_BGR2RGB, VideoCapture, circle, cvtColor, imread, imwrite, line
from json import dump as writeJson
from tempfile import gettempdir
from mediapipe.python.solutions.holistic import Holistic
from numpy import array, ndarray, uint8, zeros, float32, random, save as numpysave
from pathlib import Path


# ------------------------
# ------------------------
# ---- contants start ----
PROJ_ROOT= Path(__file__).resolve().parent.parent.parent
HYBRID_DS_DIR: Path= PROJ_ROOT /"dataset" /"clean_dataset"
VIDEO_DIR: Path= HYBRID_DS_DIR /"videos"
def get_tmp_folder() -> Path:
    out: Path= Path(gettempdir()).resolve() /f"p012_investigate__{random.randint(1, 1000)}"
    while out.exists():
        out= Path(gettempdir()).resolve() /f"p012_investigate__{random.randint(1, 4_000)}"
    out.mkdir()
    return out
IMAGE_tmp_dir: Path= Path('to_init')
LANDMARK_dir: Path= HYBRID_DS_DIR /"investigate_landmark"
SKELETON_dir: Path= HYBRID_DS_DIR /"investigate_skeleton"
IMG_SIZE: int= 158
MPH_fph: Holistic= Holistic(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.68,
    min_tracking_confidence=0.68,
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
KEY_G: str= 'gloss'
KEY_VID: str= 'videos'
HYBRID_DATA_SRC: list= [
    {
        KEY_G: '00_book',
        KEY_VID: [
            '69241.mp4',
            '65225.mp4',
            '68011.mp4',
            '68208.mp4',
            '68012.mp4',
            '70212.mp4',
            '70266.mp4',
            '07069.mp4',
            '07068.mp4',
            '07097.mp4',
            '07099.mp4',
            '07071.mp4',
            '67424.mp4',
            '07075.mp4',
            '07076.mp4',
            '07070.mp4',
            '07072.mp4',
            '07074.mp4',
            '07077.mp4',
            '07078.mp4',
            '07079.mp4',
            '07080.mp4',
            '07081.mp4',
            '07082.mp4',
            '07083.mp4',
            '07084.mp4',
            '07085.mp4',
            '07086.mp4',
            '07087.mp4',
            '07088.mp4',
            '07089.mp4',
            '07090.mp4',
            '07091.mp4',
            '07092.mp4',
            '07093.mp4',
            '07094.mp4',
            '07095.mp4',
            '07096.mp4',
            '07098.mp4',
        ]
    },
    {
        KEY_G: '01_drink',
        KEY_VID: [
            '69302.mp4',
            '70173.mp4',
            '68538.mp4',
            '68042.mp4',
            '68660.mp4',
            '68041.mp4',
            '17725.mp4',
            '17726.mp4',
            '17727.mp4',
            '17710.mp4',
            '17729.mp4',
            '17730.mp4',
            '17731.mp4',
            '17732.mp4',
            '17734.mp4',
            '17711.mp4',
            '17712.mp4',
            '17714.mp4',
            '17715.mp4',
            '17717.mp4',
            '17709.mp4',
            '67594.mp4',
            '17719.mp4',
            '17713.mp4',
            '17720.mp4',
            '17723.mp4',
            '17724.mp4',
            '65539.mp4',
        ]
    },
    {
        KEY_G: '02_computer',
        KEY_VID: [
            '12306.mp4',
            '68028.mp4',
            '69054.mp4',
            '12331.mp4',
            '12336.mp4',
            '12314.mp4',
            '12312.mp4',
            '12326.mp4',
            'computer_0004.mp4',
            'computer_0005.mp4',
        ]
    },
    {
        KEY_G: '03_before',
        KEY_VID: [
            '05744.mp4',
            '05729.mp4',
            '05730.mp4',
            '65167.mp4',
            '05727.mp4',
            '05733.mp4',
            '05739.mp4',
            '05747.mp4',
            '05748.mp4',
            '05750.mp4',
            '05740.mp4',
        ]
    },
    {
        KEY_G: '04_chair',
        KEY_VID: [
            '09847.mp4',
            '70230.mp4',
            '68580.mp4',
            '70263.mp4',
            '68019.mp4',
            '09865.mp4',
            '09866.mp4',
            '09848.mp4',
            '09867.mp4',
            '09869.mp4',
            '09850.mp4',
            '09851.mp4',
            '65328.mp4',
            '09853.mp4',
            '09855.mp4',
            '09856.mp4',
            '67483.mp4',
            '09857.mp4',
            '09858.mp4',
            '09860.mp4',
            '09862.mp4',
            '09863.mp4',
            '09849.mp4',
            '09854.mp4',
            '09859.mp4',
            '09861.mp4',
        ]
    },
    {
        KEY_G: '05_go',
        KEY_VID: [
            '24857.mp4',
            '69345.mp4',
            '68292.mp4',
            '24941.mp4',
            '24965.mp4',
            '24969.mp4',
            '24970.mp4',
            '24971.mp4',
            '24973.mp4',
            '24943.mp4',
            '24946.mp4',
            '24948.mp4',
            '24940.mp4',
            '67715.mp4',
            '24951.mp4',
            '24947.mp4',
            '24954.mp4',
            '24955.mp4',
            '24956.mp4',
            '65824.mp4',
        ]
    },
    {
        KEY_G: '06_clothes',
        KEY_VID: [
            '11305.mp4',
            '68870.mp4',
            '68024.mp4',
            '11327.mp4',
            '11328.mp4',
            '11310.mp4',
            '11329.mp4',
            '11330.mp4',
            '11312.mp4',
            '11314.mp4',
            '11315.mp4',
            '11309.mp4',
            '11316.mp4',
            '11324.mp4',
            '11311.mp4',
            '11313.mp4',
            '11317.mp4',
            '11318.mp4',
            '11319.mp4',
            '11320.mp4',
            '11321.mp4',
            '11322.mp4',
            '11323.mp4',
            '11325.mp4',
            '11326.mp4',
        ]
    },
    {
        KEY_G: '07_who',
        KEY_VID: [
            '63240.mp4',
            '63242.mp4',
            '63228.mp4',
            '63229.mp4',
            '63234.mp4',
            '63237.mp4',
            '63232.mp4',
            '70380.mp4',
            'who_0003.mp4',
            'who_0005.mp4',
        ]
    },
    {
        KEY_G: '08_candy',
        KEY_VID: [
            '68018.mp4',
            '70326.mp4',
            '68790.mp4',
            '08929.mp4',
            '08916.mp4',
            '08919.mp4',
            '08921.mp4',
            '08923.mp4',
            '65299.mp4',
            '08926.mp4',
            '08927.mp4',
            '08915.mp4',
            '08925.mp4',
            '67468.mp4',
        ]
    },
    {
        KEY_G: '09_cousin',
        KEY_VID: [
            '65415.mp4',
            '70332.mp4',
            '68592.mp4',
            '13647.mp4',
            '13648.mp4',
            '13631.mp4',
            '13632.mp4',
            '13633.mp4',
            '13630.mp4',
            '67535.mp4',
            '13643.mp4',
            '13644.mp4',
            '13645.mp4',
            '13646.mp4',
            '13634.mp4',
            '13635.mp4',
            '13636.mp4',
            '13637.mp4',
            '13638.mp4',
            '13639.mp4',
            '13640.mp4',
            '13641.mp4',
            '13642.mp4',
        ]
    },
    {
        KEY_G: '10_mine_my',
        KEY_VID: [
            '36139.mp4',
            '36141.mp4',
            '36144.mp4',
            '36145.mp4',
            '36146.mp4',
            '37464.mp4',
            '69404.mp4',
            '37477.mp4',
            '37469.mp4',
            '67920.mp4',
            '37470.mp4',
            '37472.mp4',
            '37474.mp4',
            '37475.mp4',
            '37476.mp4',
            '36140.mp4',
            '36142.mp4',
            '37473.mp4',
            '37471.mp4',
        ]
    },
    {
        KEY_G: '11_me_i',
        KEY_VID: [
            '35267.mp4',
            '35544.mp4',
            '67879.mp4',
            '35545.mp4',
            '35546.mp4',
            '35547.mp4',
            '35549.mp4',
            '35550.mp4',
            '35548.mp4',
            'me_i_0001.mp4',
        ]
    },
    {
        KEY_G: '12_stomach',
        KEY_VID: [
            '54870.mp4',
            '54885.mp4',
            '66560.mp4',
            '54878.mp4',
            '54880.mp4',
            '54881.mp4',
            '54883.mp4',
            '54884.mp4',
            'stomach_0001.mp4',
            'stomach_0002.mp4',
        ]
    },
    {
        KEY_G: '13_have',
        KEY_VID: [
            '26757.mp4',
            '69360.mp4',
            '68069.mp4',
            '70221.mp4',
            '68984.mp4',
            '69088.mp4',
            '26775.mp4',
            '26766.mp4',
            '67746.mp4',
            '26776.mp4',
            '26777.mp4',
            '26778.mp4',
            '26779.mp4',
            '26767.mp4',
            '26768.mp4',
            '26770.mp4',
            '26773.mp4',
        ]
    },
    {
        KEY_G: '14_need',
        KEY_VID: [
            '37879.mp4',
            '70237.mp4',
            '68544.mp4',
            '37885.mp4',
            '37887.mp4',
            '37888.mp4',
            '37889.mp4',
            '37890.mp4',
            '37881.mp4',
            '67925.mp4',
            '37891.mp4',
            '37892.mp4',
            '37893.mp4',
            '37894.mp4',
            '37882.mp4',
            '37883.mp4',
            '37884.mp4',
            '37886.mp4',
        ]
    },
    {
        KEY_G: '15_see',
        KEY_VID: [
            '50107.mp4',
            '69460.mp4',
            '68444.mp4',
            '50120.mp4',
            '67178.mp4',
            '50123.mp4',
            '50126.mp4',
            '50127.mp4',
            '50128.mp4',
            '50122.mp4',
            '50125.mp4',
        ]
    },
    {
        KEY_G: '16_feel',
        KEY_VID: [
            '21425.mp4',
            '69319.mp4',
            '21438.mp4',
            '21439.mp4',
            '21440.mp4',
            '21441.mp4',
            '65696.mp4',
            '21432.mp4',
            '67653.mp4',
            '21442.mp4',
            '21433.mp4',
            '21435.mp4',
            '21437.mp4',
            '21434.mp4',
            '21436.mp4',
        ]
    },
    {
        KEY_G: '17_hurt',
        KEY_VID: [
            '28438.mp4',
            '70147.mp4',
            '28441.mp4',
            '28443.mp4',
            '28444.mp4',
            '28450.mp4',
            '28452.mp4',
            '28453.mp4',
            '28454.mp4',
            '28440.mp4',
            '28445.mp4',
            '28446.mp4',
        ]
    },
    {
        KEY_G: '18_fever',
        KEY_VID: [
            '78219.mp4',
            '78220.mp4',
            '78221.mp4',
            '78222.mp4',
            '78223.mp4',
            '78224.mp4',
            '78225.mp4',
            '78226.mp4',
            'fever_0001.mp4',
            'fever_0002.mp4',
        ]
    },
    {
        KEY_G: '19_dizzy',
        KEY_VID: [
            '16980.mp4',
            '65501.mp4',
            '16982.mp4',
            '16984.mp4',
            '16985.mp4',
            '16986.mp4',
            '16987.mp4',
            '16988.mp4',
            '78227.mp4',
            '78228.mp4',
            '78229.mp4',
        ]
    },
    {
        KEY_G: '20_headache',
        KEY_VID: [
            '26832.mp4',
            '26835.mp4',
            '26836.mp4',
            '26837.mp4',
            '26838.mp4',
            '26841.mp4',
            '65881.mp4',
            '67747.mp4',
            '26842.mp4',
            '26843.mp4',
            '26844.mp4',
            '26845.mp4',
            '26846.mp4',
            '26833.mp4',
            '26834.mp4',
            '26839.mp4',
            '26840.mp4',
        ]
    },
    {
        KEY_G: '21_doctor',
        KEY_VID: [
            '17007.mp4',
            '68842.mp4',
            '70049.mp4',
            '17015.mp4',
            '17016.mp4',
            '17017.mp4',
            '17018.mp4',
            '17022.mp4',
            '65503.mp4',
            '65504.mp4',
            '17013.mp4',
            '67579.mp4',
            '17023.mp4',
            '17024.mp4',
            '17025.mp4',
            '17026.mp4',
            '17014.mp4',
            '17019.mp4',
            '17020.mp4',
        ]
    }
]
# ---- contants end ------
# ------------------------
# ------------------------








def get_images_abs_path(video_file_name: str) -> list[str]:
    video_abs_file_dir: Path= VIDEO_DIR /video_file_name
    if video_abs_file_dir.exists():
        try:
            video_ocv: VideoCapture= VideoCapture(str(video_abs_file_dir))
            images_list: list[str]= list()
            if video_ocv.isOpened():
                for _ in range(  int(video_ocv.get(CAP_PROP_FRAME_COUNT))  ):
                    isNotEmpty, obj_image= video_ocv.read()
                    if isNotEmpty and 0<len(obj_image):
                        filename: str= f"{str(len(images_list)+1).zfill(8)}.jpg"
                        imwrite(f"{IMAGE_tmp_dir /filename}", obj_image)
                        images_list.append(f"{IMAGE_tmp_dir /filename}")
                if len(images_list)<1:
                    raise ValueError(f"Video {VIDEO_DIR /video_file_name} has No images exist.")
                return images_list


        except Exception as e:
            print(f"error at video {VIDEO_DIR /video_file_name}: {e}")
    raise FileNotFoundError(f"Video {video_file_name} Does Not Exist --> No such file {video_abs_file_dir}")
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
        out_landmark.extend(zeros(
            (
                len(WORTHY_FACE_IDX),
                2
            ), dtype=float32).tolist()
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
        out_landmark.extend(zeros(
            (
                len(WORTHY_POSE_IDX),
                2
            ), dtype=float32).tolist())

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
        out_landmark.extend(zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            ), dtype=float32).tolist())

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
        out_landmark.extend(zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            ), dtype=float32).tolist())

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
    landmark__face_pose_left_right_hand: list= zeros((
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +(QUANTITY_HAND_LMARK*2),
        2
    ), dtype=float32).tolist()
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
def init_dirs() -> None:
    global IMAGE_tmp_dir
    global LANDMARK_dir
    global SKELETON_dir
    IMAGE_tmp_dir= get_tmp_folder()
    if LANDMARK_dir.exists():
        print(f"Please delete {LANDMARK_dir}, this will automatically be created for you")
    if SKELETON_dir.exists():
        print(f"Please delete {SKELETON_dir}, this will automatically be created for you")
    if LANDMARK_dir.exists() or SKELETON_dir.exists():
        raise NotImplementedError(f"Can't proceed, due to {LANDMARK_dir} or {SKELETON_dir} exist.")

    LANDMARK_dir.mkdir()
    SKELETON_dir.mkdir()
def main() -> None:
    init_dirs()
    data2write_landmark: list= []
    data2write_skeleton: list= []
    for a_gloss in HYBRID_DATA_SRC:
        data2write_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VID: []
        })
        data2write_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VID: []
        })
        for a_video_fname in a_gloss[KEY_VID]:
            data2write_landmark[-1][KEY_VID].append({
                'video_file': a_video_fname,
                'landmark': []
            })
            data2write_skeleton[-1][KEY_VID].append({
                'video_file': a_video_fname,
                'skeleton': []
            })
            abs_landmark_under_folder: Path= LANDMARK_dir /f"{a_gloss[KEY_G]}_{Path(a_video_fname).stem}"
            abs_skeleton_under_folder: Path= SKELETON_dir /f"{a_gloss[KEY_G]}_{Path(a_video_fname).stem}"
            abs_landmark_under_folder.mkdir()
            abs_skeleton_under_folder.mkdir()
            images: list[str]= get_images_abs_path(a_video_fname)
            for idx, an_image_abs in enumerate(images):
                img_loaded= imread(an_image_abs)
                img_loaded= array(img_loaded, dtype=uint8)
                fph: Any= MPH_fph.process(cvtColor(src=img_loaded, code=COLOR_BGR2RGB))
                img_skeleton, lmark_face_pose_lr_hand= drawFacePoseHand(
                    img_write_to=zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8),
                    lmark_mph=fph,
                    orig_shape=img_loaded.shape
                )
                img_landmark_skeleton_prefix: str= f"{idx+1}".zfill(8)
                with open(f"{abs_landmark_under_folder /img_landmark_skeleton_prefix}.npy", 'wb') as f:
                    numpysave(f, lmark_face_pose_lr_hand)
                imwrite(f"{abs_skeleton_under_folder /img_landmark_skeleton_prefix}.jpg", img_skeleton)
                data2write_landmark[-1][KEY_VID][-1]['landmark'].append({
                    'file': f"{img_landmark_skeleton_prefix}.npy",
                    'face': fph.face_landmarks != None,
                    'pose': fph.pose_landmarks != None,
                    'left_hand': fph.left_hand_landmarks != None,
                    'right_hand': fph.right_hand_landmarks != None,
                })
                data2write_skeleton[-1][KEY_VID][-1]['skeleton'].append({
                    'file': f"{img_landmark_skeleton_prefix}.jpg",
                    'face': fph.face_landmarks != None,
                    'pose': fph.pose_landmarks != None,
                    'left_hand': fph.left_hand_landmarks != None,
                    'right_hand': fph.right_hand_landmarks != None,
                })
                Path(an_image_abs).unlink()
    IMAGE_tmp_dir.rmdir()
    with open(f"{HYBRID_DS_DIR /"hybrid_data.investigate.landmark.json"}", 'w') as f:
        writeJson(data2write_landmark, f, indent=4)
    with open(f"{HYBRID_DS_DIR /"hybrid_data.investigate.skeleton.json"}", 'w') as f:
        writeJson(data2write_skeleton, f, indent=4)

if __name__=="__main__":
    main()
    # took about 28 minutes to finish
