from concurrent.futures import ProcessPoolExecutor
import cv2
from json import load as loadJson
from keras.models import load_model as loadModelKerasFile
from mediapipe.python.solutions.holistic import Holistic
import numpy
from pathlib import Path
import pygame
# from time import sleep
from typing import Any




# 22 images
# get_landmarks_on_an_image
#   each image 86 landmarks
#     21 each hand landmarks
#     8 for worthy pose landmarks
#     36 for worthy face landmarks
#     -- order
#       1 - face
#       2 - pose
#       3 - left hand
#       4 - right hand
VIDEO_IN: cv2.VideoCapture|None= None
SCREEN: pygame.Surface|None= None
WINDOW_HIGHT: int|None= None
WINDOW_WIDTH: int|None= None

MPH_fph: Holistic|None= None
ASL2EN: Any|None= None
ASL_GLOSS: tuple[str]|None= None

FACE_CONNECTIONS: tuple|None= None
POSE_CONNECTIONS: tuple|None= None
HAND_CONNECTIONS: tuple|None= None
WORTHY_FACE_IDX: tuple|None= None
WORTHY_POSE_IDX: tuple|None= None
WORTHY_HANDS_QUANTITY: int= 21

def configs() -> None:
    global VIDEO_IN, SCREEN, MPH_fph
    global FACE_CONNECTIONS, POSE_CONNECTIONS, HAND_CONNECTIONS
    global WORTHY_FACE_IDX, WORTHY_POSE_IDX
    global WINDOW_HIGHT, WINDOW_WIDTH
    global ASL2EN, ASL_GLOSS


    projectDirectory: Path= Path(__file__).parent.parent.parent
    ASL2EN= loadModelKerasFile(Path(projectDirectory/"model"/"aslvid2gloss_v41.keras"))
    with open(Path(projectDirectory/"dataset"/"glasl"/"glasl.annotation.landmark.json"), 'r') as f:
        assert ASL2EN is not None
        ASL2EN.predict(
            x=numpy.zeros((1, *ASL2EN.input_shape[1:])),
            batch_size=1
        )
        tmpdata= loadJson(f)
        ASL_GLOSS= tuple(tmpdata['id2gloss'])
        if len(ASL_GLOSS)!=ASL2EN.output_shape[-1]:
            raise NotImplementedError('Incorrect implementation, len(ASL_GLOSS) should be equals to ASL2EN.output_shape[-1]')


    VIDEO_IN= cv2.VideoCapture(0) #, cv2.CAP_V4L2)
    # VIDEO_IN.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Force MJPG format
    ok, frame= VIDEO_IN.read()
    if not ok:
        raise RuntimeError("can't access camera")
    pygame.init()


    # _, desktop_h= pygame.display.get_desktop_sizes()[0]
    # WINDOW_HIGHT: int= int(math.ceil(0.9*desktop_h))
    # # WINDOW_WIDTH: int= WINDOW_HIGHT/camera_img_height * camera_img_width
    # WINDOW_WIDTH: int= int(math.ceil((WINDOW_HIGHT/frame.shape[0]) *frame.shape[1]))


    WINDOW_HIGHT= int(frame.shape[0])
    WINDOW_WIDTH= int(frame.shape[1])
    SCREEN= pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HIGHT))
    MPH_fph= Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


    WORTHY_FACE_IDX= (
        0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
        105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
        263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
        386, 389, 397
    )
    FACE_CONNECTIONS= (
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

    # before use of POSE_CONNECTIONS modify landmark 1st
    # modify to use index to be used only: 11,12,13,14,15,16,23,24
    # so new index: 0,1,2,3,4,5,6,7
    WORTHY_POSE_IDX= (11,12,13,14,15,16,23,24)
    POSE_CONNECTIONS= ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6), (1, 7))

    HAND_CONNECTIONS= (
        (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), # palm connections
        (1, 2), (2, 3), (3, 4),           # thumb finger connections
        (5, 6), (6, 7), (7, 8),           # index finger connections
        (9, 10), (10, 11), (11, 12),      # middle finger connections
        (13, 14), (14, 15), (15, 16),     # ring finger connections
        (17, 18), (18, 19), (19, 20)      # pinky finger connections
    )
def wantExit() -> bool:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            return True
    pygame.event.clear()
    return False
def cleanGlobal() -> None:
    global VIDEO_IN
    if VIDEO_IN is not None:
        VIDEO_IN.release()
    pygame.quit()
    print('Resources cleaned up safely')
def updateImgDisplay(frame: numpy.ndarray) -> None:
    global SCREEN
    assert SCREEN is not None
    surf= pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    SCREEN.blit(surf, (0, 0))
    pygame.display.flip()
def extractWorthyLandmarks(lmark_facePoseLeftRightHand) -> tuple:
    assert WORTHY_FACE_IDX is not None and isinstance(WORTHY_FACE_IDX, tuple)
    assert WORTHY_POSE_IDX is not None and isinstance(WORTHY_POSE_IDX, tuple)
    lmarkExist: dict= {
        'face': False,
        'pose': False,
        'left_hand': False,
        'right_hand': False
    }
    outLandmarksWorthy: list= []
    if lmark_facePoseLeftRightHand.face_landmarks==None \
        and lmark_facePoseLeftRightHand.pose_landmarks==None \
        and lmark_facePoseLeftRightHand.left_hand_landmarks==None \
        and lmark_facePoseLeftRightHand.right_hand_landmarks==None:
        return (
            numpy.zeros((86, 2), dtype=numpy.float32),
            lmarkExist
        )
    lmarkExist['face']=       lmark_facePoseLeftRightHand.face_landmarks is not None
    lmarkExist['pose']=       lmark_facePoseLeftRightHand.pose_landmarks is not None
    lmarkExist['left_hand']=  lmark_facePoseLeftRightHand.left_hand_landmarks is not None
    lmarkExist['right_hand']= lmark_facePoseLeftRightHand.right_hand_landmarks is not None
    if lmarkExist['face']:
        for worthyIdx in WORTHY_FACE_IDX:
            outLandmarksWorthy.append((
                lmark_facePoseLeftRightHand.face_landmarks.landmark[worthyIdx].x,
                lmark_facePoseLeftRightHand.face_landmarks.landmark[worthyIdx].y
            ))
    else:
        outLandmarksWorthy.extend(numpy.zeros((len(WORTHY_FACE_IDX), 2)).tolist())

    if lmarkExist['pose']:
        for worthyIdx in WORTHY_POSE_IDX:
            outLandmarksWorthy.append((
                lmark_facePoseLeftRightHand.pose_landmarks.landmark[worthyIdx].x,
                lmark_facePoseLeftRightHand.pose_landmarks.landmark[worthyIdx].y
            ))
    else:
        outLandmarksWorthy.extend(numpy.zeros((len(WORTHY_POSE_IDX), 2)).tolist())

    if lmarkExist['left_hand']:
        for value in lmark_facePoseLeftRightHand.left_hand_landmarks.landmark:
            outLandmarksWorthy.append((
                value.x,
                value.y
            ))
    else:
        outLandmarksWorthy.extend(numpy.zeros((WORTHY_HANDS_QUANTITY, 2)).tolist())

    if lmarkExist['right_hand']:
        for value in lmark_facePoseLeftRightHand.right_hand_landmarks.landmark:
            outLandmarksWorthy.append((
                value.x,
                value.y
            ))
    else:
        outLandmarksWorthy.extend(numpy.zeros((WORTHY_HANDS_QUANTITY, 2)).tolist())


    return (
        [tuple(el) for el in outLandmarksWorthy],
        lmarkExist
    )
def isOKplt(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return 0.0<=coord[0] and coord[0]<=1.0 and \
           0.0<=coord[1] and coord[1]<=1.0
def drawSkeletonOnImage(imageSource: numpy.ndarray, \
                    landmarkCoordinates: tuple, \
                    connections_list_idxs: tuple, \
                    thickness: int=2, \
                    color_lines: tuple=(255,0,255), \
                    color_dots: tuple=(255,255,0), \
                    drawJoint: bool=True) -> numpy.ndarray:
    img_wh: dict= {
        'height_y': imageSource.shape[0],
        'width_x':  imageSource.shape[1]
    }
    # drawing the lines between 2 landmark connections
    for lmark_idx in connections_list_idxs:
        pA: tuple= (
            landmarkCoordinates[  lmark_idx[0]  ][0], # x
            landmarkCoordinates[  lmark_idx[0]  ][1]  # y
        )
        pB: tuple= (
            landmarkCoordinates[  lmark_idx[1]  ][0], # x
            landmarkCoordinates[  lmark_idx[1]  ][1]  # y
        )
        okPlotPointA: bool= isOKplt(pA)
        okPlotPointB: bool= isOKplt(pB)
        if okPlotPointA and okPlotPointB:
            cv2.line(
                img=imageSource,
                pt1=(int(pA[0]*img_wh['width_x']), int(pA[1]*img_wh['height_y'])),
                pt2=(int(pB[0]*img_wh['width_x']), int(pB[1]*img_wh['height_y'])),
                color=color_lines,
                thickness=thickness
            )
        if drawJoint:
            if okPlotPointA:
                cv2.circle(
                    img=imageSource,
                    center=(int(pA[0]*img_wh['width_x']), int(pA[1]*img_wh['height_y'])),
                    radius=0,
                    color=color_dots,
                    thickness=thickness*2
                )
            if okPlotPointB:
                cv2.circle(
                    img=imageSource,
                    center=(int(pB[0]*img_wh['width_x']), int(pB[1]*img_wh['height_y'])),
                    radius=0,
                    color=color_dots,
                    thickness=thickness*2
                )
    return imageSource
def drawLandmarksAndLines(dataIn: tuple) -> numpy.ndarray:
    imageSource: numpy.ndarray= dataIn[0]
    lmark_facePoseLeftRightHand: list= dataIn[1]
    lmarkExist: dict= dataIn[2]
    assert WORTHY_FACE_IDX is not None and isinstance(WORTHY_FACE_IDX, tuple)
    assert FACE_CONNECTIONS is not None and isinstance(FACE_CONNECTIONS, tuple)
    assert WORTHY_POSE_IDX is not None and isinstance(WORTHY_POSE_IDX, tuple)
    assert POSE_CONNECTIONS is not None and isinstance(POSE_CONNECTIONS, tuple)
    assert HAND_CONNECTIONS is not None and isinstance(HAND_CONNECTIONS, tuple)
    # def drawSkeletonOnImage(imageSource: numpy.ndarray, \
    #                     landmarkCoordinates: tuple, \
    #                     connections_list_idxs: tuple, \
    #                     thickness: int=2, \
    #                     color_lines: tuple=(255,0,255), \
    #                     color_dots: tuple=(255,255,0), \
    #                     drawJoint: bool=True) -> numpy.ndarray:
    idxStartPose: int= len(WORTHY_FACE_IDX)
    idxStartLeftHand: int= idxStartPose      +len(WORTHY_POSE_IDX)
    idxStartRightHand: int= idxStartLeftHand +WORTHY_HANDS_QUANTITY
    if lmarkExist['face']:
        imageSource= drawSkeletonOnImage(
            imageSource=imageSource,
            landmarkCoordinates=tuple(lmark_facePoseLeftRightHand[:idxStartPose]),
            connections_list_idxs=FACE_CONNECTIONS,
            thickness=5,
            color_lines=(0, 153, 0),
            color_dots=(0, 153, 0),
            drawJoint=True
        )
    if lmarkExist['pose']:
        imageSource= drawSkeletonOnImage(
            imageSource=imageSource,
            landmarkCoordinates=tuple(lmark_facePoseLeftRightHand[
                idxStartPose: idxStartLeftHand
            ]),
            connections_list_idxs=POSE_CONNECTIONS,
            thickness=5,
            color_lines=(0, 0, 153),
            color_dots=(0, 0, 153),
            drawJoint=True
        )
    if lmarkExist['left_hand']:
        imageSource= drawSkeletonOnImage(
            imageSource=imageSource,
            landmarkCoordinates=tuple(lmark_facePoseLeftRightHand[
                idxStartLeftHand: idxStartRightHand
            ]),
            connections_list_idxs=HAND_CONNECTIONS,
            thickness=5,
            color_lines=(255, 255, 255),
            color_dots=(255, 255, 255),
            drawJoint=True
        )
    if lmarkExist['right_hand']:
        imageSource= drawSkeletonOnImage(
            imageSource=imageSource,
            landmarkCoordinates=tuple(lmark_facePoseLeftRightHand[
                idxStartRightHand:
            ]),
            connections_list_idxs=HAND_CONNECTIONS,
            thickness=5,
            color_lines=(204, 204, 14),
            color_dots=(204, 204, 14),
            drawJoint=True
        )


    return imageSource
def part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not landmarks:
        return landmarks

    xs, ys= zip(*landmarks)
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
def part2_beSquareRatioOnImage(landmarks: list, original_shape: tuple) -> list:
    height, width= original_shape
    if height==width or not landmarks:
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
def part3_zoomInOutForPadding(landmarks: list) -> list:
    if not landmarks:
        return landmarks

    ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    # zoom in/out for padding be 10% each side with respect to original aspect ratio
    # ie.:
    # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
    # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
    # pad: float= 0.05
    pad: float= 4.0/158.0
    xs, ys = zip(*landmarks)
    min_x, min_y=    min(xs), min(ys)
    max_x, max_y=    max(xs), max(ys)
    scale: float= (1  -2*pad)/max(
        max_x -min_x,
        max_y -min_y
    )
    return [(
        (x -min_x)    *scale    +pad,
        (y -min_y)    *scale    +pad
    ) for x, y in landmarks]
def part4_centerLandmarkVerticallyHorizontally(landmarks: list) -> list:
    if not landmarks:
        return landmarks

    ### 3) center landmark with same aspect ratio as original
    # center horizontally and vertically, since done padding then just
    # move to right/down
    xs, ys = zip(*landmarks)
    shift_x: float=  0.5    -(min(xs) +max(xs))  /2
    shift_y: float=  0.5    -(min(ys) +max(ys))  /2

    return [(
        x +shift_x,
        y +shift_y
    ) for x, y in landmarks]
def normalizeLandmarks(dataIn: tuple) -> list:
    landmarks: list= dataIn[0]
    original_shape= dataIn[1]
    landmarks= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks)
    landmarks= part2_beSquareRatioOnImage(
        landmarks,
        (original_shape[0], original_shape[1])
    )
    landmarks= part3_zoomInOutForPadding(landmarks)
    landmarks= part4_centerLandmarkVerticallyHorizontally(landmarks)


    return landmarks
def doRecognitionAslAlgorithm(tmpLandmarks: dict, anImageLandmarks: list|None) -> dict:
    assert ASL2EN is not None
    assert ASL_GLOSS is not None and isinstance(ASL_GLOSS, tuple)
    GAP_BE_HOW_MANY_THEN_APPEND: int= 4
    MAX_LAST_HAS_HAND: int= 9
    assert MAX_LAST_HAS_HAND < ASL2EN.input_shape[1]
    # tmpLandmarks: dict= {
    #     'lastAnImageLandmarks': [],
    #     'landmarksPredictLater': [],
    #     'lastHasHand': 0,
    #     'skippedBy': 0
    # }
    if anImageLandmarks is not None:
        tmpLandmarks['lastAnImageLandmarks']= anImageLandmarks
    if MAX_LAST_HAS_HAND < tmpLandmarks['lastHasHand'] and \
        0 < len(tmpLandmarks['landmarksPredictLater']):
        # ------------------------------------------------------------
        tmpLandmarks['landmarksPredictLater']= []
        tmpLandmarks['lastAnImageLandmarks']= []
        tmpLandmarks['skippedBy']= 0
    elif tmpLandmarks['skippedBy']%GAP_BE_HOW_MANY_THEN_APPEND==0:
        if anImageLandmarks is not None:
            tmpLandmarks['landmarksPredictLater'].append(anImageLandmarks)
            tmpLandmarks['lastAnImageLandmarks']= []
            tmpLandmarks['skippedBy']= 0
        elif anImageLandmarks is None and \
            0 < len(tmpLandmarks['lastAnImageLandmarks']):
            # ------------------------------------------------------------
            tmpLandmarks['landmarksPredictLater'].append(tmpLandmarks['lastAnImageLandmarks'])
            tmpLandmarks['lastAnImageLandmarks']= []
            tmpLandmarks['skippedBy']= 0
        # ------------------------------------------------------------
        if len(tmpLandmarks['landmarksPredictLater'])==ASL2EN.input_shape[1]:
            # DONE: do prediction asl2en
            manyImages2predictAsl2en: numpy.ndarray= numpy.array(
                tmpLandmarks['landmarksPredictLater'],
                dtype=numpy.float32
            )
            assert tuple(manyImages2predictAsl2en.shape)==tuple(ASL2EN.input_shape[1:])
            predictedAsl= ASL2EN.predict(
                x=manyImages2predictAsl2en.reshape((1, *manyImages2predictAsl2en.shape)),
                batch_size=1
            )[0]
            asl2enIdx: int= numpy.argmax(predictedAsl, axis=-1)
            print(f"accuracy: {predictedAsl[asl2enIdx]*100.0 : .3f}%")
            print(f"idx --> {asl2enIdx}")
            print(f"gloss: --> {ASL_GLOSS[asl2enIdx]}")
            tmpLandmarks['landmarksPredictLater']= tmpLandmarks['landmarksPredictLater'][MAX_LAST_HAS_HAND:]


    return tmpLandmarks




def main() -> None:
    configs()
    assert VIDEO_IN is not None and isinstance(VIDEO_IN, cv2.VideoCapture)
    assert SCREEN is not None and isinstance(SCREEN, pygame.Surface)
    assert MPH_fph is not None and isinstance(MPH_fph, Holistic)
    assert ASL2EN is not None
    assert ASL_GLOSS is not None and isinstance(ASL_GLOSS, tuple)


    tmpLandmarks: dict= {
        'lastAnImageLandmarks': [],
        'landmarksPredictLater': [],
        'lastHasHand': 0,
        'skippedBy': 0
    }
    with ProcessPoolExecutor() as exec:
        try:
            while not wantExit():
                ok, an_image= VIDEO_IN.read()
                # sleep(0.2)
                if not ok:
                    break
                an_image= cv2.cvtColor(an_image, cv2.COLOR_BGR2RGB)
                lmark_facePoseLeftRightHand= MPH_fph.process(an_image)
                landmarks_worthy, lmarkExist= extractWorthyLandmarks(lmark_facePoseLeftRightHand)
                results: list|None= None
                if lmarkExist['left_hand'] or lmarkExist['right_hand']:
                    futures= [
                        exec.submit(
                            drawLandmarksAndLines,
                            (
                                an_image,
                                landmarks_worthy,
                                lmarkExist
                            )
                        ),
                        exec.submit(
                            normalizeLandmarks,
                            (
                                landmarks_worthy,
                                an_image.shape
                            )
                        )
                    ]
                    results= [f.result() for f in futures]
                tmpLandmarks['skippedBy']+= 1
                tmpLandmarks= doRecognitionAslAlgorithm(
                    tmpLandmarks=tmpLandmarks,
                    anImageLandmarks=None if results is None else results[1]
                )
                if results is not None:
                    an_image= results[0]
                    tmpLandmarks['lastHasHand']= 0
                else:
                    tmpLandmarks['lastHasHand']+= 1
                updateImgDisplay(an_image)
        finally:
            # due to if exception occurs still be able to do clean up
            cleanGlobal()
if __name__=="__main__":
    main()
