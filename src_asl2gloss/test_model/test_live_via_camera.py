import cv2
from mediapipe.python.solutions.holistic import Holistic
import numpy
import pygame




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
MPH_fph: Holistic|None= None
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

    VIDEO_IN= cv2.VideoCapture(0) #, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Force MJPG format
    ok, frame= VIDEO_IN.read()
    if not ok:
        raise RuntimeError("can't access camera")
    pygame.init()


    # _, desktop_h= pygame.display.get_desktop_sizes()[0]
    # WINDOW_HIGHT: int= int(math.ceil(0.9*desktop_h))
    # # WINDOW_WIDTH: int= WINDOW_HIGHT/camera_img_height * camera_img_width
    # WINDOW_WIDTH: int= int(math.ceil((WINDOW_HIGHT/frame.shape[0]) *frame.shape[1]))


    WINDOW_HIGHT: int= frame.shape[0]
    WINDOW_WIDTH: int= frame.shape[1]
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
        numpy.array(outLandmarksWorthy, dtype=numpy.float32),
        lmarkExist
    )






def main() -> None:
    configs()
    assert VIDEO_IN is not None and isinstance(VIDEO_IN, cv2.VideoCapture)
    assert SCREEN is not None and isinstance(SCREEN, pygame.Surface)
    assert MPH_fph is not None and isinstance(MPH_fph, Holistic)
    while not wantExit():
        ok, frame= VIDEO_IN.read()
        if not ok:
            break
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lmark_facePoseLeftRightHand= MPH_fph.process(frame)
        lmark_facePoseLeftRightHand, lmarkExist= extractWorthyLandmarks(lmark_facePoseLeftRightHand)
        print(f"face: --> {lmarkExist['face']}")
        print(f"pose: --> {lmarkExist['pose']}")
        print(f"left_hand: --> {lmarkExist['left_hand']}")
        print(f"right_hand: --> {lmarkExist['right_hand']}")
        print(f"shape landmark: {lmark_facePoseLeftRightHand.shape}")
        print(f"val[37]: --> {lmark_facePoseLeftRightHand[37]}\n\n")
        updateImgDisplay(frame)
    # due to if exception occurs still be able to do clean up
    cleanGlobal()
if __name__=="__main__":
    main()
