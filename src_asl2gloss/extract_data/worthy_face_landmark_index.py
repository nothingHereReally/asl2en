from cv2 import circle, imwrite, line
from numpy import array, float32, load as loadnp, ndarray, zeros


from ..lmark_constant import K_TRAIN, WLASL_LANDMARK_DIR, wlasl_landmark
FACE_CONNECTIONS_FULL: tuple= (
    # oval face, final oval face
    # (10, 10), # top most
    # (332, 332), # left 4th floor
    # (389, 389), # left 3rd floor
    # (323, 323), # left 2nd floor
    # (397, 397), # left 1st floor
    # (103, 103), # right 4th floor
    # (162, 162), # right 3rd floor
    # (93, 93),   # right 2nd floor
    # (172, 172), # right 1st floor
    # (152, 152), # chin lowest part, 0th floor
    # _________________________________________
    (10, 332), (332, 389), (389, 323), (323, 397), (397, 152),
    (10, 103), (103, 162), (162, 93), (93, 172), (172, 152),


    # left eyebrow, final left eyebrow
    # (300, 300), # left most
    # (334, 334), # middle
    # (336, 336), # right most
    # _______________________
    (300, 334), (334, 336),


    # left eye, final left eye
    # (263, 263), # left most
    # (374, 374), # down
    # (386, 386), # up
    # (362, 362), # right most
    # ________________________
    (263, 374), (374, 362),
    (362, 386), (386, 263),


    # right eyebrow, final right eyebrow
    # (70, 70), # right most
    # (105, 105), # middle
    # (107, 107), # left most
    # ______________________
    (70, 105), (105, 107),


    # right eye, final right eye
    # (33, 33), # right most
    # (145, 145), # down
    # (159, 159), # up
    # (133, 133), # left most
    # _______________________
    (33, 145), (145, 133),
    (133, 159), (159, 33),


    # nose, final nose
    # (168, 168), # top most
    # (195, 195), # between( top most, middle vertical )
    # (4, 4), # middle vertical
    # (2, 2), # lowest
    # (294, 294), # left most
    # (64, 64), # right most
    # ____________________
    (168, 195), (195, 4),
    (4, 294), (294, 2),
    (2, 64), (64, 4),


    # lips
    # (61, 61), # right most
    # (17, 17), # down lip down
    # (291, 291), # left most
    # (0, 0), # up lip up
    # (14, 14),   # up/down lip up/down a
    # (13, 13), # up/down lip up/down b
    # _______________________________
    (61, 17), (17, 291), # down lip edge down
    (291, 0), (0, 61), # up lip edge up
    (61, 14), (14, 291), # up/down lip inner a
    (291, 13), (13, 61), # up/down lip inner b
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
def drawSkeletonImg(img_orig: ndarray, \
                    lmark_cords: tuple, \
                    conn_idxs_list: tuple, \
                    thick: int=2, \
                    color_conn: tuple=(255,0,255), \
                    color_lmark: tuple=(255,255,0), \
                    drawJoint: bool=True) -> ndarray:
    def isOKplt(coord: tuple) -> bool:
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
def recalcDrawFace(img_orig: ndarray, lmark_face: tuple) -> ndarray:
    img: ndarray= img_orig.copy()
    return drawSkeletonImg(
        img_orig=img,
        lmark_cords=lmark_face,
        conn_idxs_list=FACE_CONNECTIONS,
        thick=1,
        color_conn=(0,255,0), # 255/255= 1.0
        # color_conn=(255,255,255), # blackNwhite
        drawJoint=False
    )


if __name__=="__main__":
    lm_= None
    lm_instance_json= wlasl_landmark[K_TRAIN][0]
    with open(f"{WLASL_LANDMARK_DIR}{lm_instance_json['video_id']}/{lm_instance_json['landmark'][0]['file']}", 'rb') as f:
        lm_= array(loadnp(f), dtype=float32)
    landmarks_face: list= []
    for idx in WORTHY_FACE_IDX:
        landmarks_face.append(lm_[idx])
    face_image= recalcDrawFace(
        img_orig=zeros((300, 300, 3)),
        lmark_face=tuple(landmarks_face)
    )
    imwrite(filename="/tmp/asdf_face_asdf.png", img=face_image)

