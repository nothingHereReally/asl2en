from typing import Any
from keras.src.saving import load_model
from numpy import argmax, array, float32, load as loadnp
from json import load as loadjson
from math import ceil


from ..lmark_constant import KEY_ID2G, KEY_TEST, KEY_TRAIN, KEY_VAL, PROJ_ROOT, GLASL_LANDMARK_DIR


LANDMARK_SHAPE: tuple= (36 +8 +21*2, 2)
QUANTITY_FRAME: int= 22
TRAIN_BATCH: int= 2



def getGreaterThan_landmark_allHasHand(lmark_: dict) -> list:
    '''
    to be used for when QUANTITY_FRAME < len(lmark_['landmark'])
    output be of shape(__ QUANTITY_FRAME, 86, 2 __)
    '''
    lmark_numpy: list= [] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

    lmark_all: list= []
    idx_init_has_hand: int= -1
    for i in range(len(lmark_['landmark'])):
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][i]['file']}", 'rb') as f:
            lmark_all.append(loadnp(f))
        if idx_init_has_hand==-1:
            if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                idx_init_has_hand= i
    if idx_init_has_hand==-1:
        return []

    # part 1, floor at index level
    o2t_ratio: float= (len(lmark_['landmark'])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        if lmark_['landmark'][idx_init_has_hand+int(i*o2t_ratio)]['left_hand'] or \
            lmark_['landmark'][idx_init_has_hand+int(i*o2t_ratio)]['right_hand']:
            lmark_numpy.append(lmark_all[idx_init_has_hand+int(i*o2t_ratio)]) # floor
        else:
            lmark_numpy.append(lmark_numpy[-1])
    if len(lmark_numpy)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_np on part 1 due to not QUANTITY_FRAME")
    del o2t_ratio

    return lmark_numpy


def getLessThanOrEqual_landmark_allHasHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) <= QUANTITY_FRAME
    '''
    def getIdxStartHand(lmarks: list) -> int:
        for i in range(len(lmarks)):
            if lmarks[i]['left_hand'] or lmarks[i]['right_hand']:
                return i
        return -1
    idx_init_has_hand: int= getIdxStartHand(lmarks=lmark_['landmark'])
    if idx_init_has_hand==-1:
        return []
    lmark_numpy: list= []
    t2o_ratio: int= int(ceil(QUANTITY_FRAME/(len(lmark_['landmark'])-idx_init_has_hand)))
    for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, len(lmark_['landmark'])), range(len(lmark_['landmark'])-idx_init_has_hand)):
        landmark_data_numpy= None
        with open(f"{GLASL_LANDMARK_DIR}{lmark_['video_id']}/{lmark_['landmark'][  i  ]['file']}", 'rb') as f:
            landmark_data_numpy= loadnp(f)
        for ii in range(t2o_ratio):
            if (i_0to_t2o_multiplier*t2o_ratio+ii)<QUANTITY_FRAME:
                if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                    lmark_numpy.append( landmark_data_numpy )
                else:
                    lmark_numpy.append( lmark_numpy[-1] )
    if len(lmark_numpy)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getLessThan_np, due to len(lmark_numpy)!=QUANTITY_FRAME")
    return lmark_numpy








if __name__=="__main__":
    glasl_landmark: dict= {}
    with open(f"{PROJ_ROOT}dataset/glasl/glasl.annotation.landmark.json", 'r') as f:
        glasl_landmark= loadjson(f)
    TRAIN_GLOSS: int= 10
    model: Any= load_model(f"{PROJ_ROOT}model/aslvid2gloss_v25.keras")


    batch: int= 4
    details: dict= {
        KEY_TRAIN: {
            'correct':   [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':  [0 for _ in range(TRAIN_GLOSS)],
            'total_vid': [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_VAL: {
            'correct':   [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':  [0 for _ in range(TRAIN_GLOSS)],
            'total_vid': [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_TEST: {
            'correct':   [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':  [0 for _ in range(TRAIN_GLOSS)],
            'total_vid': [0 for _ in range(TRAIN_GLOSS)]
        },
    }
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        batch_vid_lm: list= []
        batch_gloss: list= []
        i: int= 0
        while i<len(glasl_landmark[tvt_idv]):
            if len(batch_vid_lm)<batch:
                tmp= []
                if len(glasl_landmark[tvt_idv][i]['landmark'])<=QUANTITY_FRAME:
                    batch_vid_lm.append(array(getLessThanOrEqual_landmark_allHasHand(glasl_landmark[tvt_idv][i]), dtype=float32))
                else:
                    batch_vid_lm.append(array(getGreaterThan_landmark_allHasHand(glasl_landmark[tvt_idv][i]), dtype=float32))
                batch_gloss.append(int(glasl_landmark[tvt_idv][i]['gloss_id']))
                i+= 1
            else:
                y_pred= model.predict(
                    x=array(batch_vid_lm, dtype=float32),
                    batch_size=len(batch_vid_lm)
                )
                for y_out, y_shouldbe in zip(y_pred, batch_gloss):
                    y_out_g_id= int(argmax(y_out, axis=-1))
                    if y_out_g_id==y_shouldbe:
                        details[tvt_idv]['correct'][y_shouldbe]+= 1
                        if details[tvt_idv]['accuracy'][y_shouldbe]==0:
                            details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                        else:
                            details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                            details[tvt_idv]['accuracy'][y_shouldbe]/= 2.0
                    details[tvt_idv]['total_vid'][y_shouldbe]+= 1
                batch_vid_lm= []
                batch_gloss= []
        if len(batch_vid_lm)>0:
            y_pred= model.predict(
                x=array(batch_vid_lm, dtype=float32),
                batch_size=len(batch_vid_lm)
            )
            for y_out, y_shouldbe in zip(y_pred, batch_gloss):
                y_out_g_id= int(argmax(y_out, axis=-1))
                if y_out_g_id==y_shouldbe:
                    details[tvt_idv]['correct'][y_shouldbe]+= 1
                    if details[tvt_idv]['accuracy'][y_shouldbe]==0:
                        details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                    else:
                        details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                        details[tvt_idv]['accuracy'][y_shouldbe]/= 2.0
                details[tvt_idv]['total_vid'][y_shouldbe]+= 1
            batch_vid_lm= []
            batch_gloss= []
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"__________________________ {tvt_idv} ____")
        q_correct: int= 0
        q_vid: int= 0
        for g_id in range(10):
            q_correct+= details[tvt_idv]['correct'][g_id]
            q_vid+= details[tvt_idv]['total_vid'][g_id]
        #     print(f"{g_id}: {glasl_landmark[KEY_ID2G][g_id]} --> {details[tvt_idv]['correct'][g_id]}/{details[tvt_idv]['total_vid'][g_id]}")
        #     print(f"precentage correct: {details[tvt_idv]['correct'][g_id] / details[tvt_idv]['total_vid'][g_id] *100}%")
        #     print(f"accuracy: {details[tvt_idv]['accuracy'][g_id] *100}%")
        #     print()
        # print("<< ----------------------------------------------------------------- >>")
        print(f"____ {tvt_idv} --> {q_correct}/{q_vid} --> {q_correct/q_vid*100}%")
        # print("<< ----------------------------------------------------------------- >>")
