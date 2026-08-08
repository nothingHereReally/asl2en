from typing import Any
from keras.src.saving import load_model
from numpy import argmax, array, float32, load as loadnp, ndarray
from json import load as loadjson
from math import ceil


from ..lmark_constant import KEY_FILE, KEY_GLOSS, KEY_ID2G, KEY_LHAND, KEY_LMARK, KEY_RHAND, KEY_TEST, KEY_TRAIN, KEY_VAL, KEY_VIDEO, PROJ_ROOT, GLASL_LANDMARK_DIR


LANDMARK_SHAPE: tuple= (36 +8 +21*2, 2)
model: Any= load_model(f"{PROJ_ROOT /"model" /"aslvid2gloss_v24.keras"}")
TRAIN_GLOSS: int= model.output_shape[-1]   # 10 categories
QUANTITY_FRAME: int= model.input_shape[-3] # 22 categories
TRAIN_BATCH: int= 2


def get_idx_start_hand(annotated_images: list) -> int:
    for idx in range(len(annotated_images)):
        if annotated_images[idx][KEY_LHAND] or annotated_images[idx][KEY_RHAND]:
            return idx
    return -1


def get_landmark4less_or_equal(a_raw_video: dict, idx_init_has_hand: int|None=None) -> list:
    '''
    output be of shape(____ QUANTITY_FRAME, 86, 2 ____)
    '''
    if idx_init_has_hand is None:
        idx_init_has_hand= get_idx_start_hand(a_raw_video[KEY_LMARK])
    if idx_init_has_hand==-1:
        return list()

    lmark_numpy_out: list= []
    ratio_what: int= int(ceil(
        QUANTITY_FRAME  /  (len(a_raw_video[KEY_LMARK])-idx_init_has_hand)
    ))
    for idx in range(idx_init_has_hand, len(a_raw_video[KEY_LMARK])):
        with open(f"{GLASL_LANDMARK_DIR /a_raw_video[KEY_VIDEO] /a_raw_video[KEY_LMARK][idx][KEY_FILE]}", 'rb') as f:
            load_an_image_landmarks= loadnp(f)
        for _ in range(min(ratio_what, QUANTITY_FRAME-len(lmark_numpy_out))):
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                lmark_numpy_out.append(load_an_image_landmarks)
            else:
                lmark_numpy_out.append(lmark_numpy_out[-1])
    check_shape: ndarray= array(lmark_numpy_out, dtype=float32)
    if check_shape.shape!=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]):
        raise NotImplementedError(
            f"incorrect implementation on get_landmark4less_or_equal(), due to lmark_numpy_out should be of shape {
            tuple((QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]))
            }, but got {check_shape.shape}"
        )
    return lmark_numpy_out


def get_landmark4greater(a_raw_video: dict) -> list:
    '''
    output be of shape(____ QUANTITY_FRAME, 86, 2 ____)
    '''
    lmark_load_ALL: list= []
    idx_init_has_hand: int= -1
    for idx in range(len(a_raw_video[KEY_LMARK])):
        with open(f"{GLASL_LANDMARK_DIR /a_raw_video[KEY_VIDEO] /a_raw_video[KEY_LMARK][idx][KEY_FILE]}", 'rb') as f:
            lmark_load_ALL.append(loadnp(f))
        if idx_init_has_hand==-1:
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                idx_init_has_hand= idx
    if idx_init_has_hand==-1:
        raise ValueError(f"video {a_raw_video[KEY_VIDEO]}.mp4 has no hands")
    len_available_images: int= len(a_raw_video[KEY_LMARK])-idx_init_has_hand

    if QUANTITY_FRAME<len_available_images:
        init_hand_idxs: tuple= tuple(range(idx_init_has_hand, len(a_raw_video[KEY_LMARK])))
        past_img_has_hand: ndarray= lmark_load_ALL[idx_init_has_hand]
        tmp_part4: list= []
        for check_mod_init_0, idx in zip(range(len_available_images), init_hand_idxs):
            if check_mod_init_0%5 == 4:
                if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or \
                    a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                    tmp_part4.append(lmark_load_ALL[idx])
                else:
                    tmp_part4.append(past_img_has_hand)
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or \
                a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                past_img_has_hand= lmark_load_ALL[idx]
        if QUANTITY_FRAME<len(tmp_part4):
            start_where: int= int((len(tmp_part4)-QUANTITY_FRAME+1)//2)
            return tmp_part4[start_where:start_where+QUANTITY_FRAME]
        else:
            tmp_for_less_or_qual: list= []
            ratio4mod_part4: int= int(ceil(QUANTITY_FRAME/len(tmp_part4)))
            for an_image_landmarks in tmp_part4:
                tmp_for_less_or_qual.extend(list([an_image_landmarks]) *ratio4mod_part4)
            tmp_for_less_or_qual= tmp_for_less_or_qual[:QUANTITY_FRAME]
            return tmp_for_less_or_qual
    # len_available_images <= QUANTITY_FRAME
    copy_a_raw_video: dict= {
        KEY_GLOSS: a_raw_video[KEY_GLOSS],
        KEY_VIDEO: a_raw_video[KEY_VIDEO],
        KEY_LMARK: a_raw_video[KEY_LMARK][idx_init_has_hand:]
    }
    return get_landmark4less_or_equal(copy_a_raw_video, 0)








if __name__=="__main__":
    glasl_landmark: dict= {}
    with open(f"{PROJ_ROOT /"dataset" /"glasl" /"glasl.annotation.landmark.45videos.json"}", 'r') as f:
        glasl_landmark= loadjson(f)


    batch: int= 8
    details: dict= {
        KEY_TRAIN: {
            'correct':          [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
            'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
            'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_VAL: {
            'correct':          [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
            'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
            'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
        },
        KEY_TEST: {
            'correct':          [0 for _ in range(TRAIN_GLOSS)],
            'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
            'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
            'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
        },
        "summary": {
            KEY_TRAIN: {
                'correct':          [0 for _ in range(TRAIN_GLOSS)],
                'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
                'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
                'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
            },
            KEY_VAL: {
                'correct':          [0 for _ in range(TRAIN_GLOSS)],
                'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
                'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
                'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
            },
            KEY_TEST: {
                'correct':          [0 for _ in range(TRAIN_GLOSS)],
                'accuracy':         [0 for _ in range(TRAIN_GLOSS)],
                'overall_accuracy': [0 for _ in range(TRAIN_GLOSS)],
                'total_vid':        [0 for _ in range(TRAIN_GLOSS)]
            },
        }
    }
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        batch_vid_lm: list= []
        batch_gloss: list= []
        i: int= 0
        while i<len(glasl_landmark[tvt_idv]) and glasl_landmark[tvt_idv][i]['gloss_id']<TRAIN_GLOSS:
            if len(batch_vid_lm)<batch:
                tmp= []
                if len(glasl_landmark[tvt_idv][i]['landmark'])<=QUANTITY_FRAME:
                    batch_vid_lm.append(array(get_landmark4less_or_equal(glasl_landmark[tvt_idv][i]), dtype=float32))
                else:
                    batch_vid_lm.append(array(get_landmark4greater(glasl_landmark[tvt_idv][i]), dtype=float32))
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
                        details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                    details[tvt_idv]['overall_accuracy'][y_shouldbe]+= y_out[y_out_g_id]
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
                    details[tvt_idv]['accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                details[tvt_idv]['overall_accuracy'][y_shouldbe]+= y_out[y_out_g_id]
                details[tvt_idv]['total_vid'][y_shouldbe]+= 1
            batch_vid_lm= []
            batch_gloss= []
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"__________________________ {tvt_idv} ____")
        q_correct: int= 0
        q_vid: int= 0
        overall_accuracy: float= 0.0
        quantity_videos: int= 0
        for g_id in range(TRAIN_GLOSS):
            q_correct+= details[tvt_idv]['correct'][g_id]
            q_vid+= details[tvt_idv]['total_vid'][g_id]
            print(f"{g_id}: {glasl_landmark[KEY_ID2G][g_id]} --> {details[tvt_idv]['correct'][g_id]}/{details[tvt_idv]['total_vid'][g_id]}")
            print(f"precentage correct: {details[tvt_idv]['correct'][g_id] / details[tvt_idv]['total_vid'][g_id] *100}%")
            print(f"accuracy: {details[tvt_idv]['accuracy'][g_id] / (details[tvt_idv]['correct'][g_id] if details[tvt_idv]['correct'][g_id]>0 else 1) *100}%")
            print(f"overall accuracy: {details[tvt_idv]['overall_accuracy'][g_id] / details[tvt_idv]['total_vid'][g_id] *100}%")
            overall_accuracy+= details[tvt_idv]['overall_accuracy'][g_id]
            quantity_videos+= details[tvt_idv]['total_vid'][g_id]
            print()
        print("<< ----------------------------------------------------------------- >>")
        print(f"____ {tvt_idv} --> {q_correct}/{q_vid} --> {q_correct/q_vid*100}%")
        print(f"____ overall accuracy {tvt_idv} --> {overall_accuracy/quantity_videos*100}%")
        print("<< ----------------------------------------------------------------- >>")
    print("\n\nsummary")
    summary_details: dict= {
        'correct': 0,
        'total_vid': 0,
        'accuracy': 0,
        'overall_accuracy': 0
    }
    for tvt_idv in (KEY_TRAIN, KEY_VAL, KEY_TEST):
        print(f"-----------------------------------------------------------------")
        print(f"____ {tvt_idv} --> {  sum(details[tvt_idv]['correct'])  }/{  sum(details[tvt_idv]['total_vid'])  } --> {sum(details[tvt_idv]['correct'])  /  sum(details[tvt_idv]['total_vid'])*100}%")
        print(f"____ overall accuracy {tvt_idv} --> {sum(details[tvt_idv]['overall_accuracy'])/sum(details[tvt_idv]['total_vid'])*100}%")

        summary_details['correct']+= sum(details[tvt_idv]['correct'])
        summary_details['total_vid']+= sum(details[tvt_idv]['total_vid'])
        summary_details['accuracy']+= sum(details[tvt_idv]['correct'])  /  sum(details[tvt_idv]['total_vid'])
        summary_details['overall_accuracy']+= sum(details[tvt_idv]['overall_accuracy'])/sum(details[tvt_idv]['total_vid'])
    print(f"-----------------------------------------------------------------")
    print(f"____ percentage correct --> {summary_details['correct']} / {summary_details['total_vid']} --> {summary_details['correct'] /summary_details['total_vid']  *100}%")
    print(f"____ hardmax accuracy --> {summary_details['accuracy']  /3  *100}%")
    print(f"____ softmax accuracy --> {summary_details['overall_accuracy']  /3  *100}%")
