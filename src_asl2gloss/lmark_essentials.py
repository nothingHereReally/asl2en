from random import shuffle
from typing import Generator
from numpy import float32, ndarray, uint16, zeros, load as loadnp
from math import ceil
from os.path import exists

from .lmark_constant import (
    KEY_TRAIN,
    LANDMARK_SHAPE,
    LEN_TRAIN,
    LEN_VAL,
    ON_TRAINING_BATCH,
    QUANTITY_FRAME,

    GLASL_LANDMARK_DIR,

    glasl_landmark,
)








def getGreaterThan_landmark_allHasHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 86, 2 ____)
    '''
    lmark_numpy_MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

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

    # part 1, floor at index level, still idx_init_has_hand
    o2t_ratio: float= (len(lmark_['landmark'])-idx_init_has_hand)/QUANTITY_FRAME
    for i in range(QUANTITY_FRAME):
        append_if_valid: int= idx_init_has_hand+int(i*o2t_ratio)
        if lmark_['landmark'][append_if_valid]['left_hand'] or \
            lmark_['landmark'][append_if_valid]['right_hand']:
            lmark_numpy_MANY_VIDS[0].append(lmark_all[append_if_valid]) # floor
        else:
            lmark_numpy_MANY_VIDS[0].append(lmark_numpy_MANY_VIDS[0][-1])
    if len(lmark_numpy_MANY_VIDS[0])!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getGreaterThan_landmark_allHasHand on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del o2t_ratio
    # lmark_numpy_MANY_VIDS[0] is of shape (QUANTITY_FRAME, 518, 2), but
    # here lmark_numpy_MANY_VIDS is of shape (1, QUANTITY_FRAME, 518, 2)

    len_available_images: int= len(lmark_['landmark'])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
        notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        # notIncludedOn_mod, due to on a single video has quantity of images( ie. len(lmark_['landmark']) )
        # then mandatory idx_init_has_hand till last has enough images for QUANTITY_FRAME
        # ie. above --> QUANTITY_FRAME<=len_available_images,
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                         ^^^^^^^^^^^^^^^^^^^^^^^____ total quanitty images on video
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^____ subtract
        # whats to be used on forward index images
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                  ^^^^^^^^^^^^^^^^^____ due below appends starts
        # at idx_init_has_hand
        # notIncludedOn_mod: int= len(lmark_['landmark'])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod)
        #                                                                     ^^^^^^^^^^^^^^^^^^^^^^____ due to
        # index below `ii`( represents mod ie. o2t_mod, ie. int(len_available_images/QUANTITY_FRAME) # floor,
        # ie. 0, 1, 2, ..., o2t_mod-1 ) and `iii`( represents 0, 1, 2, ..., QUANTITY_FRAME-1 ), ie.
        # for combo o2t_mod*`iii` on last part of images as (QUANTITY_FRAME, 86, 2)
        for i in range(notIncludedOn_mod+1):
            for ii in range(o2t_mod):
                lmark_numpy_MANY_VIDS.append([])
                for iii in range(QUANTITY_FRAME):
                    append_if_valid: int= idx_init_has_hand +(iii*o2t_mod+ii) +i
                    if lmark_['landmark'][append_if_valid]['left_hand'] or \
                        lmark_['landmark'][append_if_valid]['right_hand']:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[
                            append_if_valid
                        ])
                    elif iii==0:
                        # due to since iii==0 then lmark_numpy_MANY_VIDS[-1][-1] does not exist,
                        # ie. len(lmark_numpy_MANY_VIDS[-1])==0 True, due to prev at
                        # lmark_numpy_MANY_VIDS.append([]) above
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[idx_init_has_hand])
                    else:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
        del o2t_mod
        del notIncludedOn_mod

        # part 3, consecutive, mandatory initial has hand
        for i in range((len_available_images-QUANTITY_FRAME)+1):
            lmark_numpy_MANY_VIDS.append([])
            # due to below appends shape (QUANTITY_FRAME, 86, 2)
            for ii in range(QUANTITY_FRAME):
                append_if_valid: int= idx_init_has_hand+ii +i
                if lmark_['landmark'][append_if_valid]['left_hand'] or \
                    lmark_['landmark'][append_if_valid]['right_hand']:
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[
                        append_if_valid
                    ])
                elif ii==0:
                    # due to since ii==0 then lmark_numpy_MANY_VIDS[-1][-1] does not exist,
                    # ie. len(lmark_numpy_MANY_VIDS[-1])==0 True, due to prev at
                    # lmark_numpy_MANY_VIDS.append([]) above
                    lmark_numpy_MANY_VIDS[-1].append(lmark_all[idx_init_has_hand])
                else:
                    lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
    elif len_available_images<=QUANTITY_FRAME:
        lmark_numpy_MANY_VIDS.append([])
        t2o_ratio: int= int(ceil(QUANTITY_FRAME/len_available_images)) # ceiling2make QUANTITY_FRAME possible
        for i, i_0to_t2o_multiplier in zip(range(idx_init_has_hand, idx_init_has_hand+len_available_images), range(len_available_images)):
            # i_0to_t2o_multiplier, for counting later due to target is (QUANTITY_FRAME, 86, 2)
            # i_0to_t2o_multiplier, ie. 0, 1, 2, ..., int(len_available_images-1)
            # i_0to_t2o_multiplier<QUANTITY_FRAME, due to len_available_images<=QUANTITY_FRAME
            for ii in range(t2o_ratio):
                if (i_0to_t2o_multiplier*t2o_ratio +ii)<QUANTITY_FRAME:
                    # i_0to_t2o_multiplier*t2o_ratio, due to since: len_available_images<=QUANTITY_FRAME,
                    # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                    # multiple times ie. int(t2o_ratio) times
                    # then +ii, due to current be added mod of from int(t2o_ratio),
                    # thus i_0to_t2o_multiplier*t2o_ratio+ii
                    if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_all[  i  ])
                    else:
                        lmark_numpy_MANY_VIDS[-1].append(lmark_numpy_MANY_VIDS[-1][-1])
        if len(lmark_numpy_MANY_VIDS[-1])!=QUANTITY_FRAME:
            raise ValueError("incorrect implementation on idx idx_init_has_hand!=-1 and len_available_images<QUANTITY_FRAME, getGreaterThan_landmark_allHasHand")
    del len_available_images

    return lmark_numpy_MANY_VIDS


def getLessThanOrEqual_landmark_allHasHand(lmark_: dict) -> list:
    '''
    to be used for when len(lmark_['landmark']) <= QUANTITY_FRAME
    '''
    def getIdxStartHand(image_list: list) -> int:
        for i in range(len(image_list)):
            if image_list[i]['left_hand'] or image_list[i]['right_hand']:
                return i
        return -1
    idx_init_has_hand: int= getIdxStartHand(image_list=lmark_['landmark'])
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
                # i_0to_t2o_multiplier*t2o_ratio, due to since: getLessThanOrEqual_landmark,
                # then mandatory be each image/frame/landmark/pose_face_lefthand_righthand be used
                # multiple/( or 1 time if equal and idx 0 has hand ) times ie. int(t2o_ratio) times
                # then +ii, due to current be added mod of from int(t2o_ratio),
                # thus i_0to_t2o_multiplier*t2o_ratio+ii
                if lmark_['landmark'][i]['left_hand'] or lmark_['landmark'][i]['right_hand']:
                    lmark_numpy.append( landmark_data_numpy )
                else:
                    lmark_numpy.append( lmark_numpy[-1] )
    if len(lmark_numpy)!=QUANTITY_FRAME:
        raise ValueError("incorrect implementation on getLessThanOrEqual_landmark_allHasHand, due to len(lmark_numpy)!=QUANTITY_FRAME")
    return lmark_numpy


def getdata_landmark(trainVal: str= 'train', batch: int=ON_TRAINING_BATCH) -> Generator[tuple, None, None]:
    # glasl_READY['train']
    # glasl_READY['val']
    # glasl_READY['test']
    # glasl_READY['id2gloss']
    # glasl_READY['gloss2id']

    # each landmark numpy file is of shape (518, 2)
    shuffle(glasl_landmark[trainVal])
    b_idxINIT: int= 0
    total_q_dataset: int= LEN_TRAIN if trainVal==KEY_TRAIN else LEN_VAL
    past_landmarks: list= [] # to hold for past landmark
    # `while True:` loop runs int(TRAIN_STEPS) for every epoch
    # total_q_count, counts the quantity of video landmarks that was and is training
    # ie. past all batch_vids on instance training, ie. `p -m src_asl2gloss.model_train`, then
    # count glasl_LM[trainVal][  idx_DS  ] including repeated( video but different
    # images, due to video has many images ) due to greater
    # than QUANTITY_FRAME
    total_q_count: int= 0
    i_0toBatchOrMore: int= 0 # for glasl_LM[TrainVal][__ b_idxINIT + i_0toBatchOrMore __]
    while True:
        batch_vids: ndarray= zeros((batch, QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]), dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)


        idx_add2batch: int= 0
        # for batch_vids[__ idx_add2batch __]
        # batch_class[__ idx_add2batch __]
        # below( ie. while idx_add2batch<batch: ) runs 1 time( 1 while loop done ) per batch,
        # below only knows batch NOTHING MORE NOTHING LESS
        # does ----> NOT <---- have control on train steps and epochs
        while idx_add2batch<batch:
            idx_DS: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<total_q_dataset else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-total_q_dataset
            ))
            lmark_nplist: list= [] # at end should be of shape 22, 86, 2
            if len(past_landmarks)==0:
                folder_landmark: str= f"{GLASL_LANDMARK_DIR}{glasl_landmark[trainVal][  idx_DS  ]['video_id']}"
                if exists(folder_landmark):
                    if len(glasl_landmark[trainVal][  idx_DS  ]['landmark'])<=QUANTITY_FRAME:
                        past_landmarks.append(getLessThanOrEqual_landmark_allHasHand(glasl_landmark[trainVal][  idx_DS  ]))
                        # past_landmarks.append(getLessThanOrEqual_landmark_initHand(glasl_landmark[trainVal][  idx_DS  ]))
                    else: # quanity of image landmark is more than QUANTITY_FRAME
                        past_landmarks= getGreaterThan_landmark_allHasHand(glasl_landmark[trainVal][  idx_DS  ])
                        # past_landmarks.extend(getGreaterThan_landmark_initHand(glasl_landmark[trainVal][  idx_DS  ]))
                        # past_landmarks.extend(getGreaterThan_landmark(glasl_landmark[trainVal][  idx_DS  ]))
            # if 0<len(past_landmarks):
            lmark_nplist= past_landmarks[0]
            past_landmarks= past_landmarks[1:]
            total_q_count+= 1

            if len(past_landmarks)==0 or len(lmark_nplist)==0:
                i_0toBatchOrMore+= 1
            if len(lmark_nplist)==QUANTITY_FRAME:
                batch_vids[idx_add2batch]= tuple(lmark_nplist) # array of shape(QUANTITY_FRAME, 86, 2)
                batch_class[idx_add2batch]= int(glasl_landmark[trainVal][  idx_DS  ]['gloss_id'])
                idx_add2batch+= 1
            elif len(lmark_nplist)!=0 and len(lmark_nplist)!=QUANTITY_FRAME:
                print(f"len of lmark_nplist: {len(lmark_nplist)}")
                raise ValueError("incorrect implementation on getdata_landmark, due to len(lmark_nplist)!=QUANTITY_FRAME and len(lmark_nplist)!=QUANTITY_FRAME")


            if idx_DS==(total_q_dataset-1) and len(past_landmarks)==0:
                # print(f"________ total_q_count: {total_q_count+len(past_landmarks)} ______ {trainVal}")
                total_q_count= 0


        if len(past_landmarks)==0:
            b_idxINIT= (b_idxINIT+batch) if (b_idxINIT+batch)<total_q_dataset else 0+( (b_idxINIT+batch)-total_q_dataset )
            i_0toBatchOrMore= 0
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))
