from json import dump
from os import makedirs
from os.path import join as pjoin
from random import choices, shuffle
from typing import Generator
from cv2 import imread
from numpy import array, float32, ndarray, uint16, uint8, zeros
from math import ceil
from os.path import exists

from .lmark_constant import EPOCHS, IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, T10_DIR_IMG, TRAIN_BATCH, TRAIN_STEPS, VAL_STEPS, wlasl_READY_10




def getFramesLessThanTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
    if int(len(vid['images']))<1 or TqFrames<=int(len(vid['images'])):
        raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Less(or zero) than {TqFrames}( TqFrames )")
    oqFrames: int= len(vid['images'])
    t2o_ratio: int= int(ceil(TqFrames/oqFrames))
    multiVids: list= [[]]
    for i in range(oqFrames):
        for ii in range(t2o_ratio):
            if (i*t2o_ratio +ii)<TqFrames:
                img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
                if exists(img_path):
                    multiVids[0].append(imread(  img_path  ).astype(uint8))
                else:
                    raise FileExistsError(f"no file exist on {img_path}")
    # by 2
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
    beEmpty: list= [i for i in range(0, TqFrames, 2)]
    for i in beEmpty:
        multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+1<TqFrames:
            multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)


    # by 3
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5

    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
    beEmpty= [i for i in range(0, TqFrames, 3)]
    for i in beEmpty:
        multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
        multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
        if i+1<TqFrames:
            multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
            multiVids[6][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
            multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
        if i+2<TqFrames:
            multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
            multiVids[7][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod


    # by 4
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
    beEmpty= [i for i in range(0, TqFrames, 4)]
    for i in beEmpty:
        multiVids[8][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+1<TqFrames:
            multiVids[9][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+2<TqFrames:
            multiVids[10][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+3<TqFrames:
            multiVids[11][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)


    # by 5
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 15
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 16
    beEmpty= [i for i in range(0, TqFrames, 5)]
    for i in beEmpty:
        multiVids[12][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+1<TqFrames:
            multiVids[13][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+2<TqFrames:
            multiVids[14][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+3<TqFrames:
            multiVids[15][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+4<TqFrames:
            multiVids[16][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)

    return tuple(multiVids)


def getFramesEqualTarget(vid: dict, TqFrames: int=QUANTITY_FRAME) -> tuple:
    if int(len(vid['images']))!=TqFrames:
        raise FileExistsError(f"files exist on {vid['video_id']} quantity images is NOT Equal to {TqFrames}( TqFrames )")
    multiVids: list= [[]]
    for i in range(TqFrames):
        img_path: str= str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))
        if exists(img_path):
            multiVids[0].append(imread(  img_path  ).astype(uint8))
        else:
            raise FileExistsError(f"no file exist on {img_path}")
    # by 2
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 1
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 2
    beEmpty: list= [i for i in range(0, TqFrames, 2)]
    for i in beEmpty:
        multiVids[1][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+1<TqFrames:
            multiVids[2][i+1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)


    # by 3
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 3
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 4
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 5

    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 6
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 7
    beEmpty= [i for i in range(0, TqFrames, 3)]
    for i in beEmpty:
        multiVids[3][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 1st mod
        multiVids[6][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
        if i+1<TqFrames:
            multiVids[4][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 2nd mod
            multiVids[6][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 1st and 2nd mod
            multiVids[7][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod
        if i+2<TqFrames:
            multiVids[5][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 1 missing 3rd mod
            multiVids[7][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8) # 2 missing 2nd and 3rd mod


    # by 4
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 8
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 9
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 10
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 11
    beEmpty= [i for i in range(0, TqFrames, 4)]
    for i in beEmpty:
        multiVids[8][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+1<TqFrames:
            multiVids[9][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+2<TqFrames:
            multiVids[10][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+3<TqFrames:
            multiVids[11][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)


    # by 5
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 12
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 13
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 14
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 15
    multiVids.append(array(multiVids[0], dtype=uint8, copy=True)) # index 16
    beEmpty= [i for i in range(0, TqFrames, 5)]
    for i in beEmpty:
        multiVids[12][i]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+1<TqFrames:
            multiVids[13][i +1]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+2<TqFrames:
            multiVids[14][i +2]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+3<TqFrames:
            multiVids[15][i +3]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)
        if i+4<TqFrames:
            multiVids[16][i +4]= zeros((IMG_SIZE, IMG_SIZE, 3), dtype=uint8)

    return tuple(multiVids)


def getFramesG10_sHand(vid: dict, initGT: int=0, q_train: int=0, TqFrames: int=QUANTITY_FRAME) -> list:
    q_minTrain2addMissing_img: int= 2
    modWhere2empty: int= choices([3,4])[0]
    if int(len(vid['images']))<1:
        raise FileExistsError(f"no files exist on {vid['video_id']}")
    imgsList: list= []
    oqFrames: int= len(vid['images'])
    o2t_ratio: int= int(oqFrames//TqFrames)
    if oqFrames<TqFrames:
        t2o_ratio: int= int(ceil(TqFrames/oqFrames))
        for i in range(oqFrames):
            if vid['images'][i]['left_hand'] or vid['images'][i]['right_hand']:
                for ii in range(t2o_ratio):
                    if (i*t2o_ratio +ii)<TqFrames:
                        if q_minTrain2addMissing_img<=q_train and \
                            (((i*t2o_ratio+ii)%modWhere2empty)==(modWhere2empty-1) or \
                             ((i*t2o_ratio+ii)%modWhere2empty)==0):
                            imgsList.append(zeros(
                                (IMG_SIZE, IMG_SIZE, 3),
                                dtype=uint8
                            ))
                        else:
                            imgsList.append(array(
                                imread(str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))),
                                dtype=uint8,
                                copy=True
                            ))
        while len(imgsList)<TqFrames:
            imgsList.append(array(imgsList[-1], dtype=uint8, copy=True))
    elif oqFrames==TqFrames:
        for i in range(TqFrames):
            if q_minTrain2addMissing_img<=q_train and \
                ((i%modWhere2empty)==(modWhere2empty-1) or \
                 (i%modWhere2empty)==0):
                imgsList.append(zeros(
                    (IMG_SIZE, IMG_SIZE, 3),
                    dtype=uint8
                ))
            else:
                if vid['images'][i]['left_hand'] or vid['images'][i]['right_hand']:
                    imgsList.append(array(
                        imread(str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i]['file']}.png"))),
                        dtype=uint8,
                        copy=True
                    ))
        while len(imgsList)<TqFrames:
            imgsList.append(array(imgsList[-1], dtype=uint8, copy=True))
    else: # TqFrames<oqFrames
        initGT= initGT%o2t_ratio
        for i in range(TqFrames):
            i_has_hands: int= 0
            for ii in range(o2t_ratio):
                if ( not vid['images'][i*o2t_ratio +i_has_hands]['left_hand'] and \
                    not vid['images'][i*o2t_ratio +i_has_hands]['right_hand'] ) and \
                    ( vid['images'][i*o2t_ratio +ii]['left_hand'] or \
                    vid['images'][i*o2t_ratio +ii]['right_hand'] ):
                    i_has_hands= ii
            if q_minTrain2addMissing_img<=q_train and \
                ((i%modWhere2empty)==(modWhere2empty-1) or \
                 (i%modWhere2empty)==0):
                imgsList.append(zeros(
                    (IMG_SIZE, IMG_SIZE, 3),
                    dtype=uint8
                ))
            else:
                if vid['images'][i*o2t_ratio +i_has_hands]['left_hand'] or vid['images'][i*o2t_ratio +i_has_hands]['right_hand']:
                    imgsList.append(array(
                        imread(str(pjoin(T10_DIR_IMG, vid['video_id'], f"{vid['images'][i*o2t_ratio +i_has_hands]['file']}.png")))
                    ))
                elif 0<len(imgsList):
                    imgsList.append(array(imgsList[-1], dtype=uint8, copy=True))
        while len(imgsList)<TqFrames:
            imgsList.append(array(imgsList[-1], dtype=uint8, copy=True))
    return [
        array(imgsList, dtype=uint8, copy=True),
        o2t_ratio if 1<o2t_ratio else 0
    ]


def getdataNotVid_10_new(TrainVal: str= 'train', batch: int=TRAIN_BATCH) -> Generator[tuple, None, None]:
    b_idxINIT: int= 0 # idx on 'train'|'val' on batch where idx start at
    batchWhat: int= 0 # pangkapila is current batch
    shuffle(wlasl_READY_10[TrainVal])
    shuffle(wlasl_READY_10[TrainVal])

    glossDist: dict= { i: {'gloss_id': i, 'quantity': 0, 'video_id': []} for i in range(len(wlasl_READY_10['label_id2gloss']))}
    glossDist['split']= TrainVal
    glossDist['split_size']= len(wlasl_READY_10[TrainVal])
    # glossDist= {
    #     0: {
    #         'gloss_id': 0,
    #         'quantity': int(on this gloss id( ie. gloss_id=0 is book ) how many training was done),
    #         'video_id': list(all video_id that training was done, later be processed as unique),
    #         'vid_q_uniq': int(quantity of video_id that are unique, ie. above right after processed unique)
    #       },
    #     1: {...},
    #     2: {...},
    #     ...
    #     9: {...},
    #     'split': str(  'train'|'val'  )
    #     'split_size': int(quantity of video_id on 'train'|'val' ie. len(wlasl_READY_10[TrainVal]))
    # }
    # print(len(wlasl_READY_10['label_id2gloss'])) # correct, it exist
    pastFrame: list= []
    while True:
        batchWhat+= 1
        batch_vids: ndarray= zeros(
            (batch, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
            dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)

        i_0toBatchOrMore: int= 0 # beUsed as b_idxINIT+i_0toBatchOrMore
        idx_add2batch: int= 0 # for batch_vids[ index ] and batch_class[ index ] itself
        while idx_add2batch<batch:
            curr_IDX: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<len(wlasl_READY_10[TrainVal]) else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-len(wlasl_READY_10[TrainVal])
            ))
            vidcurr_ann: dict= wlasl_READY_10[TrainVal][  curr_IDX  ]
            # trainThisClass: int= sorted(glossDist, key=lambda x: x['quantity'], reverse=False)[0]['gloss_id']
            # while vidcurr_ann['gloss_id']!=trainThisClass:
            #     i_0toBatchOrMore+= 1
            #     curr_IDX= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<len(wlasl_READY_10[TrainVal]) else (0 +(
            #         (b_idxINIT+i_0toBatchOrMore)-len(wlasl_READY_10[TrainVal])
            #     ))
            #     vidcurr_ann= wlasl_READY_10[TrainVal][  curr_IDX  ]
            if exists(str(pjoin(T10_DIR_IMG, vidcurr_ann['video_id']))):
                try:
                    # len(vidcurr_ann['images']) < QUANTITY_FRAME
                    # len(vidcurr_ann['images']) == QUANTITY_FRAME
                    # len(vidcurr_ann['images']) > QUANTITY_FRAME
                    vidframes_data, _= getFramesG10_sHand(vidcurr_ann, q_train=glossDist[
                        vidcurr_ann['gloss_id']
                    ]['quantity'])
                    batch_vids[idx_add2batch]= vidframes_data.astype(float32)/255.0
                    batch_class[idx_add2batch]= int(vidcurr_ann['gloss_id'])/1.0
                    glossDist[int(vidcurr_ann['gloss_id'])]['quantity']+= 1
                    glossDist[int(vidcurr_ann['gloss_id'])]['video_id'].append(
                        vidcurr_ann['video_id']
                    )
                    idx_add2batch+= 1
                except FileExistsError as e:
                    del e
            if len(pastFrame)==0:
                i_0toBatchOrMore+= 1
        if len(pastFrame)==0:
            b_idxINIT= (b_idxINIT+batch) if (b_idxINIT+batch)<len(wlasl_READY_10[TrainVal]) else 0+( (b_idxINIT+batch)-int(len(wlasl_READY_10[TrainVal])) )
        if batchWhat==(TRAIN_STEPS*EPOCHS) or batchWhat==VAL_STEPS:
            for i in range(len(wlasl_READY_10['label_id2gloss'])):
                glossDist[ i ]['video_id']= list(set(
                    glossDist[ i ]['video_id']
                ))
                glossDist[ i ]['vid_q_uniq']= int(len(glossDist[ i ]['video_id']))
            if not exists(str(pjoin(PROJ_ROOT, f"training_{TrainVal}"))):
                makedirs(str(pjoin(PROJ_ROOT, f"training_{TrainVal}")))
            with open(str(pjoin(PROJ_ROOT, f"training_{TrainVal}", f"{TrainVal}_{batchWhat}.json")), 'w') as f:
                dump(glossDist, f)
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))

