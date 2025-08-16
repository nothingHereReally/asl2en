from json import dump
from os import listdir, makedirs
from os.path import join as pjoin
from random import choices, shuffle
from typing import Generator
from cv2 import imread
from numpy import array, float32, ndarray, uint16, uint8, zeros
from math import ceil
from os.path import exists

from .lmark_constant import EPOCHS, IMG_SIZE, PROJ_ROOT, QUANTITY_FRAME, T10_DIR_IMG, TRAIN_BATCH, TRAIN_STEPS, VAL_STEPS, wlasl_READY_10




def getdataNotVid_10_new(TrainVal: str= 'train', batch: int=TRAIN_BATCH) -> Generator[tuple, None, None]:
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
    b_idxINIT: int= 0 # idx on 'train'|'val' on batch where idx start at
    batchWhat: int= 0 # pangkapila is current batch
    shuffle(wlasl_READY_10[TrainVal])
    shuffle(wlasl_READY_10[TrainVal])

    glossDist: dict= { i: {'quantity': 0, 'video_id': []} for i in range(len(wlasl_READY_10['label_id2gloss']))}
    glossDist['split']= TrainVal
    glossDist['split_size']= len(wlasl_READY_10[TrainVal])
    # glossDist= {
    #     0: {
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
    while True:
        batchWhat+= 1
        batch_vids: ndarray= zeros(
            (batch, QUANTITY_FRAME, IMG_SIZE, IMG_SIZE, 3),
            dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)

        i_0toBatchOrMore: int= 0
        idx_add2batch: int= 0
        modWhat: int= 0
        while idx_add2batch<batch:
            curr_IDX_USE: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<len(wlasl_READY_10[TrainVal]) else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-len(wlasl_READY_10[TrainVal])
            ))
            vidcurr_ann: dict= wlasl_READY_10[TrainVal][  curr_IDX_USE  ]
            min_q: int= 99999
            trainThisClass: int= -1
            for i in range(len(wlasl_READY_10['label_id2gloss'])):
                if glossDist[i]['quantity']<min_q:
                    min_q= glossDist[i]['quantity']
                    trainThisClass= i
            if exists(str(pjoin(T10_DIR_IMG, vidcurr_ann['video_id']))) and \
                int(vidcurr_ann['gloss_id'])==trainThisClass:
                try:
                    vidframes_data, o2tRatio= getFramesG10_sHand(vidcurr_ann, initGT=modWhat, q_train=glossDist[
                        vidcurr_ann['gloss_id']
                    ]['quantity'])
                    batch_vids[idx_add2batch]= vidframes_data.astype(float32)/255.0
                    batch_class[idx_add2batch]= int(vidcurr_ann['gloss_id'])/1.0
                    glossDist[int(vidcurr_ann['gloss_id'])]['quantity']+= 1
                    glossDist[int(vidcurr_ann['gloss_id'])]['video_id'].append(
                        vidcurr_ann['video_id']
                    )
                    idx_add2batch+= 1
                    # if true below, then worthy be balik to igbaw same vidfile_dir as previous
                    # due to original_frames_quantity//target_frames_quantity > 1
                    # ie. daghag original frames compare to target frames( QUANTITY_FRAME )
                    if 1<o2tRatio:
                        if modWhat==0:
                            modWhat= o2tRatio
                        # modWhat==1 meaning has just recently processed the last mod
                        # due to modWhat==0 is already done and was the 1st 1 to be
                        # processed
                        if modWhat!=1:
                            # meaning modWhat be 0, 2, 3, 4, 5, 6, 7, 8, ...
                            # then go back, due process be 0, ..., 4, 3, 2
                            # to go back, but on 1 since last part, then dili na due to 2nd last
                            i_0toBatchOrMore-= 1
                        modWhat-= 1
                except FileExistsError as e:
                    del e
            i_0toBatchOrMore+= 1
            if len(wlasl_READY_10[TrainVal])<i_0toBatchOrMore:
                i_0toBatchOrMore= 0
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

