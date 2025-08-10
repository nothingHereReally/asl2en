from ..lmark_constant import IMG_SIZE, QUANTITY_FRAME, wlasl_READY_10


if __name__=='__main__':
    for train in wlasl_READY_10['train']:
        print(f"{train['video_id']} ---- {train['gloss_id']}")
    for val in wlasl_READY_10['val']:
        print(f"{val['video_id']} ---- {val['gloss_id']}")
    for test in wlasl_READY_10['test']:
        print(f"{test['video_id']} ---- {test['gloss_id']}")


    b_idxINIT: int= 0
    shape_vidBatch: tuple= (IMG_SIZE, IMG_SIZE, 3)
    for train in wlasl_READY_10['train']:
        batch_vids: ndarray= zeros(shape_vidBatch, dtype=float32)
        batch_class: ndarray= zeros((batch), dtype=uint16)
        i_0toBatchOrMore: int= 0
        idx_add2batch: int= 0
        modWhat: int= 0
        # below( ie. while idx_add2batch<batch: ) runs 1 time per epoch
        while idx_add2batch<batch:
            curr_IDX_USE: int= (b_idxINIT+i_0toBatchOrMore) if (b_idxINIT+i_0toBatchOrMore)<len(wlasl_READY_10[TrainVal]) else (0 +(
                (b_idxINIT+i_0toBatchOrMore)-len(wlasl_READY_10[TrainVal])
            ))
            vidfile_dir: str= f"{WLASL_VID_DIR}{wlasl_READY_10[TrainVal][  curr_IDX_USE  ]['video_id']}.mp4"
            if exists(vidfile_dir):
                try:
                    vidframes_data, o2tRatio= getSkeletonFrames(vidfile_dir, isSingleImg=isSimg, initGT=modWhat)
                    batch_vids[idx_add2batch]= vidframes_data.astype(float32)/255.0
                    batch_class[idx_add2batch]= int(wlasl_READY_10[TrainVal][  curr_IDX_USE  ]['gloss_id'])/1.0
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
        b_idxINIT= (b_idxINIT+batch) if (b_idxINIT+batch)<len(wlasl_READY_10[TrainVal]) else 0+( (b_idxINIT+batch)-int(len(wlasl_READY_10[TrainVal])) )
        yield (batch_vids.astype(float32), batch_class.astype(dtype=uint16))

