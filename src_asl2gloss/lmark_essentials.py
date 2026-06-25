from math import ceil
from numpy import load as loadnp
from typing import Generator
from random import sample
from numpy import array, float32, ndarray, uint16


from .lmark_constant import (
    GLASL_LANDMARK_DIR,
    KEY_FILE,
    KEY_GLOSS,
    KEY_VIDEO,
    # LANDMARK_SHAPE,
    PART4_MOD2USE,
    glasl_landmark as GLASL_LM_DS,
    KEY_LHAND,
    KEY_LMARK,
    KEY_RHAND,
    KEY_TRAIN,
    ON_TRAINING_BATCH,
    QUANTITY_FRAME,
)


def get_idx_start_hand(annotated_images: list) -> int:
    for idx in range(len(annotated_images)):
        if annotated_images[idx][KEY_LHAND] or annotated_images[idx][KEY_RHAND]:
            return idx
    return -1


def calculate_steps_needed(train_val: str=KEY_TRAIN, batch_size: int=ON_TRAINING_BATCH) -> int:
    total_DS: int= 0
    for a_video in GLASL_LM_DS[train_val]:
        idx_init_has_hand: int= get_idx_start_hand(a_video[KEY_LMARK])
        if idx_init_has_hand!=-1:
            if len(a_video[KEY_LMARK])<=QUANTITY_FRAME:
                total_DS+= 1
            else:
                len_available_images: int= len(a_video[KEY_LMARK])-idx_init_has_hand
                o2t_mod: int= int(len_available_images/QUANTITY_FRAME) # floor
                total_DS+= 1 # for part 1
                if QUANTITY_FRAME<len_available_images:
                    total_DS+= int(
                        o2t_mod *(len(a_video[KEY_LMARK]) -(idx_init_has_hand+ QUANTITY_FRAME*o2t_mod))
                    ) # for part 2
                    # total_DS+= (len_available_images-QUANTITY_FRAME)+1 # for part 3
                    for work4mod_what in PART4_MOD2USE: # for part 4
                        if 0<int(len_available_images//work4mod_what):
                            total_DS+= (len_available_images//work4mod_what) -QUANTITY_FRAME +1
                        else:
                            total_DS+= 1
                else:
                    total_DS+= 1
    return int(ceil(total_DS/float(batch_size)))


def get_landmark4less_or_equal(a_raw_video: dict, idx_init_has_hand: int|None=None) -> list:
    '''
    on a raw video the quantity of images is less than or
    equal to the target frames which is `QUANTITY_FRAME: int`
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
        lmark_numpy_out.extend([load_an_image_landmarks] *ratio_what)
        # for _ in range(min(ratio_what, QUANTITY_FRAME-len(lmark_numpy_out))):
        #     if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
        #         lmark_numpy_out.append(load_an_image_landmarks)
        #     else:
        #         lmark_numpy_out.append(lmark_numpy_out[-1])
    lmark_numpy_out= lmark_numpy_out[:QUANTITY_FRAME]
    # check_shape: ndarray= array(lmark_numpy_out, dtype=float32)
    # if check_shape.shape!=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]):
    #     raise NotImplementedError(
    #         f"incorrect implementation on get_landmark4less_or_equal(), due to lmark_numpy_out should be of shape {
    #         tuple((QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]))
    #         }, but got {check_shape.shape}"
    #     )
    return lmark_numpy_out


def get_landmark4greater(a_raw_video: dict) -> list:
    '''
    to be used for when len(a_raw_video[KEY_LMARK]) > QUANTITY_FRAME
    output be of shape(____ int, QUANTITY_FRAME, 86, 2 ____)
    '''
    lmark_numpy_out__MANY_VIDS: list= [[]] # be of shape(__ int, QUANTITY_FRAME, 86, 2 __)

    lmark_load_ALL: list= []
    idx_init_has_hand: int= -1
    for idx in range(len(a_raw_video[KEY_LMARK])):
        with open(f"{GLASL_LANDMARK_DIR /a_raw_video[KEY_VIDEO] /a_raw_video[KEY_LMARK][idx][KEY_FILE]}", 'rb') as f:
            lmark_load_ALL.append(loadnp(f))
        if idx_init_has_hand==-1:
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                idx_init_has_hand= idx
    if idx_init_has_hand==-1:
        return list()

    # part 1, floor at index level, still idx_init_has_hand
    ratio: float= (len(a_raw_video[KEY_LMARK]) -idx_init_has_hand) /QUANTITY_FRAME
    for idx in range(QUANTITY_FRAME):
        append_if_valid: int= idx_init_has_hand +int(idx*ratio) # floor
        if a_raw_video[KEY_LMARK][append_if_valid][KEY_LHAND] or \
            a_raw_video[KEY_LMARK][append_if_valid][KEY_RHAND]:
            lmark_numpy_out__MANY_VIDS[0].append(lmark_load_ALL[append_if_valid])
        else:
            lmark_numpy_out__MANY_VIDS[0].append(lmark_numpy_out__MANY_VIDS[0][-1])
    # if len(lmark_numpy_out__MANY_VIDS[0])!=QUANTITY_FRAME:
    #     raise ValueError("incorrect implementation on get_landmark4greater() on part 1 due to NOT QUANTITY_FRAME, when should be QUANTITY_FRAME")
    del ratio
    # lmark_numpy_out__MANY_VIDS[0] is of shape (QUANTITY_FRAME, 86, 2), but
    # here lmark_numpy_out__MANY_VIDS is of shape (1, QUANTITY_FRAME, 86, 2)

    len_available_images: int= len(a_raw_video[KEY_LMARK])-idx_init_has_hand
    # len_available_images, quantity of images starting from idx_init_has_hand till end
    if QUANTITY_FRAME<len_available_images:
        # part 2, evenly spaced via mod, floor at orig/target ratio level
        o2t_ratio: int= int(len_available_images/QUANTITY_FRAME) # floor >= 1
        notIncludedOn_mod: int= len(a_raw_video[KEY_LMARK])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_ratio)
        # '''
        # notIncludedOn_mod, due to on a single video has quantity of images( ie. len(a_raw_video[KEY_LMARK]) )
        # then mandatory idx_init_has_hand till last has enough images for QUANTITY_FRAME
        # ie. above --> QUANTITY_FRAME<=len_available_images,
        # notIncludedOn_mod: int= len(a_raw_video[KEY_LMARK])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_ratio)
        #                         ^^^^^^^^^^^^^^^^^^^^^^^____ total quanitty images on video
        # notIncludedOn_mod: int= len(a_raw_video[KEY_LMARK])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_ratio)
        #                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^____ subtract
        # whats to be used on forward index images
        # notIncludedOn_mod: int= len(a_raw_video[KEY_LMARK])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_ratio)
        #                                                  ^^^^^^^^^^^^^^^^^____ due below appends starts
        # at idx_init_has_hand
        # notIncludedOn_mod: int= len(a_raw_video[KEY_LMARK])-(idx_init_has_hand+ QUANTITY_FRAME*o2t_ratio)
        #                                                                     ^^^^^^^^^^^^^^^^^^^^^^____ due to
        # index below `ii`( represents mod ie. o2t_ratio, ie. int(len_available_images/QUANTITY_FRAME) --> floor,
        # ie. 0, 1, 2, ..., o2t_ratio-1 ) and `iii`( represents 0, 1, 2, ..., QUANTITY_FRAME-1 ), ie.
        # for combo o2t_ratio*`iii` on last part of images as (QUANTITY_FRAME, 86, 2)
        # '''
        for shift_right in range(notIncludedOn_mod+1):
            for idx_mod in range(o2t_ratio):
                lmark_numpy_out__MANY_VIDS.append([])
                for idx_img2target in range(QUANTITY_FRAME):
                    append_if_valid: int= idx_init_has_hand +(idx_img2target*o2t_ratio+idx_mod) +shift_right
                    if a_raw_video[KEY_LMARK][append_if_valid][KEY_LHAND] or \
                        a_raw_video[KEY_LMARK][append_if_valid][KEY_RHAND]:
                        lmark_numpy_out__MANY_VIDS[-1].append(lmark_load_ALL[
                            append_if_valid
                        ])
                    elif idx_img2target==0:
                        # due to since idx_img2target==0 then lmark_numpy_out__MANY_VIDS[-1][-1] does not exist,
                        # ie. len(lmark_numpy_out__MANY_VIDS[-1])==0 True, due to prev at
                        # lmark_numpy_out__MANY_VIDS.append([]) above
                        lmark_numpy_out__MANY_VIDS[-1].append(lmark_load_ALL[idx_init_has_hand])
                    else:
                        lmark_numpy_out__MANY_VIDS[-1].append(lmark_numpy_out__MANY_VIDS[-1][-1])
        del notIncludedOn_mod

        # # part 3, consecutive, mandatory initial has hand
        # for start_where in range((len_available_images-QUANTITY_FRAME)+1):
        #     lmark_numpy_out__MANY_VIDS.append([])
        #     # due to below appends shape (QUANTITY_FRAME, 86, 2)
        #     for idx in range(QUANTITY_FRAME):
        #         append_if_valid: int= idx_init_has_hand+idx +start_where
        #         if a_raw_video[KEY_LMARK][append_if_valid][KEY_LHAND] or \
        #             a_raw_video[KEY_LMARK][append_if_valid][KEY_RHAND]:
        #             lmark_numpy_out__MANY_VIDS[-1].append(lmark_load_ALL[
        #                 append_if_valid
        #             ])
        #         elif idx==0:
        #             # due to since ii==0 then lmark_numpy_out__MANY_VIDS[-1][-1] does not exist,
        #             # ie. len(lmark_numpy_out__MANY_VIDS[-1])==0 True, due to prev at
        #             # lmark_numpy_out__MANY_VIDS.append([]) above
        #             lmark_numpy_out__MANY_VIDS[-1].append(lmark_load_ALL[idx_init_has_hand])
        #         else:
        #             lmark_numpy_out__MANY_VIDS[-1].append(lmark_numpy_out__MANY_VIDS[-1][-1])

        # part 4, all has hand and use past if current no hand
        init_hand_idxs: tuple= tuple(range(idx_init_has_hand, len(a_raw_video[KEY_LMARK])))
        past_img_has_hand: ndarray= lmark_load_ALL[idx_init_has_hand]
        tmp_part4: dict= {i: [] for i in PART4_MOD2USE}
        for check_mod_init_0, idx in zip(range(len_available_images), init_hand_idxs):
            for work4mod_what in PART4_MOD2USE:
                if check_mod_init_0%work4mod_what == work4mod_what-1:
                    if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or \
                        a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                        tmp_part4[work4mod_what].append(lmark_load_ALL[idx])
                    else:
                        tmp_part4[work4mod_what].append(past_img_has_hand)
            if a_raw_video[KEY_LMARK][idx][KEY_LHAND] or \
                a_raw_video[KEY_LMARK][idx][KEY_RHAND]:
                past_img_has_hand= lmark_load_ALL[idx]
        for work4mod_what in PART4_MOD2USE:
            if QUANTITY_FRAME<len(tmp_part4[work4mod_what]):
                for idx_start_at in range(len(tmp_part4[work4mod_what])-QUANTITY_FRAME +1):
                    lmark_numpy_out__MANY_VIDS.append(tmp_part4[work4mod_what][
                        idx_start_at:idx_start_at+QUANTITY_FRAME
                    ])
            else:
                lmark_numpy_out__MANY_VIDS.append([])
                ratio4mod_part4: int= int(ceil(QUANTITY_FRAME/len(tmp_part4[work4mod_what])))
                for an_image_landmarks in tmp_part4[work4mod_what]:
                    lmark_numpy_out__MANY_VIDS[-1].extend(list([an_image_landmarks]) *ratio4mod_part4)
                lmark_numpy_out__MANY_VIDS[-1]= lmark_numpy_out__MANY_VIDS[-1][:QUANTITY_FRAME]
        for idx_start_at in range(len(tmp_part4)-QUANTITY_FRAME +1):
            lmark_numpy_out__MANY_VIDS.append(tmp_part4[idx_start_at:idx_start_at+QUANTITY_FRAME])
    else: # len_available_images <= QUANTITY_FRAME
        copy_a_raw_video: dict= {
            KEY_GLOSS: a_raw_video[KEY_GLOSS],
            KEY_VIDEO: a_raw_video[KEY_VIDEO],
            KEY_LMARK: a_raw_video[KEY_LMARK][idx_init_has_hand:]
        }
        lmark_numpy_out__MANY_VIDS.append(get_landmark4less_or_equal(copy_a_raw_video, 0))
    # check_shape: ndarray= array(lmark_numpy_out__MANY_VIDS, dtype=float32)
    # if check_shape.shape[-3:]!=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]):
    #     raise NotImplementedError(f"incorrect implementation get_landmark4greater(), lmark_numpy_out__MANY_VIDS should of shape {
    #     (QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1])
    #     } but got {tuple(check_shape.shape[-3:])}")

    return lmark_numpy_out__MANY_VIDS


def get_data_landmark(
    train_val: str= KEY_TRAIN,
    batch_size: int=ON_TRAINING_BATCH
) -> Generator:
    dataset_idxs: tuple= tuple(sample(
        range(len(GLASL_LM_DS[train_val])),
        len(GLASL_LM_DS[train_val])
    ))
    batch_videos: list= []
    batch_class: list= []
    while True:
        for idx_ds in dataset_idxs:
            a_raw_video: dict= GLASL_LM_DS[train_val][idx_ds]
            if QUANTITY_FRAME<len(a_raw_video[KEY_LMARK]):
                tmp: list= get_landmark4greater(a_raw_video)
                if 0<len(tmp):
                    batch_videos.extend(tmp)
                    batch_class.extend(
                        [a_raw_video[KEY_GLOSS]] *len(tmp)
                    )
            else:
                tmp: list= get_landmark4less_or_equal(a_raw_video)
                if 0<len(tmp):
                    batch_videos.append(tmp)
                    batch_class.append(a_raw_video[KEY_GLOSS])
            # ---- now ready ----
            while batch_size<=len(batch_videos):
                out_inputs: ndarray= array(batch_videos[:batch_size], dtype=float32)
                out_expected_outputs: ndarray= array(batch_class[:batch_size], dtype=uint16)
                batch_videos= batch_videos[batch_size:]
                batch_class= batch_class[batch_size:]
                yield (out_inputs, out_expected_outputs)
