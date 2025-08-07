from cv2 import CAP_PROP_FRAME_COUNT, VideoCapture
from json import load as jsonload
from os.path import exists
from pickle import dump as pdump, load as pload

from ..lmark_constant import PROJ_ROOT, WLASL_VID_DIR


if __name__=="__main__":
    wlasl_ann: list= []
    with open(file= f"{WLASL_VID_DIR}../wlasl.annotation.clean.json", mode='r') as f:
        wlasl_ann= jsonload(f)
    if 0<len(wlasl_ann):
        print("processing...")
        min_qframe: int= 90909
        max_qframe: int= 0
        q_vids_successOpened: int= 0
        qframe_dist: dict= {}
        qframe_dist_vidfile: dict= {}
        # for gloss in wlasl_ann:
        #     for svideo in gloss['instances']:
        #         vidfile= f"{WLASL_VID_DIR}{svideo['video_id']}.mp4"
        #         if exists(vidfile):
        #             vidfile= VideoCapture(vidfile)
        #             if vidfile.isOpened():
        #                 curr_vidqframe: int= int(vidfile.get(CAP_PROP_FRAME_COUNT))
        #                 min_qframe= curr_vidqframe if curr_vidqframe<min_qframe else min_qframe
        #                 max_qframe= curr_vidqframe if max_qframe<curr_vidqframe else max_qframe
        #                 q_vids_successOpened +=1
        #                 try:
        #                     qframe_dist[curr_vidqframe] +=1
        #                     qframe_dist_vidfile[curr_vidqframe].append(svideo['video_id'])
        #                 except KeyError:
        #                     qframe_dist[curr_vidqframe]= 1
        #                     qframe_dist_vidfile[curr_vidqframe]= [svideo['video_id']]
        #             vidfile.release()
        # # min_qframe
        # # max_qframe
        # # q_vids_successOpened
        # # qframe_dist
        # with open(f"{PROJ_ROOT}src_asl2gloss/checks/data_frame_vids.py.pickle", "wb") as f:
        #     pdump([min_qframe, max_qframe, q_vids_successOpened, qframe_dist, qframe_dist_vidfile], f)
        with open(f"{PROJ_ROOT}src_asl2gloss/checks/data_frame_vids.py.pickle", "rb") as f:
            tmp: list= pload(f)
            min_qframe= tmp[0]
            max_qframe= tmp[1]
            q_vids_successOpened= tmp[2]
            qframe_dist= tmp[3]
            qframe_dist_vidfile= tmp[4]
        print(f"MINimum quantity of frames amongs ALL video( MP4 ) files --> {min_qframe}")
        print(f"MINimum quantity of frames files are {qframe_dist_vidfile[min_qframe]}")
        print(f"MAXimum quantity of frames amongs ALL video( MP4 ) files --> {max_qframe}")
        print(f"MAXimum quantity of frames files are {qframe_dist_vidfile[max_qframe]}")
        print(f"quantity video files successfully opened {q_vids_successOpened}")
        # for k in sorted(list(qframe_dist.keys())):
        #     print(f"qframe_dist {k} --> {qframe_dist[k]}")
        sqf_minQvids: dict= { "k": [], "val": 909090} # min quantity videos has same frame, sameQuantityFrame minQuantityVideos
        sqf_maxQvids: dict= { "k": [], "val": 0} # max quantity videos has same frame, sameQuantityFrame maxQuantityVideos
        qframe_dist_keyList: list= []
        for k in sorted(list(qframe_dist.keys())):
            qframe_dist_keyList.append(k)
            if qframe_dist[k] < sqf_minQvids['val']:
                sqf_minQvids['val']= qframe_dist[k]
                sqf_minQvids['k']= [k]
            elif qframe_dist[k] == sqf_minQvids['val']:
                sqf_minQvids['k'].append(k)
            if sqf_maxQvids['val'] < qframe_dist[k]:
                sqf_maxQvids['val']= qframe_dist[k]
                sqf_maxQvids['k']= [k]
            elif qframe_dist[k] == sqf_maxQvids['val']:
                sqf_maxQvids['k'].append(k)
        print(f"MINimum quantity videos has same frame of {sqf_minQvids['k']} -- quantity of videos {sqf_minQvids['val']}")
        print(f"MAXimum quantity videos has same frame of {sqf_maxQvids['k']} -- quantity of videos {sqf_maxQvids['val']}")
        print(f"---- box plot on frames with quantity of videos ----")
        print(f"quarter 1 ( minimum ) --> {qframe_dist_keyList[0]} frames( multiImg ) --> {qframe_dist[qframe_dist_keyList[0]]} qvids")
        print(f"Q1 files: {qframe_dist_vidfile[qframe_dist_keyList[0]]}")


        bp_mid: int= int(len(qframe_dist_keyList)//2)
        print(f"quarter 2 ( median ) --> {qframe_dist_keyList[bp_mid//2]} frames( multiImg ) --> {qframe_dist[qframe_dist_keyList[bp_mid//2]]} qvids")
        print(f"Q2 files: {qframe_dist_vidfile[qframe_dist_keyList[bp_mid//2]]}")


        print(f"quarter 3 ( median ) --> {qframe_dist_keyList[bp_mid]} frames( multiImg ) --> {qframe_dist[qframe_dist_keyList[bp_mid]]} qvids")
        print(f"Q3 files: {qframe_dist_vidfile[qframe_dist_keyList[bp_mid]]}")


        print(f"quarter 4 ( median ) --> {qframe_dist_keyList[bp_mid+bp_mid//2]} frames( multiImg ) --> {qframe_dist[qframe_dist_keyList[bp_mid+bp_mid//2]]} qvids")
        print(f"Q4 files: {qframe_dist_vidfile[qframe_dist_keyList[bp_mid+bp_mid//2]]}")


        print(f"quarter 5 ( maximum ) --> {qframe_dist_keyList[-1]} frames( multiImg ) --> {qframe_dist[qframe_dist_keyList[-1]]} qvids")
        print(f"Q5 files: {qframe_dist_vidfile[qframe_dist_keyList[-1]]}")

