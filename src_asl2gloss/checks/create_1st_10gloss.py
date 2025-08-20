from json import load as jsonload, dump as jsondump

from ..lmark_constant import WLASL_VID_DIR, PROJ_ROOT

if __name__=="__main__":
    wlasl_ann: list= []
    with open(file= f"{WLASL_VID_DIR}../wlasl.annotation.clean.json", mode='r') as f:
        wlasl_ann= jsonload(f)
    wrk_train: str= "train"
    wrk_val: str= "val"
    wrk_test: str= "test"
    wrk_label_id2gloss: str= "label_id2gloss"
    wrk_label_gloss2id: str= "label_gloss2id"
    wlasl_ready: dict= {
        wrk_train: [],
        wrk_val: [],
        wrk_test: [],
        wrk_label_id2gloss: [str(wlasl_ann[i]['gloss']) for i in range(10)]
    }
    wlasl_ready[wrk_label_gloss2id]= { str(wlasl_ready[wrk_label_id2gloss][  i  ]): int(  i  )
        for i in range(len(wlasl_ready[wrk_label_id2gloss]))}
    for idx_gloss in range(10):
        for idx_insVid in range(len(wlasl_ann[idx_gloss]['instances'])):
            if wlasl_ann[idx_gloss]['instances'][idx_insVid]['split']==wrk_train:
                wlasl_ready[wrk_train].append({
                    "gloss_id": int(idx_gloss),
                    "video_id": str(wlasl_ann[idx_gloss]['instances'][idx_insVid]['video_id'])
                })
            elif wlasl_ann[idx_gloss]['instances'][idx_insVid]['split']==wrk_val:
                wlasl_ready[wrk_val].append({
                    "gloss_id": int(idx_gloss),
                    "video_id": str(wlasl_ann[idx_gloss]['instances'][idx_insVid]['video_id'])
                })
            else:
                wlasl_ready[wrk_test].append({
                    "gloss_id": int(idx_gloss),
                    "video_id": str(wlasl_ann[idx_gloss]['instances'][idx_insVid]['video_id'])
                })
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.1st_10.json", "w") as f:
        jsondump(wlasl_ready, f)
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.1st_10.json", "r") as f:
        wlasl_ready= jsonload(f)


    print("---- training 1st 10 gloss ----")
    for insTrain in wlasl_ready[wrk_train]:
        print(f"gloss_id( {insTrain['gloss_id']} ) ---- video_id( {insTrain['video_id']} ) {wlasl_ready[wrk_label_id2gloss][insTrain['gloss_id']]}")
    print("\n\n---- val 1st 10 gloss ----")
    for insVal in wlasl_ready[wrk_val]:
        print(f"gloss_id( {insVal['gloss_id']} ) ---- video_id( {insVal['video_id']} ) {wlasl_ready[wrk_label_id2gloss][insVal['gloss_id']]}")
    print("\n\n---- test 1st 10 gloss ----")
    for insTest in wlasl_ready[wrk_test]:
        print(f"gloss_id( {insTest['gloss_id']} ) ---- video_id( {insTest['video_id']} ) {wlasl_ready[wrk_label_id2gloss][insTest['gloss_id']]}")

