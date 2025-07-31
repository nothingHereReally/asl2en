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
        wrk_label_id2gloss: [str(i['gloss']) for i in wlasl_ann]
    }
    wlasl_ready[wrk_label_gloss2id]= { str(wlasl_ready[wrk_label_id2gloss][  i  ]): int(  i  )
        for i in range(len(wlasl_ready[wrk_label_id2gloss]))}
    for sgloss in wlasl_ann:
        for svid in sgloss['instances']:
            if svid['split']==wrk_train:
                wlasl_ready[wrk_train].append({
                    "gloss_id": int(wlasl_ready[wrk_label_gloss2id][  sgloss['gloss']  ]),
                    "video_id": str(svid['video_id'])
                })
            elif svid['split']==wrk_val:
                wlasl_ready[wrk_val].append({
                    "gloss_id": int(wlasl_ready[wrk_label_gloss2id][  sgloss['gloss']  ]),
                    "video_id": str(svid['video_id'])
                })
            else:
                wlasl_ready[wrk_test].append({
                    "gloss_id": int(wlasl_ready[wrk_label_gloss2id][  sgloss['gloss']  ]),
                    "video_id": str(svid['video_id'])
                })
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "w") as f:
        jsondump(wlasl_ready, f)
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
        wlasl_ready= jsonload(f)
    print(f"quantity train {len(wlasl_ready[wrk_train])} -- sample {wlasl_ready[wrk_train][2]}")
    print(f"---- type(video_id) {type(wlasl_ready[wrk_train][3]['video_id'])} -- type(gloss_id) {type(wlasl_ready[wrk_train][3]['gloss_id'])}")
    print(f"quantity val {len(wlasl_ready[wrk_val])} -- sample {wlasl_ready[wrk_val][3]}")
    print(f"quantity test {len(wlasl_ready[wrk_test])} -- sample {wlasl_ready[wrk_test][4]}")
    print(f"quantity id2gloss {len(wlasl_ready[wrk_label_id2gloss])} -- sample {wlasl_ready[wrk_label_id2gloss][5]}")
    print(f"quantity gloss2id {len(wlasl_ready[wrk_label_gloss2id])} -- sample {wlasl_ready[wrk_label_gloss2id][  wlasl_ready[wrk_label_id2gloss][5]  ]}")

