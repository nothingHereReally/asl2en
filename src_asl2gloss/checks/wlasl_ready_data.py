from json import load as jsonload, dump as jsondump

from ..lmark_constant import WLASL_VID_DIR, PROJ_ROOT

if __name__=="__main__":
    wlasl_ann: list= []
    with open(file= f"{WLASL_VID_DIR}../wlasl.annotation.clean.json", mode='r') as f:
        wlasl_ann= jsonload(f)
    wlasl_ready: dict= {
        "train": [],
        "val": [],
        "test": [],
        "label_id2gloss": [i['gloss'] for i in wlasl_ann],
    }
    wlasl_ready["label_gloss2id"]= { str(wlasl_ready['label_id2gloss'][i]): int(i) for i in range(len(wlasl_ready['label_id2gloss']))}
    for sgloss in wlasl_ann:
        for svid in sgloss['instances']:
            if svid['split']=='train':
                wlasl_ready['train'].append({
                    "gloss_id": wlasl_ready['label_gloss2id'][  sgloss['gloss']  ],
                    "video_id": svid['video_id']
                })
            elif svid['split']=='val':
                wlasl_ready['val'].append({
                    "gloss_id": wlasl_ready['label_gloss2id'][  sgloss['gloss']  ],
                    "video_id": svid['video_id']
                })
            else:
                wlasl_ready['test'].append({
                    "gloss_id": wlasl_ready['label_gloss2id'][  sgloss['gloss']  ],
                    "video_id": svid['video_id']
                })
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "w") as f:
        jsondump(wlasl_ready, f)
    with open(f"{PROJ_ROOT}dataset/wlasl_dataset/wlasl.annotation.ready.json", "r") as f:
        wlasl_ready= jsonload(f)
    print(f"quantity train {len(wlasl_ready['train'])} -- sample {wlasl_ready['train'][0]}")
    print(f"quantity val {len(wlasl_ready['val'])} -- sample {wlasl_ready['val'][0]}")
    print(f"quantity test {len(wlasl_ready['test'])} -- sample {wlasl_ready['test'][0]}")
    print(f"quantity id2gloss {len(wlasl_ready['label_id2gloss'])} -- sample {wlasl_ready['label_id2gloss'][0]}")
    print(f"quantity gloss2id {len(wlasl_ready['label_gloss2id'])} -- sample {wlasl_ready['label_gloss2id'][ wlasl_ready['label_id2gloss'][0] ]}")

