from json import load as jsonload

from ..lmark_constant import WLASL_VID_DIR

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
    print(f"id2gloss len {len(wlasl_ready['label_id2gloss'])}")
    print(f"gloss2id len {len(wlasl_ready['label_gloss2id'])}")
    if len(wlasl_ready['label_id2gloss'])==len(wlasl_ready['label_gloss2id']):
        min_qtrain: int= 99999
        max_qtrain: int= 0
        min_qtrain_idx: int= -1
        max_qtrain_idx: int= -1
        print("blah---------------------- True")
        for i in range(len(wlasl_ready['label_id2gloss'])):
            if i != wlasl_ready['label_gloss2id'][ wlasl_ready['label_id2gloss'][i] ]:
                print(f"problem mapping id {i} id2gloss {wlasl_ready['label_id2gloss'][i]} 1st process")
                print(f"_______ mapping id {i} gloss2id {wlasl_ready['label_gloss2id'][ wlasl_ready['label_id2gloss'][i] ]} 2nd process")
            print(f"{i} ---- {wlasl_ann[i]['gloss']} ---- q: {len(wlasl_ann[i]['instances'])}")
            if len(wlasl_ann[i]['instances'])<min_qtrain:
                min_qtrain= len(wlasl_ann[i]['instances'])
                min_qtrain_idx= i
            if max_qtrain<len(wlasl_ann[i]['instances']):
                max_qtrain= len(wlasl_ann[i]['instances'])
                max_qtrain_idx= i
        print(f"{min_qtrain} min quantity of files on train, gloss id {min_qtrain_idx} ---- {wlasl_ann[min_qtrain_idx]['gloss']}")
        print(f"{max_qtrain} max quantity of files on train, gloss id {max_qtrain_idx} ---- {wlasl_ann[max_qtrain_idx]['gloss']}")
    else:
        print("blah---------------------- False")

