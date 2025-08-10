from ..lmark_constant import wlasl_READY_10


if __name__=='__main__':
    for train in wlasl_READY_10['train']:
        print(f"{train['video_id']} ---- {train['gloss_id']}")
    for val in wlasl_READY_10['val']:
        print(f"{val['video_id']} ---- {val['gloss_id']}")
    for test in wlasl_READY_10['test']:
        print(f"{test['video_id']} ---- {test['gloss_id']}")

