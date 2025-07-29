from typing import Generator
from cv2 import imwrite


from ..lmark_essential_draw import getdata


if __name__=="__main__":
    gen: Generator= getdata()
    batch1= next(gen)
    for i in range(len(batch1[0]['batch_vid'][0])):
        if (i+1)<10:
            imwrite(f"/tmp/wlasl_vid_test/00{i+1}.png", batch1[0]['batch_vid'][0][i])
        else:
            imwrite(f"/tmp/wlasl_vid_test/0{i+1}.png", batch1[0]['batch_vid'][0][i])
    print(batch1[0]['batch_vid'].shape)
    print(batch1[0]['batch_vid'][0].shape)
    print(batch1[1]['batch_class'])

