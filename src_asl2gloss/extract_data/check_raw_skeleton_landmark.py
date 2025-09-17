from json import load as loadjson
from os import listdir


PROJ_ROOT: str= f"{"/".join(__file__.rsplit("/")[:-3])}/"


if __name__=='__main__':
    raw_image_json: str= f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.train_val_test.json"
    skeleton_json: str= f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.skeleton_image.train_val_test.json"
    landmark_json: str= f"{PROJ_ROOT}dataset/wlasl/wlasl.annotation.landmark_numpy.train_val_test.json"
    RAW_IMAGE_dir: str= f"{PROJ_ROOT}dataset/wlasl/raw_image/"
    SKELETON__dir: str= f"{PROJ_ROOT}dataset/wlasl/skeleton_image/"
    LANDMARK__dir: str= f"{PROJ_ROOT}dataset/wlasl/landmark_numpy/"
    raw_image: dict= {}
    skeleton_: dict= {}
    landmark_: dict= {}
    with open(raw_image_json, 'r') as f:
        raw_image= loadjson(f)
    with open(skeleton_json, 'r') as f:
        skeleton_= loadjson(f)
    with open(landmark_json, 'r') as f:
        landmark_= loadjson(f)

    tvt: tuple= ('train', 'val', 'test')
    all_good: bool= True
    for tvt_idv in tvt:
        for i in range(len(raw_image[tvt_idv])):
            # check on all 3 if each video folder
            # has same quanitty of file ie.
            # png file to png file to numpy file
            q_raw_image: int= len(listdir(
                f"{RAW_IMAGE_dir}{raw_image[tvt_idv][i]['video_id']}"
            ))
            q_skeleton: int= len(listdir(
                f"{SKELETON__dir}{skeleton_[tvt_idv][i]['video_id']}"
            ))
            q_landmark: int= len(listdir(
                f"{LANDMARK__dir}{landmark_[tvt_idv][i]['video_id']}"
            ))
            if q_raw_image!=q_skeleton or q_raw_image!=q_landmark or q_skeleton!=q_landmark:
                all_good= False
                print(f"folder video_id: {raw_image[tvt_idv][i]['video_id']} all 3 are not of equal quantity of files")
                print(f"raw image quantity files: {q_raw_image}")
                print(f"skeleton image quantity files: {q_skeleton}")
                print(f"landmark quantity files: {q_landmark}")
            if len(skeleton_[tvt_idv][i]['image'])!=len(landmark_[tvt_idv][i]['landmark']):
                all_good= False
                print(f"json file at video_id {raw_image[tvt_idv][i]['video_id']} skeleton and landmark are not of equal quantity of images/landmarks")
                print(f"skeleton image quantity files on json: {len(skeleton_[tvt_idv][i]['image'])}")
                print(f"landmark quantity files on json: {len(landmark_[tvt_idv][i]['landmark'])}")
    if all_good:
        print("all files per video_id are of equal quantity, all is good")
    else:
        print("sorry somethings wrong on the dataset")

