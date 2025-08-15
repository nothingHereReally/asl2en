from ..lmark_constant import wlasl_READY_10


if __name__=="__main__":
    trainValTest: tuple= ('train', 'val', 'test')
    print(f"wlasl_READY_10 keys: {', '.join(list(wlasl_READY_10.keys()))}")
    for tvt in trainValTest:
        g10_dist: list= [{
            'q_video': 0,
            'q_images': 0, # q_video < q_images
            'q_face': 0,
            'q_pose': 0,
            'q_left_hand': 0,
            'q_right_hand': 0
        } for _ in range(len(wlasl_READY_10['label_id2gloss']))]
        min_img: int= 99999
        min_face: int= 99999
        min_pose: int= 99999
        min_left_hand: int= 99999
        min_right_hand: int= 99999
        max_img: int= 0
        max_face: int= 0
        max_pose: int= 0
        max_left_hand: int= 0
        max_right_hand: int= 0
        for vid in wlasl_READY_10[tvt]:
            g10_dist[  vid['gloss_id']  ]['q_video']+= 1
            g10_dist[  vid['gloss_id']  ]['q_images']+= int(len(vid['images']))
            g10_dist[  vid['gloss_id']  ]['q_face']+= int(len(list(filter(lambda x: x['face']==True, vid['images']))))
            g10_dist[  vid['gloss_id']  ]['q_pose']+= int(len(list(filter(lambda x: x['pose']==True, vid['images']))))
            g10_dist[  vid['gloss_id']  ]['q_left_hand']+= int(len(list(filter(lambda x: x['left_hand']==True, vid['images']))))
            g10_dist[  vid['gloss_id']  ]['q_right_hand']+= int(len(list(filter(lambda x: x['right_hand']==True, vid['images']))))
            min_img= min_img if min_img<int(len(vid['images'])) else int(
                len(vid['images'])
            )
            min_face= min_face if min_face<int(len(list(filter(lambda x: x['face']==True, vid['images'])))) else int(
                len(list(filter(lambda x: x['face']==True, vid['images'])))
            )
            min_pose= min_pose if min_pose<int(len(list(filter(lambda x: x['pose']==True, vid['images'])))) else int(
                len(list(filter(lambda x: x['pose']==True, vid['images'])))
            )
            min_left_hand= min_left_hand if min_left_hand<int(len(list(filter(lambda x: x['left_hand']==True, vid['images'])))) else int(
                len(list(filter(lambda x: x['left_hand']==True, vid['images'])))
            )
            min_right_hand= min_right_hand if min_right_hand<int(len(list(filter(lambda x: x['right_hand']==True, vid['images'])))) else int(
                len(list(filter(lambda x: x['right_hand']==True, vid['images'])))
            )

            max_img= max_img if int(len(vid['images']))<max_img else int(
                len(vid['images'])
            )
            max_face= max_face if int(len(list(filter(lambda x: x['face']==True, vid['images']))))<max_face else int(
                len(list(filter(lambda x: x['face']==True, vid['images'])))
            )
            max_pose= max_pose if int(len(list(filter(lambda x: x['pose']==True, vid['images']))))<max_pose else int(
                len(list(filter(lambda x: x['pose']==True, vid['images'])))
            )
            max_left_hand= max_left_hand if int(len(list(filter(lambda x: x['left_hand']==True, vid['images']))))<max_left_hand else int(
                len(list(filter(lambda x: x['left_hand']==True, vid['images'])))
            )
            max_right_hand= max_right_hand if int(len(list(filter(lambda x: x['right_hand']==True, vid['images']))))<max_right_hand else int(
                len(list(filter(lambda x: x['right_hand']==True, vid['images'])))
            )
        print(f"----> {tvt} <----")
        print(f"quantity of elements {int(len(wlasl_READY_10[tvt]))} {tvt} <======")
        for i in range(len(wlasl_READY_10['label_id2gloss'])):
            print(f"{wlasl_READY_10['label_id2gloss'][i]} ---- {i}")
            print(f"____ quantity video: {g10_dist[i]['q_video']}")
            print(f"____ quantity images( on all vides combined ): {g10_dist[i]['q_images']}")
            print(f"____ quantity face( on all vides combined ): {g10_dist[i]['q_face']}")
            print(f"____ quantity pose( on all vides combined ): {g10_dist[i]['q_pose']}")
            print(f"____ quantity left hand( on all vides combined ): {g10_dist[i]['q_left_hand']}")
            print(f"____ quantity right hand( on all vides combined ): {g10_dist[i]['q_right_hand']}")
            print(f"______ on single video quantity of images has min --> q images <--( compare to other videos ) {min_img}")
            print(f"______ on single video quantity of images has min --> FACE <--( compare to other videos ) {min_face}")
            print(f"______ on single video quantity of images has min --> POSE <--( compare to other videos ) {min_pose}")
            print(f"______ on single video quantity of images has min --> LEFT HAND <--( compare to other videos ) {min_left_hand}")
            print(f"______ on single video quantity of images has min --> RIGHT HAND <--( compare to other videos ) {min_right_hand}")
            print(f"________ on single video quantity of images has max --> q images <--( compare to other videos ) {max_img}")
            print(f"________ on single video quantity of images has max --> FACE <--( compare to other videos ) {max_face}")
            print(f"________ on single video quantity of images has max --> POSE <--( compare to other videos ) {max_pose}")
            print(f"________ on single video quantity of images has max --> LEFT HAND <--( compare to other videos ) {max_left_hand}")
            print(f"________ on single video quantity of images has max --> RIGHT HAND <--( compare to other videos ) {max_right_hand}\n\n")
        print(f"__________________________________________")

