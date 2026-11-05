from json import dump as writeJson
from mediapipe.python.solutions.holistic import Holistic
from pathlib import Path
from typing import Any
import cv2
import numpy as np


PROJ_ROOT: Path= Path(__file__).parent.parent.parent
DS_DIR: Path= PROJ_ROOT /"dataset" /"asl_citizen"
VIDEO_DIR: Path= DS_DIR /"video"
LANDMARK_DIR: Path= DS_DIR /"investigate_landmark"
SKELETON_DIR: Path= DS_DIR /"investigate_skeleton"
KEY_G: str= 'gloss'
KEY_VIDS: str= 'videos'
KEY_LANDMARK: str= 'landmark'
KEY_SKELETON: str= 'skeleton'
# ---------------------------
KEY_SPLIT: str= 'split'
VAL_TRAIN: str= 'train'
VAL_TEST: str= 'test'
RATIO4TRAIN: float= 0.8
# ---------------------------
KEY_FILE: str= 'file'
KEY_FACE: str= 'face'
KEY_POSE: str= 'pose'
KEY_LHAND: str= 'left_hand'
KEY_RHAND: str= 'right_hand'
KEY_IMG_VALID: str= 'valid_images'
KEY_IMGSTART: str= 'start' # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_IMGEND: str= 'end'     # counting is 1, 2, 3, ..., NOT --> 0, 1, 2, ....
KEY_PFOLDER: str= 'parent_folder'
IMG_SIZE: int= 158
MPH_fph: Holistic= Holistic(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)
FACE_CONNECTIONS: tuple= (
    (3, 28), (28, 34), (34, 27), (27, 35), (35, 17), # left oval face
    (3, 12), (12, 19), (19, 11), (11, 21), (21, 17), # right oval face

    (26, 29), (29, 30), # left eyebrow

    (23, 32), (32, 31), # left eye down
    (31, 33), (33, 23), # left eye up

    (10, 13), (13, 14), # right eyebrow

    (7, 16), (16, 15), # right eye down
    (15, 18), (18, 7), # rght eye up

    (20, 22), (22, 2), # nose vertical line
    (2, 25), (25, 1), # left half nose
    (1, 9), (9, 2), # rigth half nose

    # mouth
    (8, 6), (6, 24), # down lip edge down
    (24, 0), (0, 8), # up lip edge up
    (8, 5), (5, 24), # up/down lip inner a
    (24, 4), (4, 8), # up/down lip inner b
)
WORTHY_FACE_IDX: tuple= (
    0, 2, 4, 10, 13, 14, 17, 33, 61, 64, 70, 93, 103,
    105, 107, 133, 145, 152, 159, 162, 168, 172, 195,
    263, 291, 294, 300, 323, 332, 334, 336, 362, 374,
    386, 389, 397
)

# before use of POSE_CONNECTIONS modify landmark 1st
# modify to use index to be used only: 11,12,13,14,15,16,23,24
# so new index: 0,1,2,3,4,5,6,7
POSE_CONNECTIONS: tuple= ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6), (1, 7))
WORTHY_POSE_IDX: tuple= (11,12,13,14,15,16,23,24)

HAND_CONNECTIONS: tuple= (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), # palm connections
    (1, 2), (2, 3), (3, 4),           # thumb finger connections
    (5, 6), (6, 7), (7, 8),           # index finger connections
    (9, 10), (10, 11), (11, 12),      # middle finger connections
    (13, 14), (14, 15), (15, 16),     # ring finger connections
    (17, 18), (18, 19), (19, 20)      # pinky finger connections
)
QUANTITY_HAND_LMARK: int= 21
VIDEO_LIST: list=[
    {
        KEY_G: 'body',
        KEY_VIDS: [
            'body_012509162398409268-BODY.mp4',
            'body_023036288555289275-BODY.mp4',
            'body_14988750046183918-BODY.mp4',
            'body_18110186267814066-BODY.mp4',
            'body_18235319744310297-BODY.mp4',
            'body_2179874717207788-BODY.mp4',
            'body_22658344717761292-BODY.mp4',
            'body_23152470677862436-BODY.mp4',
            'body_26985590858242636-BODY.mp4',
            'body_3003852815347694-seedBODY.mp4',
            'body_31812874442193184-BODY.mp4',
            'body_3219216092683568-BODY.mp4',
            'body_3470900852703691-BODY.mp4',
            'body_38673461744862414-BODY.mp4',
            'body_5487980874423544-BODY.mp4',
            'body_5857440141605472-BODY.mp4',
            'body_6459896690781501-BODY.mp4',
            'body_7111482896558119-BODY.mp4',
            'body_7291708075931664-BODY.mp4',
            'body_7381642274284421-BODY.mp4',
            'body_7403536775582238-BODY.mp4',
            'body_7518098133669218-BODY.mp4',
            'body_7530476182907826-BODY.mp4',
            'body_7716445637163099-BODY.mp4',
            'body_8329852027760176-BODY.mp4',
            'body_8676508765182491-BODY.mp4',
            'body_8918008593331341-BODY.mp4',
            'body_92382278646442-BODY.mp4',
            'body_9670696859371226-BODY.mp4',
            'body_9931262095524651-BODY.mp4',
        ]
    },
    {
        KEY_G: 'ear',
        KEY_VIDS: [
            'ear_004887131733942285-EAR.mp4',
            'ear_015143605161060059-EAR.mp4',
            'ear_06251353179823571-EAR.mp4',
            'ear_156747591554246-EAR.mp4',
            'ear_1677354883522144-EAR.mp4',
            'ear_1856278575298309-seedEAR.mp4',
            'ear_22124278737999714-EAR.mp4',
            'ear_23747369865179402-EAR.mp4',
            'ear_24470512221089535-EAR.mp4',
            'ear_3550234595479076-EAR.mp4',
            'ear_3597923622170567-EAR.mp4',
            'ear_393295857925948-EAR.mp4',
            'ear_4402580501014326-EAR.mp4',
            'ear_4496497579994534-EAR.mp4',
            'ear_468644667228318-EAR.mp4',
            'ear_48472331371206123-EAR.mp4',
            'ear_4886596064274962-EAR.mp4',
            'ear_4980479911230302-EAR.mp4',
            'ear_6904017055031177-EAR.mp4',
            'ear_7453873486095204-EAR.mp4',
            'ear_7486834557903128-EAR.mp4',
            'ear_8233537761275598-EAR.mp4',
            'ear_8272672764484996-EAR.mp4',
            'ear_8332413026628822-EAR.mp4',
            'ear_8438085614372515-EAR.mp4',
            'ear_8672274027954561-EAR.mp4',
            'ear_8789814378451983-EAR.mp4',
            'ear_9043673723315442-EAR.mp4',
            'ear_9689943177012852-EAR.mp4',
            'ear_9957354005644234-EAR.mp4',
        ]
    },
    {
        KEY_G: 'eyes',
        KEY_VIDS: [
            'eyes_015907943367508448-EYES.mp4',
            'eyes_024859836566530236-EYES.mp4',
            'eyes_0259965808147955-EYES.mp4',
            'eyes_06163237568266666-EYES.mp4',
            'eyes_0825936687116755-EYES.mp4',
            'eyes_09936405951023719-EYES.mp4',
            'eyes_11765090581788384-EYES.mp4',
            'eyes_1417646504520429-EYES.mp4',
            'eyes_1710485018536383-seedEYES.mp4',
            'eyes_18542508287605441-EYES.mp4',
            'eyes_23137361727286043-EYES.mp4',
            'eyes_2960811103126946-EYES.mp4',
            'eyes_31593048576495075-EYES.mp4',
            'eyes_32332228660933526-EYES.mp4',
            'eyes_32695601160288645-EYES.mp4',
            'eyes_32913239320491705-EYES.mp4',
            'eyes_3703962814561417-EYES.mp4',
            'eyes_4324249049090094-EYES.mp4',
            'eyes_4798839631383074-EYES.mp4',
            'eyes_4799020574000763-EYES.mp4',
            'eyes_48476800256701247-EYES.mp4',
            'eyes_5262600936263233-EYES.mp4',
            'eyes_5370606838669136-EYES.mp4',
            'eyes_5656636230859979-EYES.mp4',
            'eyes_5776607071720377-EYES.mp4',
            'eyes_7364724475858522-EYES.mp4',
            'eyes_7915772963155006-EYES.mp4',
            'eyes_9263181285873419-EYES.mp4',
            'eyes_9739227363076881-EYES.mp4',
            'eyes_9784462808238372-EYES.mp4',
        ]
    },
    {
        KEY_G: 'head',
        KEY_VIDS: [
            'head_04384724186611-HEAD.mp4',
            'head_09712564478415864-HEAD.mp4',
            'head_1434058469566546-HEAD.mp4',
            'head_14486467848103057-HEAD.mp4',
            'head_18306126220160568-HEAD.mp4',
            'head_209949774531083-HEAD.mp4',
            'head_22456983792819685-HEAD.mp4',
            'head_24673566947760395-HEAD.mp4',
            'head_24680684027267863-HEAD.mp4',
            'head_2820341113576026-HEAD.mp4',
            'head_3011506857442785-HEAD.mp4',
            'head_42134820560428166-HEAD.mp4',
            'head_49019167048619505-HEAD.mp4',
            'head_5670994066372224-HEAD.mp4',
            'head_6435251723193283-HEAD.mp4',
            'head_6723633822967117-HEAD.mp4',
            'head_6931138867364373-HEAD.mp4',
            'head_708769017680714-HEAD.mp4',
            'head_7133987301547997-HEAD.mp4',
            'head_7216744097791985-HEAD.mp4',
            'head_7571183732047406-HEAD.mp4',
            'head_820003400952684-seedHEAD.mp4',
            'head_8543671956618997-HEAD.mp4',
            'head_858079173472031-HEAD.mp4',
            'head_875536918629837-HEAD.mp4',
            'head_9165906112034756-HEAD.mp4',
            'head_9180504598913761-HEAD.mp4',
            'head_9415520083603939-HEAD.mp4',
            'head_9453054211167886-HEAD.mp4',
            'head_9516322979360861-HEAD.mp4',
            'head_9902361145649847-HEAD.mp4',
        ]
    },
    {
        KEY_G: 'nose',
        KEY_VIDS: [
            'nose_024011464372713398-NOSE.mp4',
            'nose_05530025137611427-NOSE.mp4',
            'nose_08859205707077078-NOSE.mp4',
            'nose_09217547618470956-NOSE.mp4',
            'nose_11100447586620077-NOSE.mp4',
            'nose_11213721786275399-NOSE.mp4',
            'nose_12789189992227112-NOSE.mp4',
            'nose_2649632981009986-NOSE.mp4',
            'nose_2691787200603868-NOSE.mp4',
            'nose_27103930782829466-NOSE.mp4',
            'nose_27763818780548877-NOSE.mp4',
            'nose_33334259968320534-NOSE.mp4',
            'nose_3712619010031373-seedNOSE.mp4',
            'nose_37138484525407467-NOSE.mp4',
            'nose_388068410054895-NOSE.mp4',
            'nose_388468441362549-NOSE.mp4',
            'nose_6032656220146979-NOSE.mp4',
            'nose_6652814499755777-NOSE.mp4',
            'nose_6657895793756088-NOSE.mp4',
            'nose_6746586649060085-NOSE.mp4',
            'nose_6926188522163574-NOSE.mp4',
            'nose_7964037565939572-NOSE.mp4',
            'nose_8013809161550702-NOSE.mp4',
            'nose_8366576122149043-NOSE.mp4',
            'nose_8590037977346539-NOSE.mp4',
            'nose_8740752793195805-NOSE.mp4',
            'nose_8812266341155683-NOSE.mp4',
            'nose_8977573353357979-NOSE.mp4',
            'nose_9573594298423673-NOSE.mp4',
            'nose_9749846209389852-NOSE.mp4',
            'nose_9814478093116983-NOSE.mp4',
        ]
    },
    {
        KEY_G: 'chin',
        KEY_VIDS: [
            'chin_028431370391587052-CHIN_2.mp4',
            'chin_09101309522440815-CHIN_2.mp4',
            'chin_09176723865512693-CHIN_2.mp4',
            'chin_11642380273775732-CHIN_2.mp4',
            'chin_167715325082177-CHIN_2.mp4',
            'chin_17378200161913648-CHIN_2.mp4',
            'chin_1921109044689493-CHIN_2.mp4',
            'chin_22956006243428684-CHIN_2.mp4',
            'chin_25405637892250943-CHIN_2.mp4',
            'chin_30852673069728254-CHIN_2.mp4',
            'chin_3145184748224372-CHIN_2.mp4',
            'chin_3969659858373005-CHIN_2.mp4',
            'chin_41965703854599123-CHIN_2.mp4',
            'chin_45079112918929276-CHIN_2.mp4',
            'chin_45798047890015803-CHIN_2.mp4',
            'chin_5804474557706663-CHIN_2.mp4',
            'chin_6126017782493465-CHIN_2.mp4',
            'chin_6388058030166044-CHIN_2.mp4',
            'chin_7475699875205186-seedCHIN_2.mp4',
            'chin_7489397758558725-CHIN_2.mp4',
            'chin_8181059620707356-CHIN_2.mp4',
            'chin_8238098597093131-CHIN_2.mp4',
            'chin_8425645911667179-CHIN_2.mp4',
            'chin_8632429403144224-CHIN_2.mp4',
            'chin_8828565299201225-CHIN_2.mp4',
            'chin_8838873748002694-CHIN_2.mp4',
            'chin_9026329674927109-CHIN_2.mp4',
            'chin_9116796811415531-CHIN_2.mp4',
            'chin_9621111268056717-CHIN_2.mp4',
            'chin_9870662940863097-CHIN_2.mp4',
        ]
    },
    {
        KEY_G: 'gum',
        KEY_VIDS: [
            'gum_018213254785876654-GUM.mp4',
            'gum_0663439230334546-GUM.mp4',
            'gum_07050828584534741-GUM.mp4',
            'gum_07804494694411823-GUM.mp4',
            'gum_15670060196174318-GUM.mp4',
            'gum_17151697988510128-GUM.mp4',
            'gum_3704551309457076-GUM.mp4',
            'gum_41310640938491505-GUM.mp4',
            'gum_49233665334372345-GUM.mp4',
            'gum_5169364593751733-GUM.mp4',
            'gum_6083769831208186-GUM.mp4',
            'gum_6458203226414239-GUM.mp4',
            'gum_6525706206661304-GUM.mp4',
            'gum_6774779032894782-GUM.mp4',
            'gum_6833405506171966-GUM.mp4',
            'gum_7307539259823796-GUM.mp4',
            'gum_7553281453781038-GUM.mp4',
            'gum_7567845252066006-GUM.mp4',
            'gum_7959535258775348-GUM.mp4',
            'gum_8098218886424609-GUM.mp4',
            'gum_8379768724841927-GUM.mp4',
            'gum_8473585521435789-GUM.mp4',
            'gum_8538998842407926-GUM.mp4',
            'gum_888256404419975-GUM.mp4',
            'gum_9021548417432745-GUM.mp4',
            'gum_9038745375939039-GUM.mp4',
            'gum_9223917641630366-GUM.mp4',
            'gum_9832027585927117-GUM.mp4',
            'gum_9942355192801713-seedGUM.mp4',
            'gum_9971342114927344-GUM.mp4',
        ]
    },
    {
        KEY_G: 'mouth',
        KEY_VIDS: [
            'mouth_03816967766642465-MOUTH.mp4',
            'mouth_07770960749445921-MOUTH.mp4',
            'mouth_08629381582280149-MOUTH.mp4',
            'mouth_11034432898849245-MOUTH.mp4',
            'mouth_16288519431904458-MOUTH.mp4',
            'mouth_17189033119247688-MOUTH.mp4',
            'mouth_27934471953770945-MOUTH.mp4',
            'mouth_3108143937298642-MOUTH.mp4',
            'mouth_36090957279237323-MOUTH.mp4',
            'mouth_36938644349320904-MOUTH.mp4',
            'mouth_3755885673667929-MOUTH.mp4',
            'mouth_3784591045062218-MOUTH.mp4',
            'mouth_38196977602640914-MOUTH.mp4',
            'mouth_38571484397842015-MOUTH.mp4',
            'mouth_43438530362822236-MOUTH.mp4',
            'mouth_5022024196847912-MOUTH.mp4',
            'mouth_5193451899614112-MOUTH.mp4',
            'mouth_5203155274355433-MOUTH.mp4',
            'mouth_5432467311144609-MOUTH.mp4',
            'mouth_5548976355283721-MOUTH.mp4',
            'mouth_6329931932403075-MOUTH.mp4',
            'mouth_6465600241429885-MOUTH.mp4',
            'mouth_7034210782927104-MOUTH.mp4',
            'mouth_7248892268066589-seedMOUTH.mp4',
            'mouth_726398082719435-MOUTH.mp4',
            'mouth_7918844361958985-MOUTH.mp4',
            'mouth_7995460004354844-MOUTH.mp4',
            'mouth_811072606700949-MOUTH.mp4',
            'mouth_8137737865216244-MOUTH.mp4',
            'mouth_9174421973113369-MOUTH.mp4',
            'mouth_9840276388705085-MOUTH.mp4',
        ]
    },
    {
        KEY_G: 'neck',
        KEY_VIDS: [
            'neck_002159742772197104-NECK.mp4',
            'neck_00689292174465983-NECK.mp4',
            'neck_010111338865819608-NECK.mp4',
            'neck_029915112474971917-NECK.mp4',
            'neck_04541132470344267-NECK.mp4',
            'neck_08986172824338423-NECK.mp4',
            'neck_10197900855285669-NECK.mp4',
            'neck_1105210028899144-NECK.mp4',
            'neck_11253959737739772-NECK.mp4',
            'neck_2117672377435793-NECK.mp4',
            'neck_2914346788925697-NECK.mp4',
            'neck_392847143355763-NECK.mp4',
            'neck_40869836172089014-NECK.mp4',
            'neck_41528846653396556-NECK.mp4',
            'neck_46050718150682335-NECK.mp4',
            'neck_5912486425716992-NECK.mp4',
            'neck_6285059634173602-NECK.mp4',
            'neck_6444945863834219-NECK.mp4',
            'neck_7132418799120459-NECK.mp4',
            'neck_7432560177781751-NECK.mp4',
            'neck_8031051338831652-NECK.mp4',
            'neck_810265626942593-NECK.mp4',
            'neck_813425015075141-NECK.mp4',
            'neck_8335950392429026-NECK.mp4',
            'neck_8662672393229258-NECK.mp4',
            'neck_9124929111701665-NECK.mp4',
            'neck_9587190259032854-NECK.mp4',
            'neck_9592682704538538-NECK.mp4',
            'neck_9632390680801999-NECK.mp4',
            'neck_9870815260603178-seedNECK.mp4',
        ]
    },
    {
        KEY_G: 'tongue',
        KEY_VIDS: [
            'tongue_10952554539506631-TONGUE.mp4',
            'tongue_11388872998746447-TONGUE.mp4',
            'tongue_16538366877078858-TONGUE.mp4',
            'tongue_1731074372958592-TONGUE.mp4',
            'tongue_18793572427530347-TONGUE.mp4',
            'tongue_25040182027203084-TONGUE.mp4',
            'tongue_2928599376925225-TONGUE.mp4',
            'tongue_29312176513514654-TONGUE.mp4',
            'tongue_2995978175034413-TONGUE.mp4',
            'tongue_3180818311436544-TONGUE.mp4',
            'tongue_3766203415307263-TONGUE.mp4',
            'tongue_43750843655956917-TONGUE.mp4',
            'tongue_4394027253628445-TONGUE.mp4',
            'tongue_44185907261489477-TONGUE.mp4',
            'tongue_4578141062450063-TONGUE.mp4',
            'tongue_4716794408227516-TONGUE.mp4',
            'tongue_5125705967911591-TONGUE.mp4',
            'tongue_5231084233910046-seedTONGUE.mp4',
            'tongue_5322853783203081-TONGUE.mp4',
            'tongue_7378948905149243-TONGUE.mp4',
            'tongue_7882190991680063-TONGUE.mp4',
            'tongue_8223991553574521-TONGUE.mp4',
            'tongue_8437488963585511-TONGUE.mp4',
            'tongue_8444805263891311-TONGUE.mp4',
            'tongue_8775868781021792-TONGUE.mp4',
            'tongue_882408493830463-TONGUE.mp4',
            'tongue_9066607607927726-TONGUE.mp4',
            'tongue_9150450405711148-TONGUE.mp4',
            'tongue_9388164161732007-TONGUE.mp4',
            'tongue_9459694545314585-TONGUE.mp4',
        ]
    },
    {
        KEY_G: 'arm',
        KEY_VIDS: [
            'arm_06875405886440444-ARM_3.mp4',
            'arm_13014078898215797-ARM_3.mp4',
            'arm_1535086938572987-ARM_3.mp4',
            'arm_1703736894296095-ARM_3.mp4',
            'arm_1766551266387144-ARM_3.mp4',
            'arm_18252662803841702-ARM_3.mp4',
            'arm_18436257304546988-ARM_3.mp4',
            'arm_20151143656361903-ARM_3.mp4',
            'arm_252084110246515-ARM_3.mp4',
            'arm_3280914345213024-ARM_3.mp4',
            'arm_3532171049708259-ARM_3.mp4',
            'arm_3875903439603874-ARM_3.mp4',
            'arm_38957968308015767-ARM_3.mp4',
            'arm_40026841881315-ARM_3.mp4',
            'arm_44775996014584885-ARM_3.mp4',
            'arm_46398054082560547-ARM_3.mp4',
            'arm_46904377801196495-ARM_3.mp4',
            'arm_5617325649329079-seedARM_3.mp4',
            'arm_6252908560318753-ARM_3.mp4',
            'arm_6451133306899228-ARM_3.mp4',
            'arm_648363559934048-ARM_3.mp4',
            'arm_7720082318308641-ARM_3.mp4',
            'arm_7743834750763563-ARM_3.mp4',
            'arm_8035752319943781-ARM_3.mp4',
            'arm_8161065721190677-ARM_3.mp4',
            'arm_8353385242068521-ARM_3.mp4',
            'arm_8490859524236458-ARM_3.mp4',
            'arm_8503669604822188-ARM_3.mp4',
            'arm_907767547176082-ARM_3.mp4',
            'arm_9195740431840584-ARM_3.mp4',
            'arm_9346317186909523-ARM_3.mp4',
        ]
    },
    {
        KEY_G: 'finger',
        KEY_VIDS: [
            'finger_006007887322393879-FINGER.mp4',
            'finger_06889282709145239-FINGER.mp4',
            'finger_19511824774736541-FINGER.mp4',
            'finger_2196507384016626-FINGER.mp4',
            'finger_30123812533804895-FINGER.mp4',
            'finger_3017633228385066-FINGER.mp4',
            'finger_31835717268957353-seedFINGER.mp4',
            'finger_3237225336922218-FINGER.mp4',
            'finger_33057973168830923-FINGER.mp4',
            'finger_35557707796449156-FINGER.mp4',
            'finger_35791960976845894-FINGER.mp4',
            'finger_3781013762752725-FINGER.mp4',
            'finger_3825663392118379-FINGER.mp4',
            'finger_4033702656472984-FINGER.mp4',
            'finger_41929710796780184-FINGER.mp4',
            'finger_42851478328968695-FINGER.mp4',
            'finger_4740066894367059-FINGER.mp4',
            'finger_47727751077906433-FINGER.mp4',
            'finger_6325675928394079-FINGER.mp4',
            'finger_6537381381903058-FINGER.mp4',
            'finger_6735292433767321-FINGER.mp4',
            'finger_6862442324594495-FINGER.mp4',
            'finger_7131946304999996-FINGER.mp4',
            'finger_7426465985678312-FINGER.mp4',
            'finger_7997056920264738-FINGER.mp4',
            'finger_8088159932133285-FINGER.mp4',
            'finger_8488425175148562-FINGER.mp4',
            'finger_922585810595768-FINGER.mp4',
            'finger_9328368891488181-FINGER.mp4',
            'finger_9626179528662269-FINGER.mp4',
        ]
    },
    {
        KEY_G: 'hands',
        KEY_VIDS: [
            'hands_02782722533629678-HANDS.mp4',
            'hands_04268194927860791-HANDS.mp4',
            'hands_07235547531459896-HANDS.mp4',
            'hands_09820261529418906-HANDS.mp4',
            'hands_13923543055899334-HANDS.mp4',
            'hands_24253465795861295-HANDS.mp4',
            'hands_31487135960578416-HANDS.mp4',
            'hands_3802234939313722-HANDS.mp4',
            'hands_41033415539817586-HANDS.mp4',
            'hands_4141697998179248-HANDS.mp4',
            'hands_44916141431303425-HANDS.mp4',
            'hands_46654728425757486-HANDS.mp4',
            'hands_5315418020402818-HANDS.mp4',
            'hands_5548063463829158-HANDS.mp4',
            'hands_6104364113963681-HANDS.mp4',
            'hands_6137014052176015-HANDS.mp4',
            'hands_6335100605902133-HANDS.mp4',
            'hands_6368495873121254-HANDS.mp4',
            'hands_6719352826133227-HANDS.mp4',
            'hands_6720783140394533-HANDS.mp4',
            'hands_7044510285087742-HANDS.mp4',
            'hands_7382172388589892-HANDS.mp4',
            'hands_739112927091981-HANDS.mp4',
            'hands_7454876674866346-HANDS.mp4',
            'hands_7494358434873458-HANDS.mp4',
            'hands_7720364098262635-HANDS.mp4',
            'hands_7773712240167525-HANDS.mp4',
            'hands_8005044888557988-seedHANDS.mp4',
            'hands_8187487299677705-HANDS.mp4',
            'hands_9694797706140053-HANDS.mp4',
            'hands_9808999675373478-HANDS.mp4',
        ]
    },
    {
        KEY_G: 'hearth',
        KEY_VIDS: [
            'hearth_0596738632892615-HEARTBODY.mp4',
            'hearth_06823095187885575-HEARTBODY.mp4',
            'hearth_14629201741355402-HEARTBODY.mp4',
            'hearth_20132491671886066-HEARTBODY.mp4',
            'hearth_20146972673867425-HEARTBODY.mp4',
            'hearth_21189655894981962-HEARTBODY.mp4',
            'hearth_23476630068245297-HEARTBODY.mp4',
            'hearth_2898262557497693-HEARTBODY.mp4',
            'hearth_32138134663150675-HEARTBODY.mp4',
            'hearth_4070974316820264-HEARTBODY.mp4',
            'hearth_47938919974598626-HEARTBODY.mp4',
            'hearth_5145420536026162-HEARTBODY.mp4',
            'hearth_5396509619343579-HEARTBODY.mp4',
            'hearth_561963088245941-HEARTBODY.mp4',
            'hearth_5710813665222216-HEARTBODY.mp4',
            'hearth_5754764115128781-HEARTBODY.mp4',
            'hearth_5885245957009604-HEARTBODY.mp4',
            'hearth_5977460365096976-HEARTBODY.mp4',
            'hearth_6466497454416145-HEARTBODY.mp4',
            'hearth_6597977577129788-HEARTBODY.mp4',
            'hearth_6650562321508888-HEARTBODY.mp4',
            'hearth_7170471283154412-HEARTBODY.mp4',
            'hearth_7178181868282436-HEARTBODY.mp4',
            'hearth_7181712901886621-HEARTBODY.mp4',
            'hearth_7433275010056222-HEARTBODY.mp4',
            'hearth_8491958419536045-HEARTBODY.mp4',
            'hearth_8641385034049041-HEARTBODY.mp4',
            'hearth_8973987821006428-HEARTBODY.mp4',
            'hearth_9095265574554456-HEARTBODY.mp4',
            'hearth_9303535424634621-HEARTBODY.mp4',
            'hearth_9540549470345112-seedHEARTBODY.mp4',
        ]
    },
    {
        KEY_G: 'lungs',
        KEY_VIDS: [
            'lungs_03849716397122993-LUNGS.mp4',
            'lungs_03926378975519795-LUNGS.mp4',
            'lungs_05843959020692013-LUNGS.mp4',
            'lungs_07014496526883707-LUNGS.mp4',
            'lungs_0987107580213471-LUNGS.mp4',
            'lungs_1361388675596844-LUNGS.mp4',
            'lungs_17927180292408784-LUNGS.mp4',
            'lungs_1822908234890963-LUNGS.mp4',
            'lungs_20190981431059507-LUNGS.mp4',
            'lungs_25611976222926836-LUNGS.mp4',
            'lungs_2960784973349957-LUNGS.mp4',
            'lungs_3156394499603383-LUNGS.mp4',
            'lungs_4858881657130556-LUNGS.mp4',
            'lungs_5064445532351614-LUNGS.mp4',
            'lungs_5351840003920145-LUNGS.mp4',
            'lungs_5647565128465428-LUNGS.mp4',
            'lungs_5830051383767008-LUNGS.mp4',
            'lungs_5847004858651264-LUNGS.mp4',
            'lungs_5897532389009943-LUNGS.mp4',
            'lungs_5915817261595504-LUNGS.mp4',
            'lungs_6321497764529505-seedLUNGS.mp4',
            'lungs_6689918265006289-LUNGS.mp4',
            'lungs_7022904784480481-LUNGS.mp4',
            'lungs_7085293214408721-LUNGS.mp4',
            'lungs_725842508167579-LUNGS.mp4',
            'lungs_7643105618563182-LUNGS.mp4',
            'lungs_7696999663258599-LUNGS.mp4',
            'lungs_8032180358658938-LUNGS.mp4',
            'lungs_8451735986687199-LUNGS.mp4',
            'lungs_8641906132122146-LUNGS.mp4',
            'lungs_982114265618838-LUNGS.mp4',
        ]
    },
    {
        KEY_G: 'shoulder',
        KEY_VIDS: [
            'shoulder_0015823648451052108-SHOULDER.mp4',
            'shoulder_0752008538127451-SHOULDER.mp4',
            'shoulder_08082334261631097-SHOULDER.mp4',
            'shoulder_10026272000810366-SHOULDER.mp4',
            'shoulder_1674492320377452-SHOULDER.mp4',
            'shoulder_24102098476174771-SHOULDER.mp4',
            'shoulder_286056050097667-SHOULDER.mp4',
            'shoulder_30007839824915705-SHOULDER.mp4',
            'shoulder_33477858047544085-SHOULDER.mp4',
            'shoulder_33812397275866224-SHOULDER.mp4',
            'shoulder_372842197823239-SHOULDER.mp4',
            'shoulder_5030770445869783-SHOULDER.mp4',
            'shoulder_5366284286897927-SHOULDER.mp4',
            'shoulder_5375547392845288-SHOULDER.mp4',
            'shoulder_564853635851184-SHOULDER.mp4',
            'shoulder_5688080856457567-SHOULDER.mp4',
            'shoulder_5710597932364143-SHOULDER.mp4',
            'shoulder_6045076691946099-SHOULDER.mp4',
            'shoulder_6101256568316258-SHOULDER.mp4',
            'shoulder_620816222847451-SHOULDER.mp4',
            'shoulder_623613771693494-SHOULDER.mp4',
            'shoulder_6534347652891821-SHOULDER.mp4',
            'shoulder_6647334800031155-SHOULDER.mp4',
            'shoulder_7235871692600837-SHOULDER.mp4',
            'shoulder_7644326857879518-seedSHOULDER.mp4',
            'shoulder_77245841642963-SHOULDER.mp4',
            'shoulder_7957897785038617-SHOULDER.mp4',
            'shoulder_8427343594525847-SHOULDER.mp4',
            'shoulder_8453144931851071-SHOULDER.mp4',
            'shoulder_8966206605069458-SHOULDER.mp4',
            'shoulder_976227036852908-SHOULDER.mp4',
        ]
    },
]
def get_face(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., :36, :].copy()
def get_pose(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., 36:36+8, :].copy()
def get_left_hand(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., 36+8:36+8+21, :].copy()
def get_right_hand(landmark: np.ndarray) -> np.ndarray:
    return landmark[..., -21:, :].copy()
def part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # xs, ys= zip(*landmarks)
    xs= list(map(lambda el: el[0], landmarks))
    ys= list(map(lambda el: el[1], landmarks))
    min_x, min_y= min(xs), min(ys)
    xNeedForward: bool= min_x<0.0
    yNeedForward: bool= min_y<0.0
    if xNeedForward or yNeedForward:
        landmarks= [(
            x -min_x    if xNeedForward    else x,
            y -min_y    if yNeedForward    else y
        ) for x, y in landmarks]
    del xs, ys, min_x, min_y, xNeedForward, yNeedForward

    max_xy: float= max(
        max(x for x, _ in landmarks),
        max(y for _, y in landmarks)
    )
    if max_xy>1:
        landmarks= [(
            x/max_xy,
            y/max_xy
        ) for x, y in landmarks]

    return landmarks
def part2_beSquareRatioOnImage(landmarks: list[tuple[float, float]], original_shape: tuple[int, int]) -> list[tuple[float, float]]:
    height, width= original_shape
    if height==width:
        return landmarks

    if width<height: # portrait, change2withRespect2Height
        scale: float= width/height
        return [(
            x*scale,
            y
        ) for x, y in landmarks]

    # landscape, change2withRespect2Width
    scale: float= height/width
    return [(
        x,
        y*scale
    ) for x, y in landmarks]
def part3_zoomInOutForPadding(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    # zoom in/out for padding be 10% each side with respect to original aspect ratio
    # ie.:
    # ---- top/bottom pad 0.02, leftSide( fromPerspectiveOfSomeoneReadingThis ) pad 0.02: if wx < hy
    # ---- top pad 0.02, leftSide/right pad 0.02: if hy < wx
    # pad: float= 0.05
    pad: float= 4.0/158.0
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    # xs= list(filter(lambda el: el!=0, xs))
    # ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    min_x, min_y=    min(xs), min(ys)
    max_x, max_y=    max(xs), max(ys)
    scale: float= (1  -2*pad)/max(
        max_x -min_x,
        max_y -min_y
    )
    return [(
        # (x -min_x)    *scale    +pad  if x!=0 else x,
        # (y -min_y)    *scale    +pad  if y!=0 else y
        (x -min_x)    *scale    +pad,
        (y -min_y)    *scale    +pad
    ) for x, y in landmarks]
def part4_centerLandmarkVerticallyHorizontally(landmarks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ### 3) center landmark with same aspect ratio as original
    # center horizontally and vertically, since done padding then just
    # move to right/down
    # xs, ys = zip(*landmarks)
    xs: list= list(map(lambda el: el[0], landmarks))
    ys: list= list(map(lambda el: el[1], landmarks))
    # xs= list(filter(lambda el: el!=0, xs))
    # ys= list(filter(lambda el: el!=0, ys))
    if len(xs)==0 or len(ys)==0:
        return landmarks
    shift_x: float=  0.5    -(min(xs) +max(xs))  /2
    shift_y: float=  0.5    -(min(ys) +max(ys))  /2

    return [(
        # x +shift_x  if x!=0 else x,
        # y +shift_y  if y!=0 else y
        x +shift_x,
        y +shift_y
    ) for x, y in landmarks]
def normalizeLandmarks(landmarks: list[tuple[float, float]], original_shape: tuple) -> list[tuple[float, float]]:
    '''
    landmarks is an array eg. of shape (86, 2)
    original_shape is tuple (HEIGHT, WIDTH)
    '''
    # lmark_fph.face_landmarks.landmark
    # lmark_fph.pose_landmarks.landmark
    # lmark_fph.left_hand_landmarks.landmark
    # lmark_fph.right_hand_landmarks.landmark
    # logic resize to --> 480 x 480 x 3
    #     0) all coords be greater than|= 0.0 and less than|= 1.0
    #         a) lmarks x,y overwrite to [0.0, 1.0] only
    #         b) has x < 0.0 then ALL_x+abs(min(x_neg)), ie. move right
    #         b) has y < 0.0 then ALL_y+abs(min(y_neg)), ie. move down
    #         c) ALL_coords_x_y/highest_value
    #         d) eg. 1.74 then ALL_coords_x_y/1.74
    #         e) for all be scaled down with same aspect ratio as orig
    #         f) NOW all( x, y ) are 0.0 to 1.0 value only
    #     1) from old img ratio to new square img ratio
    #         a) if owx < ohy: all_x= all_x* (480*owx/ohy)/480
    #         b) if ohy < owx: all_y= all_y* (480*ohy/owx)/480
    #     2) zoom in/out with padding 0.05 each side( with respecting orig aspect ratio )
    #         a) if too far zoom in
    #         b) if too close zoom out
    #         c) goal lowest val 0.05 both(x,y) .ie padding
    #         d) goal highest val 0.95 both(x,y)
    #         e) ie. max(lm_wx, lm_hy) == 0.9
    #     3) center landmark with same aspect ratio as original
    #         a) min_wx_hy= min( wx, hy ); max_wx_hy= max( wx, hy )
    #         b) min_wx_hy as mn; max_wx_hy as mx
    #         c) if mn is wx, all X +( (mx-mn)/(mx*2) )
    #         d) if mn is hy, all Y +( (mx-mn)/(mx*2) )
    assert 1<len(original_shape) # incorrect use of normalizeLandmarks(...), mandatory 1<len(original_shape)
    landmarks= part1_beGreaterThanOrEqual0_and_lessThanOrEqual1(landmarks)
    landmarks= part2_beSquareRatioOnImage(
        landmarks,
        (original_shape[0], original_shape[1])
    )
    landmarks= part3_zoomInOutForPadding(landmarks)
    landmarks= part4_centerLandmarkVerticallyHorizontally(landmarks)


    return landmarks
def normalizeWorthyLandmarkWrapper(
    landmarks: list[tuple[float, float]], # order --> face_pose_left_right_hand
    original_shape: tuple,
    hasLandmarks: dict
) -> list[tuple[float, float]]:
    if not (hasLandmarks["face"] or hasLandmarks["pose"] or hasLandmarks["left_hand"] or hasLandmarks["right_hand"]):
        return landmarks
    norm: list= []
    if hasLandmarks['face']:
        norm.extend(landmarks[
            :len(WORTHY_FACE_IDX)
        ])
    if hasLandmarks['pose']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
        ])
    if hasLandmarks['left_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
        ])
    if hasLandmarks['right_hand']:
        norm.extend(landmarks[
            len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
        ])
    norm= normalizeLandmarks(landmarks=norm, original_shape=original_shape)

    out_landmark: list= []
    # -- face landmarks --
    if hasLandmarks["face"]:
        out_landmark.extend(norm[:len(WORTHY_FACE_IDX)])
    else:
        out_landmark.extend(np.zeros(
            (
                len(WORTHY_FACE_IDX),
                2
            )).tolist()
        )

    # -- pose landmarks --
    if hasLandmarks["pose"]:
        if hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX)
            ])
        else:
            out_landmark.extend(norm[
                :len(WORTHY_POSE_IDX)
            ])
    else:
        out_landmark.extend(np.zeros(
            (
                len(WORTHY_POSE_IDX),
                2
            )).tolist())

    # -- left hand landmarks --
    if hasLandmarks["left_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
                len(WORTHY_FACE_IDX) +QUANTITY_HAND_LMARK
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
                len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK
            ])
        else:
            out_landmark.extend(norm[
                :QUANTITY_HAND_LMARK
            ])
    else:
        out_landmark.extend(np.zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            )).tolist())

    # -- right hand landmarks --
    if hasLandmarks["right_hand"]:
        if hasLandmarks["face"] and hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["face"] and hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["face"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["pose"] and hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX) +QUANTITY_HAND_LMARK:
            ])
        elif hasLandmarks["face"]:
            out_landmark.extend(norm[
                len(WORTHY_FACE_IDX):
            ])
        elif hasLandmarks["pose"]:
            out_landmark.extend(norm[
                len(WORTHY_POSE_IDX):
            ])
        elif hasLandmarks["left_hand"]:
            out_landmark.extend(norm[
                QUANTITY_HAND_LMARK:
            ])
        else:
            out_landmark.extend(norm)
    else:
        out_landmark.extend(np.zeros(
            (
                QUANTITY_HAND_LMARK,
                2
            )).tolist())

    # out_landmark shape:
    # (
    #     len(WORTHY_FACE_IDX)
    #     +len(WORTHY_POSE_IDX)
    #     +QUANTITY_HAND_LMARK *2,
    #     2
    # )
    return out_landmark # ie. output shape (86, 2)
def isOkPlot(coord: tuple) -> bool:
    # x and y coordinates
    # mandatory be greater than or equal to Zero
    # and less than or equal to One
    return coord[0]<=1.0 and coord[1]<=1.0 and 0.0<=coord[0] and 0.0<=coord[1]
def drawSkeletonImg(image: np.ndarray, \
                    lmark_coordinates: list, \
                    connections_idxs: tuple, \
                    thick: int=2, \
                    color_line: tuple|None=None, \
                    color_dot: tuple|None=None) -> np.ndarray:
    img_wh: dict= {"wx": image.shape[1], "hy": image.shape[0]}


    # drawing the lines between 2 landmark connections
    if color_line!=None or color_dot!=None:
        for lmark_idx_pair in connections_idxs:
            pA: tuple= (
                lmark_coordinates[  lmark_idx_pair[0]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[0]  ][1]  # y
            )
            pB: tuple= (
                lmark_coordinates[  lmark_idx_pair[1]  ][0], # x
                lmark_coordinates[  lmark_idx_pair[1]  ][1]  # y
            )
            if isOkPlot(pA) and isOkPlot(pB):
                if color_dot!=None:
                    cv2.circle(
                        img=image,
                        center=(
                            int(pA[0]*img_wh['wx']),
                            int(pA[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                    cv2.circle(
                        img=image,
                        center=(
                            int(pB[0]*img_wh['wx']),
                            int(pB[1]*img_wh['hy'])
                        ),
                        radius=0,
                        color=color_dot,
                        thickness=thick*2
                    )
                if color_line!=None:
                    cv2.line(
                        img=image,
                        pt1=(int(pA[0]*img_wh['wx']), int(pA[1]*img_wh['hy'])),
                        pt2=(int(pB[0]*img_wh['wx']), int(pB[1]*img_wh['hy'])),
                        color=color_line,
                        thickness=thick
                    )
            else:
                raise NotImplementedError("Has landmark_coordinate<0.0 or 1.0<landmark_coordinate which is not allowed, it should be 0.0<= landmark_coordinate <=1.0, on both x and y coordinates")
            del pA
            del pB
    return image
def extract_lm_from_mph(lmark_mph, orig_shape: tuple) -> list[tuple[float, float]]:
    landmark__face_pose_left_right_hand: list= np.zeros((
        len(WORTHY_FACE_IDX) +len(WORTHY_POSE_IDX) +(QUANTITY_HAND_LMARK*2),
        2
    )).tolist()
    if lmark_mph.face_landmarks!=None \
        or lmark_mph.pose_landmarks!=None \
        or lmark_mph.left_hand_landmarks!=None \
        or lmark_mph.right_hand_landmarks!=None:
        landmark__face_pose_left_right_hand= list()
        # here possible -2.0<= i[1].x <=2.0, mostly on pose
        # here possible -2.0<= i[1].y <=2.0, mostly on pose
        # that's why next force be 0.0<= all <=1.0

        # ---- face landmarks ----
        if lmark_mph.face_landmarks != None:
            for idx, el in enumerate(lmark_mph.face_landmarks.landmark):
                if idx in WORTHY_FACE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((len(WORTHY_FACE_IDX), 2)).tolist())

        # ---- pose landmarks ----
        if lmark_mph.pose_landmarks != None:
            for idx, el in enumerate(lmark_mph.pose_landmarks.landmark):
                if idx in WORTHY_POSE_IDX:
                    landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((len(WORTHY_POSE_IDX), 2)).tolist())

        # ---- left hand landmarks ----
        if lmark_mph.left_hand_landmarks != None:
            for el in lmark_mph.left_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((QUANTITY_HAND_LMARK, 2)).tolist())

        # ---- right hand landmarks ----
        if lmark_mph.right_hand_landmarks != None:
            for el in lmark_mph.right_hand_landmarks.landmark:
                landmark__face_pose_left_right_hand.append((  el.x, el.y  ))
        else:
            landmark__face_pose_left_right_hand.extend(np.zeros((QUANTITY_HAND_LMARK, 2)).tolist())


        landmark__face_pose_left_right_hand= normalizeWorthyLandmarkWrapper(
            landmark__face_pose_left_right_hand,
            orig_shape,
            {
                'face': lmark_mph.face_landmarks != None,
                'pose': lmark_mph.pose_landmarks != None,
                'left_hand': lmark_mph.left_hand_landmarks != None,
                'right_hand': lmark_mph.right_hand_landmarks != None,
            }
        )
    # return list of length 86, each is tuple of length 2
    return landmark__face_pose_left_right_hand
def drawFacePoseHandFromLM(img_write_to: np.ndarray, landmark: list[tuple[float, float]], hasLandmark: dict) -> np.ndarray:
    # ---- face landmarks ----
    if hasLandmark[KEY_FACE]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_face(np.array(landmark)).tolist(),
            connections_idxs=FACE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 153, 0), # 153/255= 0.6
        )

    # ---- pose landmarks ----
    if hasLandmark[KEY_POSE]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_pose(np.array(landmark)).tolist(),
            connections_idxs=POSE_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(0, 0, 153), # 153/255= 0.6
        )

    # ---- left hand landmarks ----
    if hasLandmark[KEY_LHAND]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_left_hand(np.array(landmark)).tolist(),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(255, 255, 255)
        )

    # ---- right hand landmarks ----
    if hasLandmark[KEY_RHAND]:
        img_write_to= drawSkeletonImg(
            image=img_write_to,
            lmark_coordinates=get_right_hand(np.array(landmark)).tolist(),
            connections_idxs=HAND_CONNECTIONS,
            thick=1,
            color_dot=None,
            color_line=(153, 204, 204), # 204/255= 0.8
        )

    # HERE ORDER OF LANDMARKS
    # order of landmarks [...face..., ...pose..., ...left_hand..., ...right_hand...]
    # return tuple(ndarray, list_of_shape_86_2)
    return img_write_to




def init_dirs() -> None:
    if LANDMARK_DIR.exists():
        raise NotImplementedError(f"Please delete {LANDMARK_DIR}, this will be created for you.")
    if SKELETON_DIR.exists():
        raise NotImplementedError(f"Please delete {SKELETON_DIR}, this will be created for you.")
    LANDMARK_DIR.mkdir()
    SKELETON_DIR.mkdir()
def has_atleast_1hand(mph_result) -> bool:
    return mph_result.face_landmarks != None and \
           mph_result.pose_landmarks != None and \
           ( mph_result.left_hand_landmarks != None or \
           mph_result.right_hand_landmarks != None )
def extract_lm_from_video(video_filename: str) -> tuple:
    '''
    video_filename, eg. file.mp4
    --------------
    return a tuple of (out_landmarks, out_annotations)
    '''
    video_data: cv2.VideoCapture= cv2.VideoCapture(f"{VIDEO_DIR /video_filename}")
    out_landmarks: list= []
    out_annotations: list= []
    for _ in range(int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))):
        has_img, image= video_data.read()
        if has_img:
            image= cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            mph_result: Any= MPH_fph.process(image)
            norm_landmarks: list[tuple[float, float]]= extract_lm_from_mph(mph_result, orig_shape=image.shape) # list of shape (86, 2)
            if len(out_landmarks) == 0 or has_atleast_1hand(mph_result):
                out_landmarks.append(norm_landmarks)
                out_annotations.append({
                    KEY_FACE: mph_result.face_landmarks != None,
                    KEY_POSE: mph_result.pose_landmarks != None,
                    KEY_LHAND: mph_result.left_hand_landmarks != None,
                    KEY_RHAND: mph_result.right_hand_landmarks != None,
                })
            elif not has_atleast_1hand(mph_result):
                out_landmarks.append(np.array(out_landmarks[-1]).copy().tolist())
                out_annotations.append(dict(out_annotations[-1]).copy())

    return (out_landmarks, out_annotations)
def process_videos_ds() -> tuple:
    ds_landmark: list= []
    ds_skeleton: list= []
    for a_gloss in VIDEO_LIST:
        ds_landmark.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        ds_skeleton.append({
            KEY_G: a_gloss[KEY_G],
            KEY_VIDS: [],
        })
        for idx_video, a_video in enumerate(a_gloss[KEY_VIDS]):
            video_landmarks, video_annotations= extract_lm_from_video(a_video)
            pfolder_name: str= f"{a_video[:-4]}"
            landmark_abs_folder: Path= LANDMARK_DIR /pfolder_name
            skeleton_abs_folder: Path= SKELETON_DIR /pfolder_name
            landmark_abs_folder.mkdir()
            skeleton_abs_folder.mkdir()
            split_what: str= VAL_TRAIN if idx_video<int(RATIO4TRAIN*len(a_gloss[KEY_VIDS])) else VAL_TEST
            ds_landmark[-1][KEY_VIDS].append({
                KEY_FILE: a_video,
                KEY_SPLIT: split_what,
                KEY_LANDMARK: [],
            })
            ds_skeleton[-1][KEY_VIDS].append({
                KEY_FILE: a_video,
                KEY_SPLIT: split_what,
                KEY_SKELETON: [],
            })
            for idx_image in range(len(video_annotations)):
                landmark_skeleton_prefix: str= str(idx_image+1).zfill(8)
                landmark= video_landmarks[     idx_image ]
                annotation= video_annotations[ idx_image ]
                with open(f"{landmark_abs_folder /landmark_skeleton_prefix}.npy", 'wb') as f:
                    np.save(f, landmark)
                cv2.imwrite(
                    filename=f"{skeleton_abs_folder /landmark_skeleton_prefix}.jpg",
                    img=drawFacePoseHandFromLM(
                        img_write_to=np.zeros( (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8 ),
                        landmark=landmark,
                        hasLandmark=annotation,
                    ),
                )
                ds_landmark[-1][KEY_VIDS][-1][KEY_LANDMARK].append({
                    KEY_FILE: f"{landmark_skeleton_prefix}.npy",
                    KEY_FACE: annotation[KEY_FACE],
                    KEY_POSE: annotation[KEY_POSE],
                    KEY_LHAND: annotation[KEY_LHAND],
                    KEY_RHAND: annotation[KEY_RHAND],
                })
                ds_skeleton[-1][KEY_VIDS][-1][KEY_SKELETON].append({
                    KEY_FILE: f"{landmark_skeleton_prefix}.jpg",
                    KEY_FACE: annotation[KEY_FACE],
                    KEY_POSE: annotation[KEY_POSE],
                    KEY_LHAND: annotation[KEY_LHAND],
                    KEY_RHAND: annotation[KEY_RHAND],
                })
    return (ds_landmark, ds_skeleton)
def main() -> None:
    init_dirs()
    ds_landmark, ds_skeleton= process_videos_ds()
    with open(f"{DS_DIR /"investigate_landmark.json"}", 'w') as f:
        writeJson(ds_landmark, f, indent=4)
    with open(f"{DS_DIR /"investigate_skeleton.json"}", 'w') as f:
        writeJson(ds_skeleton, f, indent=4)
if __name__=="__main__":
    main()
