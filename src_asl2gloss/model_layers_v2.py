from keras.src.activations.activations import ReLU
from keras.src.layers import LSTM, Add, Attention, BatchNormalization, Concatenate, Reshape, Dense, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32, float64


from .lmark_constant_v2 import LM_Q_FACE, LM_Q_HAND, LM_Q_POSE, QUANTITY_FRAME, T10_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LM_Q_FACE+LM_Q_POSE+(LM_Q_HAND*2), 2),
    dtype=float32,
    name='batch_vid',
)
x= Reshape(
    target_shape=(QUANTITY_FRAME, -1),
    dtype=float64,
    name='reshape'
)(data_in)


q_ann_p1= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h1_1')(x)
# q_ann_p1= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h1_2')(q_ann_p1)
q_ann_p1= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h1_3')(q_ann_p1)
# q_ann_p1= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h1_4')(q_ann_p1)
q_ann_p1= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h1_5')(q_ann_p1)

v_ann_p1= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h1_1')(x)
# v_ann_p1= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h1_2')(v_ann_p1)
v_ann_p1= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h1_3')(v_ann_p1)
# v_ann_p1= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h1_4')(v_ann_p1)
v_ann_p1= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h1_5')(v_ann_p1)


q_ann_p2= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h2_1')(x)
# q_ann_p2= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h2_2')(q_ann_p2)
q_ann_p2= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h2_3')(q_ann_p2)
# q_ann_p2= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h2_4')(q_ann_p2)
q_ann_p2= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h2_5')(q_ann_p2)

v_ann_p2= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h2_1')(x)
# v_ann_p2= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h2_2')(v_ann_p2)
v_ann_p2= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h2_3')(v_ann_p2)
# v_ann_p2= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h2_4')(v_ann_p2)
v_ann_p2= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h2_5')(v_ann_p2)


q_ann_p3= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h3_1')(x)
# q_ann_p3= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h3_2')(q_ann_p3)
q_ann_p3= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h3_3')(q_ann_p3)
# q_ann_p3= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h3_4')(q_ann_p3)
q_ann_p3= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h3_5')(q_ann_p3)

v_ann_p3= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h3_1')(x)
# v_ann_p3= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h3_2')(v_ann_p3)
v_ann_p3= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h3_3')(v_ann_p3)
# v_ann_p3= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h3_4')(v_ann_p3)
v_ann_p3= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h3_5')(v_ann_p3)


q_ann_p4= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h4_1')(x)
# q_ann_p4= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h4_2')(q_ann_p4)
q_ann_p4= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h4_3')(q_ann_p4)
# q_ann_p4= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h4_4')(q_ann_p4)
q_ann_p4= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h4_5')(q_ann_p4)

v_ann_p4= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h4_1')(x)
# v_ann_p4= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h4_2')(v_ann_p4)
v_ann_p4= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h4_3')(v_ann_p4)
# v_ann_p4= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h4_4')(v_ann_p4)
v_ann_p4= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h4_5')(v_ann_p4)


q_ann_p5= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h5_1')(x)
# q_ann_p5= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h5_2')(q_ann_p5)
q_ann_p5= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h5_3')(q_ann_p5)
# q_ann_p5= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h5_4')(q_ann_p5)
q_ann_p5= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h5_5')(q_ann_p5)

v_ann_p5= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h5_1')(x)
# v_ann_p5= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h5_2')(v_ann_p5)
v_ann_p5= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h5_3')(v_ann_p5)
# v_ann_p5= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h5_4')(v_ann_p5)
v_ann_p5= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h5_5')(v_ann_p5)


q_ann_p6= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h6_1')(x)
# q_ann_p6= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h6_2')(q_ann_p6)
q_ann_p6= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h6_3')(q_ann_p6)
# q_ann_p6= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h6_4')(q_ann_p6)
q_ann_p6= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h6_5')(q_ann_p6)

v_ann_p6= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h6_1')(x)
# v_ann_p6= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h6_2')(v_ann_p6)
v_ann_p6= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h6_3')(v_ann_p6)
# v_ann_p6= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h6_4')(v_ann_p6)
v_ann_p6= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h6_5')(v_ann_p6)


q_ann_p7= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h7_1')(x)
# q_ann_p7= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h7_2')(q_ann_p7)
q_ann_p7= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h7_3')(q_ann_p7)
# q_ann_p7= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h7_4')(q_ann_p7)
q_ann_p7= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h7_5')(q_ann_p7)

v_ann_p7= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h7_1')(x)
# v_ann_p7= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h7_2')(v_ann_p7)
v_ann_p7= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h7_3')(v_ann_p7)
# v_ann_p7= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h7_4')(v_ann_p7)
v_ann_p7= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h7_5')(v_ann_p7)


q_ann_p8= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h8_1')(x)
# q_ann_p8= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h8_2')(q_ann_p8)
q_ann_p8= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h8_3')(q_ann_p8)
# q_ann_p8= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q_ann_h8_4')(q_ann_p8)
q_ann_p8= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q_ann_h8_5')(q_ann_p8)

v_ann_p8= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h8_1')(x)
# v_ann_p8= TimeDistributed(Dense(
#     units=64,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h8_2')(v_ann_p8)
v_ann_p8= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h8_3')(v_ann_p8)
# v_ann_p8= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v_ann_h8_4')(v_ann_p8)
v_ann_p8= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v_ann_h8_5')(v_ann_p8)


att_h1= Attention(
    name='att_h1',
    use_scale=True
)([q_ann_p1, v_ann_p1, v_ann_p1])
att_h2= Attention(
    name='att_h2',
    use_scale=True
)([q_ann_p2, v_ann_p2, v_ann_p2])
att_h3= Attention(
    name='att_h3',
    use_scale=True
)([q_ann_p3, v_ann_p3, v_ann_p3])
att_h4= Attention(
    name='att_h4',
    use_scale=True
)([q_ann_p4, v_ann_p4, v_ann_p4])
att_h5= Attention(
    name='att_h5',
    use_scale=True
)([q_ann_p5, v_ann_p5])
att_h6= Attention(
    name='att_h6',
    use_scale=True
)([q_ann_p6, v_ann_p6])
att_h7= Attention(
    name='att_h7',
    use_scale=True
)([q_ann_p7, v_ann_p7])
att_h8= Attention(
    name='att_h8',
    use_scale=True
)([q_ann_p8, v_ann_p8])


att_h1= Add(name='add_h1')([q_ann_p1, v_ann_p1, att_h1])
att_h2= Add(name='add_h2')([q_ann_p2, v_ann_p2, att_h2])
att_h3= Add(name='add_h3')([q_ann_p3, v_ann_p3, att_h3])
att_h4= Add(name='add_h4')([q_ann_p4, v_ann_p4, att_h4])
att_h5= Add(name='add_h5')([q_ann_p5, v_ann_p5, att_h5])
att_h6= Add(name='add_h6')([q_ann_p6, v_ann_p6, att_h6])
att_h7= Add(name='add_h7')([q_ann_p7, v_ann_p7, att_h7])
att_h8= Add(name='add_h8')([q_ann_p8, v_ann_p8, att_h8])
att_h1= BatchNormalization(name='norm_h1')(att_h1)
att_h2= BatchNormalization(name='norm_h2')(att_h2)
att_h3= BatchNormalization(name='norm_h3')(att_h3)
att_h4= BatchNormalization(name='norm_h4')(att_h4)
att_h5= BatchNormalization(name='norm_h5')(att_h5)
att_h6= BatchNormalization(name='norm_h6')(att_h6)
att_h7= BatchNormalization(name='norm_h7')(att_h7)
att_h8= BatchNormalization(name='norm_h8')(att_h8)
conct= Concatenate(
    axis=-1,
    name='concat_8heads'
)([att_h1, att_h2, att_h3, att_h4, att_h5, att_h6, att_h7, att_h8]) # shape is (22, 8*8) is (22, 64)
# conct= SpatialDropout1D(
#     rate=0.45,
#     name='do_8heads'
# )(conct)


q2_ann_p1= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h1_2')(conct)
q2_ann_p1= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h1_3')(q2_ann_p1)
# q2_ann_p1= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q2_ann_h1_4')(q2_ann_p1)
q2_ann_p1= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h1_5')(q2_ann_p1)

v2_ann_p1= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h1_2')(conct)
v2_ann_p1= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h1_3')(v2_ann_p1)
# v2_ann_p1= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v2_ann_h1_4')(v2_ann_p1)
v2_ann_p1= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h1_5')(v2_ann_p1)


q2_ann_p2= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h2_2')(conct)
q2_ann_p2= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h2_3')(q2_ann_p2)
# q2_ann_p2= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q2_ann_h2_4')(q2_ann_p2)
q2_ann_p2= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h2_5')(q2_ann_p2)

v2_ann_p2= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h2_2')(conct)
v2_ann_p2= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h2_3')(v2_ann_p2)
# v2_ann_p2= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v2_ann_h2_4')(v2_ann_p2)
v2_ann_p2= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h2_5')(v2_ann_p2)


q2_ann_p3= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h3_2')(conct)
q2_ann_p3= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h3_3')(q2_ann_p3)
# q2_ann_p3= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q2_ann_h3_4')(q2_ann_p3)
q2_ann_p3= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h3_5')(q2_ann_p3)

v2_ann_p3= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h3_2')(conct)
v2_ann_p3= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h3_3')(v2_ann_p3)
# v2_ann_p3= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v2_ann_h3_4')(v2_ann_p3)
v2_ann_p3= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h3_5')(v2_ann_p3)


q2_ann_p4= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h4_2')(conct)
q2_ann_p4= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h4_3')(q2_ann_p4)
# q2_ann_p4= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='q2_ann_h4_4')(q2_ann_p4)
q2_ann_p4= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='q2_ann_h4_5')(q2_ann_p4)

v2_ann_p4= TimeDistributed(Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h4_2')(conct)
v2_ann_p4= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h4_3')(v2_ann_p4)
# v2_ann_p4= TimeDistributed(Dense(
#     units=16,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ),
#     name='v2_ann_h4_4')(v2_ann_p4)
v2_ann_p4= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
),
    name='v2_ann_h4_5')(v2_ann_p4)


att2_h1= Attention(
    name='att2_h1',
    use_scale=True
)([q2_ann_p1, v2_ann_p1, v2_ann_p1])
att2_h2= Attention(
    name='att2_h2',
    use_scale=True
)([q2_ann_p2, v2_ann_p2, v2_ann_p2])
att2_h3= Attention(
    name='att2_h3',
    use_scale=True
)([q2_ann_p3, v2_ann_p3, v2_ann_p3])
att2_h4= Attention(
    name='att2_h4',
    use_scale=True
)([q2_ann_p4, v2_ann_p4, v2_ann_p4])


att2_h1= Add(name='add2_h1')([q2_ann_p1, v2_ann_p1, att2_h1])
att2_h2= Add(name='add2_h2')([q2_ann_p2, v2_ann_p2, att2_h2])
att2_h3= Add(name='add2_h3')([q2_ann_p3, v2_ann_p3, att2_h3])
att2_h4= Add(name='add2_h4')([q2_ann_p4, v2_ann_p4, att2_h4])
att2_h1= BatchNormalization(name='norm2_h1')(att2_h1)
att2_h2= BatchNormalization(name='norm2_h2')(att2_h2)
att2_h3= BatchNormalization(name='norm2_h3')(att2_h3)
att2_h4= BatchNormalization(name='norm2_h4')(att2_h4)
conct= Concatenate(
    axis=-1,
    name='concat_4heads'
)([att2_h1, att2_h2, att2_h3, att2_h4]) # shape is (22, 4*8) is (22, 32)
# conct= SpatialDropout1D(
#     rate=0.2,
#     name='do_4heads'
# )(conct)
ann= LSTM(
    units=32
)(conct)


# ann= Flatten(
#     name='flat_4heads'
# )(conct)
# ann= Dense(
#     units=512,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
#     name='ann2_512'
# )(ann)
# ann= Dense(
#     units=128,
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
#     name='ann2_128'
# )(ann)
ann= Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
    name='ann2_32'
)(ann)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(ann)
