from keras.src.activations.activations import ReLU
from keras.src.layers import Add, Attention, Concatenate, Flatten, Normalization, Reshape, Dense, Input, TimeDistributed
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
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h1_1')(x)
q_ann_p1= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h1_3')(q_ann_p1)
q_ann_p1= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h1_4')(q_ann_p1)

v_ann_p1= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h1_1')(x)
v_ann_p1= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h1_3')(v_ann_p1)
v_ann_p1= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h1_4')(v_ann_p1)


q_ann_p2= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h2_1')(x)
q_ann_p2= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h2_3')(q_ann_p2)
q_ann_p2= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h2_4')(q_ann_p2)

v_ann_p2= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h2_1')(x)
v_ann_p2= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h2_3')(v_ann_p2)
v_ann_p2= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h2_4')(v_ann_p2)


q_ann_p3= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h3_1')(x)
q_ann_p3= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h3_3')(q_ann_p3)
q_ann_p3= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h3_4')(q_ann_p3)

v_ann_p3= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h3_1')(x)
v_ann_p3= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h3_3')(v_ann_p3)
v_ann_p3= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h3_4')(v_ann_p3)


q_ann_p4= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h4_1')(x)
q_ann_p4= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h4_3')(q_ann_p4)
q_ann_p4= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h4_4')(q_ann_p4)

v_ann_p4= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h4_1')(x)
v_ann_p4= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h4_3')(v_ann_p4)
v_ann_p4= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h4_4')(v_ann_p4)


q_ann_p5= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h5_1')(x)
q_ann_p5= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h5_3')(q_ann_p5)
q_ann_p5= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h5_4')(q_ann_p5)

v_ann_p5= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h5_1')(x)
v_ann_p5= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h5_3')(v_ann_p5)
v_ann_p5= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h5_4')(v_ann_p5)


q_ann_p6= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h6_1')(x)
q_ann_p6= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h6_3')(q_ann_p6)
q_ann_p6= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h6_4')(q_ann_p6)

v_ann_p6= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h6_1')(x)
v_ann_p6= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h6_3')(v_ann_p6)
v_ann_p6= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h6_4')(v_ann_p6)


q_ann_p7= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h7_1')(x)
q_ann_p7= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h7_3')(q_ann_p7)
q_ann_p7= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h7_4')(q_ann_p7)

v_ann_p7= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h7_1')(x)
v_ann_p7= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h7_3')(v_ann_p7)
v_ann_p7= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h7_4')(v_ann_p7)


q_ann_p8= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h8_1')(x)
q_ann_p8= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h8_3')(q_ann_p8)
q_ann_p8= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='q_ann_h8_4')(q_ann_p8)

v_ann_p8= TimeDistributed(Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h8_1')(x)
v_ann_p8= TimeDistributed(Dense(
    units=32,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h8_3')(v_ann_p8)
v_ann_p8= TimeDistributed(Dense(
    units=8,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0)
),
    name='v_ann_h8_4')(v_ann_p8)


att_h1= Attention(
    name='att_h1',
    use_scale=True
)([q_ann_p1, v_ann_p1])
att_h2= Attention(
    name='att_h2',
    use_scale=True
)([q_ann_p2, v_ann_p2])
att_h3= Attention(
    name='att_h3',
    use_scale=True
)([q_ann_p3, v_ann_p3])
att_h4= Attention(
    name='att_h4',
    use_scale=True
)([q_ann_p4, v_ann_p4])
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
att_h1= Normalization(name='norm_h1')(att_h1)
att_h2= Normalization(name='norm_h2')(att_h2)
att_h3= Normalization(name='norm_h3')(att_h3)
att_h4= Normalization(name='norm_h4')(att_h4)
att_h5= Normalization(name='norm_h5')(att_h5)
att_h6= Normalization(name='norm_h6')(att_h6)
att_h7= Normalization(name='norm_h7')(att_h7)
att_h8= Normalization(name='norm_h8')(att_h8)
conct= Concatenate(
    axis=-1,
    name='concat_16heads'
)([att_h1, att_h2, att_h3, att_h4, att_h5, att_h6, att_h7, att_h8]) # shape is (22, 8*16) is (22, 128)
# conct= LSTM(
#     units=256,
#     name='lstm'
# )(conct)


ann= Flatten(
    name='flat'
)(conct)
ann= Dense(
    units=1024,
    name='ann1024'
)(ann)
ann= Dense(
    units=512,
    name='ann512'
)(ann)
ann= Dense(
    units=256,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='ann256'
)(ann)
ann= Dense(
    units=128,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='ann128'
)(ann)
ann= Dense(
    units=64,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    name='ann64'
)(ann)


data_out = Dense(T10_GLOSS, activation=softmax, name='batch_class')(ann)
