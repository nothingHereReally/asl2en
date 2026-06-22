from math import ceil
from keras.src.activations.activations import ReLU
from keras.src.layers import Add, Attention, Flatten, Reshape, Dense, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32, float64


from .lmark_constant import LANDMARK_SHAPE, QUANTITY_FRAME, LEN_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]),
    dtype=float32,
    name='batch_vid',
)
x= Reshape(
    target_shape=(QUANTITY_FRAME, -1),
)(data_in) # now shape be (22, 86*2) or (22, 172)




att_q_h1= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h1= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h1= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h2= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h2= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h2= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h3= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h3= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h3= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h4= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h4= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h4= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h5= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h5= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h5= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h6= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h6= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h6= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h7= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h7= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h7= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h8= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h8= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h8= TimeDistributed(Dense(
    units=ceil(LANDMARK_SHAPE[0]),
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

# att_q_h9= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_v_h9= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_k_h9= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
#
# att_q_h10= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_v_h10= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_k_h10= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
#
# att_q_h11= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_v_h11= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_k_h11= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
#
# att_q_h12= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_v_h12= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)
# att_k_h12= TimeDistributed(Dense(
#     units=ceil(LANDMARK_SHAPE[0]),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# ))(x)

att_h1= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_1",
)([att_q_h1, att_v_h1, att_k_h1])
att_h2= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_2",
)([att_q_h2, att_v_h2, att_k_h2])
att_h3= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_3",
)([att_q_h3, att_v_h3, att_k_h3])
att_h4= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_4",
)([att_q_h4, att_v_h4, att_k_h4])
att_h5= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_5",
)([att_q_h5, att_v_h5, att_k_h5])
att_h6= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_6",
)([att_q_h6, att_v_h6, att_k_h6])
att_h7= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_7",
)([att_q_h7, att_v_h7, att_k_h7])
att_h8= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
    name="att_8",
)([att_q_h8, att_v_h8, att_k_h8])
# att_h9= Attention(
#     use_scale=True,
#     dropout=0.1,
#     dtype=float64,
#     name="att_9",
# )([att_q_h9, att_v_h9, att_k_h9])
# att_h10= Attention(
#     use_scale=True,
#     dropout=0.1,
#     dtype=float64,
#     name="att_10",
# )([att_q_h10, att_v_h10, att_k_h10])
# att_h11= Attention(
#     use_scale=True,
#     dropout=0.1,
#     dtype=float64,
#     name="att_11",
# )([att_q_h11, att_v_h11, att_k_h11])
# att_h12= Attention(
#     use_scale=True,
#     dropout=0.1,
#     dtype=float64,
#     name="att_12",
# )([att_q_h12, att_v_h12, att_k_h12])

x= Add()([att_h1, att_h2, att_h3, att_h4, att_h5, att_h6, att_h7, att_h8])
# x= Add()([att_h1, att_h2, att_h3, att_h4, att_h5, att_h6, att_h7, att_h8, att_h9, att_h10, att_h11, att_h12])




ann= Flatten()(x)
# ann= Dense(
#     units=int((QUANTITY_FRAME*LANDMARK_SHAPE[0])//2),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# )(ann)
# ann= Dense(
#     units=int((QUANTITY_FRAME*LANDMARK_SHAPE[0])//5),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# )(ann)
# ann= Dense(
#     units=int((QUANTITY_FRAME*LANDMARK_SHAPE[0])//10),
#     activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
#     dtype=float64,
# )(ann)


data_out = Dense(LEN_GLOSS, activation=softmax, name='batch_class')(ann)
