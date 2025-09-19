from keras.src.activations.activations import ReLU
from keras.src.layers import Add, Attention, Flatten, Normalization, Reshape, Dense, Input, TimeDistributed
from keras.src.activations import softmax
from numpy import float32, float64


from .lmark_constant import LANDMARK_SHAPE, QUANTITY_FRAME, TRAIN_GLOSS


data_in= Input(
    shape=(QUANTITY_FRAME, LANDMARK_SHAPE[0], LANDMARK_SHAPE[1]),
    dtype=float32,
    name='batch_vid',
)
x= Reshape(
    target_shape=(QUANTITY_FRAME, -1),
    dtype=float64,
    name='reshape'
)(data_in)






att_q_h1= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h1= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h1= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h2= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h2= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h2= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h3= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h3= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h3= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h4= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h4= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h4= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h5= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h5= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h5= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h6= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h6= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h6= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h7= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h7= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h7= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)

att_q_h8= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_v_h8= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)
att_k_h8= TimeDistributed(Dense(
    units=512,
    activation=ReLU(negative_slope=0.0, max_value=256.0, threshold=0.0),
    dtype=float64,
))(x)


att_h1= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h1, att_v_h1, att_k_h1])
att_h2= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h2, att_v_h2, att_k_h2])
att_h3= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h3, att_v_h3, att_k_h3])
att_h4= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h4, att_v_h4, att_k_h4])
att_h5= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h5, att_v_h5, att_k_h5])
att_h6= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h6, att_v_h6, att_k_h6])
att_h7= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h7, att_v_h7, att_k_h7])
att_h8= Attention(
    use_scale=True,
    dropout=0.1,
    dtype=float64,
)([att_q_h8, att_v_h8, att_k_h8])

x= Add()([att_h1, att_h2, att_h3, att_h4, att_h5, att_h6, att_h7, att_h8])
x= Normalization(axis=-1)(x)




ann= Flatten()(x)


data_out = Dense(TRAIN_GLOSS, activation=softmax, name='batch_class')(ann)
