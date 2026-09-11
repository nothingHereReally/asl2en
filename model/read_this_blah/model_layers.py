from keras.src.layers import Dense, Dropout, GlobalAveragePooling1D, Input, Layer, LayerNormalization, MultiHeadAttention, Embedding
from keras.src.activations import softmax
from keras.src.saving import register_keras_serializable
from keras.src import ops


from .lmark_constant import LANDMARK_SHAPE, QUANTITY_FRAME, LEN_GLOSS




# ============================================================
# Custom layer:
# Add landmark positional information
#
# Input:
#   (B, T, 86, D)
#
# Output:
#   (B, T, 86, D)
# ============================================================

@register_keras_serializable()
class LandmarkPositionEmbedding( Layer ):

    def __init__(
        self,
        landmark_count: int,
        embedding_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.landmark_count= landmark_count
        self.embedding_dim= embedding_dim

        self.embedding= Embedding(
            input_dim=landmark_count,
            output_dim=embedding_dim,
            name="landmark_embedding",
        )

    def call(self, inputs):

        positions= ops.arange(
            self.landmark_count
        )

        position_embedding= self.embedding(
            positions
        )

        # Broadcasting:
        #
        # inputs:
        #     (B,T,86,D)
        #
        # position_embedding:
        #     (86,D)
        #
        # result:
        #     (B,T,86,D)

        return inputs + position_embedding

    def get_config(self):

        return {
            **super().get_config(),
            "landmark_count": self.landmark_count,
            "embedding_dim": self.embedding_dim,
        }


# ============================================================
# Custom layer:
# Reshape
#
# (B,T,L,D)
#      ↓
# (B*T,L,D)
#
# Used so spatial attention sees landmarks independently
# for every frame.
# ============================================================

@register_keras_serializable()
class SpatialReshape(Layer):

    def call(self, inputs):

        shape= ops.shape(inputs)

        batch= shape[0]
        frames= shape[1]
        landmarks= shape[2]
        channels= shape[3]

        return ops.reshape(
            inputs,
            (
                batch * frames,
                landmarks,
                channels,
            ),
        )


# ============================================================
# Custom layer:
# Restore spatial shape
#
# (B*T,L,D)
#      ↓
# (B,T,L,D)
# ============================================================

@register_keras_serializable()
class SpatialRestore(Layer):

    def call(self, inputs, original):

        shape= ops.shape(original)

        batch= shape[0]
        frames= shape[1]
        landmarks= shape[2]
        channels= shape[3]

        return ops.reshape(
            inputs,
            (
                batch,
                frames,
                landmarks,
                channels,
            ),
        )


# ============================================================
# Custom layer:
# Spatial Transformer block
#
# Attention is across landmarks.
#
# Input:
#   (B,T,86,D)
#
# Output:
#   (B,T,86,D)
# ============================================================

@register_keras_serializable()
class SpatialTransformerBlock(Layer):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: int= 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim= embedding_dim
        self.num_heads= num_heads
        self.dropout= dropout
        self.mlp_ratio= mlp_ratio

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.norm1= LayerNormalization(
            epsilon=1e-6,
            name="attention_norm",
        )

        self.attention= MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout,
            use_bias=True,
            name="attention",
        )

        self.dropout1= Dropout(
            dropout,
            name="attention_dropout",
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        self.norm2= LayerNormalization(
            epsilon=1e-6,
            name="ffn_norm",
        )

        self.ffn1= Dense(
            embedding_dim * mlp_ratio,
            activation="gelu",
            name="ffn_expand",
        )

        self.ffn_dropout1= Dropout(
            dropout,
            name="ffn_dropout1",
        )

        self.ffn2= Dense(
            embedding_dim,
            name="ffn_project",
        )

        self.ffn_dropout2= Dropout(
            dropout,
            name="ffn_dropout2",
        )

        # ----------------------------------------------------
        # Reshaping
        # ----------------------------------------------------

        self.reshape_spatial= SpatialReshape(
            name="reshape_spatial",
        )

        self.restore_spatial= SpatialRestore(
            name="restore_spatial",
        )

    def call(
        self,
        inputs,
        training=None,
    ):

        # ====================================================
        # (B,T,86,D)
        #       ↓
        # (B*T,86,D)
        # ====================================================

        x= self.reshape_spatial(
            inputs
        )

        # ====================================================
        # Self Attention
        # ====================================================

        residual= x

        x= self.norm1(
            x
        )

        x= self.attention(
            query=x,
            value=x,
            key=x,
            training=training,
        )

        x= self.dropout1(
            x,
            training=training,
        )

        x= residual + x

        # ====================================================
        # Feed Forward Network
        # ====================================================

        residual= x

        x= self.norm2(
            x
        )

        x= self.ffn1(
            x
        )

        x= self.ffn_dropout1(
            x,
            training=training,
        )

        x= self.ffn2(
            x
        )

        x= self.ffn_dropout2(
            x,
            training=training,
        )

        x= residual + x

        # ====================================================
        # (B*T,86,D)
        #       ↓
        # (B,T,86,D)
        # ====================================================

        x= self.restore_spatial(
            x,
            inputs,
        )

        return x

    def get_config(self):

        return {
            **super().get_config(),
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "mlp_ratio": self.mlp_ratio,
        }


# ============================================================
# Custom layer:
# Pool landmarks
#
# (B,T,86,D)
#       ↓
# (B,T,D)
#
# Uses mean instead of sum so the magnitude doesn't depend
# on the number of landmarks.
# ============================================================

@register_keras_serializable()
class LandmarkMeanPooling(Layer):

    def call(self, inputs):

        return ops.mean(
            inputs,
            axis=2,
        )


# ============================================================
# Custom layer:
# Temporal positional embedding
#
# Input:
#   (B,T,D)
#
# Output:
#   (B,T,D)
# ============================================================

@register_keras_serializable()
class TemporalPositionEmbedding(Layer):

    def __init__(
        self,
        frame_count: int,
        embedding_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.frame_count= frame_count
        self.embedding_dim= embedding_dim

        self.embedding= Embedding(
            input_dim=frame_count,
            output_dim=embedding_dim,
            name="temporal_embedding",
        )

    def call(self, inputs):

        positions= ops.arange(
            self.frame_count
        )

        position_embedding= self.embedding(
            positions
        )

        return inputs + position_embedding

    def get_config(self):

        return {
            **super().get_config(),
            "frame_count": self.frame_count,
            "embedding_dim": self.embedding_dim,
        }


# ============================================================
# Custom layer:
# Temporal Transformer block
#
# Input:
#   (B,T,D)
#
# Attention is across frames.
# ============================================================

@register_keras_serializable()
class TemporalTransformerBlock(Layer):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: int= 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim= embedding_dim
        self.num_heads= num_heads
        self.dropout= dropout
        self.mlp_ratio= mlp_ratio

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.norm1= LayerNormalization(
            epsilon=1e-6,
            name="attention_norm",
        )

        self.attention= MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout,
            use_bias=True,
            name="attention",
        )

        self.dropout1= Dropout(
            dropout,
            name="attention_dropout",
        )

        # ----------------------------------------------------
        # Feed Forward
        # ----------------------------------------------------

        self.norm2= LayerNormalization(
            epsilon=1e-6,
            name="ffn_norm",
        )

        self.ffn1= Dense(
            embedding_dim * mlp_ratio,
            activation="gelu",
            name="ffn_expand",
        )

        self.ffn_dropout1= Dropout(
            dropout,
            name="ffn_dropout1",
        )

        self.ffn2= Dense(
            embedding_dim,
            name="ffn_project",
        )

        self.ffn_dropout2= Dropout(
            dropout,
            name="ffn_dropout2",
        )

    def call(
        self,
        inputs,
        training=None,
    ):

        # ====================================================
        # Temporal self attention
        # ====================================================

        residual= inputs

        x= self.norm1(
            inputs
        )

        x= self.attention(
            query=x,
            value=x,
            key=x,
            training=training,
        )

        x= self.dropout1(
            x,
            training=training,
        )

        x= residual + x

        # ====================================================
        # Feed Forward
        # ====================================================

        residual= x

        x= self.norm2(
            x
        )

        x= self.ffn1(
            x
        )

        x= self.ffn_dropout1(
            x,
            training=training,
        )

        x= self.ffn2(
            x
        )

        x= self.ffn_dropout2(
            x,
            training=training,
        )

        x= residual + x

        return x

    def get_config(self):

        return {
            **super().get_config(),
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "mlp_ratio": self.mlp_ratio,
        }


# ============================================================
# INPUT
#
# (B, QUANTITY_FRAME, 86, 2)
# ============================================================

data_in= Input(
    shape=(
        QUANTITY_FRAME,
        LANDMARK_SHAPE[0],
        LANDMARK_SHAPE[1],
    ),
    dtype="float32",
    name="batch_vid",
)


# ============================================================
# Configuration
# ============================================================

EMBED_DIM= 128
NUM_HEADS= 8
DROPOUT= 0.10


# ============================================================
# Coordinate -> embedding
#
# (B,T,86,2)
#       ↓
# (B,T,86,128)
# ============================================================

x= Dense(
    EMBED_DIM,
    activation="gelu",
    name="landmark_projection",
)(data_in)


# ============================================================
# Landmark positional information
#
# (B,T,86,128)
# ============================================================

x= LandmarkPositionEmbedding(
    landmark_count=LANDMARK_SHAPE[0],
    embedding_dim=EMBED_DIM,
    name="landmark_position",
)(x)


# ============================================================
# SPATIAL TRANSFORMER
#
# Attention between the 86 landmarks.
# ============================================================

x= SpatialTransformerBlock(
    embedding_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    mlp_ratio=4,
    name="spatial_transformer_1",
)(x)

x= SpatialTransformerBlock(
    embedding_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    mlp_ratio=4,
    name="spatial_transformer_2",
)(x)


# ============================================================
# Normalize after spatial processing
# ============================================================

x= LayerNormalization(
    epsilon=1e-6,
    name="spatial_output_norm",
)(x)


# ============================================================
# LANDMARK POOLING
#
# (B,T,86,128)
#       ↓
# (B,T,128)
# ============================================================

x= LandmarkMeanPooling(
    name="landmark_mean_pooling",
)(x)


# ============================================================
# TEMPORAL POSITION
#
# (B,T,128)
# ============================================================

x= TemporalPositionEmbedding(
    frame_count=QUANTITY_FRAME,
    embedding_dim=EMBED_DIM,
    name="temporal_position",
)(x)


# ============================================================
# TEMPORAL TRANSFORMER
#
# Attention between frames.
# ============================================================

x= TemporalTransformerBlock(
    embedding_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    mlp_ratio=4,
    name="temporal_transformer_1",
)(x)

x= TemporalTransformerBlock(
    embedding_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    mlp_ratio=4,
    name="temporal_transformer_2",
)(x)


# ============================================================
# Normalize temporal representation
# ============================================================

x= LayerNormalization(
    epsilon=1e-6,
    name="temporal_output_norm",
)(x)


# ============================================================
# Temporal pooling
#
# (B,T,128)
#       ↓
# (B,128)
# ============================================================

x= GlobalAveragePooling1D(
    name="temporal_global_pool",
)(x)


# ============================================================
# Classification
# ============================================================

x= Dense(
    EMBED_DIM,
    activation="gelu",
    name="classifier_projection",
)(x)

x= Dropout(
    0.30,
    name="classifier_dropout",
)(x)

data_out= Dense(
    LEN_GLOSS,
    activation=softmax,
    name="batch_class",
)(x)
