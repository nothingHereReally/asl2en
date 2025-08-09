from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from .lmark_constant import PROJ_ROOT


d_lr: ReduceLROnPlateau= ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=1,
    verbose=1,
    mode='max',
    min_delta=0.01,
    cooldown=1,
    min_lr=1.0e-8
)
sTraining: EarlyStopping= EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
    verbose=1,
    mode='min'
)
tf_board: TensorBoard= TensorBoard(
    log_dir=f"{PROJ_ROOT}model/tfboard_logs",
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch'
)

