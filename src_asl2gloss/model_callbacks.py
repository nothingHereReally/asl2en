from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from .lmark_constant import PROJ_ROOT


d_lr: ReduceLROnPlateau= ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=1,
    verbose=1,
    mode='max',
    min_delta=0.0001,
    cooldown=1,
    min_lr=1.0e-8
)
sTraining: EarlyStopping= EarlyStopping(
    monitor='loss',
    min_delta=0.0001,
    patience=2,
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

