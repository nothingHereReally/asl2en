from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam

from .constant import (
    EPOCHS,
    VAL_TRAIN,
    VAL_TEST,
    STEPS_TRAIN,
    STEPS_TEST,
    PROJ_ROOT,
    Q_PERVIDEO_NOTSTRICT,
)
from .algo import calculate_steps, dataset_generator
from .layers import layer_in, layer_out
from .callbacks import d_lr, sTraining, tf_board


def main():
    model: Model= Model(
        inputs=layer_in,
        outputs=layer_out
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sparse_categorical_crossentropy,
        metrics=['accuracy'],
    )
    model.summary()
    if 0 < Q_PERVIDEO_NOTSTRICT:
        model.fit(
            x=dataset_generator(train_test=VAL_TRAIN),
            epochs=EPOCHS,
            callbacks=[d_lr, sTraining, tf_board],
            validation_data=dataset_generator(train_test=VAL_TEST),
            steps_per_epoch=STEPS_TRAIN,
            validation_steps=STEPS_TEST,
            validation_freq=1
        )
    else:
        model.fit(
            x=dataset_generator(train_test=VAL_TRAIN),
            epochs=EPOCHS,
            callbacks=[d_lr, sTraining, tf_board],
            validation_data=dataset_generator(train_test=VAL_TEST),
            steps_per_epoch=calculate_steps(VAL_TRAIN),
            validation_steps=calculate_steps(VAL_TEST),
            validation_freq=1
        )
    model_file_name: str= "aslvid2gloss_v83.keras"
    print(f"Quantity images --> {model.input_shape[1]}")
    print(f"Quantity categories --> {model.output_shape[-1]}")
    print(f"model( --> {model_file_name} <-- )")
    model.save(f"{PROJ_ROOT /"model" /model_file_name}")


if __name__=="__main__":
    main()
