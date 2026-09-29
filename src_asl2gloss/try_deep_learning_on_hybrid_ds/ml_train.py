from keras.src.losses import sparse_categorical_crossentropy
from keras.src.models import Model
from keras.src.optimizers import Adam

from .constant import EPOCHS, VAL_TRAIN, VAL_TEST, PROJ_ROOT
from .algo import calculate_steps_needed, get_data_landmark
from .layers import data_in, data_out
from .callbacks import d_lr, sTraining, tf_board


def main():
    model: Model= Model(
        inputs=data_in,
        outputs=data_out
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        x=get_data_landmark(train_val=VAL_TRAIN),
        epochs=EPOCHS,
        callbacks=[d_lr, sTraining, tf_board],
        validation_data=get_data_landmark(train_val=VAL_TEST),
        steps_per_epoch=calculate_steps_needed(VAL_TRAIN),
        validation_steps=calculate_steps_needed(VAL_TEST),
        validation_freq=1
    )
    model_file_name: str= "aslvid2gloss_v68.keras"
    print(f"Quantity images --> {model.input_shape[1]}")
    print(f"Quantity categories --> {model.output_shape[-1]}")
    print(f"model( --> {model_file_name} <-- )")
    model.save(f"{PROJ_ROOT /"model" /model_file_name}")


if __name__=="__main__":
    main()
