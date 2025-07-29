from keras.src.models import Model


from .modelasl2gloss_layers import data_in, data_out

if __name__=="__main__":


    model: Model= Model(
        inputs=data_in,
        outputs=[data_out]
    )
    model.summary()
