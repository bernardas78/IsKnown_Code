# Prepares autoencoder and classifier-based-on-encoder architecture

import tensorflow.keras.models
from tensorflow.keras.layers import Dropout, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Conv2DTranspose, Input, Dense

################################################
#  Autoencoder stuff
###############################################
def encoder (input_img):
    # 1st - encoder
    x = Convolution2D(6, (3, 3), padding="same")(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 2nd - encoder
    x = Convolution2D(8, (3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 3rd - encoder
    x = Convolution2D(16, (3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 4th - encoder
    x = Convolution2D(32, (3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 5th - encoder
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 6th - encoder
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # trivial layers: after Flatten - get activations
    #unflat_shape = x.shape[1:]
    #x = Flatten()(x)

    return x#, unflat_shape

def decoder ( x ): #, unflat_shape ):
    #x = Reshape( unflat_shape )(flat_img)
    #x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)

    # -6th decoder
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = Activation('relu')(x)

    # -5th decoder
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
    x = Activation('relu')(x)

    # -4th decoder
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
    x = Activation('relu')(x)

    # -3rd decoder
    x = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same")(x)
    x = Activation('relu')(x)

    # -2nd decoder
    x = Conv2DTranspose(6, (3, 3), strides=(2,2), padding="same")(x)
    x = Activation('relu')(x)

    # -1st decoder
    x = Conv2DTranspose(3, (3, 3), strides=(2,2), padding="same")(x)
    x = Activation('sigmoid')(x)
    return x

def prepModel_autoenc( **argv ):
    input_shape = argv["input_shape"]
    input_img = Input(shape=input_shape)
    model = tensorflow.keras.Model(input_img, decoder(encoder(input_img)) )
    print (model.summary())
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[])
    return model

