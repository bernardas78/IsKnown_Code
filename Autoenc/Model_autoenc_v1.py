# Prepares autoencoder
#
# To run:
#   model = m_v1.prepModel()

import tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Flatten, Reshape, Conv2DTranspose, Input
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model

def prepModel( **argv ):
#def prepModel( input_shape, bn_layers, dropout_layers, l2_layers, padding, dense_sizes ):
    input_shape = argv["input_shape"]
    #bn_layers = argv["bn_layers"]
    #dropout_layers = argv["dropout_layers"]
    #l2_layers = argv["l2_layers"]
    #padding = argv["padding"]
    #dense_sizes = argv["dense_sizes"]
    #   bn_layers - list of indexes of Dense layers (-1 and down) and CNN layers (1 and up) where Batch Norm should be applied
    #   dropout_layers - list of indexes of Dense layers (-1 and down) where Dropout should be applied
    #   bn_layers - list of indexes of Dense layers (-1 and down) where L2 regularization should be applied
    #   padding - changed to "same" to keep 2^n feature map sizes
    #   dense_sizes - dictionary of dense layer sizes (cnt of neurons)

    print ("Model_autoenc_v1")
    #model = Sequential()


    input_img = Input(shape=input_shape)

## ENCODER
    # 1st - encoder
    #model.add(Convolution2D(8, (3,3), input_shape=input_shape, padding="same"))
    x = Convolution2D(6, (3,3), input_shape=input_shape, padding="same")(input_img)
    #if "c+1" in bn_layers:
    #    model.add (BatchNormalization())
    #model.add(Activation('relu'))
    x = Activation('relu')(x)
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    x = MaxPooling2D(pool_size=(2, 2))(x)

# 2nd - encoder
    #model.add(Convolution2D(16, (3, 3), padding="same"))
    x = Convolution2D(8, (3,3), padding="same")(x)
    #if "c+2" in bn_layers:
    #    model.add (BatchNormalization())
    #model.add(Activation('relu'))
    x = Activation('relu')(x)
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 3rd - encoder
    #model.add(Convolution2D(8, (3,3), input_shape=input_shape, padding="same"))
    x = Convolution2D(16, (3,3), input_shape=input_shape, padding="same")(x)
    #if "c+1" in bn_layers:
    #    model.add (BatchNormalization())
    #model.add(Activation('relu'))
    x = Activation('relu')(x)
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 4th - encoder
    #model.add(Convolution2D(8, (3,3), input_shape=input_shape, padding="same"))
    x = Convolution2D(32, (3,3), input_shape=input_shape, padding="same")(x)
    #if "c+1" in bn_layers:
    #    model.add (BatchNormalization())
    #model.add(Activation('relu'))
    x = Activation('relu')(x)
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 5th - encoder
    #model.add(Convolution2D(128, (3, 3), padding="same"))
    #if "c+5" in bn_layers:
    #    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    # trivial layers: after Flatten - get activations
    #model.add (Flatten())
    #print ("x.shape: {}".format(x.shape[1:]))
    unflat_shape = x.shape[1:]
    x = Flatten()(x)
    #model.add (Reshape( (8,8,128) ))
    x = Reshape( unflat_shape )(x)

    ## DECODER
    # -5th decoder
    #model.add(UpSampling2D(size=(2, 2)))
    #model.add(Convolution2D(128, (3, 3), padding="same"))
    #if "c+5" in bn_layers:
    #    model.add(BatchNormalization())
    #model.add(Activation('relu'))


    # -4th decoder
    # model.add(Conv2DTranspose(16, (3, 3), strides=(2,2), padding="same"))
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)
    # if "c+2" in bn_layers:
    #    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    x = Activation('relu')(x)


    # -3rd decoder
    # model.add(Conv2DTranspose(16, (3, 3), strides=(2,2), padding="same"))
    x = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding="same")(x)
    # if "c+2" in bn_layers:
    #    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    x = Activation('relu')(x)

    # -2nd decoder
    #model.add(Conv2DTranspose(16, (3, 3), strides=(2,2), padding="same"))
    x = Conv2DTranspose(6, (3, 3), strides=(2,2), padding="same")(x)
    #if "c+2" in bn_layers:
    #    model.add(BatchNormalization())
    #model.add(Activation('relu'))
    x = Activation('relu')(x)


    # -1st decoder
    #model.add(Conv2DTranspose(3, (3, 3), strides=(2,2), padding="same"))
    x = Conv2DTranspose(3, (3, 3), strides=(2,2), padding="same")(x)
    #if "c+1" in bn_layers:
    #    model.add(BatchNormalization())
    #model.add(Activation('sigmoid'))
    x = Activation('sigmoid')(x)

    model = tensorflow.keras.Model(input_img, x )

    print (model.summary())
    #plot_model(model, to_file='A:\\IsKnown_Results\\model_plot.png', show_shapes=True, show_layer_names=True)


    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model