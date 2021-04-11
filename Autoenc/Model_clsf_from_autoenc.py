import Model_autoenc_v1, Model_autoenc_v2_xtraconv, Model_autoenc_v3_bn
import tensorflow.keras.models
from tensorflow.keras.layers import Dropout, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Conv2DTranspose, Input, Dense


################################################
#  Classifier stuff
###############################################
def fc_1 (latent_img,softmax_size):
    x = Flatten() (latent_img)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    out = Dense(softmax_size, activation='softmax')(x)
    return out

def fc_2 (latent_img,softmax_size):
    x = Flatten() (latent_img)
    x = Dense(128, activation='relu')(x)

    out = Dense(softmax_size, activation='softmax')(x)
    return out

def prepModel_clsf( **argv ):
    input_shape = argv["input_shape"]
    softmax_size = argv["softmax_size"]
    fc_version = argv["fc_version"]
    autoenc_version = argv["autoenc_version"]

    if fc_version == 1:
        fc = fc_1
    elif fc_version == 2:
        fc = fc_2
    else:
        raise Exception("Unknown fc_version. Model_clsf_from_autoenc.py")

    if autoenc_version == 1:
        Model_autoenc = Model_autoenc_v1
    elif autoenc_version == 2:
        Model_autoenc = Model_autoenc_v2_xtraconv
    elif autoenc_version == 3:
        Model_autoenc = Model_autoenc_v3_bn
    else:
        raise Exception("Unknown autoenc_version. Model_clsf_from_autoenc.py")

    input_img = Input(shape=input_shape)
    model = tensorflow.keras.Model(input_img, fc(Model_autoenc.encoder(input_img), softmax_size) )

    print (model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
