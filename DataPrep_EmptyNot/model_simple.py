# Prepares a model same arch as best IsVisible

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

def myBN ():
    return BatchNormalization( momentum=0.95)

def prepModel( target_size, Softmax_size ) :

    padding= "same"

    model = Sequential()

    # 1st CNN
    model.add(Convolution2D(32, (3,3), input_shape=(target_size,target_size,3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2nd CNN
    model.add(Convolution2D(64, (3,3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3rd CNN
    model.add(Convolution2D(128, (3, 3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 5th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 6th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))

    # 7th CNN
    model.add(Convolution2D(256, (3, 3), padding=padding))
    model.add (myBN())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # -3rd dense
    model.add(Dense(256, kernel_regularizer=None, bias_regularizer=None))
    model.add (myBN())
    model.add (Dropout(rate=0.5))
    model.add(Activation('relu'))

    # -2nd dense
    model.add(Dense(128, kernel_regularizer=None, bias_regularizer=None))
    model.add (myBN())
    model.add (Dropout(rate=0.5))
    model.add(Activation('relu'))

    # -1st dense
    model.add(Dense(Softmax_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4), #'adam', # default LR: 1e-3
                  metrics=['accuracy'])

    return model