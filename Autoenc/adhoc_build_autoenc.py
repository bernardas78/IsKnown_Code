from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Flatten, Reshape, Conv2DTranspose


model = Sequential()

model.add(Convolution2D(8, (3,3), input_shape=(100,100,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, (3,3), input_shape=(100,100,3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2DTranspose(16, (3,3), strides=2, padding="same"))
#model.add(Conv2DTranspose(8, (3,3), strides=2, padding="same"))

model.summary()
