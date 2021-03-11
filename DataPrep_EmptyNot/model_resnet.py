# Prepares a simple model
#
# To run:
#   model = m_resnet.prepModel()
# To load a trained model:
#   model = load_model("D:\ILSVRC14\models\model_v55.h5", custom_objects={'top_5': m_vgg.top_5})

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam


def prepModel(train_d1=False, train_d2=False, Softmax_size=10, momentum=0.0, lr=0.01):
    # Load pretrained model - except the last softmax layer
    base_model = ResNet50()
    print ("base_model.layers[0].input_shape: {}".format(base_model.layers[0].input_shape))

    # Remove last (softmax) layer
    #print ("len(base_model.layers): {}".format(len(base_model.layers)))
    #base_model_no_last = Sequential(base_model.layers[:-1])
    print ("len(base_model.layers[:-1]): {}".format(len(base_model.layers[:-1])))
    #base_model.layers.pop()

    # Add xtra dense layer
    #d_prelast = Dense(1000, activation='relu')(base_model.layers[-2].output)

    # Add a softmax layer with my classes to pre-last layer
    #predictions = Dense(Softmax_size, activation='softmax')(d_prelast)
    predictions = Dense(Softmax_size, activation='softmax')(base_model.layers[-2].output)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    #d2_layer_index = len(base_model.layers) - 1
    #base_model.layers[d2_layer_index].trainable = train_d2
    #model.layers[-2].trainable = True#train_d2

    #d1_layer_index = len(base_model.layers) - 2
    #base_model.layers[d1_layer_index].trainable = train_d1
    #model.layers[-3].trainable = True#train_d1

    #print ("model_resnet.py, lr, momentum: " + str(lr) + "," + str(momentum))
    #optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=False)

    # def top_5(y_true, y_pred):
    #    return top_k_categorical_accuracy(y_true, y_pred, k=5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001), #'adam', # default LR: 0.001
                  metrics=['accuracy'])

    return model


def top_5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
