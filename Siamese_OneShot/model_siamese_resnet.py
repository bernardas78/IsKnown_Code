# Prepares a simple model
#
# To run:
#   model = m_resnet.prepModel()
# To load a trained model:
#   model = load_model("D:\ILSVRC14\models\model_v55.h5", custom_objects={'top_5': m_vgg.top_5})

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Subtract, Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam


def prepModel():

    # Load pretrained model - except the last softmax layer
    base_model = ResNet50()
    print ("base_model.layers[0].input_shape: {}".format(base_model.layers[0].input_shape))

    input_shape = (224, 224, 3)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    encoded_l = base_model(left_input)
    encoded_r = base_model(right_input)

    subtracted = Subtract()([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(subtracted)

    # this is the model we will train
    model = Model(inputs=[left_input, right_input], outputs=prediction)

    # first: train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001), #'adam', # default LR: 0.001
                  metrics=['accuracy'])

    return model

