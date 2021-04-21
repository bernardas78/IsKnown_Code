from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.backend import function
#import tensorflow.keras.layers.core
import tensorflow.python.keras.layers.core
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

def get_data_iterator (data_folder,
                       target_size,
                       is_categorical=True,
                       is_resnet=False):
    data_gen = \
        ImageDataGenerator(preprocessing_function=resnet_preprocess_input) \
        if is_resnet \
        else ImageDataGenerator(rescale=1. / 255)
    data_iterator = data_gen.flow_from_directory(
        directory=data_folder,
        target_size=(target_size, target_size),
        batch_size=32,
        shuffle=False,
        class_mode='categorical' if is_categorical else None)
    return data_iterator

def get_pred_actual_classes (model, data_iterator):
    predictions = model.predict_generator(data_iterator, steps=len(data_iterator))
    pred_classes = np.argmax(predictions, axis=1)
    return (pred_classes, data_iterator.classes)

def get_last_dense_activations (model, data_iterator):
    predictions = model.predict_generator(data_iterator, steps=len(data_iterator))
    pred_classes = np.argmax(predictions, axis=1)
    return (data_iterator.classes, pred_classes, predictions)


def get_preds_top1 (model, data_iterator):
    predictions = model.predict_generator(data_iterator, steps=len(data_iterator))
    preds_top1 = np.max(predictions, axis=1)
    return preds_top1

def get_preds (model, data_iterator):
    predictions = model.predict_generator(data_iterator, steps=len(data_iterator))
    return predictions

# Given model, extract pre-last dense layer activations of data_iterator
def get_prelast_dense_activations(model, data_iterator, is_categorical):
    # discover pre-last dense layer

    # in resnet, 2nd last is tensorflow.python.keras.layers.pooling.GlobalAveragePooling2; doesn't really matter - just take 2nd last
    # dense_layer_ids = np.where(['keras.layers.core.Dense' in str(type(layer)) for layer in model.layers])[0]
    dense_layer_ids = np.arange( len(model.layers) )

    prelast_dense_layer = model.layers [ dense_layer_ids[-2] ]
    prelast_func_activation = function([model.input], [prelast_dense_layer.output])

    output_activations = np.zeros ( (0, prelast_dense_layer.output_shape[1] ) )
    classes = np.array([], dtype=int)
    pred_classes = np.array([], dtype=int)

    # get activations
    for i in range(len(data_iterator)):
        #print("Getting activations of X shaped ", X.shape)
        print ("get_prelast_dense_activations {}/{}".format (i, len(data_iterator)))
        X = data_iterator[i][0] if is_categorical else data_iterator[i]
        output_activations = np.concatenate ( (output_activations, prelast_func_activation([X])[0] ) )

        #print ("X.shape:{}".format(X.shape))
        #print ("np.argmax(model.predict(X).shape: {}",format(np.argmax(model.predict(X)).shape))
        pred_classes = np.concatenate ( (pred_classes, np.argmax(model.predict(X), axis=1) ) )

        # Preserve classes together with actications
        new_classes = np.argmax(data_iterator[i][1], axis=1) if is_categorical else np.ones( X.shape[0] )*-1 #fake class -1 if not categorical
        classes = np.concatenate ( (classes, new_classes ) )

    return (classes,pred_classes,output_activations)

