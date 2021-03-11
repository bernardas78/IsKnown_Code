# Trains 2 class classifier: Empty/NotEmpty

from DataPrep_EmptyNot import model_is_visible as makeModel_is_visible, model_resnet as makeModel_resnet, model_simple as makeModel_simple
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from datetime import date
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import os

# model file
model_file_name = "A:\IsKnown_Results\\model_emptyNot_{}.h5".format(date.today().strftime("%Y%m%d"))

# learning curve file
lc_file_name = "A:\\IsKnown_Results\\lc_emptyNot_{}.csv".format(date.today().strftime("%Y%m%d"))

# data folder
data_folder = "C:\\EmptyNot"

train_data_folder = os.path.join (data_folder, "Train")
#train_data_folder = os.path.join (data_folder, "Train_NotEmptyUndersampled")
val_data_folder = os.path.join (data_folder, "Val")
test_data_folder = os.path.join (data_folder, "Test")

def trainModel(epochs=1, isresnet=False):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns:
    #   model: trained Keras model

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)

    if isresnet:
        target_size = 224
        data_gen = ImageDataGenerator(preprocessing_function = resnet_preprocess_input)
    else:
        target_size = 256
        data_gen = ImageDataGenerator(rescale=1. / 255)


    f_make_iterator = lambda data_folder: data_gen.flow_from_directory(
        directory=data_folder,
        target_size=(target_size, target_size),
        batch_size=59,
        shuffle=True,   # very important: similar images in batches cause wrong accuracy during fit() to be shown (batches' averagres are biased)
        class_mode='categorical')
    train_iterator = f_make_iterator(train_data_folder)
    val_iterator = f_make_iterator(val_data_folder)
    test_iterator = f_make_iterator(test_data_folder)

    Softmax_size = len(train_iterator.class_indices)

    if isresnet:
        model = makeModel_resnet.prepModel(train_d1=False, train_d2=True, Softmax_size=Softmax_size)
    else:
        #model = makeModel_is_visible.prepModel(target_size=target_size, Softmax_size=Softmax_size)
        model = makeModel_simple.prepModel(target_size=target_size, Softmax_size=Softmax_size)
    model.summary()

    # prepare a validation data generator, used for early stopping
    callback_earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20, verbose=1, mode='max', restore_best_weights=True )
    callback_csv_logger = CSVLogger(lc_file_name, separator=",", append=False)
    mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_acc', mode='max')

    model.fit(train_iterator, steps_per_epoch=len(train_iterator),
              validation_data=val_iterator, validation_steps=len(val_iterator),
              epochs=epochs, verbose=2,
              callbacks=[  mcp_save, callback_earlystop, callback_csv_logger])
    model.save (model_file_name)

    train_iterator1 = f_make_iterator(train_data_folder)
    print("Evaluation on train set (1 frame)")
    train_metrics = model.evaluate_generator(train_iterator1)
    print ("Train1: {}".format(train_metrics))

    # Loading best saved model
    #model = load_model(model_file_name)

    print("Evaluation on validation set (1 frame)")
    val_metrics = model.evaluate_generator(val_iterator)
    print("Evaluation on test set (1 frame)")
    test_metrics = model.evaluate_generator(test_iterator)
    print ("Val: {}; Test: {}".format(val_metrics,test_metrics))

    return model

trainModel(epochs=100, isresnet=False)