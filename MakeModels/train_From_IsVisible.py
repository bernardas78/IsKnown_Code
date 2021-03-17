# Trains a model for 200 epochs: 12x12 shifts

#from MakeModels import model_is_visible as makeModel_is_visible
#from MakeModels import model_resnet as makeModel_resnet
import model_resnet as makeModel_resnet

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from datetime import date
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from globalvars import Glb

#isvisible_model_version = sys.argv[1]
#hier_lvl = int(sys.argv[2]) # Hierarchy level: 0: ind. product "12345"; ...; 4: cateogory "1"
#aff_aug_lvl = sys.argv[3]
#val_acc_name = sys.argv[4]
#model_id = sys.argv[5]

def trainModel(epochs,
               isvisible_model_version,
               hier_lvl,
               aff_aug_lvl,
               val_acc_name,
               model_id):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns:
    #   model: trained Keras model


    # model file
    model_file_name = os.path.join (Glb.results_folder, "model_isvisible_v{}_aff{}_Ind-{}_model{}_{}.h5".format(isvisible_model_version, aff_aug_lvl, hier_lvl, model_id, date.today().strftime("%Y%m%d") ) )

    # learning curve file
    lc_file_name = os.path.join (Glb.results_folder, "lc_isvisible_v{}_aff{}_Ind-{}_model{}_{}.csv".format(isvisible_model_version, aff_aug_lvl, hier_lvl, model_id, date.today().strftime("%Y%m%d") ) )

    metrics_file_name = os.path.join (Glb.results_folder, "metrics.csv" )

    # data folder
    data_folder = os.path.join (Glb.images_folder, "affAug{}_v{}".format(aff_aug_lvl,isvisible_model_version), "Ind-{}".format (hier_lvl) )


    train_data_folder = os.path.join (data_folder, "Train")
    val_data_folder = os.path.join (data_folder, "Val")
    test_data_folder = os.path.join (data_folder, "Test")


    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)

    #target_size = 256
    target_size = 224

    #data_gen = ImageDataGenerator(rescale=1. / 255)
    data_gen = ImageDataGenerator(preprocessing_function = resnet_preprocess_input)

    f_make_iterator = lambda data_folder: data_gen.flow_from_directory(
        directory=data_folder,
        target_size=(target_size, target_size),
        batch_size=32,
        shuffle=False,
        class_mode='categorical')
    train_iterator = f_make_iterator(train_data_folder)
    val_iterator = f_make_iterator(val_data_folder)
    test_iterator = f_make_iterator(test_data_folder)

    Softmax_size = len(train_iterator.class_indices)

    #model = makeModel_is_visible.prepModel(target_size=target_size, Softmax_size=Softmax_size)

    if epochs==0:
        model = load_model ( model_file_name)
    else:
        model = makeModel_resnet.prepModel(train_d1=False, train_d2=True, Softmax_size=Softmax_size)

        model.summary()

        # prepare a validation data generator, used for early stopping
        callback_earlystop = EarlyStopping(monitor=val_acc_name, min_delta=0.0001, patience=10, verbose=1, mode='max', restore_best_weights=True )
        callback_csv_logger = CSVLogger(lc_file_name, separator=",", append=False)

        mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor=val_acc_name, mode='max')

        model.fit(train_iterator, steps_per_epoch=len(train_iterator),
                  validation_data=val_iterator, validation_steps=len(val_iterator),
                  epochs=epochs, verbose=2,
                  callbacks=[callback_earlystop, mcp_save, callback_csv_logger])

    # Loading best saved model (for python 3.6 and older keras only)
    #model_file_name = "A:\\IsKnown_Results\\model_isvisible_v14_Ind-0_20210109.h5"
    #model = load_model(model_file_name)

    #print("Evaluation on train set (1 frame)")
    #train_metrics = model.evaluate_generator(train_iterator)
    #print ("Train: {}".format(train_metrics))

    print("Evaluation on validation set (1 frame)")
    val_metrics = model.evaluate_generator(val_iterator)
    print ("Val: {}".format(val_metrics))

    print("Evaluation on test set (1 frame)")
    test_metrics = model.evaluate_generator(test_iterator)
    print ("Test: {}".format(test_metrics))

    column_names = ['isVisible_version','Ind-X','aff_aug_lvl','model_id', 'date_trained', 'val_acc', 'val_loss', 'test_acc', 'test_loss']
    df_metrics = pd.DataFrame(columns=column_names,
                                data=[np.hstack([isvisible_model_version, hier_lvl, aff_aug_lvl, model_id, date.today().strftime("%Y%m%d"),
                                                 val_metrics[1], val_metrics[0], test_metrics[1], test_metrics[0]])] )
    df_metrics.to_csv(metrics_file_name, index=False, header=True, mode='a')

    return model

#trainModel(epochs=0)
#trainModel(epochs=200)