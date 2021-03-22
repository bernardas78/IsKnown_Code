# Trains a model for 200 epochs: 12x12 shifts

#from MakeModels import model_is_visible as makeModel_is_visible
#from MakeModels import model_resnet as makeModel_resnet
import model_resnet as makeModel_resnet

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
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
               #aff_aug_lvl,
               emptyness_prefix,
               val_acc_name,
               model_id):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns:
    #   model: trained Keras model


    # model file
    #model_file_name = os.path.join (Glb.results_folder, "model_isvisible_v{}_aff{}_Ind-{}_model{}_{}.h5".format(isvisible_model_version, aff_aug_lvl, hier_lvl, model_id, date.today().strftime("%Y%m%d") ) )
    model_file_name = os.path.join (Glb.results_folder, "model_empty{}_isvisible_v{}_model{}_{}.h5".format(emptyness_prefix, isvisible_model_version, model_id, date.today().strftime("%Y%m%d") ) )

    # learning curve file
    lc_file_name = os.path.join (Glb.results_folder, "lc_empty{}_isvisible_v{}_model{}_{}.csv".format(emptyness_prefix, isvisible_model_version, model_id, date.today().strftime("%Y%m%d") ) )

    metrics_file_name = os.path.join (Glb.results_folder, "metrics.csv" )

    # data folder
    data_folder = os.path.join (Glb.images_folder, "{}_v{}".format(emptyness_prefix,isvisible_model_version), "Ind-{}".format (hier_lvl) )
    #data_folder = os.path.join (Glb.images_folder, "affAug{}_v{}".format(aff_aug_lvl,isvisible_model_version), "Ind-{}".format (hier_lvl) )
    #data_folder = os.path.join (Glb.images_folder, "affAug{}_v{}_small".format(aff_aug_lvl,isvisible_model_version), "Ind-{}".format (hier_lvl) )


    train_data_folder = os.path.join (data_folder, "Train")
    val_data_folder = os.path.join (data_folder, "Val")
    test_data_folder = os.path.join (data_folder, "Test")


    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)

    #target_size = 256
    target_size = 224

    #data_gen = ImageDataGenerator(rescale=1. / 255)
    data_gen = ImageDataGenerator(preprocessing_function = resnet_preprocess_input)
    #data_gen = ImageDataGenerator()

    #f_make_ds = lambda data_folder: tf.keras.preprocessing.image_dataset_from_directory(
    #    data_folder,
    #    #validation_split=0.2,
    #    #subset="training",
    #    label_mode="categorical",
    #    seed=123,
    #    image_size=(target_size, target_size),
    #    batch_size=Glb.batch_size )
    #train_ds = f_make_ds(train_data_folder)
    #val_ds = f_make_ds(val_data_folder)
    #test_ds = f_make_ds(test_data_folder)

    #Softmax_size = len(train_ds.class_names)

    #AUTOTUNE = tf.data.AUTOTUNE
    #train_ds = train_ds.cache( os.path.join( Glb.cache_folder, 'cache_train.bin') ).prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.cache( os.path.join( Glb.cache_folder, 'cache_val.bin') ).prefetch(buffer_size=AUTOTUNE)
    #train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    #resnet_preprocess_value = np.zeros( (1,target_size,target_size,3), dtype=np.float32)
    #resnet_preprocess_value[:,:,:,0] = -103.939
    #resnet_preprocess_value[:,:,:,1] = -116.779
    #resnet_preprocess_value[:,:,:,2] = -123.68
    #resnet_preprocess_tensor = tf.constant (resnet_preprocess_value, tf.float32)

    ## Throws OOM after a few epoxhs, so commenting out
    #resnet_preprocess_fun = lambda  x: tf.math.add(tf.reverse(x,axis=[3]),resnet_preprocess_tensor)
    ##resnet_preprocess_fun = resnet_preprocess_input


    #preprocess_fun_full = lambda x, y: (resnet_preprocess_fun(x), y)
    #train_iterator = train_ds.map(preprocess_fun_full)
    #val_iterator = val_ds.map(preprocess_fun_full)
    #test_iterator = test_ds.map(preprocess_fun_full)

    #train_iterator = train_ds.map(lambda x, y: (resnet_preprocess_input(x), y))
    #val_iterator = val_ds.map(lambda x, y: (resnet_preprocess_input(x), y))


    #train_iterator = train_ds.map(lambda x, y: (tf.reverse(x,axis=3)+resnet_preprocess_value, y))
    #val_iterator = val_ds.map(lambda x, y: (tf.reverse(x,axis=3)+resnet_preprocess_value, y))
    #train_iterator = train_ds.map(lambda x, y: (x,y))
    #val_iterator = val_ds.map(lambda x, y: (x,y))

    f_make_iterator = lambda data_folder: data_gen.flow_from_directory(
        directory=data_folder,
        target_size=(target_size, target_size),
        batch_size=Glb.batch_size ,
        shuffle=False,
        class_mode='categorical')
    train_iterator = f_make_iterator(train_data_folder)
    val_iterator = f_make_iterator(val_data_folder)
    test_iterator = f_make_iterator(test_data_folder)

    Softmax_size = len(test_iterator.class_indices)

    if epochs==0:
        model = load_model ( model_file_name)
    else:
        model = makeModel_resnet.prepModel(train_d1=False, train_d2=True, Softmax_size=Softmax_size)

        model.summary()

        # prepare a validation data generator, used for early stopping
        callback_earlystop = EarlyStopping(monitor=val_acc_name, min_delta=0.0001, patience=10, verbose=1, mode='max', restore_best_weights=True )
        callback_csv_logger = CSVLogger(lc_file_name, separator=",", append=False)

        mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor=val_acc_name, mode='max')
        #tb_callback = TensorBoard(log_dir=Glb.tensorboard_logs_folder, profile_batch=(4,8) )

        steps_per_train_epoch = len(train_iterator)
        steps_per_val_epoch = len(val_iterator)

        print ("Starting fit()")
        model.fit(train_iterator, steps_per_epoch=steps_per_train_epoch,
                  validation_data=val_iterator, validation_steps=steps_per_val_epoch,
                  epochs=epochs, verbose=2,
                  callbacks=[callback_earlystop, mcp_save, callback_csv_logger])    #, tb_callback

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