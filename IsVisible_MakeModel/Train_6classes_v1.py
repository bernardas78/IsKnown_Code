import Model_6classes_c5plus_d3_v1 as m_6classes_c5plus_d3_v1
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from Globals.globalvars import Glb, Glb_Iterators
import os
from datetime import date

modelVersions_dic = {
    "Model_6classes_c5plus_d3_v1": m_6classes_c5plus_d3_v1.prepModel
}

def trainModel(epochs,bn_layers, dropout_layers, l2_layers,
               padding, target_size, dense_sizes,
               architecture, conv_layers_over_5, use_maxpool_after_conv_layers_after_5th, version, load_existing):

    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    #   bn_layers - list of indexes of Dense layers (-1 and down) and CNN layers (1 and up) where Batch Norm should be applied
    #   dropout_layers - list of indexes of Dense layers (-1 and down) where Dropout should be applied
    #   bn_layers - list of indexes of Dense layers (-1 and down) where L2 regularization should be applied
    #   padding - changed to "same" to keep 2^n feature map sizes
    #   dense_sizes - dictionary of dense layer sizes (cnt of neurons)
    #   architecture - one of:  Model_6classes_c4_d3_v1, Model_6classes_c5_d2_v1, Model_6classes_c5_d3_v1
    #   conv_layers_over_5 - number of convolutional layers after 5th
    #   use_maxpool_after_conv_layers_after_5th - list of boolean values whether to use maxpooling after 5th layer
    #   version - used to name a learning curve file
    #   load_existing - whether to load an existing model file
    # Returns:
    #   model: trained Keras model
    #
    # To call:
    #   model = Train_v1.trainModel(epochs=20)

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    #target_size = 224
    batch_size = 32
    #datasrc = "visible"

    # Manually copied to C: to speed up training
    data_dir = os.path.join(Glb.images_folder, "Bal_v14", "Ind-0")
    data_dir_train = os.path.join(data_dir, "Train")
    data_dir_val = os.path.join(data_dir, "Val")
    data_dir_test = os.path.join(data_dir, "Test")

    train_iterator = Glb_Iterators.get_iterator(data_dir_train,"div255")
    val_iterator = Glb_Iterators.get_iterator(data_dir_val,"div255")
    test_iterator = Glb_Iterators.get_iterator(data_dir_test,"div255")

    Softmax_size = len (train_iterator.class_indices)
    dense_sizes["d-1"] = Softmax_size

    model_file_name = os.path.join(Glb.results_folder,
                                   "model_clsf_from_isVisible_{}.h5".format(date.today().strftime("%Y%m%d")))
    lc_filename = os.path.join(Glb.results_folder,
                               "lc_clsf_from_isVisible_{}.csv".format(date.today().strftime("%Y%m%d")))
    # Create or load model
    if not load_existing:
        print ("Creating model")
        prepModel = modelVersions_dic[architecture]
        prep_model_params = {
            "input_shape": (target_size,target_size,3),
            "bn_layers": bn_layers,
            "dropout_layers": dropout_layers,
            "l2_layers": l2_layers,
            "padding": padding,
            "dense_sizes": dense_sizes,
            "conv_layers_over_5": conv_layers_over_5,
            "use_maxpool_after_conv_layers_after_5th": use_maxpool_after_conv_layers_after_5th
        }
        model = prepModel (**prep_model_params)
    else:
        print ("Loading model")
        #model_file_name = r"J:\Visible_models\6class\model_6classes_v" + str(version) + ".h5"
        model = load_model(model_file_name)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001), # default LR: 0.001
                      metrics=['accuracy'])

    print (model.summary())

    callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=1, mode='max',
                                           restore_best_weights=True)
    callback_csv_logger = CSVLogger(lc_filename, separator=",", append=False)

    mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_accuracy', mode='max')

    model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                    validation_data=val_iterator, validation_steps=len(val_iterator),
                    callbacks=[callback_csv_logger, callback_earlystop, mcp_save]) #

    print("Evaluation on test set (1 frame)")
    test_metrics = model.evaluate(test_iterator)
    print("Test: {}".format(test_metrics))

    print("Evaluation on validation set (1 frame)")
    val_metrics = model.evaluate(val_iterator)
    print("Val: {}".format(val_metrics))

    return model