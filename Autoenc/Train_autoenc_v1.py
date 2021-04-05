from Autoenc import Model_autoenc_v1 as m_autoenc_v1
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import date
import os

def trainModel(epochs):

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
    # Returns:
    #   model: trained Keras model
    #
    # To call:
    #   model = Train_v1.trainModel(epochs=20)

    model_file_name = "A:\\IsKnown_Results\\model_autoenc_{}.h5".format( date.today().strftime("%Y%m%d") )

    crop_range = 1  # number of pixels to crop image (if size is 235, crops are 0-223, 1-224, ... 11-234)
    target_size = 256
    batch_size = 32
    #datasrc = "visible"

    # Balanced dataset
    data_dir = "C:\\IsKnown_Images_IsVisible\\Bal_v14\\Ind-0"

    # Unbalanced dataset
    #data_dir = r"C:\AutoEnc_ImgsTmp\Bal_v14\Ind-0"

    data_dir_train = os.path.join ( data_dir, "Train" )
    data_dir_val = os.path.join ( data_dir, "Val" )

    #data_dir_6classes_test = r"C:\TrainAndVal_6classes\Test"
    #data_dir_6classes_train = r"D:\Visible_Data\4.Augmented\Train"
    #data_dir_6classes_val = r"D:\Visible_Data\4.Augmented\Val"

    # define train and validation sets
    dataGen = ImageDataGenerator(
        rescale=1./255
    )
    f_make_iter = lambda data_dir : dataGen.flow_from_directory(
        directory=data_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='input')

    train_iterator = f_make_iter ( data_dir_train )
    val_iterator = f_make_iter ( data_dir_val )

    # Create model
    prepModel_autoenc = m_autoenc_v1.prepModel_autoenc
    prep_model_params = {
        "input_shape": (target_size,target_size,3),
    }
    model = prepModel_autoenc (**prep_model_params)

    callback_earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=20, verbose=1, mode='min', restore_best_weights=True)
    callback_csv_logger = CSVLogger('A:/IsKnown_results/lc_autoenc_{}.csv'.format(date.today().strftime("%Y%m%d")), separator=",", append=False)
    mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_loss', mode='min')

    model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                        validation_data=val_iterator, validation_steps=len(val_iterator),
                        callbacks=[callback_csv_logger, callback_earlystop, mcp_save])

    return model

trainModel(epochs=100)
