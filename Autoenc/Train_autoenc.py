from Autoenc import Model_autoenc_v1 as m_autoenc_v1
from Autoenc import Model_autoenc_v2_xtraconv as m_autoenc_v2
from Autoenc import Model_autoenc_v3_bn as m_autoenc_v3
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import date
import os
from Globals.globalvars import Glb

def trainModel(**argv):

    autoenc_version = argv["autoenc_version"]


    model_file_name = os.path.join(Glb.results_folder, "model_autoenc_{}_v{}.h5".format( date.today().strftime("%Y%m%d"), str(autoenc_version) ) )

    epochs=100
    target_size = 256
    batch_size = 32
    #datasrc = "visible"

    # Balanced dataset
    data_dir = os.path.join(Glb.images_folder, "Bal_v14", "Ind-0")

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
    if autoenc_version == 1:
        prepModel_autoenc = m_autoenc_v1.prepModel_autoenc
    elif autoenc_version == 2:
        prepModel_autoenc = m_autoenc_v2.prepModel_autoenc
    elif autoenc_version == 3:
        prepModel_autoenc = m_autoenc_v3.prepModel_autoenc
    else:
        raise Exception("Unknown autoenc_version. Train_autoenc_v1.py")

    prep_model_params = {
        "input_shape": (target_size,target_size,3),
    }
    model = prepModel_autoenc (**prep_model_params)

    callback_earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=20, verbose=1, mode='min', restore_best_weights=True)
    lc_filename = os.path.join(Glb.results_folder, 'lc_autoenc_{}.csv'.format(date.today().strftime("%Y%m%d")))
    callback_csv_logger = CSVLogger(lc_filename, separator=",", append=False)
    mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_loss', mode='min')

    model.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                        validation_data=val_iterator, validation_steps=len(val_iterator),
                        callbacks=[callback_csv_logger, callback_earlystop, mcp_save])

    return model

