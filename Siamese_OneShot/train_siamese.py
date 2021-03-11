# Trains a model for 200 epochs: 12x12 shifts

from Siamese_OneShot import model_siamese_resnet as m, PairSequence as ps
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from datetime import date
import os
from keras.utils import plot_model

# model file
model_file_name = "A:\\IsKnown_Results\\model_siamese_emptyNot_{}.h5".format(date.today().strftime("%Y%m%d"))

# learning curve file
lc_file_name = "A:\\IsKnown_Results\\lc_siamese_emptyNot_{}.csv".format(date.today().strftime("%Y%m%d"))

# data folder
data_folder = "C:\\EmptyNot"

train_data_folder = os.path.join (data_folder, "Train")
val_data_folder = os.path.join (data_folder, "Val")
test_data_folder = os.path.join (data_folder, "Test")

def trainModel(epochs=1):
    # Trains a model
    #   model = optional parameter; creates new if not passed; otherwise keeps training
    #   epochs - number of max epochs to train (subject to early stopping)
    # Returns:
    #   model: trained Keras model

    #data_gen = ImageDataGenerator(preprocessing_function = resnet_preprocess_input)

    f_make_iterator = lambda data_folder: ps.PairSequence(
        #batch_size=32,
        #target_size=224,
        #is_resnet=True,
        #debug=True,
        data_folder_pos=os.path.join(data_folder,"Empty"),
        data_folder_neg=os.path.join(data_folder,"NotEmpty") )
    train_iterator = f_make_iterator(train_data_folder)
    val_iterator = f_make_iterator(val_data_folder)
    test_iterator = f_make_iterator(test_data_folder)

    siamese_net = m.prepModel()

    siamese_net.summary()
    #plot_model(siamese_net, show_shapes=True, show_layer_names=True)

    # prepare a validation data generator, used for early stopping
    callback_earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=200, verbose=1, mode='max', restore_best_weights=True )
    callback_csv_logger = CSVLogger(lc_file_name, separator=",", append=False)
    mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_acc', mode='max')

    siamese_net.fit(train_iterator, steps_per_epoch=len(train_iterator),
                    validation_data=val_iterator, validation_steps=len(val_iterator),
                    epochs=epochs, verbose=2,
                    callbacks=[ mcp_save, callback_csv_logger])
                    #callbacks = [callback_earlystop, mcp_save, callback_csv_logger])

    # Loading best saved model
    #model = load_model(model_file_name)

    print("Evaluation on validation set (1 frame)")
    val_metrics = siamese_net.evaluate_generator(val_iterator)
    print("Evaluation on test set (1 frame)")
    test_metrics = siamese_net.evaluate_generator(test_iterator)
    print ("Val: {}; Test: {}".format(val_metrics,test_metrics))

    return siamese_net

trainModel(epochs=1000)