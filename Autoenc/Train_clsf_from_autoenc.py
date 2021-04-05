from Autoenc import Model_autoenc_v1 as m_autoenc_v1
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from datetime import date
import os

autoenc_file_path = "A:\\IsKnown_Results\\model_autoenc_20210402.h5"
model_file_name = "A:\\IsKnown_Results\\model_clsf_from_autoenc_20210402.h5"

data_path = "C:\\IsKnown_Images_IsVisible\\Bal_v14\\Ind-0"

epochs=100
target_size = 256
batch_size = 32
# datasrc = "visible"

# Manually copied to C: to speed up training
data_dir_train = os.path.join (data_path, "Train")
data_dir_val = os.path.join (data_path, "Val")
data_dir_test = os.path.join (data_path, "Test")
# data_dir_6classes_test = r"C:\TrainAndVal_6classes\Test"
# data_dir_6classes_train = r"D:\Visible_Data\4.Augmented\Train"
# data_dir_6classes_val = r"D:\Visible_Data\4.Augmented\Val"

# define train and validation sets
dataGen = ImageDataGenerator ( rescale=1./255 )
f_make_iter = lambda data_dir : dataGen.flow_from_directory(
        directory=data_dir,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')
train_iterator = f_make_iter ( data_dir_train )
val_iterator = f_make_iter ( data_dir_val )
test_iterator = f_make_iter ( data_dir_test )

Softmax_size = len (train_iterator.class_indices)

print ("Autoencoder loading..." )
autoenc = load_model ( autoenc_file_path )
print (autoenc.summary())

# Find flat layer
#flat_layer_ind = [layer.name for layer in autoenc.layers].index('flatten')
#flat_layer = autoenc.layers [ flat_layer_ind ]

# Construct classifier model
prepModel_clsf = m_autoenc_v1.prepModel_clsf
prep_model_params = {
    "input_shape": (target_size,target_size,3),
    "softmax_size": Softmax_size
}
model_clsf = prepModel_clsf (**prep_model_params)

# Copy weights from autoencoder
flat_layer_ind = [layer.name for layer in model_clsf.layers].index('flatten')
for l1,l2 in zip(model_clsf.layers[:flat_layer_ind],autoenc.layers[0:flat_layer_ind]):
    l1.set_weights(l2.get_weights())
    l1.trainable = False


# vldDataGen = dg_v1.prepDataGen( target_size=target_size, test=True, batch_size=128, datasrc=datasrc )
callback_earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=20, verbose=1, mode='max',
                                       restore_best_weights=True)
callback_csv_logger = CSVLogger('A:/IsKnown_results/lc_clsf_from_autoenc_{}.csv'.format(date.today().strftime("%Y%m%d")), separator=",", append=False)

mcp_save = ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_accuracy', mode='max')

model_clsf.fit(train_iterator, steps_per_epoch=len(train_iterator), epochs=epochs, verbose=2,
                validation_data=val_iterator, validation_steps=len(val_iterator),
                callbacks=[callback_csv_logger, callback_earlystop, mcp_save]) #

print("Evaluation on test set (1 frame)")
test_metrics = model_clsf.evaluate_generator(test_iterator)
print("Test: {}".format(test_metrics))

print("Evaluation on validation set (1 frame)")
val_metrics = model_clsf.evaluate_generator(val_iterator)
print("Val: {}".format(val_metrics))

