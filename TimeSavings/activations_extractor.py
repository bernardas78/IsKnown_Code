import os
from Common import common as cm
import pickle
from datetime import datetime
import trained_clsfs as tc
import numpy as np
from tensorflow.keras.models import load_model

models_path = "A:\\IsKnown_Results"
data_folder = "C:\\IsKnown_Images_IsVisible"
target_size = 224
is_resnet=True
last_activations_filepattern = "A:\\IsKnown_Results\\lastAct_v{}_Ind-{}_{}.h5"
prelast_activations_filepattern = "A:\\IsKnown_Results\\preLastAct_v{}_Ind-{}_{}.h5"

#####################
def calc_save_last_activations(model, data_iter, last_activations_file_name):

    now = datetime.now()
    (actual_classes, pred_classes, activations) = cm.get_last_dense_activations(model, data_iter)
    print("Got predictions in {} sec".format((datetime.now() - now).total_seconds()))

    act_file = open(last_activations_file_name, 'wb')
    pickle.dump( (actual_classes, pred_classes, activations), act_file)
    act_file.close()
    print("Results saved to file {}".format(last_activations_file_name))
    return

########################
def calc_save_prelast_activations(model, data_iter, prelast_activations_file_name):

    now = datetime.now()
    (actual_classes, pred_classes, activations) = cm.get_prelast_dense_activations(model, data_iter, is_categorical=True)
    print("Got pre-last activations in {} sec".format((datetime.now() - now).total_seconds()))

    act_file = open(prelast_activations_file_name, 'wb')
    pickle.dump((actual_classes, pred_classes, activations), act_file)
    act_file.close()
    print("Results saved to file {}".format(prelast_activations_file_name))
    return


Visible_versions = [14, 62]
Hier_lvls = np.arange(5)
Extract_sets = ["Val", "Test"]

for version in Visible_versions:
    for hier in Hier_lvls:
        for the_set in Extract_sets:

            # Load model
            model_file_key = "v" + str(version) + "_Ind-" + str(hier)
            model_filename = os.path.join( models_path, tc.clsfs[model_file_key] )
            model = load_model(model_filename)

            # data iterator
            set_data_folder = os.path.join( data_folder, "v"+str(version), "Ind-"+str(hier), the_set )
            data_iter = cm.get_data_iterator(data_folder=set_data_folder, target_size=target_size, is_categorical=True, is_resnet=is_resnet)

            # calc/save last activations
            last_activations_filename = last_activations_filepattern.format(version,hier,the_set)
            calc_save_last_activations (model, data_iter, last_activations_filename)

            # calc/save pre-last activations
            prelast_activations_filename = prelast_activations_filepattern.format(version,hier,the_set)
            calc_save_prelast_activations(model, data_iter, prelast_activations_filename)
