# Calculates mean, sigma of known classes
#   of pre-last dense layer of given a classifier
#   using train data only
#   result: .\meansigmas.csv of structure: class, mean, sigma
#

train_dir = "C:\\RetelLectImages\\Train"

# dictinary with key=class and value = (means, sigmas)
results_dic = {}

from Prototype_preLastDense_Dist import common as cm
import pickle
import os
import numpy as np

# Load model
model = cm.get_model()
prelast_dense_layer = cm.get_prelast_dense(model)

# Extract layer's output shape
pre_last_dense_out_shape = int(prelast_dense_layer.output.shape[1]) # shape is (m,n), where m-number of samples; n-number neurons

for classs in os.listdir(train_dir):
    print ("Processing class {}".format(classs) )
    train_class_dir = "\\".join ([train_dir, classs])

    # init array of all activations; shape [m,n], m - #samples; n - #neurons in pre-last layer
    all_activations_preLast = np.empty((0,pre_last_dense_out_shape))

    # collect activations of training images (this class)
    i=0
    for file_name in os.listdir(train_class_dir):
        img_preped = cm.prepareImage ( os.path.join ( train_class_dir,file_name ), cm.get_target_size(model) )
        img_activations_preLast = cm.get_layer_activations (model, img_preped, prelast_dense_layer)
        # add last image activations to all activations
        all_activations_preLast = np.vstack ( [ all_activations_preLast, img_activations_preLast])
        if i % 20==0:
            print ("Processed {} images".format(i) )
        i += 1

    #mus and sigmas for each neuron
    mus = np.mean(all_activations_preLast, axis=0)
    sigmas = np.std(all_activations_preLast, axis=0)
    #print (mus, sigmas)

    # add to results dictionary
    results_dic[classs] = ( mus, sigmas )

 #   # flush results
 #   df_metrics.to_csv(results_csv, header=None, index=None, mode='a')
results_filehandler = open( cm.get_meansigmas_dic_filename(), 'wb')
pickle.dump(results_dic, results_filehandler)
results_filehandler.close()
print ("Results saved to file {}".format (cm.get_meansigmas_dic_filename() ))
