# Distance from Top1 prediction
#   Euclidean
#   Of pre-last dense layer
#   Data: validation TopX + Unknown (Other) classes
#
# Result: ".\\distances.csv" of str: 'actual','top1','dist_eucl','dist_mahalanobis'

other_dir = "C:\\RetelectImages\\NotTop24"
val_dir = "C:\\RetelectImages\\Val"

from Prototype_preLastDense_Dist import common as cm
import pickle
import os
import numpy as np
import pandas as pd
import scipy.spatial.distance

# Load model
model = cm.get_model()
prelast_dense_layer = cm.get_prelast_dense(model)

# Load dictionary: key=class and value = (means, sigmas)
mus_sigmas_dic = pickle.load( open( cm.get_meansigmas_dic_filename(), 'rb') )

# get pandas frame of products
df_products = cm.get_products()

# overwrite results file
column_names = ['actual','top1','dist_eucl','dist_mahalanobis','dist_cosine','dist_mahalanobis_excl_0']
df_distances = pd.DataFrame (columns=column_names )
df_distances.to_csv( cm.get_distances_filename(), index=False, header=True, mode='w')

def process_leaf_folder (classs, imgs_folder):
    print ("processing class {}".format(classs) )
    i=0
    for filename in os.listdir (imgs_folder):
        img_preped = cm.prepareImage(os.path.join(imgs_folder, filename), cm.get_target_size(model))

        # get pre-last dense activations
        img_activations_preLast = cm.get_layer_activations(model, img_preped, prelast_dense_layer)

        # top 1 guess and related top 1's mus, sigmas
        top1 = np.argmax(cm.get_layer_activations(model, img_preped, model.layers[-1]))
        #print ("Top1"+str(top1))
        top1_class = str(df_products["barcode"][top1])
        #print ("Top1_class"+str(top1_class))
        #print ("top 1 guess:" + top1_class)
        (top1_mus, top1_sigmas) = mus_sigmas_dic [top1_class]

        # Calculate euclidean distance and mahalandobis distance
        dist = np.sum(np.square( (img_activations_preLast - top1_mus) ))
        # How many sigmas in each dimension varies from mean? (0 sigmas are added epsilon)
        dist_mahalanobis = np.sum(np.square( (img_activations_preLast - top1_mus) / (top1_sigmas + 1e-7) ))
        # cosine distance
        dist_cosine = scipy.spatial.distance.cosine ( img_activations_preLast,top1_mus )
        # How many sigmas in each dimension varies from mean? (0 sigmas are excluded)
        non_zero_sigmas = np.where (top1_sigmas>1e-7)[0]
        dist_mahalanobis_excl_0 = np.sum(np.square( (img_activations_preLast[:,non_zero_sigmas] - top1_mus[non_zero_sigmas]) / (top1_sigmas[non_zero_sigmas]) ))
        #print ("Actual: {}, Top1_pred: {}, Dist: {}, Dist_mah: {}".format (classs, top1_class, dist,dist_mahalanobis) )

        # Result to file
        df_distances = pd.DataFrame(
            data=[np.hstack([classs, top1_class, dist,dist_mahalanobis,dist_cosine,dist_mahalanobis_excl_0])],
            columns=column_names)
        df_distances.to_csv( cm.get_distances_filename(), header=None, index=None, mode='a')

        print ("Processed {} files".format(i)) if i%20==0 else 0
        i+=1

# process validation folder
for classs in os.listdir (val_dir):
    process_leaf_folder (classs, os.path.join( val_dir, classs ) )

# process Other (not TopX) folder
process_leaf_folder ("", other_dir )