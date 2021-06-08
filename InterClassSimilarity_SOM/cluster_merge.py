import pickle
import os
from InterClassSimilarity_SOM import cluster_merge_functions as somf
from Globals.globalvars import Glb

set_name = "Test"
#set_name = "Train"

dim_size = 8

results_folder = Glb.results_folder
clusters_filename = os.path.join ( results_folder,"{}_clstrs_{}x{}_Orange.tab".format ( set_name, str(dim_size), str(dim_size) ) )
distmat_filename = os.path.join ( results_folder,"{}_distmat_{}x{}.h5".format ( set_name, str(dim_size), str(dim_size) ) )


(pred_winner_neurons, lbls) = pickle.load( open(clusters_filename, 'rb') )

# Merge all classes
#for clstr_id in range( len(lbls) )
purity_impr_mat = somf.hypotheticalMergePurity(pred_winner_neurons, lbls)
purity_impr_mat_filename = os.path.join ( results_folder,"{}_purity_impr_mat_{}x{}.h5".format ( set_name, str(dim_size), str(dim_size) ) )
pickle.dump(purity_impr_mat,open(purity_impr_mat_filename, 'wb'))




