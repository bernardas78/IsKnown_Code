# Make SOM clusters:
#   Input: Activation files (made by InterClassSimilarity\adhoc_dist_mat.py
#   Output: SOM clusterers

from Orange.projection import som
import pickle
import os
import time
from Globals.globalvars import Glb
from InterClassSimilarity_SOM.som_common import loadActivations


#set_name = "Test"
#set_name = "Val"
#set_name = "Train"

def make_som_clusterers( hier_lvl, dim_size, n_iters):
    #dim_size = 15 #8
    l_rate = 0.5
    #n_iters = 20


    # SOM clusterer
    som_filename = os.path.join(Glb.results_folder, "som_clusterer_{}x{}_hier{}".format(dim_size, dim_size, hier_lvl) )

    # Load train activations
    train_orange_tab,_ = loadActivations("Train", hier_lvl=hier_lvl)

    # Train SOM
    mysom = som.SOM(dim_x=dim_size, dim_y=dim_size, hexagonal=True)
    now = time.time()
    mysom.fit(x=train_orange_tab.X, n_iterations=n_iters, learning_rate=l_rate)
    print("Trained SOM for {} seconds".format(time.time() - now))

    # Save for later inference
    pickle.dump (mysom, open(som_filename, 'wb'))

