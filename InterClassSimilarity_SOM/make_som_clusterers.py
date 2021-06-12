# Make SOM clusters:
#   Input: Activation files (made by InterClassSimilarity\adhoc_dist_mat.py
#   Output: SOM clusterers

from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.projection import som
import Orange
import pickle
import os
import time
import numpy as np
from Globals.globalvars import Glb
from InterClassSimilarity_SOM.som_common import loadActivations, makeOrangeTable


#set_name = "Test"
#set_name = "Val"
#set_name = "Train"

def make_som_clusterers( hier_lvl, dim_size, n_iters, incl_filenames):
    #dim_size = 15 #8
    l_rate = 0.5
    #n_iters = 20


    # SOM clusterer
    som_filename = os.path.join(Glb.results_folder, "som_clusterer_{}x{}_hier{}".format(dim_size, dim_size, hier_lvl) )

    # Load train activations
    #train_orange_tab,_ = loadActivations("Train", hier_lvl=hier_lvl)
    tuple_contents = loadActivations("Train", hier_lvl=hier_lvl, incl_filenames=incl_filenames)
    act_prelast, lbls = tuple_contents[0], tuple_contents[1]
    #act_prelast,lbls = loadActivations("Train", hier_lvl=hier_lvl)

    # Make Orange table
    train_orange_tab = makeOrangeTable(act_prelast, lbls)

    # Train SOM
    mysom = som.SOM(dim_x=dim_size, dim_y=dim_size, hexagonal=True)
    now = time.time()
    mysom.fit(x=train_orange_tab.X, n_iterations=n_iters, learning_rate=l_rate)
    print("Trained SOM for {} seconds".format(time.time() - now))

    # Save for later inference
    pickle.dump (mysom, open(som_filename, 'wb'))

