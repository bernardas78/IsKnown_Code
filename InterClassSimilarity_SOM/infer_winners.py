#   Input: SOM clusterers
#   Output: SOM clusters (assignment of each sample to a cluster); in file
#       For Train, Val, Test

import pickle
import os
from InterClassSimilarity_SOM.som_common import loadActivations
from Globals.globalvars import Glb
from InterClassSimilarity_SOM.pie_cluster_purity import purity_pie
import time

dim_size = 14

set_names = ["Train", "Val", "Test"]
do_predict = False
do_piecharts = True

# SOM clusterer
som_filename = os.path.join(Glb.results_folder, "som_clusterer_{}x{}".format(dim_size, dim_size) )

# Results file: assigned clusters
clusters_filename_pattern = "som_clstrs_{}_{}x{}.h5"

# Load SOM clusterer
mysom = pickle.load ( open(som_filename, 'rb') )

for set_name in set_names:
    if do_predict:
        # Load activations
        orange_tab = loadActivations(set_name)

        # Inference: get winner neurons (i.e. cluster asssignment) for each sample
        now = time.time()
        pred_winner_neurons = mysom.winners ( orange_tab.X )
        print("Prdicted winners in {} seconds".format(time.time() - now))

        # Save cluster assignment files
        clusters_filename = os.path.join ( Glb.results_folder, clusters_filename_pattern.format ( set_name, str(dim_size), str(dim_size) ) )
        now = time.time()
        pickle.dump( (pred_winner_neurons, orange_tab.Y), open(clusters_filename, 'wb'))
        print("Saved winners in {} seconds".format(time.time() - now))

    if do_piecharts:
        # Calculate purity and draw pie charts
        purity_pie(set_name,dim_size)
