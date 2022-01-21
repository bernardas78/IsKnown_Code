#   Input: SOM clusterers
#   Output:
#       SOM clusters (assignment of each sample to a cluster); in file
#       Pie charts
#    , for Train, Val, Test

import pickle
import os
from InterClassSimilarity_SOM.som_common import loadActivations, makeOrangeTable
from Globals.globalvars import Glb
from InterClassSimilarity_SOM.pie_cluster_purity import purity_pie
from InterClassSimilarity_SOM.dist_from_winners_to_csv import cluster_filenames_dists_to_csv
import time

#dim_size = 15

#set_names = ["Test"] #["Train", "Val", "Test"]
#do_predict = True
#do_piecharts = True

def infer_winners (set_names, dim_size, hier_lvl, do_predict, do_piecharts, do_clstr_str, do_clstr_dist, incl_filenames, trained_on_set_name):
    # SOM clusterer
    som_filename = os.path.join(Glb.results_folder, "som_clusterer_{}x{}_hier{}_{}.h5".format(dim_size, dim_size, hier_lvl, trained_on_set_name) )

    # Results file: assigned clusters
    clusters_filename_pattern = "som_clstrs_{}_{}x{}_hier{}.h5"

    for set_name in set_names:
        if do_predict:
            # Load SOM clusterer
            mysom = pickle.load(open(som_filename, 'rb'))

            # Load activations
            #orange_tab,filenames = loadActivations(set_name,hier_lvl)
            #act_prelast, lbls, filenames = loadActivations(set_name,hier_lvl)
            tuple_contents = loadActivations(set_name, hier_lvl,incl_filenames)
            act_prelast, lbls = tuple_contents[0], tuple_contents[1]
            orange_tab = makeOrangeTable(act_prelast,lbls)

            # Inference: get winner neurons (i.e. cluster asssignment) for each sample
            now = time.time()
            print ("starting to predict")
            pred_winner_neurons = mysom.winners ( orange_tab.X )
            print("Prdicted winners in {} seconds".format(time.time() - now))

            # Save cluster assignment files
            clusters_filename = os.path.join ( Glb.results_folder, clusters_filename_pattern.format ( set_name, dim_size, dim_size, hier_lvl ) )
            now = time.time()
            #pickle.dump( (pred_winner_neurons, orange_tab.Y, filenames), open(clusters_filename, 'wb'))
            pickle.dump( (pred_winner_neurons, orange_tab.Y), open(clusters_filename, 'wb'))
            print("Saved winners in {} seconds".format(time.time() - now))

            if do_clstr_dist and incl_filenames:
                cluster_filenames_dists_to_csv(set_name, dim_size, hier_lvl, orange_tab, pred_winner_neurons, filenames=tuple_contents[2], mysom_weights=mysom.weights)

        if do_piecharts:
            # Calculate purity and draw pie charts
            purity_pie(set_name,dim_size,hier_lvl, do_clstr_str)


