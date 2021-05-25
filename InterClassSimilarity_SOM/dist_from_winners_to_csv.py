import pandas as pd
import os
import numpy as np
from Globals.globalvars import Glb
import time

cluster_dist_folder_pattern = "{}_{}x{}_hier{}"
cluster_dist_filename_pattern = "dist_[{}_{}].csv"


# saves filenames and distances from cluster center to csv
def cluster_filenames_dists_to_csv(set_name, dim_size, hier_lvl, orange_tab, pred_winner_neurons, filenames, mysom_weights):
    ## Load product names
    #df_prodnames = pd.read_csv("df_prods_194.csv", header=0)["product"].tolist()
    #prods = [df_prodnames[lbl] for lbl in this_neuron_lbls]
    now = time.time()

    for i in range(dim_size):
        for j in range(dim_size):
            #print ("Starting distance measure in SOM cluster {},{}".format(i,j))

            # filter samples where this is winner neuron
            inds_this_clstr = np.where( (pred_winner_neurons[:, 0] == i) & (pred_winner_neurons[:, 1] == j) )[0]
            this_neuron_activations = orange_tab.X [ inds_this_clstr ]
            this_neuron_lbls = orange_tab.Y [ inds_this_clstr ].astype(int)
            this_neuron_filenames = [filenames[ind] for ind in inds_this_clstr]

            # Calculate distances from
            dists = np.linalg.norm(this_neuron_activations - mysom_weights[i, j, :], axis=1)

            # save to file
            df_clstr_items = pd.DataFrame(#columns=[,,],
                                        data= {
                                            "Prod_ID": this_neuron_lbls,
                                            "Dist": dists,
                                            "Filename": this_neuron_filenames
                                        } )
            cluster_dist_folder = cluster_dist_folder_pattern.format( set_name, dim_size, dim_size, hier_lvl )
            cluster_dist_filename = cluster_dist_filename_pattern.format(i, j)

            cluster_dist_filepath = os.path.join ( Glb.results_folder, "SOM_Clstr_Dist", cluster_dist_folder, cluster_dist_filename)
            df_clstr_items.to_csv(cluster_dist_filepath, index=False, header=True, mode='w')

    print ("Saved SOM Distances from center in {}sec".format ( time.time()-now ) )
