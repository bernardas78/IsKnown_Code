import pickle
import numpy as np
from collections import Counter
import pandas as pd
from Globals.globalvars import Glb
import os

import matplotlib
matplotlib.use('Agg')   # otherwise, on 1080 fails importing pyplot
from matplotlib import pyplot as plt

purity_filename_pattern = "purity_{}_{}x{}_hier{}.jpg"
clusters_filename_pattern = "som_clstrs_{}_{}x{}_hier{}.h5"

cluster_structure_folder_pattern = "{}_{}x{}_hier{}"
cluster_structure_filename_pattern = "purity_{}_[{}_{}].csv"

show_labels = False
max_cnt_classes = 194   #5
pct_other = 0.99 #0.8

df_prodnames = pd.read_csv("df_prods_194.csv", header=0)["product"].tolist()

# Saves cluster structure to file
def cluster_structure_to_csv(set_name, dim_size,hier_lvl,i,j,most_common_classes):
    clstr_size = np.sum([item[1] for item in most_common_classes])

    # percentages and product names of all products of the cluster
    pcts = [item[1]/clstr_size for item in most_common_classes]
    prods = [ df_prodnames[item[0]] for item in most_common_classes]
    prod_ids = [ item[0] for item in most_common_classes]

    # pct of most common class
    purity = pcts[0]

    # save to file
    df_clstr_str = pd.DataFrame(#columns=[,,],
                                data= {
                                    "Product": prods,
                                    "Prod_ID": prod_ids,
                                    "Pct": pcts
                                } )
    cluster_structure_folder = cluster_structure_folder_pattern.format( set_name, dim_size, dim_size, hier_lvl )
    cluster_structure_filename = cluster_structure_filename_pattern.format("{:.3f}".format(purity), i, j)

    cluster_structure_filepath = os.path.join ( Glb.results_folder, "SOM_Clstr_Str", cluster_structure_folder, cluster_structure_filename)
    df_clstr_str.to_csv(cluster_structure_filepath, index=False, header=True, mode='w')




def purity_pie(set_name, dim_size,hier_lvl, do_clstr_str):

    purity = []

    clusters_filename = os.path.join ( Glb.results_folder, clusters_filename_pattern.format ( set_name, dim_size, dim_size, hier_lvl ) )

    #(pred_winner_neurons, lbls, filenames) = pickle.load( open(clusters_filename, 'rb') )
    (pred_winner_neurons, lbls) = pickle.load( open(clusters_filename, 'rb') )

    figure, axes = plt.subplots(nrows=dim_size, ncols=dim_size)
    colors=('b', 'g', 'r', 'c', 'm', 'y')
    for i in range(dim_size):
        for j in range(dim_size):
            # filter samples where this is winner neuron
            inds_this_clstr = np.where( (pred_winner_neurons[:, 0] == i) & (pred_winner_neurons[:, 1] == j) )[0]
            this_neuron_lbls = lbls[ inds_this_clstr ].astype(int)

            most_common_classes = Counter(this_neuron_lbls).most_common()
            if len(most_common_classes) > 0:
                if do_clstr_str:
                    # Save cluster structure to csv for further analysis
                    cluster_structure_to_csv(set_name, dim_size, hier_lvl, i, j, most_common_classes)

                # highest class
                purity.append( most_common_classes[0][1] )
                #print ("DEBUG: Node {},{} purity {}/{}".format(i,j, most_common_classes[0][1], len(this_neuron_lbls) ) )

                # top 80% containing classes
                cum_probs = np.cumsum([item[1]/len(this_neuron_lbls) for item in most_common_classes])
                max_cnt = np.where(cum_probs >= pct_other)[0][0] + 1
                max_cnt = np.minimum(max_cnt,max_cnt_classes)
                #print ("i={},j={},max_cnt={}".format(i,j,max_cnt))
                #if max_cnt==0:
                #    print (cum_probs[:4])

                sizes_first_n = [item[1] for item in most_common_classes[:max_cnt]]
                sizes = sizes_first_n + [len(this_neuron_lbls) - np.sum(sizes_first_n)] # Add "Other"
                prod_indices = [ item[0] for item in most_common_classes[:max_cnt]]
                if hier_lvl==0:
                    labels = [ df_prodnames[ind] for ind in prod_indices] + [""] # Add "Other"
                    labels = [ lbl[:15] for lbl in labels ] # left10
                else:
                    labels = prod_indices + [""] # Add "Other"

                #print ("i={}, j={}, Len={}".format(i,j,len(this_neuron_lbls)))
                radius = np.sqrt( len(this_neuron_lbls) / len(lbls)) * dim_size
                labels=labels if show_labels else None
                axes[i,j].pie(sizes, labels=labels, colors=colors, radius=radius, textprops={'fontsize': 6})#, textprops={'fontsize': 5}) #, autopct='%1.1f%%')
            else:
                #axes[i, j].set_xticks([])
                #axes[i, j].set_yticks([])
                axes[i, j].axis('off')

    #plt.show()
    purity_filename = os.path.join (Glb.graphs_folder, purity_filename_pattern.format(set_name, dim_size, dim_size, hier_lvl) )
    #plt.suptitle("SOM cluster structure", fontsize=14, fontweight="bold")
    plt.savefig(purity_filename)
    plt.close()
    print ("Saved piechart in {}".format(purity_filename))

    print ("Total purity in {}: {}".format (set_name, np.sum(purity)/len(lbls) ) )