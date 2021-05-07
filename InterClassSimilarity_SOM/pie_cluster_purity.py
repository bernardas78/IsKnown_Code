import pickle
import os
import numpy as np
from collections import Counter
from Globals.globalvars import Glb
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # otherwise, on 1080 fails importing pyplot
from matplotlib import pyplot as plt

purity_filename_pattern = "temp/purity_{}_{}x{}.jpg"
clusters_filename_pattern = "som_clstrs_{}_{}x{}.h5"

#set_name = "Test"
#set_name = "Train"

#dim_size = 14

def purity_pie(set_name, dim_size):
    df_prodnames = pd.read_csv("df_prods_194.csv", header=0)["product"].tolist()
    clusters_filename = os.path.join ( Glb.results_folder, clusters_filename_pattern.format ( set_name, str(dim_size), str(dim_size) ) )

    (pred_winner_neurons, lbls) = pickle.load( open(clusters_filename, 'rb') )

    figure, axes = plt.subplots(nrows=dim_size, ncols=dim_size)
    colors=('b', 'g', 'r', 'c', 'm', 'y')
    for i in range(dim_size):
        for j in range(dim_size):
            # print ("i={}, j={}".format(i,j))
            # filter samples where this is winner neuron
            this_neuron_lbls = lbls[(pred_winner_neurons[:, 0] == i) & (pred_winner_neurons[:, 1] == j)].astype(int)
            most_common_classes = Counter(this_neuron_lbls).most_common()
            if len(most_common_classes) > 0:
                # top 80% containing classes
                cum_probs = np.cumsum([item[1]/len(this_neuron_lbls) for item in most_common_classes])
                max_cnt = np.where(cum_probs > 0.8)[0][0] + 1
                max_cnt = np.minimum(max_cnt,5)
                #print ("i={},j={},max_cnt={}".format(i,j,max_cnt))
                #if max_cnt==0:
                #    print (cum_probs[:4])

                sizes_first_n = [item[1] for item in most_common_classes[:max_cnt]]
                sizes = sizes_first_n + [len(this_neuron_lbls) - np.sum(sizes_first_n)] # Add "Other"
                prod_indices = [ item[0] for item in most_common_classes[:max_cnt]]
                labels = [ df_prodnames[ind] for ind in prod_indices] + [""] # Add "Other"
                labels = [ lbl[:10] for lbl in labels ] # left10

                #print ("i={}, j={}, Len={}".format(i,j,len(this_neuron_lbls)))
                radius = np.sqrt( len(this_neuron_lbls) / len(lbls)) * dim_size
                axes[i,j].pie(sizes, labels=labels, colors=colors, radius=radius, textprops={'fontsize': 4})#, textprops={'fontsize': 5}) #, autopct='%1.1f%%')
            else:
                #axes[i, j].set_xticks([])
                #axes[i, j].set_yticks([])
                axes[i, j].axis('off')

    #plt.show()
    plt.savefig(purity_filename_pattern.format(set_name, dim_size,dim_size))
    plt.close()
