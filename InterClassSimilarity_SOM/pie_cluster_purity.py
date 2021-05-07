import pickle

import Orange
import os
from Orange.projection import som
import time
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

#set_name = "Test"
set_name = "Train"

dim_size = 8

results_folder = r"a:\IsKnown_Results"
clusters_filename = os.path.join ( results_folder,"{}_clstrs_{}x{}_Orange.tab".format ( set_name, str(dim_size), str(dim_size) ) )

(pred_winner_neurons, lbls) = pickle.load( open(clusters_filename, 'rb') )

figure, axes = plt.subplots(nrows=8, ncols=8)
colors=('b', 'g', 'r', 'c', 'm', 'y')
for i in range(8):
    for j in range(8):
        # print ("i={}, j={}".format(i,j))
        # filter samples where this is winner neuron
        this_neuron_lbls = lbls[(pred_winner_neurons[:, 0] == i) & (pred_winner_neurons[:, 1] == j)].astype(int)
        most_common_classes = Counter(this_neuron_lbls).most_common()
        if len(most_common_classes) > 0:
            max_cnt = np.minimum(len(most_common_classes),5)
            sizes_first_n = [item[1] for item in most_common_classes[:max_cnt]]
            sizes = sizes_first_n + [len(this_neuron_lbls) - np.sum(sizes_first_n)] # Add "Other"
            labels = [item[0] for item in most_common_classes[:max_cnt]] + ["Other"]

            print ("i={}, j={}, Len={}".format(i,j,len(this_neuron_lbls)))
            radius = np.sqrt( len(this_neuron_lbls) / len(lbls)) * 8
            axes[i,j].pie(sizes, labels=labels, colors=colors, radius=radius)#, textprops={'fontsize': 5}) #, autopct='%1.1f%%')

plt.show()
plt.savefig("purity_8x8.jpg")
plt.close()