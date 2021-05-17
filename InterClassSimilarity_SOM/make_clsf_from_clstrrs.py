# Make "classifier" from clusterers:
#   Clsf: pred_class = most frequent class of the cluster, (2nd most freq, etc)
# Save Top 1 frequent class of TRAIN set of each cluster to file
import numpy as np

from Globals.globalvars import Glb
import os
import pandas as pd
import pickle
from collections import Counter

dim_size = 15
hier_lvl = 0

clusters_filename_pattern = "som_clstrs_{}_{}x{}_hier{}.h5"

# Load TRAIN winner neurons
clusters_filename = os.path.join(Glb.results_folder, clusters_filename_pattern.format("Train", dim_size, dim_size, hier_lvl))
(pred_winner_neurons, lbls) = pickle.load(open(clusters_filename, 'rb'))

# Make "Classifier": cluster's class = most frequent class
top1_class = np.zeros( (dim_size,dim_size), dtype=np.int )
for i in range(dim_size):
    for j in range(dim_size):
        # filter samples where this is winner neuron
        this_neuron_lbls = lbls[(pred_winner_neurons[:, 0] == i) & (pred_winner_neurons[:, 1] == j)].astype(int)
        most_common_classes = Counter(this_neuron_lbls).most_common()
        top1_class[i,j] = most_common_classes[0][0] if len(most_common_classes)>0 else -1

# Evaluate "Accuracy", Purity on all sets
for set_name in ["Test", "Val", "Train"]:
    # Load winner neurons of the set
    clusters_filename = os.path.join(Glb.results_folder,
                                     clusters_filename_pattern.format(set_name, dim_size, dim_size, hier_lvl))
    (pred_winner_neurons, lbls) = pickle.load(open(clusters_filename, 'rb'))

    # Make a vector of "correct" classes for each predicted label
    y_hat_lbls = [ top1_class[winner_neuron[0],winner_neuron[1]] for winner_neuron in pred_winner_neurons ]

    # Calc acc: how many of the classes match
    acc = len(np.where ( y_hat_lbls==lbls )[0]) / len(lbls)

    print ("{} acc: {}".format(set_name,acc))

    # Calc purity
