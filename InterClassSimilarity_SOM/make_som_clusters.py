# Make SOM clusters:
#   Input: Orange table files, made by make_orange_tables.py
#   Output: SOM clusters (assignment of each sample to a cluster); in file
import pickle

import Orange
import os
from Orange.projection import som
import time

#set_name = "Test"
set_name = "Train"

results_folder = r"a:\IsKnown_Results"

# Load activation tables (~12:17-min train set)
now = time.time()
activations_orangeTable_filename = os.path.join (results_folder,"{}_activations_preLast_Orange.tab".format(set_name))
orange_tab = Orange.data.Table.from_file(activations_orangeTable_filename)
print("Loaded activations in {} seconds".format(time.time() - now))

dim_size = 14 #8
l_rate = 0.5
n_iters = 50
mysom = som.SOM(dim_x=dim_size, dim_y=dim_size, hexagonal=True)

# Train SOM
now = time.time()
mysom.fit(x=orange_tab.X, n_iterations=n_iters, learning_rate=l_rate)
print("Trained SOM for {} seconds".format(time.time() - now))

# Inference: get winner neurons (i.e. cluster asssignment) for each sample
pred_winner_neurons = mysom.winners ( orange_tab.X )

# Save cluster assignment files
clusters_filename = os.path.join ( results_folder,"{}_clstrs_{}x{}_Orange.tab".format ( set_name, str(dim_size), str(dim_size) ) )
pickle.dump( (pred_winner_neurons, orange_tab.Y), open(clusters_filename, 'wb'))
