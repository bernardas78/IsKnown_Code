# Make SOM clusters:
#   Input: Activation files
#   Output: SOM clusterers
#   Output: SOM clusters (assignment of each sample to a cluster); in file

from Orange.projection import som
import pickle
import os
import time
from Globals.globalvars import Glb
from InterClassSimilarity_SOM.som_common import loadActivations


#set_name = "Test"
set_name = "Train"
dim_size = 14 #8
l_rate = 0.5
n_iters = 50


# SOM clusterer
som_filename = os.path.join(Glb.results_folder, "som_clusterer_{}x{}".format(dim_size, dim_size) )

# Load train activations
train_orange_tab = loadActivations("Train")

# Train SOM
mysom = som.SOM(dim_x=dim_size, dim_y=dim_size, hexagonal=True)
now = time.time()
mysom.fit(x=train_orange_tab.X, n_iterations=n_iters, learning_rate=l_rate)
print("Trained SOM for {} seconds".format(time.time() - now))

# Save for later inference
pickle.dump (mysom, open(som_filename, 'wb'))

