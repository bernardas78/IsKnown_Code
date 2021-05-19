import numpy as np
import pickle
import os
from InterClassSimilarity_SOM.som_common import loadActivations
from Globals.globalvars import Glb

som_filename = os.path.join(Glb.results_folder, "som_clusterer_{}x{}_hier{}".format(15,15,0) )
mysom = pickle.load(open(som_filename, 'rb'))
orange_tab,_ = loadActivations("Train",0)
pred_winner_neurons = mysom.winners ( orange_tab.X )

np.linalg.norm(orange_tab.X[0]-mysom.weights[0,0,:])

for sample_id in range(20):
    #sample_id=0
    best_i,best_j=-1,-1
    best_dist=1e+7
    for i in range(mysom.weights.shape[0]):
        for j in range(mysom.weights.shape[1]):
            dist=np.linalg.norm(orange_tab.X[sample_id] - mysom.weights[i, j, :])
            if dist<best_dist:
                best_i,best_j=i,j
                best_dist=dist
    print ("Sample_id={}, best i,j={},{}; predicted winner:{},{}".format(sample_id,best_i,best_j,pred_winner_neurons[sample_id,0],pred_winner_neurons[sample_id,1]))


