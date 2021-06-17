# Agglomerative clustering based on distance matrix
#   Inputs: dist_mat and conf_mat
#   Outputs: accuracy=f(#classes)   CSV: #classes,acc
#            clusters               merged_str/merged_classes_{}.csv: #classes,clstrId,prekes_id,prekes_barcode,preke_name (PK:#classes+prekes_id)
#

import pickle
from InterClassSimilarity_SOM import cluster_merge_functions as somf
from Globals.globalvars import Glb
import os
import numpy as np
from matplotlib import pyplot as plt
import graph_functions
import hypot_acc_f1_functions

#dist_method = "manhattan"
#dist_method = "euclidean"
dist_method = "cosine"
#dist_method = "rbf"


############### Prep confusion matrix (used to Test) ########################
test_conf_mat = hypot_acc_f1_functions.get_conf_mat("Test")

####################### EMBEDDINGS DISTANCE ########################
dist_mat_emb_dist_filename_pattern = r"a:\IsKnown_Results\distmat_prelast_clsf_from_isVisible_20210415_gpu1_{}_{}.h5"
dist_mat_emb_dist_filename = dist_mat_emb_dist_filename_pattern.format("Val",dist_method)
dist_mat_emb_dist = pickle.load(open(dist_mat_emb_dist_filename, 'rb'))
hypot_acc_emb_dist,hypot_f1_emb_dist = hypot_acc_f1_functions.acc_f1_based_on_class_merge(dist_mat=dist_mat_emb_dist, conf_mat=test_conf_mat, file_suffix="emb_dist")
print ("####################### EMBEDDINGS DISTANCE ########################")

####################### CONF MAT BIGGEST CONTRIBUTORS ########################
val_conf_mat = hypot_acc_f1_functions.get_conf_mat("Val")
dist_mat_conf_mat_big_err = 1/(val_conf_mat+val_conf_mat.T+1e-7)
hypot_acc_conf_mat,hypot_f1_conf_mat = hypot_acc_f1_functions.acc_f1_based_on_class_merge(dist_mat=dist_mat_conf_mat_big_err, conf_mat=test_conf_mat, file_suffix="conf_mat")
print ("####################### CONF MAT BIGGEST CONTRIBUTORS ########################")


####################### SOM PURITY ##############################################
dim_size = 15
purity_impr_mat_filename_pattern = r"a:\IsKnown_Results\som_purity_impr_mat_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5"
purity_impr_mat_filename = purity_impr_mat_filename_pattern.format("Val")
if not os.path.exists(purity_impr_mat_filename):
    clusters_filename = os.path.join ( Glb.results_folder,"som_clstrs_{}_{}x{}_hier0.h5".format ( "Val", str(dim_size), str(dim_size) ) )
    (pred_winner_neurons, lbls) = pickle.load( open(clusters_filename, 'rb') )
    purity_impr_mat = somf.hypotheticalMergePurity(pred_winner_neurons, lbls)
    pickle.dump(purity_impr_mat, open(purity_impr_mat_filename, 'wb'))
    print ("Saved SOM purity improvement matrix {}".format(purity_impr_mat_filename))
else:
    purity_impr_mat = pickle.load( open(purity_impr_mat_filename, 'rb') )
    print ("Loaded SOM purity improvement matrix from {}".format(purity_impr_mat_filename))

dist_mat_purity_impr = 1/(purity_impr_mat+1e-7)
hypot_acc_purity_impr,hypot_f1_purity_impr = hypot_acc_f1_functions.acc_f1_based_on_class_merge(dist_mat=dist_mat_purity_impr, conf_mat=test_conf_mat, file_suffix="som_purity_impr")
print ("####################### SOM PURITY ########################")

####################### BARCODE STRUCTURE ########################################
bc_structure_cnt_classes = [194,109,26,5,2]
# Values by running adhoc_hier1-4_metrics.py
bc_structure_acc = [0.8303471444568868,
                    0.8355729749906682,
                    0.8266144083613288,
                    0.9759238521836506,
                    0.979096677864875]
bc_structure_f1 = [0.5539615212579857,
                   0.5483381249067844,
                   0.6127574834934619,
                   0.5619988851741187,
                   0.9504490578900224]

####################### GRAPHICS ######################################

#          accuracy
x = np.arange(test_conf_mat.shape[0],0,-1)
plt.plot( x, hypot_acc_emb_dist, label="Embeddings distance" )
plt.plot( x, hypot_acc_conf_mat, label="Error contribution" )
plt.plot( x, hypot_acc_purity_impr, label="SOM purity" )
plt.plot( bc_structure_cnt_classes, bc_structure_acc, label="Barcode hierarchy" )
plt.xlabel ("Number of classes")
plt.ylabel ("Accuracy")
plt.legend(loc="upper right")
plt.title("Hypothetical accuracy by merging classes")
plt.show()

#       f-score
x = np.arange(test_conf_mat.shape[0],0,-1)
plt.plot( x, hypot_f1_emb_dist, label="Embeddings distance" )
plt.plot( x, hypot_f1_conf_mat, label="Error contribution" )
plt.plot( x, hypot_f1_purity_impr, label="SOM purity" )
plt.plot( bc_structure_cnt_classes, bc_structure_f1, label="Barcode hierarchy" )
plt.xlabel ("Number of classes")
plt.ylabel ("F-score")
plt.legend(loc="upper right")
plt.title("Hypothetical F-score by merging classes")
plt.show()

# Graph: smoothed acc + intersections: average of 12 neighbours
cnt_neighbours = 12
lst_lines = [hypot_acc_emb_dist, hypot_acc_conf_mat, hypot_acc_purity_impr]
lst_lines_smooth12 = graph_functions.smooth_lines(lst_lines, cnt_neighbours)
intersection_pts = graph_functions.find_all_intersection_points(lst_lines_smooth12)
lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
y_label = "F-score"
title = "Hypothetical accuracy by merging classes (moving avg. of {})".format(2*cnt_neighbours+1)
graph_functions.metric_lines_and_intersections ( lst_lines_smooth12, lst_labels, y_label, title, intersection_pts )



# Graph: raw F1 + intersections
lst_lines = [hypot_f1_emb_dist, hypot_f1_conf_mat, hypot_f1_purity_impr]
intersection_pts = graph_functions.find_all_intersection_points(lst_lines)
lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
y_label = "F-score"
title = "Hypothetical F-score by merging classes"
graph_functions.metric_lines_and_intersections ( lst_lines, lst_labels, y_label, title, intersection_pts )



# Graph: smoothed F1 + intersections: average of 7 neighbours
lst_lines = [hypot_f1_emb_dist, hypot_f1_conf_mat, hypot_f1_purity_impr]
lst_lines_smooth7 = graph_functions.smooth_lines(lst_lines, 3)
intersection_pts = graph_functions.find_all_intersection_points(lst_lines_smooth7)
lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
y_label = "F-score"
title = "Hypothetical F-score by merging classes (moving avg. of 7)"
graph_functions.metric_lines_and_intersections ( lst_lines_smooth7, lst_labels, y_label, title, intersection_pts )




# Local maxima
lst_lines_local_max = [hypot_f1_emb_dist, hypot_f1_conf_mat, hypot_f1_purity_impr]
extrema_pts = graph_functions.find_local_maximums(lst_lines_local_max)
lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
y_label = "F-score"
title = "Hypothetical F-score by merging classes (local maxima)"
graph_functions.metric_lines_and_intersections ( lst_lines_local_max, lst_labels, y_label, title, extrema_pts )


# Graph: smoothed F1 + Local maxima: average of 12 neighbours, weights ratio mid/size=1.7
cnt_neighbours = 12
lst_lines = [hypot_f1_emb_dist, hypot_f1_conf_mat, hypot_f1_purity_impr]
lst_lines_smooth7 = graph_functions.smooth_lines_weighted(lst_lines, cnt_neighbours, 1.7)
extrema_pts = graph_functions.find_local_maximums(lst_lines_smooth7)
lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
y_label = "F-score"
title = "Hypothetical F-score by merging classes (local maxima - avg. of {})".format(cnt_neighbours*2+1)
graph_functions.metric_lines_and_intersections ( lst_lines_smooth7, lst_labels, y_label, title, extrema_pts )




# Local maxima F1
lst_lines = [hypot_f1_emb_dist, hypot_f1_conf_mat, hypot_f1_purity_impr]
filename_pattern = r"A://IsKnown_Results//Graph//f1_var_cntNeigh_LocalMax//pts_{}_{}.jpg"
for cnt_neighbors in range(15):
    for weights_big_small_ratio in np.arange(1,3,0.1):
        lst_lines_smooth_i = graph_functions.smooth_lines_weighted(lst_lines, cnt_neighbors=cnt_neighbors,weights_big_small_ratio=weights_big_small_ratio)
        extrema_pts_smooth_i = graph_functions.find_local_maximums(lst_lines_smooth_i)
        if len(extrema_pts_smooth_i) < 6:
            lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
            y_label = "F-score"
            title = "Hypothetical F-score by merging classes (local maxima - avg.{})".format(cnt_neighbors*2+1)
            graph_functions.metric_lines_and_intersections ( lst_lines_smooth_i, lst_labels, y_label, title, extrema_pts_smooth_i )
            filename = filename_pattern.format(cnt_neighbors,weights_big_small_ratio)
            plt.savefig(filename)
            plt.close()




# Intersection points, accuracy
filename_pattern = r"A://IsKnown_Results//Graph//acc_var_cntNeigh_Intersec//pts_{}_{}.jpg"
for cnt_neighbors in range(15):
    for weights_big_small_ratio in np.arange(1.0,2.5,0.1):
        lst_lines = [hypot_acc_emb_dist, hypot_acc_conf_mat, hypot_acc_purity_impr]
        lst_lines_smooth3 = graph_functions.smooth_lines_weighted(lst_lines, cnt_neighbors=cnt_neighbors, weights_big_small_ratio=weights_big_small_ratio)
        # lst_lines_smooth3 = lst_lines
        intersection_pts = graph_functions.find_all_intersection_points(lst_lines_smooth3)
        if len(intersection_pts)<6:
            lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
            y_label = "Accuracy"
            title = "Hypothetical accuracy by merging classes (moving avg. of {})".format(2 * cnt_neighbors + 1)
            graph_functions.metric_lines_and_intersections(lst_lines_smooth3, lst_labels, y_label, title, intersection_pts)
            filename = filename_pattern.format(cnt_neighbors, weights_big_small_ratio)
            plt.savefig(filename)
            plt.close()


# Intersection points, F1
filename_pattern = r"A://IsKnown_Results//Graph//f1_var_cntNeigh_Intersec//pts_{}_{}.jpg"
for cnt_neighbors in range(15):
    for weights_big_small_ratio in np.arange(1.0,2.5,0.1):
        lst_lines = [hypot_f1_emb_dist, hypot_f1_conf_mat, hypot_f1_purity_impr]
        lst_lines_smooth3 = graph_functions.smooth_lines_weighted(lst_lines, cnt_neighbors=cnt_neighbors, weights_big_small_ratio=weights_big_small_ratio)
        # lst_lines_smooth3 = lst_lines
        intersection_pts = graph_functions.find_all_intersection_points(lst_lines_smooth3)
        if len(intersection_pts)<6:
            lst_labels = ["Embeddings distance", "Error contribution", "SOM purity"]
            y_label = "Accuracy"
            title = "Hypothetical F-score by merging classes (moving avg. of {})".format(2 * cnt_neighbors + 1)
            graph_functions.metric_lines_and_intersections(lst_lines_smooth3, lst_labels, y_label, title, intersection_pts)
            filename = filename_pattern.format(cnt_neighbors, weights_big_small_ratio)
            plt.savefig(filename)
            plt.close()