import numpy as np
from matplotlib import pyplot as plt
from InterClassSimilarity import graph_functions, hypot_acc_f1_functions
import pandas as pd

df_hypoth_merge_metrics = pd.read_csv("metrics_hypoth_merge.csv", header=0)

class_cnt = df_hypoth_merge_metrics.class_cnt
hypot_acc_emb_dist = df_hypoth_merge_metrics.hypot_acc_emb_dist
hypot_f1_emb_dist = df_hypoth_merge_metrics.hypot_f1_emb_dist
hypot_acc_conf_mat = df_hypoth_merge_metrics.hypot_acc_conf_mat
hypot_f1_conf_mat = df_hypoth_merge_metrics.hypot_f1_conf_mat
hypot_acc_purity_impr = df_hypoth_merge_metrics.hypot_acc_purity_impr
hypot_f1_purity_impr = df_hypoth_merge_metrics.hypot_f1_purity_impr

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
#x = np.arange(test_conf_mat.shape[0],0,-1)
x = class_cnt
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
#x = np.arange(test_conf_mat.shape[0],0,-1)
x = class_cnt
plt.plot( x, hypot_f1_emb_dist, label="Embeddings distance", color="blue" )
plt.plot( x, hypot_f1_conf_mat, label="Error contribution", color="orange" )
plt.plot( x, hypot_f1_purity_impr, label="SOM purity", color="green" )
plt.plot( bc_structure_cnt_classes, bc_structure_f1, label="Barcode hierarchy", color="red" )
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