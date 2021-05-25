import pandas as pd
import os
from Globals.globalvars import Glb
from collections import Counter
import matplotlib
matplotlib.use('Agg')   # otherwise, on 1080 fails importing pyplot
from matplotlib import pyplot as plt


set_name = "Val"
dim_size = 15
hier_lvl = 0

# Cluster IDs (top5 sorted by purity asc from IsKnown_Results\SOM_Clstr_Str/Train_15x15_hier0/ )
i, j = 11,3
#i, j = 10,13
#i, j = 9,6
#i, j = 4,7
#i, j = 6,2

# how many most frequent classes to show historgrams for
cnt_most_freq = 3

# Read product names file
prodnames = pd.read_csv("df_prods_194.csv", header=0)["product"].tolist()

# Read distances of data points assigned to this cluster
cluster_dist_folder_pattern = "{}_{}x{}_hier{}"
cluster_dist_filename_pattern = "dist_[{}_{}].csv"
cluster_dist_folder = cluster_dist_folder_pattern.format(set_name, dim_size, dim_size, hier_lvl)
cluster_dist_filename = cluster_dist_filename_pattern.format(i, j)
cluster_dist_filepath = os.path.join(Glb.results_folder, "SOM_Clstr_Dist", cluster_dist_folder, cluster_dist_filename)
df_clstr_items = pd.read_csv(cluster_dist_filepath)

# Get N most frequent classes
most_common_classes = Counter(df_clstr_items.Prod_ID).most_common()
most_common_N_prod_ids = [ most_common_classes[ind][0] for ind in range(cnt_most_freq) ]
most_common_N_prod_names = [prodnames[lbl] for lbl in most_common_N_prod_ids]

# Draw historgrams of most frequent product on a single graph
for prod_id, prod_name in zip(most_common_N_prod_ids,most_common_N_prod_names):
    # Filter this product's distances
    dists_this_product = df_clstr_items.Dist [ df_clstr_items.Prod_ID==prod_id]
    plt.hist (dists_this_product, bins=50, alpha=0.5, label=prod_name)
plt.legend()
plt.title("Distance from SOM cluster center, {} most frequent products".format(cnt_most_freq))
plt.ylabel("Image count")
plt.xlabel("Distance from center")
plt.savefig( os.path.join(Glb.results_folder, "SOM_Clstr_Dist", "hist_{}_{}x{}_hier{}_[{}_{}].png".format(set_name, dim_size, dim_size, hier_lvl,i,j)) )
plt.close()