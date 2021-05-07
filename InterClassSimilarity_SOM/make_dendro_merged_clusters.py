
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np

#set_name = "Test"
set_name = "Train"

dim_size = 8

results_folder = r"a:\IsKnown_Results"
dendro_filename_pattern = "dendro.{}.{}.png"
purity_impr_mat_filename = os.path.join ( results_folder,"{}_purity_impr_mat_{}x{}.h5".format ( set_name, str(dim_size), str(dim_size) ) )


# product names
#df_prodnames = pd.DataFrame(data={"class":classnames,"product":prod_names} )
#df_prodnames.to_csv("df_prods_194.csv", header=True, index=False)
df_prodnames = pd.read_csv("df_prods_194.csv", header=0)        # csv made from InterClassSimilarity\adhoc_dist_matrix.py

purity_impr_mat = pickle.load(open(purity_impr_mat_filename, 'rb'))



dist_mat = np.exp(-purity_impr_mat*10)
dis_mat_vectorized = squareform (dist_mat, force='tovector', checks=False)

#linkage_method='centroid'
linkage_method='single'         #nearest
#linkage_method='complete'      #farthest

clstrs = linkage(y=dis_mat_vectorized, method=linkage_method) # method='centroid' ==> new cluster is in the middle of sub-clusters

# Dendrogram
fig = plt.figure(figsize=(30, 10))
dn = dendrogram(Z=clstrs, labels=df_prodnames["product"].tolist(), leaf_font_size=6)
plt.tight_layout()
#plt.show()
plt.savefig(dendro_filename_pattern.format(set_name,linkage_method))
plt.close()
