# Makes using Embeddings (pre-last activations):
#   1.inter-class distance matrix
#   2.dendrogram based on inter-class distances
from tensorflow.keras.models import load_model
from Globals.globalvars import Glb, Glb_Iterators
from tensorflow.keras.backend import function
import numpy as np
from datetime import datetime
#from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances, rbf_kernel
import pickle
import os

model = load_model( os.path.join(Glb.results_folder,"model_clsf_from_isVisible_20210415_gpu1.h5") ) # 83% test accuracy
act_filename_pattern = os.path.join(Glb.results_folder,"activations_prelast_clsf_from_isVisible_20210415_gpu1_{}_hier0.filenames.h5")
dist_mat_filename_pattern = os.path.join(Glb.results_folder, "distmat_prelast_clsf_from_isVisible_20210415_gpu1_{}_{}.h5")
dendro_filename_pattern = "temp/dendro.{}.{}.{}.png"

# which layer is needed?
#   model.summary()
prelast_dense_layer = model.layers[-2]  #model.layers[dense_layer_ids[-2]]
prelast_func_activation = function([model.input], [prelast_dense_layer.output])
prelast_output_shape = prelast_dense_layer.output_shape[1]

# Data iterator
batch_size = 350

#set_name = "Test"
#set_name = "Train"
set_name = "Val"


#dist_method = "manhattan"
#dist_method = "euclidean"
dist_method = "cosine"
#dist_method = "rbf"

#linkage_method='centroid'
linkage_method='single'
#linkage_method='complete'

data_iterator = Glb_Iterators.get_iterator(os.path.join( r"C:\IsKnown_Images_IsVisible\Bal_v14\Ind-0", set_name), "div255", batch_size=batch_size)
# Total number of images
cnt_imgs = len(data_iterator.classes)
cnt_classes = len(data_iterator.class_indices)

act_filename = act_filename_pattern.format(set_name)
#if not os.path.exists (act_filename):
#    # Allocate buffer for storing activations and labels
#    act_prelast = np.zeros ((cnt_imgs, prelast_output_shape), dtype=np.float32)
#    lbls = np.zeros ((cnt_imgs), dtype=np.int)
#
#    cntr = 0
#    now = datetime.now()
#
#    # Save activations
#    for X,y in data_iterator:
#        cnt_samples_in_batch = y.shape[0]
#        print ("Batch {}/{}".format(cntr, len(data_iterator)))
#        act_prelast[ (cntr*batch_size):(cntr*batch_size+cnt_samples_in_batch),:] = prelast_func_activation([X])[0]
#        lbls [ (cntr*batch_size):(cntr*batch_size+cnt_samples_in_batch) ] = np.argmax(y, axis=1)
#        cntr += 1
#        if cntr >= len(data_iterator):
#            break
#    print ("Total seconds: {}".format((datetime.now() - now).seconds))
#    pickle.dump( (act_prelast,lbls), open(act_filename, 'wb') )
#else:
print ("Loading {}".format(act_filename) )
(act_prelast,lbls,filenames) = pickle.load(open(act_filename, 'rb'))
print ("Loaded act_prelast,lbls" )


# Distance matrix (upper right only for dendrogram; for possible need, calc diagonal too - intra class distance)
rbf_inv = lambda X,Y: 1/rbf_kernel(X,Y)
dist_method_dic = {"cosine": cosine_distances,
                   "euclidean": euclidean_distances,
                   "rbf": rbf_inv,
                   "manhattan": manhattan_distances}
f_distances = dist_method_dic[dist_method]

dist_mat_filename = dist_mat_filename_pattern.format(set_name,dist_method)
if not os.path.exists(dist_mat_filename):
    dist_mat = np.zeros ( (cnt_classes,cnt_classes), dtype=np.float32)
    now = datetime.now()
    for class_a_ind in range(cnt_classes):
        for class_b_ind in range(class_a_ind,cnt_classes):
            #print ("{} {}".format(class_a_ind,class_b_ind))
            sample_inds_class_a = np.where (lbls==class_a_ind)[0]
            sample_inds_class_b = np.where (lbls==class_b_ind)[0]
            act_class_a = act_prelast[sample_inds_class_a]
            act_class_b = act_prelast[sample_inds_class_b]
            #dist_a_c = mean_dist_cosine (a=act_class_a, b=act_class_b)
            dist_mat[class_a_ind,class_b_ind] = np.mean(f_distances (X=act_class_a, Y=act_class_b))
            print ("Dist[{},{}]:{}; sec elapsed:{}, cnts: {},{}".format(class_a_ind, class_b_ind,
                                                                        dist_mat[class_a_ind,class_b_ind],
                                                                        (datetime.now()-now).seconds,
                                                                        len(sample_inds_class_a), len(sample_inds_class_b) ) )
    print ("Saving {}".format(dist_mat_filename) )
    pickle.dump( dist_mat, open(dist_mat_filename, 'wb') )
    print("Saved")
else:
    print ("Loading {}".format(dist_mat_filename) )
    dist_mat = pickle.load(open(dist_mat_filename, 'rb'))
    print ("Loaded dist_mat" )



# product names
import pandas as pd
df_prods = pd.read_csv ('../Dendro/dendrogramai.csv', header=None, names=["ProductName","ProductCode","Cnt"])
classnames = list(data_iterator.class_indices.keys())
prod_names = []
for productCode in classnames:
    prod_name_series = df_prods.loc[df_prods.ProductCode == int(productCode)]["ProductName"].values
    prod_names.append( prod_name_series[0] if len(prod_name_series)>0 else productCode)
#prod_names = [  df_prods.loc[df_prods.ProductCode == int(productCode)]["ProductName"].values[0]   for productCode in test_iter_classnames]

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

dis_mat_vectorized = squareform (dist_mat, checks=False, force='tovector')
clstrs = linkage(y=dis_mat_vectorized, method=linkage_method) # method='centroid' ==> new cluster is in the middle of sub-clusters

fig = plt.figure(figsize=(30, 10))
dn = dendrogram(Z=clstrs, labels=prod_names, leaf_font_size=6)
plt.tight_layout()
#plt.show()
plt.savefig(dendro_filename_pattern.format(set_name,dist_method,linkage_method))
plt.close()

import seaborn as sns
sns.set(font_scale=0.3)
ax = sns.heatmap(dist_mat, cbar=False, xticklabels=prod_names, yticklabels=prod_names, square=True)
plt.savefig(r"D:\IsKnown_Code\InterClassSimilarity\temp\dist_mat.png")
plt.close()