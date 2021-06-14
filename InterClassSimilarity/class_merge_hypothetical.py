# Agglomerative clustering based on distance matrix
#   Inputs: dist_mat and conf_mat
#   Outputs: accuracy=f(#classes)   CSV: #classes,acc
#            clusters               CSV: #classes,clstrId,prekes_id,prekes_barcode,preke_name (PK:#classes+prekes_id)

import pandas as pd
import pickle
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from tensorflow.keras.models import load_model
from InterClassSimilarity_SOM import cluster_merge_functions as somf

#dist_method = "manhattan"
#dist_method = "euclidean"
dist_method = "cosine"
#dist_method = "rbf"

#linkage_method='centroid'  # new cluster is in the middle of sub-clusters
linkage_method='single'     # nearest
#linkage_method='complete'  # farthest

# Hypoth acc calc based on dist mat
def acc_based_on_class_merge(dist_mat, conf_mat, file_suffix):
    cnt_products = dist_mat.shape[0]
    acc = np.sum([conf_mat[i, i] for i in range(cnt_products)]) / np.sum(conf_mat)

    # Agglomerative clustering
    dis_mat_vectorized = squareform(dist_mat, checks=False, force='tovector')
    clstrs = linkage(y=dis_mat_vectorized, method=linkage_method)

    # read product names and barcodes
    df_products = pd.read_csv("../InterClassSimilarity_SOM/df_prods_194.csv", header=0)

    # Overwrite merged_classes file and append to it later
    merged_classes_filename = "merged_str/merged_classes_{}.csv".format(file_suffix)
    df_clstrs = pd.DataFrame(columns=["cnt_classes", "clstr_id", "product_id", "product_barcode", "product_name"])
    df_clstrs.to_csv(merged_classes_filename, mode="w", header=True, index=False)

    # Calc based on merged classes:
    #       hypothetical accuracy
    #       cluster structure > csv
    hypot_acc = [acc]
    classes_to_clstr = {classs: classs for classs in range(cnt_products)}  # Initially classes are their own clusters
    now = time.time()
    for clstr_lvl in range(len(clstrs)):  # each merge: new grouping of classes len(clstrs)

        # assign merged cluster id to merged classes
        new_clstr_id = clstr_lvl + len(classes_to_clstr)
        this_merge_clstrs = clstrs[clstr_lvl][0:2]
        classes_to_clstr = {i: new_clstr_id if classes_to_clstr[i] in this_merge_clstrs else classes_to_clstr[i] for i
                            in range(cnt_products)}
        this_merge_lbls = np.unique(list(classes_to_clstr.values()))

        # Export cluster structure to csv: #classes,clstrId,prekes_id,prekes_barcode,preke_name
        df_clstrs = pd.DataFrame(data={
            "cnt_classes":np.repeat(len(this_merge_lbls),cnt_products),
            "clstr_id":list(classes_to_clstr.values()),
            "product_id": list(classes_to_clstr.keys()),
            "product_barcode": df_products["class"].tolist(),
            "product_name": df_products["product"].tolist()
        } )
        df_clstrs.to_csv(merged_classes_filename, mode="a", header=False, index=False)

        # calc hypot conf mat, acc
        conf_mat_hypot_size = cnt_products - clstr_lvl - 1
        conf_mat_hypot = np.zeros((conf_mat_hypot_size, conf_mat_hypot_size), dtype=float)
        for i in range(conf_mat_hypot_size):
            clstr_i_classes = this_merge_lbls[i] == list(classes_to_clstr.values())
            for j in range(conf_mat_hypot_size):
                clstr_j_classes = this_merge_lbls[j] == list(classes_to_clstr.values())
                conf_mat_hypot[i, j] = np.sum(conf_mat[clstr_i_classes, :][:, clstr_j_classes])

        hypot_acc.append(np.sum([conf_mat_hypot[i, i] for i in range(len(this_merge_lbls))]) / np.sum(conf_mat_hypot))
        print("Hypothetical accuracy of {} classes: {}".format(len(this_merge_lbls), hypot_acc[-1]))
    print("Merged in {} sec".format(time.time() - now))
    return hypot_acc



from Common import common as cm
from Globals.globalvars import Glb_Iterators, Glb
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import time

def get_conf_mat(set_name):
    # Prepare data generator
    data_folder = os.path.join( Glb.images_folder, "Bal_v14", "Ind-0", set_name )
    data_iterator = Glb_Iterators.get_iterator (data_folder, div255_resnet="div255", shuffle=False)

    # Load moddel
    model_filename = "model_clsf_from_isVisible_20210415_gpu1.h5"
    print ("Loading model {}".format(model_filename))
    now=time.time()
    model = load_model( os.path.join( Glb.results_folder, model_filename ) ) # 83% test accuracy
    print ("Loaded in {} sec".format(time.time()-now))

    # Predict highest classes and get conf_mat
    print ("Predicting...")
    now=time.time()
    (y_pred, y_true) = cm.get_pred_actual_classes(model, data_iterator)
    print ("Predicted in {} sec".format(time.time()-now))
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # sanity check: should be 83%
    acc = np.sum( [ conf_mat[i,i] for i in range(194) ]) / np.sum(conf_mat )
    print ("Acc: {}".format( acc ) )
    del model
    return conf_mat

############### Prep confusion matrix (used to Test) ########################
test_conf_mat = get_conf_mat("Test")

####################### EMBEDDINGS DISTANCE ########################

dist_mat_emb_dist_filename_pattern = r"a:\IsKnown_Results\distmat_prelast_clsf_from_isVisible_20210415_gpu1_{}_{}.h5"
dist_mat_emb_dist_filename = dist_mat_emb_dist_filename_pattern.format("Val",dist_method)
dist_mat_emb_dist = pickle.load(open(dist_mat_emb_dist_filename, 'rb'))

hypot_acc_emb_dist = acc_based_on_class_merge(dist_mat=dist_mat_emb_dist, conf_mat=test_conf_mat, file_suffix="emb_dist")


####################### CONF MAT BIGGEST CONTRIBUTORS ########################
val_conf_mat = get_conf_mat("Val")
dist_mat_conf_mat_big_err = 1/(val_conf_mat+val_conf_mat.T+1e-7)
hypot_acc_conf_mat = acc_based_on_class_merge(dist_mat=dist_mat_conf_mat_big_err, conf_mat=test_conf_mat, file_suffix="conf_mat")


####################### SOM PURITY ##############################################
dim_size = 15
#clusters_filename = os.path.join ( Glb.results_folder,"{}_clstrs_{}x{}_Orange.tab".format ( set_name, str(dim_size), str(dim_size) ) )
#distmat_filename = os.path.join ( Glb.results_folder,"{}_distmat_{}x{}.h5".format ( set_name, str(dim_size), str(dim_size) ) )
clusters_filename = os.path.join ( Glb.results_folder,"som_clstrs_{}_{}x{}_hier0.h5".format ( "Val", str(dim_size), str(dim_size) ) )
(pred_winner_neurons, lbls) = pickle.load( open(clusters_filename, 'rb') )
purity_impr_mat = somf.hypotheticalMergePurity(pred_winner_neurons, lbls)
dist_mat_purity_impr = 1/(purity_impr_mat+1e-7)
hypot_acc_purity_impr = acc_based_on_class_merge(dist_mat=dist_mat_purity_impr, conf_mat=test_conf_mat, file_suffix="som_purity_impr")

####################### BARCODE STRUCTURE ########################################
bc_structure_cnt_classes = [194,109,26,5,2]
bc_structure_acc = [0.83,0.837,0.829,0.975,0.971]

####################### GRAPHICS ######################################
from matplotlib import pyplot as plt
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