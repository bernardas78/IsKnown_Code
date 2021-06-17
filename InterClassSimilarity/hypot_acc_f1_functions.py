import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from tensorflow.keras.models import load_model
import time
from Globals.globalvars import Glb_Iterators, Glb
from Common import common as cm
from sklearn.metrics import confusion_matrix
import os

#linkage_method='centroid'  # new cluster is in the middle of sub-clusters
linkage_method='single'     # nearest
#linkage_method='complete'  # farthest

def f1_from_conf_mat(conf_mat):
    eps=1e-7
    prec = np.diag(conf_mat) / np.sum(conf_mat + eps, axis=0)
    rec = np.diag(conf_mat) / np.sum(conf_mat + eps, axis=1)
    f1 = 2*prec*rec/(prec+rec+eps)
    return np.mean(f1)


# Hypoth acc and F1 calc based on dist mat
def acc_f1_based_on_class_merge(dist_mat, conf_mat, file_suffix):
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
    hypot_f1 = [f1_from_conf_mat(conf_mat)]
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
        hypot_f1.append(f1_from_conf_mat(conf_mat_hypot))
        print("Hypothetical of {} classes: accuracy  {}, F1 {}".format(len(this_merge_lbls), hypot_acc[-1], hypot_f1[-1]))
    print("Merged in {} sec".format(time.time() - now))
    return hypot_acc,hypot_f1


# Calc/Save or Load confusion matrix of Train/Val/Test
def get_conf_mat(set_name):

    conf_mat_filename = "temp/conf_mat_{}.h5".format(set_name)

    if not os.path.exists (conf_mat_filename):
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
        del model
        print ("Predicted in {} sec".format(time.time()-now))
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        pickle.dump(conf_mat, open(conf_mat_filename, 'wb'))
        print ("Saved conf mat {}".format(set_name))
    else:
        conf_mat = pickle.load( open(conf_mat_filename, 'rb') )
        print ("Loaded conf mat {}".format(set_name))

    # sanity check: should be 83% (Test), 49.8 (Val)
    acc = np.sum( [ conf_mat[i,i] for i in range(194) ]) / np.sum(conf_mat )
    print ("Acc: {}".format( acc ) )
    return conf_mat