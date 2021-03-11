# Given a model and images folders: [Known, Unknown]:
#   1.K-means clusters.
#   2. Assign class to cluster (more of samples known/unknown)
#   3. Measure how well train data (known/unknown) separated: acc, conf mat

import pickle
from Common import common as cm
from keras.models import load_model
from os import path
from datetime import datetime
import numpy as np
from scipy.cluster.vq import whiten,kmeans2
from sklearn.metrics import confusion_matrix, accuracy_score

class Last_Kmeans_Analysis:

    def __init__(self,
                 known_data_folder,     # folder where know class images are located of structure \class\file.[jpg,...]
                 unknown_data_folder,   # folder where unknown class images are located of structure \file.[jpg,...]
                 model_file             # model file location
                 ):
        self.known_data_folder = known_data_folder
        self.unknown_data_folder = unknown_data_folder
        self.model_file = model_file
        self.model_loaded = False


    def calc_save_last_activations(self, last_activations_file_name):
        # make sure model is loaded
        self.__load_model()

        known_data_iterator = cm.get_data_iterator( self.known_data_folder, self.target_size, is_categorical=True)
        unknown_data_iterator = cm.get_data_iterator( self.unknown_data_folder, self.target_size, is_categorical=False)

        now = datetime.now()
        known_preds = cm.get_preds(self.model, known_data_iterator)
        print ("Got known predictions in {} sec".format((datetime.now()-now).total_seconds() ))

        now = datetime.now()
        unknown_preds = cm.get_preds(self.model, unknown_data_iterator)
        print ("Got unknown predictions in {} sec".format((datetime.now()-now).total_seconds() ))

        act_file = open(last_activations_file_name, 'wb')
        pickle.dump( (known_preds,unknown_preds), act_file)
        act_file.close()
        print("Results saved to file {}".format(last_activations_file_name))
        return

    def make_clusters_kmeans(self, last_activations_file_name):
        if not path.exists (last_activations_file_name):
            self.calc_save_last_activations(last_activations_file_name)
        (known_preds, unknown_preds) =  pickle.load(open(last_activations_file_name, 'rb'))
        print ("Loaded activations: known {} and unknown {}".format (known_preds.shape, unknown_preds.shape))

        # Combine Known+Unknown to a single structure
        all_preds = np.concatenate (  (known_preds,               unknown_preds),                axis=0 )
        all_labels = np.concatenate ( (np.ones(len(known_preds)), np.zeros(len(unknown_preds))), axis=0)

        # Normalize (per recommendation in kmeans description)
        all_preds_white = whiten(all_preds)

        # Cluster centers
        k_s = [2,3,5,8,13,21,34,54]

        for k in k_s:
            # Cluster classes: assign the most frequest class
            cluster_classes = np.zeros ( (k), dtype=int)

            # Run k-means clustering
            (kmeans_cntrs, cluster_lbls) = kmeans2(data=all_preds_white, k=k, iter=1000)

            # Proportions of known, unknown samples assigned to each cluster
            for kmeans_cntr_ind,kmeans_cntr in enumerate(kmeans_cntrs):
                cnt_known_this_cluster = len( np.where((all_labels==1) & (cluster_lbls==kmeans_cntr_ind))[0] )
                cnt_unknown_this_cluster = len( np.where((all_labels==0) & (cluster_lbls==kmeans_cntr_ind))[0] )

                # Assign most frequest class (unknown,known) to this cluster
                cluster_classes[kmeans_cntr_ind] = np.argmax( [cnt_unknown_this_cluster, cnt_known_this_cluster] )

                #print ("Debug: {},{}".format(cnt_known_this_cluster, cnt_unknown_this_cluster) )
                print ("Cluster {}: {} ({:.1%}) known, {} ({:.1%}) unknown. Assigned class {}".format(kmeans_cntr_ind,
                                                                             cnt_known_this_cluster,
                                                                             cnt_known_this_cluster/(cnt_known_this_cluster+cnt_unknown_this_cluster) if cnt_known_this_cluster+cnt_unknown_this_cluster>0 else 0,
                                                                             cnt_unknown_this_cluster,
                                                                             cnt_unknown_this_cluster/(cnt_known_this_cluster+cnt_unknown_this_cluster) if cnt_known_this_cluster+cnt_unknown_this_cluster>0 else 0,
                                                                             cluster_classes[kmeans_cntr_ind]) )
            # Make confusion matrix, get accuracy
            pred_lbls = np.array([ cluster_classes[cluster_lbl] for cluster_lbl in cluster_lbls ])
            acc=accuracy_score( all_labels, pred_lbls)
            print ("Accuracy: {}".format(acc))
            conf_mat = confusion_matrix(all_labels, pred_lbls)
            print (conf_mat)


    ###################################
    ### PRIVATE METHODS
    ###################################

    def __load_model(self):
        if not self.model_loaded:
            print ("Loading model {}".format(self.model_file) )
            self.model = load_model(self.model_file)
            print ("Loaded model" )

            # extract input size
            self.target_size = self.model.layers[0].input_shape[1]
