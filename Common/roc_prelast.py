# Given a model and a images folders: [Known, Unknown], make a ROC of known vs. unknown predictions

from Common import common as cm
from datetime import datetime
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd
import scipy.spatial.distance
import math

class Roc_prelast:
    def __init__(self,
                 known_data_folder,     # folder where know class images are located of structure \class\file.[jpg,...]
                 unknown_data_folder,   # folder where unknown class images are located of structure \file.[jpg,...]
                 model_file             # model file location
                 ):

        self.known_data_folder = known_data_folder
        self.unknown_data_folder = unknown_data_folder

        print ("Loading model {}".format(model_file) )
        self.model = load_model(model_file)
        print ("Loaded model" )

        # extract input size
        self.target_size = self.model.layers[0].input_shape[1]


    def make_roc_prelast(self, distances_file_name, roc_file_pattern):  # ROC graph (result). Can contain up to 2 placeholders {} for date/time for generation, #products:
        # Read distances file
        df_distances = pd.read_csv(distances_file_name).fillna('')

        # Positive = class is known (one of topX classes)
        y_true = (df_distances["actual"] == "Known").astype(np.float64)

        # Predicted = normalized distance from top 1 prediction
        # y_pred_abs = 1. - df_distances["dist_eucl"] / np.max(df_distances["dist_eucl"])
        # y_pred_mah = 1. - np.log(df_distances["dist_mahalanobis"]) / np.log(np.max(df_distances["dist_mahalanobis"]) )
        # use sigmoid
        y_pred_mah = [1. - 1. / (1 + math.exp(-np.log(curr_dist))) for curr_dist in df_distances["dist_mahalanobis"]]
        # cosine is 0-1
        y_pred_cosine = [(1. - curr_dist) for curr_dist in df_distances["dist_cosine"]]

        # Display ROC for mahalanobis ;
        (fpr_mah, tpr_mah, thresholds_mah) = roc_curve(y_score=y_pred_mah, y_true=y_true)
        roc_auc_mah = auc(fpr_mah, tpr_mah)

        # Display ROC for cosine
        (fpr_cosine, tpr_cosine, thresholds_cosine) = roc_curve(y_score=y_pred_cosine, y_true=y_true)
        roc_auc_cosine = auc(fpr_cosine, tpr_cosine)

        # Find best accuracy
        accuracy_scores = []
        for thresh in thresholds_cosine:
            accuracy_scores.append(accuracy_score(y_true, [1 if m > thresh else 0 for m in y_pred_cosine]))
        best_acc_ind = np.argmax(accuracy_scores)
        best_acc = accuracy_scores[best_acc_ind]
        threshhold_to_use = thresholds_cosine[best_acc_ind]
        #print("Threshold to use = {}".format(threshhold_to_use))

        plt.figure()
        plt.plot(fpr_mah, tpr_mah, color='blue', lw=2, label='ROC (Mahalanobis) (area = %0.2f)' % roc_auc_mah)
        plt.plot(fpr_cosine, tpr_cosine, color='green', lw=2, label='ROC (Cosine) (area = %0.2f)' % roc_auc_cosine)
        plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Postive Rate')
        plt.ylabel('True Postive Rate')
        #cnt_class = len(np.unique(df_distances["actual"])) - 1  # 1-unknown class
        samples_known = len(np.where(df_distances["actual"] == "Known")[0])
        samples_unknown = len(np.where(df_distances["actual"] == "Unknown")[0])
        plt.title('{} samples of known, {} samples of unknown'.format( samples_known, samples_unknown))
        plt.legend(loc="lower right")
        # plt.show()

        # Draw a point for best accuracy
        plt.plot(fpr_cosine[best_acc_ind], tpr_cosine[best_acc_ind], marker="s", color="red")
        plt.text(fpr_cosine[best_acc_ind] + 0.02, tpr_cosine[best_acc_ind] - 0.02,
                 "Best accuracy: {:.1f}% @Thr={:.5f}".format(best_acc * 100, threshhold_to_use))

        # Save
        roc_file_name = roc_file_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S"))
        plt.savefig(roc_file_name)
        plt.close()

    def calc_save_prelast_activations(self, prelast_activations_file_name):
        known_data_iterator = cm.get_data_iterator( self.known_data_folder, self.target_size, is_categorical=True)
        unknown_data_iterator = cm.get_data_iterator( self.unknown_data_folder, self.target_size, is_categorical=False)

        (known_classes,_,known_activations) = cm.get_prelast_dense_activations(self.model, known_data_iterator, is_categorical=True)
        #(_,_,unknown_activations) = cm.get_prelast_dense_activations(self.model, unknown_data_iterator, is_categorical=False)

        act_file = open(prelast_activations_file_name, 'wb')
        #pickle.dump( (known_classes,known_activations, unknown_activations), act_file)
        pickle.dump( (known_classes,known_activations), act_file)
        act_file.close()
        print("Results saved to file {}".format(prelast_activations_file_name))
        return

    def calc_save_meansigmas_known(self, prelast_activations_file_name, meansigmas_file_name):
        # Dictionary: key=class; value=(mus,sigmas)
        meansigmas_dic = {}

        # Load activations saved by self.calc_save_prelast_activations()
        #(known_classes,known_activations, unknown_activations) = pickle.load(open(prelast_activations_file_name, 'rb'))
        (known_classes,known_activations) = pickle.load(open(prelast_activations_file_name, 'rb'))

        # Calc means/sigmas by class
        for theclass in np.unique(known_classes):

            # filter
            theclass_activations = known_activations [known_classes==theclass, :]

            # means, sigmas
            # mus and sigmas for each neuron
            mus = np.mean(theclass_activations, axis=0)
            sigmas = np.std(theclass_activations, axis=0)
            # print (mus, sigmas)

            # add to results dictionary
            meansigmas_dic[theclass] = (mus, sigmas)

        # flush results
        results_filehandler = open(meansigmas_file_name, 'wb')
        pickle.dump(meansigmas_dic, results_filehandler)
        results_filehandler.close()
        print("Results saved to file {}".format(meansigmas_file_name))

    def __process_leaf_folder(self, meansigmas_dic, known_or_unknown, data_folder, distances_file_name):

        data_iterator = cm.get_data_iterator(data_folder, self.target_size, is_categorical=False)

        (_,top1,prelast_activations) = cm.get_prelast_dense_activations(self.model, data_iterator, is_categorical=False)

        i=0
        for (sample_top1,sample_prelast_activations) in zip(top1,prelast_activations):
            (top1_mus, top1_sigmas) = meansigmas_dic[sample_top1]
            #print ("top1_mus: {}, sample_prelast_activations: {}".format(top1_mus[:2], sample_prelast_activations[:2]))

            # Calculate euclidean distance and mahalandobis distance
            dist = np.sum(np.square((sample_prelast_activations - top1_mus)))
            # How many sigmas in each dimension varies from mean? (0 sigmas are added epsilon)
            dist_mahalanobis = np.sum(np.square((sample_prelast_activations - top1_mus) / (top1_sigmas + 1e-7)))
            # cosine distance
            dist_cosine = scipy.spatial.distance.cosine(sample_prelast_activations, top1_mus)

            # Result to file
            df_distances = pd.DataFrame(
                data=[np.hstack([known_or_unknown, sample_top1, dist, dist_mahalanobis, dist_cosine])] )
            df_distances.to_csv(distances_file_name, header=None, index=None, mode='a')

            print("Processed {} files".format(i)) if i % 100 == 0 else 0
            i += 1

    def calc_save_dist_from_top1 (self, meansigmas_file_name, distances_file_name):
        # Load mean.sigmas for classes
        meansigmas_dic = pickle.load(open(meansigmas_file_name, 'rb'))

        # overwrite results file
        column_names = ['actual', 'top1', 'dist_eucl', 'dist_mahalanobis', 'dist_cosine']
        df_distances = pd.DataFrame(columns=column_names)
        df_distances.to_csv(distances_file_name, index=False, header=True, mode='w')

        # calc distance and save to file of known and unknown folders
        self.__process_leaf_folder(meansigmas_dic, "Known", self.known_data_folder, distances_file_name)
        self.__process_leaf_folder(meansigmas_dic, "Unknown", self.unknown_data_folder, distances_file_name)
