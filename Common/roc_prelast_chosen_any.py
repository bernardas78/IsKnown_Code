# Given a model and a images folders: [Known, Unknown],
#   hypothetically customer chooses each of the model's classes
#   make a ROC: IsSelected vs. IsNotSelected - by comparing distance to the selected class

from Common import common as cm
from datetime import datetime
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pandas as pd
import scipy.spatial.distance
import math
import itertools
import seaborn as sns

class Roc_prelast_chosen_any:
    def __init__(self,
                 known_data_folder,     # folder where know class images are located of structure \class\file.[jpg,...]
                 model_file):            # model file location


        self.known_data_folder = known_data_folder

        print ("Loading model {}".format(model_file) )
        self.model = load_model(model_file)
        print ("Loaded model" )

        # extract input size
        self.target_size = self.model.layers[0].input_shape[1]


    def make_roc_prelast_chosen_any(self, distances_chosen_any_file_name,
                                    roc_file_pattern,   # ROC graph (result). Can contain up to 2 placeholders {} for date/time for generation, #products:
                                    hist_file_pattern,
                                    conf_mat_file_pattern,
                                    threshold_to_use=None):
        # Read distances file
        df_distances = pd.read_csv(distances_chosen_any_file_name).fillna('')

        # Positive = class is same as selected
        y_true = (df_distances["is_selected"]).astype(np.float64)

        # Calculate sample weights:
        #   First coefficent - to balance selected vs not-selected
        correct_samples_pct = np.sum(y_true)/len(y_true)
        #   Second coefficent - 95%-99% of selections are corrent
        correct_selection_pct = 0.95
        eps = 1e-7

        sample_weight = y_true * ( 1./(correct_samples_pct+eps) *  correct_selection_pct) + \
                       (1-y_true) * ( 1./(1.-correct_samples_pct+eps) * (1-correct_selection_pct))
        #print (sample_weight)

        # Predicted = normalized distance from top 1 prediction
        # y_pred_abs = 1. - df_distances["dist_eucl"] / np.max(df_distances["dist_eucl"])
        # y_pred_mah = 1. - np.log(df_distances["dist_mahalanobis"]) / np.log(np.max(df_distances["dist_mahalanobis"]) )
        # use sigmoid
        y_pred_mah = [1. - 1. / (1 + math.exp(-np.log(curr_dist))) for curr_dist in df_distances["dist_mahalanobis"]]
        # cosine is 0-2
        y_pred_cosine = np.array( [(2. - curr_dist) for curr_dist in df_distances["dist_cosine"]] )

        # Display ROC for mahalanobis ;
        (fpr_mah, tpr_mah, thresholds_mah) = roc_curve(y_score=y_pred_mah, y_true=y_true, sample_weight=sample_weight)
        roc_auc_mah = auc(fpr_mah, tpr_mah)

        # Display ROC for cosine
        (fpr_cosine, tpr_cosine, thresholds_cosine) = roc_curve(y_score=y_pred_cosine, y_true=y_true, sample_weight=sample_weight)
        roc_auc_cosine = auc(fpr_cosine, tpr_cosine)

        # Find best accuracy
        accuracy_scores = []
        for thresh in thresholds_cosine:
            accuracy_scores.append(accuracy_score(y_true, [1 if m > thresh else 0 for m in y_pred_cosine], sample_weight=sample_weight))
        best_acc_ind = np.argmax(accuracy_scores)
        best_acc = accuracy_scores[best_acc_ind]

        if threshold_to_use is None:
            threshold_to_use = thresholds_cosine[best_acc_ind]
        #print("Threshold to use = {}".format(threshold_to_use))

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
        plt.title('{}% correct selection'.format( int(correct_selection_pct*100) ))
        plt.legend(loc="lower right")
        # plt.show()

        # Draw a point for best accuracy
        plt.plot(fpr_cosine[best_acc_ind], tpr_cosine[best_acc_ind], marker="s", color="red")
        plt.text(fpr_cosine[best_acc_ind] + 0.02, tpr_cosine[best_acc_ind] - 0.02,
                 "Best accuracy: {:.1f}% @Thr={:.5f}".format(best_acc * 100, threshold_to_use))

        # Save ROC
        roc_file_name = roc_file_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S"))
        plt.savefig(roc_file_name)
        plt.close()

        # also, histograms of correct vs. incorrect selections
        dists_selected = list ( itertools.compress(y_pred_cosine, (y_true==1.0).to_list() ) )
        dists_not_selected = list ( itertools.compress(y_pred_cosine, (y_true==0.0).to_list() ) )
        plt.hist (dists_selected, 200, alpha=0.5, label='Correct selections')
        plt.hist (dists_not_selected, 200, alpha=0.5, label='Incorrect selections')
        plt.title("Cosine similarity from selected Correct vs. Incorrect")
        plt.legend(loc='upper right')
        # plt.show()
        hist_file = hist_file_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S"))
        plt.savefig(hist_file)
        plt.close()
        print ("Hist at: {}".format(hist_file))

        # also, show conf matrix for the best threshold
        # Draw confusion matrix
        #conf_mat = confusion_matrix(y_true=y_true, y_pred=(y_pred_cosine>threshold_to_use)*1 )
        correct_weight = (1. / (correct_samples_pct+eps) * correct_selection_pct)
        incorrect_weight = (1. / (1. - correct_samples_pct+eps) * (1 - correct_selection_pct))

        print ("type(y_true): {}, type(y_pred_cosine):{}, type(threshold_to_use):{}".format(type(y_true),type(y_pred_cosine), type(threshold_to_use)))
        tp = np.sum(np.bitwise_and (y_true>1-1e-7, y_pred_cosine > threshold_to_use)) * correct_weight
        tn = np.sum(np.bitwise_and (y_true<  1e-7, y_pred_cosine < threshold_to_use)) * incorrect_weight
        fn = np.sum(np.bitwise_and (y_true>1-1e-7, y_pred_cosine < threshold_to_use)) * correct_weight
        fp = np.sum(np.bitwise_and (y_true<  1e-7, y_pred_cosine > threshold_to_use)) * incorrect_weight

        print ("tp: {}".format(np.sum(np.bitwise_and (y_true>1-1e-7, y_pred_cosine > threshold_to_use))))
        print ("tn: {}".format(np.sum(np.bitwise_and (y_true<  1e-7, y_pred_cosine < threshold_to_use))))
        print ("fn: {}".format(np.sum(np.bitwise_and (y_true>1-1e-7, y_pred_cosine < threshold_to_use))))
        print ("fp: {}".format(np.sum(np.bitwise_and (y_true<  1e-7, y_pred_cosine > threshold_to_use))))
        total = (tp+tn+fn+fp)/100

        conf_mat = np.array(
            [[tp/total,fn/total],
             [fp/total,tn/total]]
        )
        #print ("Shape: {}".format(conf_mat.shape))
        ax = sns.heatmap(conf_mat, annot=True, cbar=False, fmt='g')
        for t in ax.texts: t.set_text(t.get_text() + " %")
        #ax.set_xticks( np.arange(len(prod_names))+0.5 )
        #ax.set_yticks( np.arange(len(prod_names))+0.5 )
        plt.title ("{}% correct selections".format(int(correct_selection_pct*100) ) )
        ax.set_yticklabels(["Correct","Incorrect"] , horizontalalignment='right', rotation = 0, size=5)
        ax.set_xticklabels(["Correct","Incorrect"] , horizontalalignment='right', rotation = 90, size=5)
        ax.set_xlabel("PREDICTED", weight="bold")#, size=20)
        ax.set_ylabel("SELECTED", weight="bold")#, size=20)
        plt.tight_layout()
        plt.savefig(conf_mat_file_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S")))
        plt.close()


    # common part from roc_prelast
    #def calc_save_prelast_activations(self, prelast_activations_file_name):

    # common part from roc_prelast
    #def calc_save_meansigmas_known(self, prelast_activations_file_name, meansigmas_file_name):

    def __process_leaf_folder(self, meansigmas_dic, data_folder, distances_file_name, is_categorical):

        data_iterator = cm.get_data_iterator(data_folder, self.target_size, is_categorical=is_categorical)

        (actual,top1,prelast_activations) = cm.get_prelast_dense_activations(self.model, data_iterator, is_categorical=is_categorical)

        i=0
        for (sample_actual,sample_top1,sample_prelast_activations) in zip(actual,top1,prelast_activations):

            #Hypothetically, customer chooses each possible product
            for chosen_id in range(len(meansigmas_dic)):
                (chosen_mus, chosen_sigmas) = meansigmas_dic[chosen_id]
                #print ("top1_mus: {}, sample_prelast_activations: {}".format(top1_mus[:2], sample_prelast_activations[:2]))

                # Calculate euclidean distance and mahalandobis distance
                dist = np.sum(np.square((sample_prelast_activations - chosen_mus)))
                # How many sigmas in each dimension varies from mean? (0 sigmas are added epsilon)
                dist_mahalanobis = np.sum(np.square((sample_prelast_activations - chosen_mus) / (chosen_sigmas + 1e-7)))
                # cosine distance
                dist_cosine = scipy.spatial.distance.cosine(sample_prelast_activations, chosen_mus)

                # Result to file
                is_selected = chosen_id==sample_actual if is_categorical else 0
                df_distances = pd.DataFrame(
                    data=[np.hstack([is_selected, sample_actual, dist, dist_mahalanobis, dist_cosine])] )
                df_distances.to_csv(distances_file_name, header=None, index=None, mode='a')

            print("Processed {} files".format(i)) if i % 100 == 0 else 0
            i += 1

    def calc_save_dist_from_chosen_any (self, meansigmas_file_name, distances_file_name, is_categorical):
        # Load mean.sigmas for classes
        meansigmas_dic = pickle.load(open(meansigmas_file_name, 'rb'))

        # overwrite results file
        column_names = ['is_selected', 'actual', 'dist_eucl', 'dist_mahalanobis', 'dist_cosine']
        df_distances = pd.DataFrame(columns=column_names)
        df_distances.to_csv(distances_file_name, index=False, header=True, mode='w')

        # calc distance and save to file of known and unknown folders
        self.__process_leaf_folder(meansigmas_dic, self.known_data_folder, distances_file_name, is_categorical)
