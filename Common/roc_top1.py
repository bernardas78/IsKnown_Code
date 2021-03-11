# Given a model and a images folders: [Known, Unknown], make a ROC of known vs. unknown predictions

from Common import common as cm
from datetime import datetime
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, accuracy_score
from matplotlib import pyplot as plt
import numpy as np

class Roc_top1:
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

    def make_roc_top1(self, roc_file_pattern):  # ROC graph (result). Can contain up to 2 placeholders {} for date/time for generation, #products:
        # Prepare data generators
        known_data_iterator = cm.get_data_iterator( self.known_data_folder, self.target_size, is_categorical=True)
        unknown_data_iterator = cm.get_data_iterator( self.unknown_data_folder, self.target_size, is_categorical=False)

        # Get top 1 probabilities
        print ("Getting predictions...")
        now=datetime.now()
        preds_known = cm.get_preds_top1(self.model, known_data_iterator)
        preds_unknown = cm.get_preds_top1(self.model, unknown_data_iterator)
        print ("Predictions done in {} sec".format( (datetime.now()-now).total_seconds() ) )

        # Combine known and unknown to same vector
        y_pred = np.concatenate ( ( preds_known,                 preds_unknown ) )
        y_true = np.concatenate ( ( np.ones( len(preds_known) ), np.zeros( len(preds_unknown) )  ) )

        # Calculate ROC
        (fpr, tpr, thresholds) = roc_curve(y_score=y_pred, y_true=y_true)
        roc_auc = auc(fpr, tpr)

        # Find best accuracy
        accuracy_scores = []
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(y_true, [1 if m > thresh else 0 for m in y_pred]))
        best_acc_ind = np.argmax(accuracy_scores)
        best_acc = accuracy_scores[best_acc_ind]
        threshhold_to_use = thresholds[best_acc_ind]
        print("Threshold to use = {}".format(threshhold_to_use))

        # Draw ROC
        plt.figure()
        plt.plot(fpr, tpr, color='green', lw=2, label='ROC AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate') #('Bandoma atpažinti %, kai nežinoma prekė')
        plt.ylabel('True Positive Rate') #('Bandoma atpažinti %, kai žinoma prekė')
        # cnt_class = len ( np.unique(df_distances["actual"]) ) - 1 #1-unknown class
        # samples_known = len(np.where (df_distances["actual"]!="")[0])
        # samples_unknown = len(np.where (df_distances["actual"]=="")[0])
        # plt.title('{} žinomos klasės; {} žinomų prekių, {} nežinomų'.format (cnt_class, samples_known, samples_unknown ) )
        plt.legend(loc="lower right")

        # Draw a point for best accuracy
        plt.plot(fpr[best_acc_ind], tpr[best_acc_ind], marker="s", color="red")
        plt.text(fpr[best_acc_ind] + 0.02, tpr[best_acc_ind] - 0.02,
                 "Best accuracy: {:.1f}%".format(best_acc * 100))

        # Save
        roc_file = roc_file_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S"),
                                           known_data_iterator.num_classes)
        plt.savefig(roc_file)

        print ("ROC at: {}".format(roc_file))
        return
